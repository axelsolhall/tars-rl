#include "TarsActor.h"

#include "Components/StaticMeshComponent.h"
#include "Dom/JsonObject.h"
#include "Engine/CollisionProfile.h"
#include "Engine/Engine.h"
#include "Engine/StaticMesh.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "PhysicalMaterials/PhysicalMaterial.h"
#include "PhysicsEngine/PhysicsConstraintComponent.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "UObject/ConstructorHelpers.h"

DEFINE_LOG_CATEGORY_STATIC(LogTars, Log, All);

namespace
{
constexpr double M2CM = 100.0;
constexpr double NM2UE = 1.0e4;  // torque: N m = kg m^2/s^2 -> kg cm^2/s^2
constexpr double FallTiltDeg = 45.0;  // same fall line as tars_rl.unreal.compare
constexpr int32 RecordEvery = 10;  // scripted mode: one pose row per 10 physics steps (50 Hz)
const TCHAR* Sides[2] = {TEXT("left"), TEXT("right")};

// MuJoCo's <position kp kv forcerange>: SI in, SI out.
double ServoForce(const FTarsServo& S, double Target, double Q, double Qd)
{
	return FMath::Clamp(S.Kp * (Target - Q) - S.Kv * Qd, -S.ForceLimit, S.ForceLimit);
}

bool ReadJson(const FString& Path, TSharedPtr<FJsonObject>& Out)
{
	FString Text;
	if (!FFileHelper::LoadFileToString(Text, *Path))
	{
		UE_LOG(LogTars, Error, TEXT("cannot read %s (run python -m tars_rl.unreal.export)"), *Path);
		return false;
	}
	const TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Text);
	if (!FJsonSerializer::Deserialize(Reader, Out) || !Out.IsValid())
	{
		UE_LOG(LogTars, Error, TEXT("bad JSON in %s"), *Path);
		return false;
	}
	return true;
}

FVector BoxSize(const TSharedPtr<FJsonObject>& B)
{
	return FVector(B->GetNumberField(TEXT("depth")), B->GetNumberField(TEXT("width")),
		B->GetNumberField(TEXT("height")));
}

FVector2D Range(const TSharedPtr<FJsonObject>& O, const TCHAR* Key)
{
	const TArray<TSharedPtr<FJsonValue>>& A = O->GetArrayField(Key);
	return FVector2D(A[0]->AsNumber(), A[1]->AsNumber());
}

FTarsServo Servo(const TSharedPtr<FJsonObject>& S)
{
	FTarsServo Out;
	Out.Kp = S->GetNumberField(TEXT("kp"));
	Out.Kv = S->GetNumberField(TEXT("kv"));
	Out.ForceLimit = S->GetNumberField(TEXT("force_limit"));
	return Out;
}

TArray<FString> Strings(const TSharedPtr<FJsonObject>& O, const TCHAR* Key)
{
	TArray<FString> Out;
	for (const TSharedPtr<FJsonValue>& V : O->GetArrayField(Key)) Out.Add(V->AsString());
	return Out;
}
}  // namespace

ATarsActor::ATarsActor()
{
	PrimaryActorTick.bCanEverTick = true;
	// Before physics: record the pose the last physics step produced, then set new targets.
	PrimaryActorTick.TickGroup = TG_PrePhysics;
	RootComponent = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
	static ConstructorHelpers::FObjectFinder<UStaticMesh> Cube(TEXT("/Engine/BasicShapes/Cube.Cube"));
	CubeMesh = Cube.Object;  // 100 cm cube, pivot at its centre
}

bool ATarsActor::LoadSpec()
{
	TSharedPtr<FJsonObject> J;
	if (!ReadJson(FPaths::ProjectContentDir() / SpecFile, J)) return false;
	if (J->GetIntegerField(TEXT("version")) != 1)
	{
		UE_LOG(LogTars, Error, TEXT("BodySpec version %d, this actor reads 1"),
			J->GetIntegerField(TEXT("version")));
		return false;
	}
	const TSharedPtr<FJsonObject> C = J->GetObjectField(TEXT("central"));
	const TSharedPtr<FJsonObject> A = J->GetObjectField(TEXT("arm"));
	Spec.CentralSizeM = BoxSize(C);
	Spec.ArmSizeM = BoxSize(A);
	Spec.CentralMass = C->GetNumberField(TEXT("mass"));
	Spec.ArmMass = A->GetNumberField(TEXT("mass"));
	Spec.ArmGap = J->GetNumberField(TEXT("arm_gap"));
	Spec.Friction = J->GetNumberField(TEXT("friction"));
	Spec.HingeRange = Range(J, TEXT("hinge_range"));
	Spec.SlideRange = Range(J, TEXT("slide_range"));
	Spec.HingeServo = Servo(J->GetObjectField(TEXT("hinge_servo")));
	Spec.SlideServo = Servo(J->GetObjectField(TEXT("slide_servo")));
	Spec.ActuatorNames = Strings(J, TEXT("actuator_names"));
	const TArray<FString> Expected = {TEXT("left_hinge_act"), TEXT("right_hinge_act"),
		TEXT("left_slide_act"), TEXT("right_slide_act")};
	if (Spec.ActuatorNames != Expected)
	{
		UE_LOG(LogTars, Error, TEXT("BodySpec actuator order changed; update TarsActor"));
		return false;
	}
	return true;
}

bool ATarsActor::LoadSequence()
{
	TSharedPtr<FJsonObject> J;
	if (!ReadJson(FPaths::ProjectContentDir() / SequenceFile, J)) return false;
	if (Strings(J, TEXT("actuator_names")) != Spec.ActuatorNames)
	{
		UE_LOG(LogTars, Error, TEXT("sequence actuators differ from the BodySpec"));
		return false;
	}
	for (const TSharedPtr<FJsonValue>& V : J->GetArrayField(TEXT("keyframes")))
	{
		const TSharedPtr<FJsonObject> K = V->AsObject();
		KeyTimes.Add(K->GetNumberField(TEXT("t")));
		TArray<double> Row;
		for (const TSharedPtr<FJsonValue>& X : K->GetArrayField(TEXT("targets"))) Row.Add(X->AsNumber());
		KeyTargets.Add(Row);
	}
	return KeyTimes.Num() > 0;
}

UStaticMeshComponent* ATarsActor::MakeBox(FName Name, const FVector& SizeCm, float MassKg,
	const FVector& LocalCm, bool bCollide)
{
	UStaticMeshComponent* B = NewObject<UStaticMeshComponent>(this, Name);
	B->SetStaticMesh(CubeMesh);
	B->SetupAttachment(RootComponent);
	B->SetRelativeLocation(LocalCm);
	B->SetRelativeScale3D(SizeCm / 100.0);
	B->RegisterComponent();
	if (bCollide)
	{
		B->SetCollisionProfileName(UCollisionProfile::PhysicsActor_ProfileName);
		B->SetPhysMaterialOverride(SlabMaterial);
	}
	else
	{
		B->SetCollisionEnabled(ECollisionEnabled::PhysicsOnly);
		B->SetCollisionResponseToAllChannels(ECR_Ignore);
		B->SetVisibility(false);
	}
	B->SetLinearDamping(0.f);  // MuJoCo has no body damping
	B->SetAngularDamping(0.f);
	B->SetSimulatePhysics(true);
	B->SetMassOverrideInKg(NAME_None, MassKg, true);
	return B;
}

UPhysicsConstraintComponent* ATarsActor::MakeConstraint(FName Name, const FVector& LocalCm,
	const FRotator& LocalRot)
{
	UPhysicsConstraintComponent* K = NewObject<UPhysicsConstraintComponent>(this, Name);
	K->SetupAttachment(RootComponent);
	K->SetRelativeLocationAndRotation(LocalCm, LocalRot);
	K->RegisterComponent();
	K->SetLinearXLimit(LCM_Locked, 0.f);
	K->SetLinearYLimit(LCM_Locked, 0.f);
	K->SetLinearZLimit(LCM_Locked, 0.f);
	K->SetAngularSwing1Limit(ACM_Locked, 0.f);
	K->SetAngularSwing2Limit(ACM_Locked, 0.f);
	K->SetAngularTwistLimit(ACM_Locked, 0.f);
	K->SetDisableCollision(true);
	return K;
}

void ATarsActor::BuildBody()
{
	SlabMaterial = NewObject<UPhysicalMaterial>(this);
	SlabMaterial->Friction = Spec.Friction;
	SlabMaterial->StaticFriction = Spec.Friction;
	SlabMaterial->bOverrideFrictionCombineMode = true;
	SlabMaterial->FrictionCombineMode = EFrictionCombineMode::Max;  // MuJoCo takes the max too
	SlabMaterial->Restitution = 0.f;  // MuJoCo contacts never bounce
	SlabMaterial->bOverrideRestitutionCombineMode = true;
	SlabMaterial->RestitutionCombineMode = EFrictionCombineMode::Min;

	// Unreal: cm, x forward, y RIGHT. MuJoCo's left (+y) is Unreal's -y (A73).
	const double HingeZ = Spec.CentralSizeM.Z / 2 * M2CM;
	const double ArmY = (Spec.CentralSizeM.Y / 2 + Spec.ArmGap + Spec.ArmSizeM.Y / 2) * M2CM;
	const FVector Centre(0, 0, HingeZ);
	Central = MakeBox(TEXT("central"), Spec.CentralSizeM * M2CM, Spec.CentralMass, Centre, true);

	const float HingeLimitDeg = static_cast<float>(FMath::RadiansToDegrees(
		FMath::Max(FMath::Abs(Spec.HingeRange.X), FMath::Abs(Spec.HingeRange.Y))));
	const float SlideLimitCm = static_cast<float>(
		FMath::Max(FMath::Abs(Spec.SlideRange.X), FMath::Abs(Spec.SlideRange.Y)) * M2CM);

	for (int32 I = 0; I < 2; ++I)
	{
		const FString Side = Sides[I];
		const FVector ArmPos(0, (I == 0 ? -1 : 1) * ArmY, HingeZ);
		UStaticMeshComponent* Carriage = MakeBox(*(Side + TEXT("_carriage")), FVector(2, 2, 2),
			CarriageMassKg, Centre, false);
		UStaticMeshComponent* Arm = MakeBox(*(Side + TEXT("_arm")), Spec.ArmSizeM * M2CM,
			Spec.ArmMass, ArmPos, true);

		// Hinge: twist is the constraint's x axis; yaw 90 turns it onto the body y axis.
		UPhysicsConstraintComponent* Hinge = MakeConstraint(*(Side + TEXT("_hinge")), Centre,
			FRotator(0, 90, 0));
		Hinge->SetAngularTwistLimit(ACM_Limited, HingeLimitDeg);  // no drive: ServoStep turns it
		Hinge->SetConstrainedComponents(Carriage, NAME_None, Central, NAME_None);

		// Slide: along the arm's length (constraint z), carried round by the hinge.
		UPhysicsConstraintComponent* Slide = MakeConstraint(*(Side + TEXT("_slide")), ArmPos,
			FRotator::ZeroRotator);
		Slide->SetLinearZLimit(LCM_Limited, SlideLimitCm);  // no drive: ServoStep moves it
		Slide->SetConstrainedComponents(Arm, NAME_None, Carriage, NAME_None);

		// Central and arm overlap in no pose MuJoCo checks, but MuJoCo excludes their contact.
		UPhysicsConstraintComponent* Free = MakeConstraint(*(Side + TEXT("_nocollide")), ArmPos,
			FRotator::ZeroRotator);
		Free->SetLinearXLimit(LCM_Free, 0.f);
		Free->SetLinearYLimit(LCM_Free, 0.f);
		Free->SetLinearZLimit(LCM_Free, 0.f);
		Free->SetAngularSwing1Limit(ACM_Free, 0.f);
		Free->SetAngularSwing2Limit(ACM_Free, 0.f);
		Free->SetAngularTwistLimit(ACM_Free, 0.f);
		Free->SetConstrainedComponents(Arm, NAME_None, Central, NAME_None);

		Arms.Add(Arm);
		Carriages.Add(Carriage);
		Hinges.Add(Hinge);
		Slides.Add(Slide);
		NoCollide.Add(Free);
	}
}

TArray<double> ATarsActor::TargetsAt(double T) const
{
	// Same rule as tars_rl.unreal.sequence.Sequence.at: linear between keyframes, held outside.
	const int32 N = KeyTimes.Num();
	if (T <= KeyTimes[0]) return KeyTargets[0];
	if (T >= KeyTimes[N - 1]) return KeyTargets[N - 1];
	int32 K = 1;
	while (KeyTimes[K] < T) ++K;
	const double A = (T - KeyTimes[K - 1]) / (KeyTimes[K] - KeyTimes[K - 1]);
	TArray<double> Out;
	for (int32 J = 0; J < KeyTargets[K].Num(); ++J)
		Out.Add(FMath::Lerp(KeyTargets[K - 1][J], KeyTargets[K][J], A));
	return Out;
}

void ATarsActor::ApplyTargets(const TArray<double>& InTargets)
{
	// Actuator order: left_hinge, right_hinge, left_slide, right_slide.
	Targets = {HingeSign * InTargets[0], HingeSign * InTargets[1], SlideSign * InTargets[2],
		SlideSign * InTargets[3]};
}

void ATarsActor::ServoStep()
{
	// Joint state from the slab states, as tars_rl.unreal.bodystate.joint_state reads it: the
	// hinge is the arm's twist about the central body's y axis, the slide the arm centre's offset
	// from the hinge point along the arm's z. Unreal units in, SI for the servo.
	const FQuat Qc = Central->GetComponentQuat();
	const FVector Xc = Central->GetComponentLocation();
	const FVector Vc = Central->GetPhysicsLinearVelocity();  // the slab's centre is its centre of mass
	const FVector Wc = Central->GetPhysicsAngularVelocityInRadians();
	const FVector Axis = Qc.GetAxisY();
	const double ArmY = (Spec.CentralSizeM.Y / 2 + Spec.ArmGap + Spec.ArmSizeM.Y / 2) * M2CM;
	for (int32 I = 0; I < 2; ++I)
	{
		UStaticMeshComponent* Arm = Arms[I];
		const FQuat Qa = Arm->GetComponentQuat();
		const FVector Xa = Arm->GetComponentLocation();
		const FVector Wa = Arm->GetPhysicsAngularVelocityInRadians();
		const double Hinge = (Qc.Inverse() * Qa).GetTwistAngle(FVector::YAxisVector);
		const double HingeRate = FVector::DotProduct(Wa - Wc, Axis);
		const FVector Pivot = Xc + Qc.RotateVector(FVector(0, (I == 0 ? -1 : 1) * ArmY, 0));
		const FVector ArmZ = Qa.GetAxisZ();
		const FVector PivotVel = Vc + FVector::CrossProduct(Wc, Pivot - Xc);
		const double Slide = FVector::DotProduct(Xa - Pivot, ArmZ) / M2CM;
		const double SlideRate = FVector::DotProduct(Arm->GetPhysicsLinearVelocity() - PivotVel, ArmZ) / M2CM;

		const double Tau = ServoForce(Spec.HingeServo, Targets[I], Hinge, HingeRate) * NM2UE;
		const double F = ServoForce(Spec.SlideServo, Targets[2 + I], Slide, SlideRate) * M2CM;
		// Equal and opposite, so the servos move TARS only by pushing on the world.
		Arm->AddTorqueInRadians(Axis * Tau);
		Central->AddTorqueInRadians(-Axis * Tau);
		Arm->AddForce(ArmZ * F);
		Central->AddForceAtLocation(-ArmZ * F, Xa);
	}
}

void ATarsActor::RecordSample()
{
	FString Row = FString::Printf(TEXT("%.6f"), SimTime);
	for (UStaticMeshComponent* B : {Central.Get(), Arms[0].Get(), Arms[1].Get()})
	{
		const FVector P = B->GetComponentLocation() - GetActorLocation();  // floor point = origin
		const FQuat Q = B->GetComponentQuat();
		Row += FString::Printf(TEXT(",%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f"),
			P.X, P.Y, P.Z, Q.X, Q.Y, Q.Z, Q.W);
	}
	Lines.Add(Row);
	const double Tilt = FMath::RadiansToDegrees(
		FMath::Acos(FMath::Clamp(Central->GetUpVector().Z, -1.0, 1.0)));
	if (StandTime < 0 && Tilt > FallTiltDeg) StandTime = SimTime;
	if (GEngine)
	{
		GEngine->AddOnScreenDebugMessage(1, 1.f, FColor::White, FString::Printf(
			TEXT("TARS  t %.1f / %.1f s   tilt %.1f deg   %s"), SimTime, KeyTimes.Last(), Tilt,
			StandTime < 0 ? TEXT("standing") : *FString::Printf(TEXT("fell at %.2f s"), StandTime)));
	}
}

void ATarsActor::SaveRecording()
{
	if (bSaved || Lines.Num() == 0) return;
	FString Header = TEXT("t");
	for (const TCHAR* B : {TEXT("central"), TEXT("left_arm"), TEXT("right_arm")})
		for (const TCHAR* K : {TEXT("px"), TEXT("py"), TEXT("pz"), TEXT("qx"), TEXT("qy"), TEXT("qz"), TEXT("qw")})
			Header += FString::Printf(TEXT(",%s_%s"), B, K);
	const FString Path = FPaths::ProjectSavedDir() / RecordFile;
	FFileHelper::SaveStringToFile(Header + TEXT("\n") + FString::Join(Lines, TEXT("\n")) + TEXT("\n"), *Path);
	UE_LOG(LogTars, Log, TEXT("recorded %d samples to %s"), Lines.Num(), *FPaths::ConvertRelativePathToFull(Path));
	if (GEngine)
		GEngine->AddOnScreenDebugMessage(2, 30.f, FColor::Green, TEXT("Recorded ") + FPaths::ConvertRelativePathToFull(Path));
	bSaved = true;
}

TArray<FTarsSlabState> ATarsActor::ReadSlabs() const
{
	TArray<FTarsSlabState> Out;
	// Non-const: GetPhysicsLinearVelocityAtPoint is not a const member.
	for (UStaticMeshComponent* B : {Central.Get(), Arms[0].Get(), Arms[1].Get()})
	{
		FTarsSlabState S;
		S.Pos = B->GetComponentLocation();
		S.Rot = B->GetComponentQuat();
		// At the slab's centre, not its centre of mass: MuJoCo reports body-origin velocities.
		S.Vel = B->GetPhysicsLinearVelocityAtPoint(S.Pos);
		S.AngVel = B->GetPhysicsAngularVelocityInRadians();
		Out.Add(S);
	}
	return Out;
}

FVector ATarsActor::CentralLocation() const
{
	return Central ? Central->GetComponentLocation() : GetActorLocation();
}

void ATarsActor::BeginPlay()
{
	Super::BeginPlay();
	if (!bScripted)
	{
		SetActorTickEnabled(false);  // the bridge drives it
		bReady = CubeMesh && LoadSpec();
		if (bReady)
		{
			BuildBody();
			ApplyTargets({0.0, 0.0, 0.0, 0.0});  // neutral: every joint at 0 (checked in Python)
		}
		return;
	}
	bReady = CubeMesh && LoadSpec() && LoadSequence();
	if (!bReady)
	{
		if (GEngine) GEngine->AddOnScreenDebugMessage(1, 60.f, FColor::Red, TEXT("TARS: spec or sequence failed to load, see Output Log"));
		return;
	}
	BuildBody();
	ApplyTargets(TargetsAt(0.0));
}

void ATarsActor::Tick(float DeltaSeconds)
{
	Super::Tick(DeltaSeconds);
	if (!bReady) return;
	if (!bSaved && Frames++ % RecordEvery == 0) RecordSample();
	if (SimTime >= KeyTimes.Last()) SaveRecording();
	// The coming physics step runs from SimTime: target its start, like MuJoCo's seq.at(d.time).
	ApplyTargets(TargetsAt(SimTime));
	ServoStep();
	SimTime += DeltaSeconds;
}

void ATarsActor::EndPlay(const EEndPlayReason::Type Reason)
{
	if (bScripted) SaveRecording();  // a run stopped early is still worth comparing
	Super::EndPlay(Reason);
}
