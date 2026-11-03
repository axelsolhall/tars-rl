#include "TarsLevel.h"

#include "Components/StaticMeshComponent.h"
#include "Dom/JsonObject.h"
#include "Engine/CollisionProfile.h"
#include "Engine/StaticMesh.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "PhysicalMaterials/PhysicalMaterial.h"
#include "UObject/ConstructorHelpers.h"

namespace
{
FVector Vec(const TSharedPtr<FJsonObject>& O, const TCHAR* Key)
{
	const TArray<TSharedPtr<FJsonValue>>& A = O->GetArrayField(Key);
	return FVector(A[0]->AsNumber(), A[1]->AsNumber(), A[2]->AsNumber());
}
}  // namespace

ATarsLevel::ATarsLevel()
{
	PrimaryActorTick.bCanEverTick = false;
	RootComponent = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
	static ConstructorHelpers::FObjectFinder<UStaticMesh> Cube(TEXT("/Engine/BasicShapes/Cube.Cube"));
	CubeMesh = Cube.Object;  // 100 cm cube, pivot at its centre
	static ConstructorHelpers::FObjectFinder<UMaterialInterface> Mat(
		TEXT("/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial"));
	BaseMaterial = Mat.Object;  // has a "Color" vector parameter
}

UMaterialInstanceDynamic* ATarsLevel::ColourMaterial(const FLinearColor& Colour)
{
	const FString Key = Colour.ToString();
	if (TObjectPtr<UMaterialInstanceDynamic>* Found = Colours.Find(Key)) return *Found;
	UMaterialInstanceDynamic* M = UMaterialInstanceDynamic::Create(BaseMaterial, this);
	M->SetVectorParameterValue(TEXT("Color"), Colour);
	Colours.Add(Key, M);
	return M;
}

UPhysicalMaterial* ATarsLevel::FrictionMaterial(double Friction)
{
	const FString Key = FString::Printf(TEXT("%.4f"), Friction);
	if (TObjectPtr<UPhysicalMaterial>* Found = Frictions.Find(Key)) return *Found;
	UPhysicalMaterial* M = NewObject<UPhysicalMaterial>(this);
	M->Friction = M->StaticFriction = static_cast<float>(Friction);
	M->bOverrideFrictionCombineMode = true;
	M->FrictionCombineMode = EFrictionCombineMode::Max;  // MuJoCo takes the max too
	M->Restitution = 0.f;
	Frictions.Add(Key, M);
	return M;
}

UStaticMeshComponent* ATarsLevel::MakeMesh(FName Name, bool bCollide)
{
	UStaticMeshComponent* B = NewObject<UStaticMeshComponent>(this, Name);
	B->SetStaticMesh(CubeMesh);
	B->SetMobility(EComponentMobility::Movable);  // created at runtime; never simulated
	B->SetupAttachment(RootComponent);
	B->RegisterComponent();
	if (bCollide)
	{
		B->SetCollisionProfileName(UCollisionProfile::BlockAll_ProfileName);
	}
	else
	{
		B->SetCollisionEnabled(ECollisionEnabled::NoCollision);
	}
	return B;
}

int32 ATarsLevel::Build(const TArray<TSharedPtr<FJsonValue>>& Json, FString& Error)
{
	for (UStaticMeshComponent* B : Boxes)
		if (B) B->DestroyComponent();
	Boxes.Reset();
	for (const TSharedPtr<FJsonValue>& V : Json)
	{
		const TSharedPtr<FJsonObject> O = V->AsObject();
		if (!O.IsValid() || !O->HasField(TEXT("p")) || !O->HasField(TEXT("q")) || !O->HasField(TEXT("s")))
		{
			Error = TEXT("level box needs p, q, s");
			return -1;
		}
		const TArray<TSharedPtr<FJsonValue>>& Q = O->GetArrayField(TEXT("q"));
		const FQuat Rot(Q[0]->AsNumber(), Q[1]->AsNumber(), Q[2]->AsNumber(), Q[3]->AsNumber());
		UStaticMeshComponent* B = MakeMesh(*FString::Printf(TEXT("box_%d"), NextId++), true);
		B->SetRelativeLocationAndRotation(Vec(O, TEXT("p")), Rot.GetNormalized());
		B->SetRelativeScale3D(Vec(O, TEXT("s")) / 100.0);
		B->SetPhysMaterialOverride(FrictionMaterial(O->GetNumberField(TEXT("mu"))));
		const FVector Rgb = O->HasField(TEXT("rgb")) ? Vec(O, TEXT("rgb")) : FVector(0.5);
		B->SetMaterial(0, ColourMaterial(FLinearColor(Rgb.X, Rgb.Y, Rgb.Z)));
		Boxes.Add(B);
	}
	return Boxes.Num();
}

void ATarsLevel::SetFlag(const FVector& PosCm, double BodyHeightCm)
{
	// Same marker as the MuJoCo arena (tars_rl.env.arena): a 1.2-body-height pole, a checker-ish
	// cloth at its top on MuJoCo's +y side (Unreal -y). Visual only.
	const double PoleH = 1.2 * BodyHeightCm, PoleR = 0.6;
	const FVector Cloth(0.4, 0.36 * BodyHeightCm, 0.24 * BodyHeightCm);  // full size
	if (!FlagPole)
	{
		FlagPole = MakeMesh(TEXT("flag_pole"), false);
		FlagPole->SetMaterial(0, ColourMaterial(FLinearColor(0.85f, 0.85f, 0.85f)));
		FlagCloth = MakeMesh(TEXT("flag_cloth"), false);
		FlagCloth->SetMaterial(0, ColourMaterial(FLinearColor(0.9f, 0.1f, 0.1f)));
	}
	FlagPole->SetRelativeScale3D(FVector(2 * PoleR, 2 * PoleR, PoleH) / 100.0);
	FlagPole->SetRelativeLocation(PosCm + FVector(0, 0, PoleH / 2));
	FlagCloth->SetRelativeScale3D(Cloth / 100.0);
	FlagCloth->SetRelativeLocation(PosCm + FVector(0, -(Cloth.Y / 2 + PoleR), PoleH - Cloth.Z / 2));
}
