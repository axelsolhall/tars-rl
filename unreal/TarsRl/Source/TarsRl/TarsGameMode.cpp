#include "TarsGameMode.h"

#include "Components/StaticMeshComponent.h"
#include "Engine/StaticMesh.h"
#include "Engine/StaticMeshActor.h"
#include "Engine/World.h"
#include "EngineUtils.h"
#include "GameFramework/PlayerController.h"
#include "GameFramework/SpectatorPawn.h"
#include "TarsActor.h"
#include "TarsBridge.h"
#include "Misc/App.h"
#include "Misc/CommandLine.h"
#include "Misc/Parse.h"
#include "Engine/Engine.h"
#include "PhysicalMaterials/PhysicalMaterial.h"

DEFINE_LOG_CATEGORY_STATIC(LogTarsGameMode, Log, All);

ATarsGameMode::ATarsGameMode()
{
	DefaultPawnClass = ASpectatorPawn::StaticClass();
}

void ATarsGameMode::StartPlay()
{
	Super::StartPlay();
	FString Mode = TEXT("bridge");
	FParse::Value(FCommandLine::Get(), TEXT("TarsMode="), Mode);
	const double FloorZ = FindOrMakeFloor();
	if (Mode == TEXT("sequence")) StartSequence(FloorZ);
	else StartBridge(FloorZ);
}

double ATarsGameMode::FindOrMakeFloor()
{
	UWorld* World = GetWorld();
	// The floor's own contact properties must not count, as MuJoCo's plane only has the body's
	// friction: friction 0 (TARS's slabs combine by max, so theirs wins) and no bounce at all.
	FloorMaterial = NewObject<UPhysicalMaterial>(this);
	FloorMaterial->Friction = FloorMaterial->StaticFriction = 0.f;
	FloorMaterial->Restitution = 0.f;
	FloorMaterial->bOverrideRestitutionCombineMode = true;
	FloorMaterial->RestitutionCombineMode = EFrictionCombineMode::Min;
	// Floor height under the origin; spawn a 100 m plain floor at z = 0 if nothing is there.
	// Static world objects only: a trace on the WorldStatic *channel* also hits the spectator pawn
	// (it blocks that channel), which put TARS 1.2 m up in the air.
	double FloorZ = 0;
	FHitResult Hit;
	if (World->LineTraceSingleByObjectType(Hit, FVector(0, 0, 10000), FVector(0, 0, -10000),
		FCollisionObjectQueryParams(ECC_WorldStatic)))
	{
		FloorZ = Hit.ImpactPoint.Z;
		if (UPrimitiveComponent* C = Hit.GetComponent()) C->SetPhysMaterialOverride(FloorMaterial);
		UE_LOG(LogTarsGameMode, Log, TEXT("floor at z %.2f cm (%s)"), FloorZ, *GetNameSafe(Hit.GetActor()));
	}
	else
	{
		AStaticMeshActor* Floor = World->SpawnActor<AStaticMeshActor>(FVector(0, 0, -50), FRotator::ZeroRotator);
		UStaticMesh* Cube = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Cube.Cube"));
		Floor->GetStaticMeshComponent()->SetMobility(EComponentMobility::Movable);  // runtime mesh set
		Floor->GetStaticMeshComponent()->SetStaticMesh(Cube);
		Floor->SetActorScale3D(FVector(100, 100, 1));
		Floor->GetStaticMeshComponent()->SetPhysMaterialOverride(FloorMaterial);
		UE_LOG(LogTarsGameMode, Log, TEXT("no floor in the map: laid a plain one at z 0"));
	}
	return FloorZ;
}

void ATarsGameMode::StartBridge(double FloorZ)
{
	ATarsBridge* Bridge = GetWorld()->SpawnActorDeferred<ATarsBridge>(ATarsBridge::StaticClass(),
		FTransform(FVector(0, 0, FloorZ)));
	FParse::Value(FCommandLine::Get(), TEXT("TarsPort="), Bridge->Port);
	FParse::Value(FCommandLine::Get(), TEXT("TarsDt="), Bridge->ControlDt);
	FParse::Value(FCommandLine::Get(), TEXT("TarsPhysicsDt="), Bridge->PhysicsDt);
	Bridge->Origin = FVector(0, 0, FloorZ);
	// Lockstep: every frame advances exactly one physics step and never sleeps to match the wall
	// clock, so Python sets the pace (watching: real time; evaluating: as fast as it goes).
	UseFixedPhysicsFrames(Bridge->PhysicsDt);
	Bridge->FinishSpawning(FTransform(FVector(0, 0, FloorZ)));
}

void ATarsGameMode::UseFixedPhysicsFrames(double PhysicsDt)
{
	// One frame = one physics step of PhysicsDt on simulated time; substepping is off in
	// DefaultEngine.ini, so ATarsActor::ServoStep runs before every step.
	GEngine->bUseFixedFrameRate = false;
	FApp::SetUseFixedTimeStep(true);
	FApp::SetFixedDeltaTime(PhysicsDt);
}

void ATarsGameMode::StartSequence(double FloorZ)
{
	UWorld* World = GetWorld();
	UseFixedPhysicsFrames(0.002);  // the BodySpec timestep, as the bridge's default

	ATarsActor* Tars = nullptr;
	for (TActorIterator<ATarsActor> It(World); It; ++It) Tars = *It;
	if (!Tars) Tars = World->SpawnActor<ATarsActor>(FVector(0, 0, FloorZ), FRotator::ZeroRotator);

	if (APlayerController* PC = World->GetFirstPlayerController())
	{
		if (APawn* Pawn = PC->GetPawn())
		{
			// 1.2 m in front of TARS (it faces +x), a little above its top, looking back at it.
			Pawn->SetActorLocation(Tars->GetActorLocation() + FVector(120, 40, 45));
			PC->SetControlRotation(FRotator(-15, 198, 0));
		}
	}
}
