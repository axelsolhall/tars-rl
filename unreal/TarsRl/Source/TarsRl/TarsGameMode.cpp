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

double ATarsGameMode::FindOrMakeFloor() const
{
	UWorld* World = GetWorld();
	// Floor height under the origin; spawn a 100 m plain floor at z = 0 if nothing is there.
	double FloorZ = 0;
	FHitResult Hit;
	if (World->LineTraceSingleByChannel(Hit, FVector(0, 0, 10000), FVector(0, 0, -10000), ECC_WorldStatic))
	{
		FloorZ = Hit.ImpactPoint.Z;
	}
	else
	{
		AStaticMeshActor* Floor = World->SpawnActor<AStaticMeshActor>(FVector(0, 0, -50), FRotator::ZeroRotator);
		UStaticMesh* Cube = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Cube.Cube"));
		Floor->GetStaticMeshComponent()->SetMobility(EComponentMobility::Movable);  // runtime mesh set
		Floor->GetStaticMeshComponent()->SetStaticMesh(Cube);
		Floor->SetActorScale3D(FVector(100, 100, 1));
	}
	return FloorZ;
}

void ATarsGameMode::StartBridge(double FloorZ)
{
	ATarsBridge* Bridge = GetWorld()->SpawnActorDeferred<ATarsBridge>(ATarsBridge::StaticClass(),
		FTransform(FVector(0, 0, FloorZ)));
	FParse::Value(FCommandLine::Get(), TEXT("TarsPort="), Bridge->Port);
	FParse::Value(FCommandLine::Get(), TEXT("TarsDt="), Bridge->ControlDt);
	Bridge->Origin = FVector(0, 0, FloorZ);
	// Lockstep: every frame advances exactly one control step and never sleeps to match the wall
	// clock, so Python sets the pace (watching: real time; evaluating: as fast as it goes).
	GEngine->bUseFixedFrameRate = false;
	FApp::SetUseFixedTimeStep(true);
	FApp::SetFixedDeltaTime(Bridge->ControlDt);
	Bridge->FinishSpawning(FTransform(FVector(0, 0, FloorZ)));
}

void ATarsGameMode::StartSequence(double FloorZ)
{
	UWorld* World = GetWorld();

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
