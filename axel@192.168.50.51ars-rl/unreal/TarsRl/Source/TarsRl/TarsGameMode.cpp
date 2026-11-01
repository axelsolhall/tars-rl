#include "TarsGameMode.h"

#include "Components/StaticMeshComponent.h"
#include "Engine/StaticMesh.h"
#include "Engine/StaticMeshActor.h"
#include "Engine/World.h"
#include "EngineUtils.h"
#include "GameFramework/PlayerController.h"
#include "GameFramework/SpectatorPawn.h"
#include "TarsActor.h"

ATarsGameMode::ATarsGameMode()
{
	DefaultPawnClass = ASpectatorPawn::StaticClass();
}

void ATarsGameMode::StartPlay()
{
	Super::StartPlay();
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
