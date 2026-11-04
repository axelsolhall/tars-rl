// Two ways to play the TarsRl project, picked on the command line:
//   bridge (default)       a Python policy drives TARS through ATarsBridge on -TarsPort=5577,
//                          in -TarsDt=0.02 s steps of fixed 2 ms frames that never wait for the clock
//   -TarsMode=sequence     T10: TARS plays the scripted sequence and records its poses
// Both find the floor of the map (or lay a plain one) and use it as the level origin.
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/GameModeBase.h"
#include "TarsGameMode.generated.h"

UCLASS()
class TARSRL_API ATarsGameMode : public AGameModeBase
{
	GENERATED_BODY()

public:
	ATarsGameMode();
	virtual void StartPlay() override;

private:
	double FindOrMakeFloor();
	void StartSequence(double FloorZ);
	void StartBridge(double FloorZ);
	static void UseFixedPhysicsFrames(double PhysicsDt);

	UPROPERTY() TObjectPtr<class UPhysicalMaterial> FloorMaterial;
};
