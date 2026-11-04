// The Unreal side of the Python bridge (tars_rl.unreal.bridge holds the protocol): a TCP server
// on 127.0.0.1:TarsPort that lets a Python policy drive TARS in lockstep.
//
// One engine frame is one physics step (PhysicsDt, 2 ms, no substeps) so that TARS's servos act
// before every step (ATarsActor::ServoStep). A `step` runs ControlDt / PhysicsDt frames (10):
// before each the bridge calls ServoStep, after the last it answers with the state physics
// produced, then blocks on the socket until the next `step` arrives (answering `hello`, `level`,
// `reset` and `flag` immediately in between). So the world advances one fixed control step
// (20 ms) per `step`, and only as fast as Python asks: paced to the wall clock to watch, or flat
// out to evaluate. Without a client it idles at about real time.
//
// The frame time must be fixed: ATarsGameMode turns on Unreal's fixed time step, and every state
// reply carries the step's dt so Python can refuse anything else.
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "TarsBridge.generated.h"

class ATarsActor;
class ATarsLevel;
class ACameraActor;
class FSocket;
class FJsonObject;

UCLASS()
class TARSRL_API ATarsBridge : public AActor
{
	GENERATED_BODY()

public:
	ATarsBridge();

	int32 Port = 5577;
	double ControlDt = 0.02;  // s per `step`
	// s per frame = per physics step: the BodySpec timestep (Python checks dt / substeps against it).
	double PhysicsDt = 0.002;
	int32 PhysicsStepsPerControl() const { return FMath::Max(1, FMath::RoundToInt(ControlDt / PhysicsDt)); }
	FVector Origin = FVector::ZeroVector;  // the floor under level point (0, 0), world cm

protected:
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type Reason) override;
	virtual void Tick(float DeltaSeconds) override;

private:
	void Accept();
	void DropClient(const FString& Why);
	bool ReadLine(FString& Line, double TimeoutS);
	void Send(const TSharedRef<FJsonObject>& Msg);
	void SendError(const FString& Error);
	TSharedRef<FJsonObject> StateMessage(float FrameDt) const;
	// Returns true when the command was a `step` (the frame may now run physics).
	bool Handle(const TSharedPtr<FJsonObject>& Cmd);
	void FollowCamera(float DeltaSeconds);

	FSocket* Listener = nullptr;
	FSocket* Client = nullptr;
	TArray<uint8> Pending;  // bytes received after the last full line
	UPROPERTY() TObjectPtr<ATarsActor> Tars;
	UPROPERTY() TObjectPtr<ATarsLevel> Level;
	UPROPERTY() TObjectPtr<ACameraActor> Camera;
	bool bStepPending = false;
	// Videos: -TarsCamera=top looks down from above the 2.4 m apartment walls (the default follow
	// camera trails TARS at 1.6 m and ends up outside a room). The camera jumps to TARS on reset.
	bool bTopCamera = false;
	bool bSnapCamera = true;
	int32 StepFrames = 0;  // physics frames run of the pending step
	double StepTime = 0;
	double EpisodeTime = 0;
	double BodyHeightCm = 36;
	FString Hud;
};
