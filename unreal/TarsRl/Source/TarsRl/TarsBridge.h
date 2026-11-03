// The Unreal side of the Python bridge (tars_rl.unreal.bridge holds the protocol): a TCP server
// on 127.0.0.1:TarsPort that lets a Python policy drive TARS in lockstep.
//
// Each frame, before physics, the bridge first answers the previous `step` with the state that
// physics produced, then blocks on the socket until the next `step` arrives (answering `hello`,
// `level`, `reset` and `flag` immediately in between). So the world advances one fixed control
// step (20 ms, physics substeps of 2 ms) per `step`, and only as fast as Python asks: paced to
// the wall clock to watch, or flat out to evaluate. Without a client it idles at about real time.
//
// The frame time must be fixed: ATarsGameMode turns on Unreal's fixed time step in bridge mode,
// and every state reply carries the frame's dt so Python can refuse anything else.
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
	double ControlDt = 0.02;  // s per step; the game mode sets the engine's fixed time step to this
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
	double EpisodeTime = 0;
	double BodyHeightCm = 36;
	FString Hud;
};
