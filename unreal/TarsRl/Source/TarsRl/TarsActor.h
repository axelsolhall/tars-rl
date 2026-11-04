// TARS as an Unreal physics actor, built at BeginPlay from the tars-rl BodySpec JSON (A57).
//
// Three simulated box slabs (central body + two arms). Each arm hangs on one Chaos constraint
// that holds both of MuJoCo's joints (A69), as MuJoCo puts both in the arm's one body: the
// hinge is its twist (about the body y axis, through the arm's centre) and the slide its linear
// z, which Chaos measures along the arm because the arm is the constraint's second component.
// The constraint also turns off central-arm collisions, like MuJoCo's <exclude>.
// Until 2026-11-04 a 100 g carriage body sat between a hinge and a slide constraint. Chaos's
// iterative joint solver made that chain soft: its free fall, push and drag probes were a few
// percent off MuJoCo and Bullet, which the single constraint matches (tars_rl.unreal.probe).
//
// The constraints only hold the joint axes and limits; the servos are ServoStep, MuJoCo's
// position actuator computed here before every physics step and applied as equal and opposite
// torques / forces: clip(kp (target - q) - kv qdot, +-limit). Chaos's own joint drives are not
// that servo (acceleration mode by default, a hidden 1.5x angular scale, limits on position
// impulses): with them the hinges hit 0.6 rad in 20 ms where MuJoCo's take 0.5 s (2026-11-04).
//
// One engine frame is one physics step (2 ms, no substeps), so ServoStep runs per step.
//
// Two modes. Scripted (T10, the default for a placed actor): each tick it plays the target
// sequence (Content/TarsData/sequence.json) and records the three slab poses to
// Saved/TarsRuns/unreal_poses.csv for `python -m tars_rl.unreal.eval`. Bridge (bScripted false,
// spawned by ATarsBridge): no tick of its own; the bridge sets servo targets each control step,
// calls ServoStep before each of its physics steps and reads the slab states back for the
// policy. Nothing is replayed in either mode: the motion is Unreal's physics answering the targets.
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "TarsActor.generated.h"

class UStaticMeshComponent;
class UStaticMesh;
class UPhysicsConstraintComponent;
class UPhysicalMaterial;
class FJsonObject;

struct FTarsServo
{
	double Kp = 0, Kv = 0, ForceLimit = 0;  // SI: N/m or Nm/rad, N s/m or Nm s/rad, N or Nm
};

// One slab's physics state, Unreal units: cm, cm/s, rad/s, world frame.
struct FTarsSlabState
{
	FVector Pos = FVector::ZeroVector;
	FQuat Rot = FQuat::Identity;
	FVector Vel = FVector::ZeroVector;  // at the slab's centre (its body frame origin in MuJoCo)
	FVector AngVel = FVector::ZeroVector;
};

struct FTarsBodySpec
{
	FVector CentralSizeM = FVector::ZeroVector;  // depth (x), width (y), height (z), metres
	FVector ArmSizeM = FVector::ZeroVector;
	double CentralMass = 0, ArmMass = 0, ArmGap = 0, Friction = 0.8;
	FVector2D HingeRange = FVector2D::ZeroVector;  // rad
	FVector2D SlideRange = FVector2D::ZeroVector;  // m
	FTarsServo HingeServo, SlideServo;
	TArray<FString> ActuatorNames;
};

UCLASS()
class TARSRL_API ATarsActor : public AActor
{
	GENERATED_BODY()

public:
	ATarsActor();

	// False: driven by ATarsBridge (set before FinishSpawning).
	UPROPERTY(EditAnywhere, Category = "TARS") bool bScripted = true;
	// Contact probes: build the central slab alone (no arms, joints or servos); the arm slots of
	// ReadSlabs then repeat the central slab. Set before FinishSpawning.
	bool bSolo = false;
	bool bSoloArm = false;  // with bSolo: the lone slab is an arm
	// Contact probes: hinges and slides locked solid and no servos (TARS as one jointed rigid body).
	bool bLocked = false;
	bool IsReady() const { return bReady; }
	// Servo targets in SI, actuator order (left_hinge, right_hinge, left_slide, right_slide).
	void SetTargets(const TArray<double>& InTargets) { ApplyTargets(InTargets); }
	// The four servo forces for the coming physics step, from the current slab states.
	void ServoStep();
	// For contact probes: a constant force (kg cm/s^2, world) at a point of the central slab
	// (cm from its centre, in its own frame), every physics step until changed.
	void SetPush(const FVector& Force, const FVector& AtLocal) { Push = Force; PushAt = AtLocal; }
	// Central, left arm, right arm.
	TArray<FTarsSlabState> ReadSlabs() const;
	// One-step probes (tars_rl.unreal.onestep): teleport the three slabs to these states (world,
	// as ReadSlabs gives them). Jointed TARS only.
	void WriteSlabs(const TArray<FTarsSlabState>& In);
	FVector CentralLocation() const;
	double BodyHeightCm() const { return Spec.CentralSizeM.Z * 100.0; }

	// Relative to the project's Content directory, written by `python -m tars_rl.unreal.export`.
	UPROPERTY(EditAnywhere, Category = "TARS") FString SpecFile = TEXT("TarsData/body_spec.json");
	UPROPERTY(EditAnywhere, Category = "TARS") FString SequenceFile = TEXT("TarsData/sequence.json");
	// Relative to the project's Saved directory.
	UPROPERTY(EditAnywhere, Category = "TARS") FString RecordFile = TEXT("TarsRuns/unreal_poses.csv");

	// Joint direction conventions (A74): flip to -1 if the eval shows a joint's corr near -1.
	// +1 for both: the mirror in y (A73) keeps a turn about y and a move along z as they are.
	UPROPERTY(EditAnywhere, Category = "TARS") float HingeSign = 1.f;
	UPROPERTY(EditAnywhere, Category = "TARS") float SlideSign = 1.f;

protected:
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type Reason) override;
	virtual void Tick(float DeltaSeconds) override;

private:
	bool LoadSpec();
	bool LoadSequence();
	void BuildBody();
	UStaticMeshComponent* MakeBox(FName Name, const FVector& SizeCm, float MassKg,
		const FVector& LocalCm);
	UPhysicsConstraintComponent* MakeConstraint(FName Name, const FVector& LocalCm,
		const FRotator& LocalRot);
	TArray<double> TargetsAt(double T) const;
	void ApplyTargets(const TArray<double>& InTargets);
	void RecordSample();
	void SaveRecording();

	UPROPERTY() TObjectPtr<UStaticMesh> CubeMesh;
	UPROPERTY() TObjectPtr<UPhysicalMaterial> SlabMaterial;
	UPROPERTY() TObjectPtr<UStaticMeshComponent> Central;
	UPROPERTY() TArray<TObjectPtr<UStaticMeshComponent>> Arms;  // left, right
	UPROPERTY() TArray<TObjectPtr<UPhysicsConstraintComponent>> Joints;  // hinge + slide per arm

	FTarsBodySpec Spec;
	TArray<double> Targets = {0, 0, 0, 0};  // SI, actuator order, signs applied
	FVector Push = FVector::ZeroVector, PushAt = FVector::ZeroVector;
	TArray<double> KeyTimes;
	TArray<TArray<double>> KeyTargets;  // per keyframe, actuator order
	double SimTime = 0;
	int64 Frames = 0;
	double StandTime = -1;  // first time tilt > 45 deg, -1 while upright
	TArray<FString> Lines;
	bool bReady = false, bSaved = false;
};
