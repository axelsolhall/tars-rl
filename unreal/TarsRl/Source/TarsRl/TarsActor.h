// TARS as an Unreal physics actor, built at BeginPlay from the tars-rl BodySpec JSON (A57).
//
// Three simulated box slabs (central body + two arms). Each arm hangs on a chain of two
// Chaos constraints, matching MuJoCo's hinge-then-slide joints (A69):
//   central --hinge (twist about the body y axis)--> carriage --slide (along the arm)--> arm
// The carriage is a tiny invisible body with no collisions. A third, fully free constraint
// between central and arm only disables their collision, like MuJoCo's <exclude>.
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
	bool IsReady() const { return bReady; }
	// Servo targets in SI, actuator order (left_hinge, right_hinge, left_slide, right_slide).
	void SetTargets(const TArray<double>& InTargets) { ApplyTargets(InTargets); }
	// The four servo forces for the coming physics step, from the current slab states.
	void ServoStep();
	// Central, left arm, right arm.
	TArray<FTarsSlabState> ReadSlabs() const;
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
	UPROPERTY(EditAnywhere, Category = "TARS") float CarriageMassKg = 0.01f;

protected:
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type Reason) override;
	virtual void Tick(float DeltaSeconds) override;

private:
	bool LoadSpec();
	bool LoadSequence();
	void BuildBody();
	UStaticMeshComponent* MakeBox(FName Name, const FVector& SizeCm, float MassKg,
		const FVector& LocalCm, bool bCollide);
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
	UPROPERTY() TArray<TObjectPtr<UStaticMeshComponent>> Carriages;
	UPROPERTY() TArray<TObjectPtr<UPhysicsConstraintComponent>> Hinges;
	UPROPERTY() TArray<TObjectPtr<UPhysicsConstraintComponent>> Slides;
	UPROPERTY() TArray<TObjectPtr<UPhysicsConstraintComponent>> NoCollide;

	FTarsBodySpec Spec;
	TArray<double> Targets = {0, 0, 0, 0};  // SI, actuator order, signs applied
	TArray<double> KeyTimes;
	TArray<TArray<double>> KeyTargets;  // per keyframe, actuator order
	double SimTime = 0;
	int64 Frames = 0;
	double StandTime = -1;  // first time tilt > 45 deg, -1 while upright
	TArray<FString> Lines;
	bool bReady = false, bSaved = false;
};
