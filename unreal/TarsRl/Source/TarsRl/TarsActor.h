// TARS as an Unreal physics actor, built at BeginPlay from the tars-rl BodySpec JSON (A57).
//
// Three simulated box slabs (central body + two arms). Each arm hangs on a chain of two
// Chaos constraints, matching MuJoCo's hinge-then-slide joints (A69):
//   central --hinge (twist about the body y axis, angular drive)--> carriage
//   carriage --slide (along the arm's length, linear drive)--> arm
// The carriage is a tiny invisible body with no collisions. A third, fully free constraint
// between central and arm only disables their collision, like MuJoCo's <exclude>.
//
// Each tick it plays the scripted target sequence (Content/TarsData/sequence.json) and records
// the three slab poses to Saved/TarsRuns/unreal_poses.csv for `python -m tars_rl.unreal.eval`.
// Nothing is replayed: the motion is Unreal's physics answering the targets.
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

	// Relative to the project's Content directory, written by `python -m tars_rl.unreal.export`.
	UPROPERTY(EditAnywhere, Category = "TARS") FString SpecFile = TEXT("TarsData/body_spec.json");
	UPROPERTY(EditAnywhere, Category = "TARS") FString SequenceFile = TEXT("TarsData/sequence.json");
	// Relative to the project's Saved directory.
	UPROPERTY(EditAnywhere, Category = "TARS") FString RecordFile = TEXT("TarsRuns/unreal_poses.csv");

	// Drive direction conventions (A74): flip to -1 if the eval shows a joint's corr near -1.
	UPROPERTY(EditAnywhere, Category = "TARS") float HingeSign = 1.f;
	UPROPERTY(EditAnywhere, Category = "TARS") float SlideSign = 1.f;
	// SI -> Unreal drive units (A72): torque and angular gains x1e4 (kg cm^2), force x100,
	// linear stiffness/damping x1 (kg/s^2 and kg/s are the same in cm). Tune here, not in code.
	UPROPERTY(EditAnywhere, Category = "TARS") float AngularGainScale = 1.0e4f;
	UPROPERTY(EditAnywhere, Category = "TARS") float LinearGainScale = 1.f;
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
	void ApplyTargets(const TArray<double>& Targets);
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
	TArray<double> KeyTimes;
	TArray<TArray<double>> KeyTargets;  // per keyframe, actuator order
	double SimTime = 0;
	double StandTime = -1;  // first time tilt > 45 deg, -1 while upright
	TArray<FString> Lines;
	bool bReady = false, bSaved = false;
};
