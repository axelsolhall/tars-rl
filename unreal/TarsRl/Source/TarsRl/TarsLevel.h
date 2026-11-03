// The world TARS walks in during a bridge session: the level's boxes and the flag marker.
//
// Built from the bridge's `level` message, which `tars_rl.env.levels` + `tars_rl.unreal.bridge`
// write: every box already in Unreal's frame (centre cm, quaternion x y z w, full size cm,
// friction, colour). MuJoCo builds the same boxes as static geoms, so both engines walk the same
// world. Boxes are static colliders with a physical material that takes the larger friction in a
// contact, like MuJoCo.
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "TarsLevel.generated.h"

class UStaticMesh;
class UStaticMeshComponent;
class UPhysicalMaterial;
class UMaterialInstanceDynamic;
class UMaterialInterface;
class FJsonValue;

UCLASS()
class TARSRL_API ATarsLevel : public AActor
{
	GENERATED_BODY()

public:
	ATarsLevel();

	// Replaces the current boxes. Positions are relative to the level origin (the actor's
	// location: the floor under level point (0, 0)). Returns the number of boxes built, or -1
	// with Error set.
	int32 Build(const TArray<TSharedPtr<FJsonValue>>& Boxes, FString& Error);
	// Flag position relative to the level origin, cm; height of TARS's central slab for scale.
	void SetFlag(const FVector& PosCm, double BodyHeightCm);

private:
	UMaterialInstanceDynamic* ColourMaterial(const FLinearColor& Colour);
	UPhysicalMaterial* FrictionMaterial(double Friction);
	UStaticMeshComponent* MakeMesh(FName Name, bool bCollide);

	UPROPERTY() TObjectPtr<UStaticMesh> CubeMesh;
	UPROPERTY() TObjectPtr<UMaterialInterface> BaseMaterial;
	UPROPERTY() TArray<TObjectPtr<UStaticMeshComponent>> Boxes;
	UPROPERTY() TObjectPtr<UStaticMeshComponent> FlagPole;
	UPROPERTY() TObjectPtr<UStaticMeshComponent> FlagCloth;
	UPROPERTY() TMap<FString, TObjectPtr<UMaterialInstanceDynamic>> Colours;
	UPROPERTY() TMap<FString, TObjectPtr<UPhysicalMaterial>> Frictions;
	int32 NextId = 0;
};
