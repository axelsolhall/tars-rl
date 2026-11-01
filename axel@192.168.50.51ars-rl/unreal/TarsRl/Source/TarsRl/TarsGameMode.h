// Puts TARS on whatever floor the level has (or a plain one if it has none) and parks the
// spectator camera in front of it, so pressing Play in any basic level runs the comparison.
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
};
