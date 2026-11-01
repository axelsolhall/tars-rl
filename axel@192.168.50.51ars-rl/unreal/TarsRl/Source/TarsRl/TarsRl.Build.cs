using UnrealBuildTool;

public class TarsRl : ModuleRules
{
	public TarsRl(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
		PublicDependencyModuleNames.AddRange(new string[] {
			"Core", "CoreUObject", "Engine", "PhysicsCore", "Json"
		});
	}
}
