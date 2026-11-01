using UnrealBuildTool;

public class TarsRlTarget : TargetRules
{
	public TarsRlTarget(TargetInfo Target) : base(Target)
	{
		Type = TargetType.Game;
		DefaultBuildSettings = BuildSettingsVersion.Latest;
		IncludeOrderVersion = EngineIncludeOrderVersion.Latest;
		ExtraModuleNames.Add("TarsRl");
	}
}
