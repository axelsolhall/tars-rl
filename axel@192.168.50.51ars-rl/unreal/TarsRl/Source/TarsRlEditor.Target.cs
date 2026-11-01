using UnrealBuildTool;

public class TarsRlEditorTarget : TargetRules
{
	public TarsRlEditorTarget(TargetInfo Target) : base(Target)
	{
		Type = TargetType.Editor;
		DefaultBuildSettings = BuildSettingsVersion.Latest;
		IncludeOrderVersion = EngineIncludeOrderVersion.Latest;
		ExtraModuleNames.Add("TarsRl");
	}
}
