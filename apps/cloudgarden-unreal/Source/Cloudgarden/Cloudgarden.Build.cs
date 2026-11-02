using UnrealBuildTool;
public class Cloudgarden : ModuleRules { public Cloudgarden(ReadOnlyTargetRules Target) : base(Target) { PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs; PublicDependencyModuleNames.AddRange(new string[] {"Core","CoreUObject","Engine","InputCore","Json","JsonUtilities"}); } }
