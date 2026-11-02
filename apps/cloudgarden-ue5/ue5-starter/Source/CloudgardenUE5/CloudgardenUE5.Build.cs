using UnrealBuildTool;

public class CloudgardenUE5 : ModuleRules
{
    public CloudgardenUE5(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
        PublicDependencyModuleNames.AddRange(new string[]
        {
            "Core", "CoreUObject", "Engine", "UMG", "Json"
        });
        PrivateDependencyModuleNames.AddRange(new string[]
        {
            "Slate", "SlateCore"
        });
    }
}
