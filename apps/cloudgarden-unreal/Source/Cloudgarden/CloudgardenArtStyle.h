#pragma once

// Cloudgarden native Unreal visual palette, intentionally independent of paper
// trades, capital, signal generation and simulation ecology. Pure, deterministic
// presentation helpers: the same canonical plant has the same look in every
// parameter-plane projection and after saving/loading the world.
#include "CoreMinimal.h"
#include "Misc/Crc.h"

struct FCloudgardenArtStyle final
{
    // Light botanical meadow, muted water and warm soil. These are Unreal
    // linear material colors, not a substitute for calibrated PBR assets.
    static FLinearColor Soil() { return FLinearColor(0.24f, 0.205f, 0.145f, 1.f); }
    static FLinearColor Water() { return FLinearColor(0.20f, 0.48f, 0.51f, 1.f); }
    static FLinearColor Meadow() { return FLinearColor(0.30f, 0.43f, 0.235f, 1.f); }
    static FLinearColor Anchor() { return FLinearColor(0.56f, 0.86f, 0.67f, 1.f); }
    static FLinearColor RootConnector() { return FLinearColor(0.36f, 0.31f, 0.20f, 1.f); }
    static FLinearColor Bee() { return FLinearColor(0.95f, 0.73f, 0.24f, 1.f); }

    static FLinearColor HudInk() { return FLinearColor(0.87f, 0.95f, 0.88f, 1.f); }
    static FLinearColor HudMint() { return FLinearColor(0.47f, 0.86f, 0.70f, 1.f); }
    static FLinearColor HudMuted() { return FLinearColor(0.60f, 0.73f, 0.66f, 1.f); }
    static FLinearColor HudPanel() { return FLinearColor(0.045f, 0.12f, 0.095f, 0.86f); }
    static FLinearColor HudLoss() { return FLinearColor(0.94f, 0.48f, 0.36f, 1.f); }

    // The gene-derived hues communicate a *visual phenotype*, not financial
    // win/loss or a second vigor metric. No index, timestamp or RNG is used.
    static FLinearColor Stem(bool bCactus, float WaterGene, float SunGene)
    {
        const float Water = FMath::Clamp(WaterGene, 0.f, 1.f);
        const float Sun = FMath::Clamp(SunGene, 0.f, 1.f);
        const FLinearColor Base = bCactus
            ? FLinearColor(0.20f, 0.45f, 0.32f, 1.f)
            : FLinearColor(0.30f, 0.41f, 0.19f, 1.f);
        return FLinearColor(
            FMath::Clamp(Base.R + 0.055f * Sun - 0.015f * Water, 0.f, 1.f),
            FMath::Clamp(Base.G + 0.040f * Water, 0.f, 1.f),
            FMath::Clamp(Base.B + 0.030f * Water - 0.012f * Sun, 0.f, 1.f), 1.f);
    }

    static FLinearColor Branch(float WaterGene, float SunGene)
    {
        const float Water = FMath::Clamp(WaterGene, 0.f, 1.f);
        const float Sun = FMath::Clamp(SunGene, 0.f, 1.f);
        return FLinearColor(0.29f + 0.035f * Sun, 0.385f + 0.045f * Water,
                            0.17f + 0.025f * Water, 1.f);
    }

    static FLinearColor Leaf(const FString& Phenotype, float Fraction, float WaterGene, float SunGene)
    {
        const float T = FMath::Clamp(Fraction, 0.f, 1.f);
        const float Water = FMath::Clamp(WaterGene, 0.f, 1.f);
        const float Sun = FMath::Clamp(SunGene, 0.f, 1.f);
        const bool bGrass = Phenotype == TEXT("Grass");
        const float R = (bGrass ? 0.27f : 0.23f) + 0.060f * Sun + 0.045f * T;
        const float G = (bGrass ? 0.47f : 0.51f) + 0.080f * Water + 0.035f * T;
        const float B = (bGrass ? 0.20f : 0.16f) + 0.055f * Water - 0.018f * Sun;
        return FLinearColor(FMath::Clamp(R, 0.f, 1.f), FMath::Clamp(G, 0.f, 1.f),
                            FMath::Clamp(B, 0.f, 1.f), 1.f);
    }

    static float LeafWidthScale(float LeafGene)
    {
        return 0.72f + 0.56f * FMath::Clamp(LeafGene, 0.f, 1.f);
    }

    static float PhyllotaxisPhase(const FString& PlantId)
    {
        // Stable across repeated projections and actor pool reassignments.
        return static_cast<float>(FCrc::StrCrc32(*PlantId) % 10000u)
            * (2.f * PI / 10000.f);
    }
};
