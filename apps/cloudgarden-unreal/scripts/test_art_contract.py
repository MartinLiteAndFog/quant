"""Source-level smoke tests for Cloudgarden native visual-only integration.

These checks prove neither compilation nor appearance in Unreal Engine.
They guard the deterministic presentation contract until a Mac UE5 native
build, interaction pass and screenshot comparison are available.
"""
from pathlib import Path
import unittest

PROJECT = Path(__file__).resolve().parents[1]
STYLE = (PROJECT / "Source" / "Cloudgarden" / "CloudgardenArtStyle.h").read_text(encoding="utf-8")
RENDER = (PROJECT / "Source" / "Cloudgarden" / "Cloudgarden.cpp").read_text(encoding="utf-8")


class ArtStyleContractTests(unittest.TestCase):
    def test_style_is_pure_and_has_no_financial_side_effects(self):
        for name in ("Stem(", "Branch(", "Leaf(", "LeafWidthScale(", "PhyllotaxisPhase("):
            self.assertIn(name, STYLE)
        for forbidden in ("FMath::Rand", "FRandomStream", "FDateTime", "GetWorld", "SpawnActor",
                          "navUsdt", "netPnl", "capitalUsdt", "paperReplay", "PlaceOrder"):
            self.assertNotIn(forbidden, STYLE)

    def test_renderer_calls_visual_gene_style_only(self):
        for gene in ('TEXT("water")', 'TEXT("sun")', 'TEXT("leaf")'):
            self.assertIn(gene, RENDER)
        for call in ("FCloudgardenArtStyle::Stem(Cactus,VisualWater,VisualSun)",
                     "FCloudgardenArtStyle::Branch(VisualWater,VisualSun)",
                     "FCloudgardenArtStyle::Leaf(Form,Fraction,VisualWater,VisualSun)",
                     "FCloudgardenArtStyle::LeafWidthScale(VisualLeaf)"):
            self.assertIn(call, RENDER)

    def test_reprojected_plant_has_stable_visual_phase(self):
        self.assertIn("FCrc::StrCrc32(*PlantId)", STYLE)
        self.assertIn("PhyllotaxisPhase(Plant->PlantId)", RENDER)
        self.assertIn("Leaf*2.39996323f+VisualPhase", RENDER)
        self.assertNotIn("Leaf*2.39996323f+Index*.3f", RENDER)

    def test_canonical_anchor_and_actor_pool_are_unchanged(self):
        for required in ("ShapePool", "BotanicalPool", "Plant->ExactAnchor=FVector(",
                         "Crown=Plant->ExactAnchor", "Plant->CrownOffset=Crown-Plant->ExactAnchor"):
            self.assertIn(required, RENDER)
        self.assertIn("if(Index>=50)break;", RENDER)

    def test_hud_pulls_snapshot_finances_not_visual_style(self):
        for expected in ("HudPanel()", "HudMint()", "HudInk()", "HudLoss()"):
            self.assertIn(expected, RENDER)
        for source in ('Num(G,TEXT("nav"))', 'Num(G,TEXT("netPnl"))',
                       'Num(G,TEXT("deposits"))', 'Num(A,TEXT("equity"))'):
            self.assertIn(source, RENDER)


if __name__ == "__main__":
    unittest.main()
