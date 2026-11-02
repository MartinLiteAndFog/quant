#!/usr/bin/env python3
"""HUD browser mock is built from the tested synthetic financial model."""
from __future__ import annotations
import json
import re
import unittest

from build_hud_snapshot import default_hud
from export_hud_preview import render_preview, embedded_json


class HUDPreviewTests(unittest.TestCase):
    def test_inlined_snapshots_match_true_generator_outputs(self):
        html = render_preview()
        self.assertNotIn("__HUD_MOCK_JSON__", html)
        self.assertNotIn("__HUD_EMPTY_JSON__", html)
        for name, finance in (("mock", True), ("empty", False)):
            match = re.search(
                r'<script type="application/json" id="data-' + name + r'">(.*?)</script>',
                html, flags=re.S)
            self.assertIsNotNone(match)
            hud = json.loads(match.group(1))
            self.assertEqual(hud, default_hud(mock_finance=finance))
            self.assertEqual(hud["authority"], "synthetic_hud_preview_only")
            self.assertIn("KEIN LIVE-HANDEL", hud["sourceBanner"])

    def test_toggles_ranking_and_world_modes_have_handlers(self):
        html = render_preview()
        for item in ('id="rank-winners"', 'id="rank-losers"', 'id="atlas"',
                     'id="explore"', 'id="show-mock"', 'id="show-missing"',
                     'id="tile-grid"', 'id="tile-control"', 'id="ranking"',
                     'id="sparkline-path"', "addEventListener('click'"):
            self.assertIn(item, html)
        self.assertIn("Keine Änderung ausgeführt", html)
        self.assertIn("KEIN UE-RENDER", html)
        self.assertIn("FIKTIVE WERTE", html)
        self.assertNotIn("BUY SOL", html)

    def test_script_breakout_escaped(self):
        s = embedded_json({"payload": "</script><img src=x onerror=evil()>", "amp": "&"})
        self.assertNotIn("</script>", s)
        self.assertIn("\\u003c", s)
        self.assertEqual(json.loads(s)["payload"],
                         "</script><img src=x onerror=evil()>")

    def test_empty_finance_is_visible_as_missing(self):
        html = render_preview()
        self.assertIn("Keine bestätigten Finanzdaten verfügbar", html)
        self.assertIn("row.netPnlUsdt", html)
        self.assertIn("tile.visiblePlantInstances", html)


if __name__ == "__main__":
    unittest.main(verbosity=2)
