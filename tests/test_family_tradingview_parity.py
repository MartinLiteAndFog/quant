"""Ensure the family Pine fork changes ONLY webhook metadata, not trading rules."""
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def normalized_lines(filename: str) -> list[str]:
    lines = (ROOT / "tradingview" / filename).read_text(encoding="utf-8").splitlines()
    # Keep Pine //@version=6 on line 1; remove only the two fork comments.
    if filename == "sol_imba_family.pine":
        if not lines or lines[0] != "//@version=6":
            raise AssertionError("Pine version directive must remain first")
        lines = [lines[0]] + lines[3:]
    result = []
    for line in lines:
        if line.startswith("strategy("):
            result.append("<strategy display name>")
        elif line.startswith("webhookSource = "):
            result.append("<strategy id source input>")
        elif line.startswith('    "{\\"token\\":\\"'):
            result.append("<JSON webhook payload only>")
        else:
            result.append(line)
    return result


class TradingViewFamilyForkTests(unittest.TestCase):
    def test_pine_trading_rules_are_byte_equivalent_after_metadata_normalization(self):
        original = normalized_lines("sol_imba_strategy.pine")
        family = normalized_lines("sol_imba_family.pine")
        self.assertEqual(family, original)

    def test_strategy_id_and_event_id_in_family_alert_payload(self):
        pine = (ROOT / "tradingview" / "sol_imba_family.pine").read_text(encoding="utf-8")
        self.assertIn(r'\"strategy_id\"', pine)
        self.assertIn(r'\"event_id\"', pine)
        self.assertIn("webhookSource", pine)
        self.assertIn("webhookToken", pine)


if __name__ == "__main__":
    unittest.main()
