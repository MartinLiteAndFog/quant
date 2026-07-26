from __future__ import annotations

import os
import subprocess
import sys
import unittest
from unittest.mock import patch

from quant.execution.runtime_mode import fleet_api_only


class FleetApiOnlyModeTests(unittest.TestCase):
    def test_truthy_values_enable_mode(self) -> None:
        for value in ("1", "true", "TRUE", "yes", "on"):
            with self.subTest(value=value), patch.dict(
                os.environ, {"FLEET_API_ONLY": value}, clear=False
            ):
                self.assertTrue(fleet_api_only())

    def test_missing_or_false_value_disables_mode(self) -> None:
        for value in (None, "", "0", "false", "off"):
            env = {} if value is None else {"FLEET_API_ONLY": value}
            with self.subTest(value=value), patch.dict(os.environ, env, clear=False):
                if value is None:
                    os.environ.pop("FLEET_API_ONLY", None)
                self.assertFalse(fleet_api_only())

    def test_worker_modules_exit_without_initializing_brokers(self) -> None:
        env = os.environ.copy()
        env["FLEET_API_ONLY"] = "1"
        for module in (
            "quant.execution.live_signal_worker",
            "quant.execution.live_executor",
        ):
            with self.subTest(module=module):
                result = subprocess.run(
                    [sys.executable, "-m", module],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
