"""The order-placing webhook must never run unauthenticated.

WEBHOOK_TOKEN was set to an empty string on the live `quant` service, which made
_auth_required() return False and left /webhook/tv-execute open to anyone who
found the URL — on an account with real funds and dry-run disabled.
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from quant.execution.webhook_server import _auth_required


class AuthRequiredTests(unittest.TestCase):
    def test_empty_token_does_not_count_as_configured(self) -> None:
        with patch.dict(os.environ, {"WEBHOOK_TOKEN": ""}, clear=False):
            self.assertFalse(_auth_required())

    def test_whitespace_token_does_not_count_as_configured(self) -> None:
        with patch.dict(os.environ, {"WEBHOOK_TOKEN": "   "}, clear=False):
            self.assertFalse(_auth_required())

    def test_real_token_enables_auth(self) -> None:
        with patch.dict(os.environ, {"WEBHOOK_TOKEN": "s3cret"}, clear=False):
            self.assertTrue(_auth_required())


if __name__ == "__main__":
    unittest.main()
