#!/usr/bin/env python3
"""Generate an offline interactive HUD HTML mock from the *same* tested snapshot.

The HTML is a browser layout/interaction mock, NOT Unreal Editor, UMG, Chaos or
financial evidence. Contains only fictional seed/finance values with DEMO label.
"""
from __future__ import annotations

import json
from pathlib import Path

from build_hud_snapshot import default_hud
from validate_gene_physics import ROOT


def embedded_json(value: dict) -> str:
    # Keep a JSON script element inert even if future sources contain e.g. </script>
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def render_preview() -> str:
    template = (ROOT / "hud" / "hud_preview_template.html").read_text(encoding="utf-8")
    if template.count("__HUD_MOCK_JSON__") != 1 or template.count("__HUD_EMPTY_JSON__") != 1:
        raise ValueError("HUD preview must have exactly one payload slot per view")
    return (template.replace("__HUD_MOCK_JSON__", embedded_json(default_hud(mock_finance=True)))
                    .replace("__HUD_EMPTY_JSON__", embedded_json(default_hud(mock_finance=False))))


if __name__ == "__main__":
    print(render_preview())
