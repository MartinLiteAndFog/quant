#!/usr/bin/env python3
"""Desktop package launcher; installed beside dist/ as launch.py. Standard library only."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import urllib.request
import webbrowser


def main():
    root = Path(__file__).resolve().parent / "dist"
    if not (root / "market/manifest.json").is_file():
        raise SystemExit("Cloudgarden's historical SOL data package is missing.")
    for port in range(5184, 5195):
        url = f"http://127.0.0.1:{port}/"
        try:
            with urllib.request.urlopen(url + "cloudgarden-build.json", timeout=.3) as response:
                marker = json.load(response)
            if isinstance(marker, dict) and marker.get("app") == "Cloudgarden" and marker.get("mode") == "historical-sol-v1":
                webbrowser.open(url)
                return
        except (OSError, ValueError):
            pass
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), partial(SimpleHTTPRequestHandler, directory=str(root)))
        except OSError:
            continue
        print(f"Cloudgarden: {url}\nKeep this window open while playing. Ctrl+C stops the local server.", flush=True)
        webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return
    raise SystemExit("Ports 5184–5194 are occupied. Close an older Cloudgarden server and start again.")


if __name__ == "__main__":
    main()
