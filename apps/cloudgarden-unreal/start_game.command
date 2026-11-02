#!/bin/zsh
set -eu
PROJECT_DIR="${0:A:h}"
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
NODE_BIN="$(command -v node || true)"
if [[ -z "$NODE_BIN" ]]; then
  echo 'Node.js fehlt für die lokale Simulationsbrücke.'
  exit 1
fi
if [[ ! -f "$PROJECT_DIR/Binaries/Mac/libUnrealEditor-Cloudgarden.dylib" ]]; then
  echo 'Das native Cloudgarden-Spielmodul muss zuerst gebaut werden.'
  exit 1
fi
cd "$PROJECT_DIR/../cloudgarden"
"$NODE_BIN" --import tsx scripts/unreal_bridge.mjs "$PROJECT_DIR/Saved" &
BRIDGE_PID=$!
trap 'kill "$BRIDGE_PID" 2>/dev/null || true' EXIT INT TERM
'/Users/Shared/Epic Games/UE_5.8/Engine/Binaries/Mac/UnrealEditor.app/Contents/MacOS/UnrealEditor' "$PROJECT_DIR/Cloudgarden.uproject" /Engine/Maps/Entry -game -windowed -ResX=1440 -ResY=900 -log
