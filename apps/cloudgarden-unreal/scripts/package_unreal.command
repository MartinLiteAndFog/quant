#!/bin/zsh
set -eu
PROJECT_DIR="${0:A:h:h}"
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
ENGINE_ROOT='/Users/Shared/Epic Games/UE_5.8'
ARCHIVE_DIR="${1:-/tmp/CloudgardenUnrealPackage}"
if ! xcrun metal --version >/dev/null 2>&1; then
  echo 'The full Xcode Metal Toolchain is required before packaging.' >&2
  exit 1
fi
exec "$ENGINE_ROOT/Engine/Build/BatchFiles/RunUAT.command" BuildCookRun \
  -project="$PROJECT_DIR/Cloudgarden.uproject" -noP4 -platform=Mac \
  -clientconfig=Development -build -cook -stage -pak -archive -UbtArgs=-MaxParallelActions=2 \
  -archivedirectory="$ARCHIVE_DIR" -map=/Engine/Maps/Entry -unattended -utf8output
