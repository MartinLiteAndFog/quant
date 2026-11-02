#!/bin/zsh
# Select the installed full Xcode for this process only.
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
ENGINE='/Users/Shared/Epic Games/UE_5.8/Engine/Binaries/Mac/UnrealEditor.app/Contents/MacOS/UnrealEditor'
if ! xcrun metal --version >/dev/null 2>&1; then
  echo 'Metal fehlt. In einem Terminal ausführen:'
  echo 'DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcodebuild -downloadComponent MetalToolchain'
  exit 1
fi
exec "$ENGINE" "$@"
