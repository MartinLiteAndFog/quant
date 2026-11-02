#!/bin/zsh
set -eu
cd "$(dirname "$0")"
exec python3 "$PWD/launch.py"
