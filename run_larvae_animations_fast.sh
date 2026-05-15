#!/usr/bin/env bash
# Quick launcher: same as python create_larvae_animations.py --fast ...
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python "$DIR/create_larvae_animations.py" --fast "$@"
