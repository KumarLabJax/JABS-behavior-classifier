#!/usr/bin/env bash
# Sync all workspace sub-package versions to match the root package version.
#
# Usage:
#   ./dev/sync-versions.sh          # apply root version to all sub-packages
#   ./dev/sync-versions.sh --dry-run  # preview without writing
set -euo pipefail

ROOT_VERSION=$(uv version --short)
DRY_RUN=""

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN="--dry-run" ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

echo "Root version: $ROOT_VERSION"

for toml in packages/*/pyproject.toml; do
    pkg_name=$(python3 -c "import tomllib; print(tomllib.load(open('$toml','rb'))['project']['name'])")
    current=$(uv version --package "$pkg_name" --short)

    if [ "$current" = "$ROOT_VERSION" ]; then
        echo "  $pkg_name: $current (already in sync)"
    else
        echo "  $pkg_name: $current -> $ROOT_VERSION"
        uv version "$ROOT_VERSION" --package "$pkg_name" --frozen $DRY_RUN
    fi
done

if [ -z "$DRY_RUN" ]; then
    echo ""
    echo "Run 'uv lock' to update the lockfile."
fi
