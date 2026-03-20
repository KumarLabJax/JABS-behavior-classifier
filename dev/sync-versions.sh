#!/usr/bin/env bash
# Sync all workspace sub-package versions to match the root package version.
#
# Usage:
#   ./dev/sync-versions.sh                     # apply root version to all sub-packages and update README.md
#   ./dev/sync-versions.sh --no-readme         # skip rewriting GitHub URLs in README.md
#   ./dev/sync-versions.sh --dry-run           # preview without writing
#   ./dev/sync-versions.sh --dry-run --no-readme  # preview package syncs only
set -euo pipefail

ROOT_VERSION=$(uv version --short)
DRY_RUN=""
UPDATE_README="yes"

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN="--dry-run" ;;
        --no-readme) UPDATE_README="" ;;
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

if [ -n "$UPDATE_README" ]; then
    echo ""
    echo "README.md GitHub URL rewrite: blob/(main|v*) -> blob/v${ROOT_VERSION}"
    if [ -n "$DRY_RUN" ]; then
        # Show lines that would change without writing
        echo "  Lines in README.md that would be updated:"
        grep -nE "(github\.com|raw\.githubusercontent\.com)/KumarLabJax/JABS-behavior-classifier/(blob/)?(main|v[^/]+)/" README.md \
            | sed "s/^/    /" \
            || echo "    (no matching URLs found)"
    else
        # Replace /blob/main/ and /blob/v<old>/ with /blob/v<new>/
        # Replace raw.githubusercontent.com/.../main/ and .../v<old>/ with .../v<new>/
        sed -i.bak -E \
            -e "s#/blob/(main|v[^/]*)/#/blob/v${ROOT_VERSION}/#g" \
            -e "s#JABS-behavior-classifier/(main|v[^/]*)/#JABS-behavior-classifier/v${ROOT_VERSION}/#g" \
            README.md
        rm -f README.md.bak
        echo "  README.md updated."
    fi
fi

if [ -z "$DRY_RUN" ]; then
    echo ""
    echo "Run 'uv lock' to update the lockfile."
fi
