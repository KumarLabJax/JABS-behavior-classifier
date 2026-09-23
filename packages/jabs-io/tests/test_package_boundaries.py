"""Guard the ``jabs-io`` package boundary.

``jabs-io`` declares exactly one JABS dependency (``jabs-core``), so every
``jabs.*`` import in its source tree must resolve to a module that ships in
``jabs-core`` or in ``jabs-io`` itself. An import of any other ``jabs.*``
namespace - most easily the root ``jabs-behavior-classifier`` distribution,
which is always importable in a workspace checkout - would not be installed
alongside ``jabs-io`` and would raise ``ModuleNotFoundError`` for anyone who
depends on the library on its own.
"""

import ast
from pathlib import Path

import pytest

import jabs.io

# top-level ``jabs`` sub-packages jabs-io is allowed to import: its own modules
# plus the single JABS distribution it declares as a dependency.
ALLOWED_JABS_PACKAGES = frozenset({"core", "io"})

_SOURCE_ROOT = Path(jabs.io.__file__).parent


def _jabs_imports(tree: ast.AST) -> set[str]:
    """Collect the absolute ``jabs.*`` module names imported by a parsed module.

    Relative imports are ignored: they can only reach modules inside the
    package being scanned.

    Args:
        tree: Parsed module to walk.

    Returns:
        Set of dotted module names beginning with ``jabs.``.
    """
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(
                alias.name for alias in node.names if alias.name.split(".")[0] == "jabs"
            )
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module
            and node.module.split(".")[0] == "jabs"
        ):
            imported.add(node.module)
    return imported


def _is_allowed(module: str) -> bool:
    """Return True if a dotted ``jabs.*`` module name is one jabs-io may import.

    Args:
        module: Dotted module name beginning with ``jabs``.

    Returns:
        True for the bare ``jabs`` namespace and for sub-packages listed in
        ``ALLOWED_JABS_PACKAGES``.
    """
    parts = module.split(".")
    return len(parts) == 1 or parts[1] in ALLOWED_JABS_PACKAGES


@pytest.mark.parametrize(
    "source_file",
    sorted(_SOURCE_ROOT.rglob("*.py")),
    ids=lambda p: str(p.relative_to(_SOURCE_ROOT)),
)
def test_only_imports_declared_jabs_packages(source_file: Path) -> None:
    """Every ``jabs.*`` import resolves to a declared dependency of jabs-io."""
    tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
    offenders = {name for name in _jabs_imports(tree) if not _is_allowed(name)}
    assert not offenders, (
        f"{source_file.relative_to(_SOURCE_ROOT)} imports {sorted(offenders)}, which "
        f"jabs-io does not depend on. Allowed: {sorted(ALLOWED_JABS_PACKAGES)}."
    )
