"""Report docstring coverage for Python modules, classes, and functions."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Counter:
    """Track documented and total counts for one metric."""

    documented: int = 0
    total: int = 0

    def add(self, has_docstring: bool) -> None:
        """Add one item to the counter.

        Args:
            has_docstring: Whether this item has a docstring.
        """
        self.total += 1
        if has_docstring:
            self.documented += 1


@dataclass
class Coverage:
    """Aggregate counters for each requested coverage category."""

    public_functions: Counter
    private_functions: Counter
    classes: Counter
    modules: Counter


@dataclass(frozen=True)
class Entity:
    """Represent one AST entity that can carry a docstring."""

    file_path: Path
    import_path: str
    kind: str
    category: str
    start_line: int
    end_line: int
    documented: bool


@dataclass
class StyleCoverage:
    """Track Google-style compliance among documented docstrings."""

    public_functions: Counter
    private_functions: Counter
    classes: Counter
    modules: Counter


@dataclass
class StyleAnalysis:
    """Store style compliance counters and detailed function paths."""

    coverage: StyleCoverage
    violating_function_paths: list[str]
    undocumented_function_paths: list[str]


MISSING_DOCSTRING_CODES = {"D100", "D101", "D102", "D103", "D104", "D105", "D106", "D107"}
CATEGORY_ORDER = [
    ("public functions", "public_functions"),
    ("private functions", "private_functions"),
    ("classes", "classes"),
    ("modules", "modules"),
]


def has_docstring(node: ast.AST) -> bool:
    """Return whether a module, class, or function has a docstring.

    Args:
        node: The AST node to inspect.

    Returns:
        True when the node has a docstring, otherwise False.
    """
    return ast.get_docstring(node, clean=False) is not None


def iter_python_files(paths: list[Path]) -> list[Path]:
    """Collect Python files from files/directories.

    Args:
        paths: Input files or directories.

    Returns:
        Sorted Python files to scan.
    """
    files: set[Path] = set()
    for path in paths:
        if path.is_file() and path.suffix == ".py":
            files.add(path)
            continue
        if path.is_dir():
            files.update(child for child in path.rglob("*.py") if child.is_file())
    return sorted(files)


def parse_python_file(path: Path) -> ast.Module | None:
    """Parse one Python file into an AST module.

    Args:
        path: File to parse.

    Returns:
        AST module, or None if parsing fails.
    """
    try:
        content = path.read_text(encoding="utf-8")
        return ast.parse(content, filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError) as exc:
        print(f"Skipping {path}: {exc}", file=sys.stderr)
        return None


def scan(paths: list[Path]) -> Coverage:
    """Compute docstring coverage across the provided paths.

    Args:
        paths: Files/directories to scan.

    Returns:
        Coverage counters for functions, classes, and modules.
    """
    coverage = Coverage(
        public_functions=Counter(),
        private_functions=Counter(),
        classes=Counter(),
        modules=Counter(),
    )

    for file_path in iter_python_files(paths):
        module = parse_python_file(file_path)
        if module is None:
            continue

        coverage.modules.add(has_docstring(module))

        for node in ast.walk(module):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                target = (
                    coverage.private_functions
                    if node.name.startswith("_")
                    else coverage.public_functions
                )
                target.add(has_docstring(node))
            elif isinstance(node, ast.ClassDef):
                coverage.classes.add(has_docstring(node))

    return coverage


def build_import_roots(paths: list[Path]) -> list[Path]:
    """Build candidate roots used to derive module import paths.

    Args:
        paths: Files/directories requested by the user.

    Returns:
        Candidate roots ordered by path depth.
    """
    cwd = Path.cwd().resolve()
    roots: set[Path] = set()
    default_src = cwd / "src"
    if default_src.is_dir():
        roots.add(default_src)

    for path in paths:
        resolved = path.resolve()
        if resolved.is_file():
            roots.add(resolved.parent)
        elif resolved.is_dir():
            roots.add(resolved)
            roots.add(resolved.parent)

    roots.add(cwd)
    return sorted(roots, key=lambda root: len(root.parts))


def module_import_path(file_path: Path, import_roots: list[Path]) -> str:
    """Resolve a Python file to its best-effort absolute import path.

    Args:
        file_path: Python source file.
        import_roots: Candidate import roots.

    Returns:
        Dotted module import path.
    """
    resolved = file_path.resolve()
    candidates: list[tuple[int, int, str]] = []
    for root in import_roots:
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            continue

        if relative.suffix != ".py":
            continue

        parts = list(relative.parts)
        if parts[-1] == "__init__.py":
            module_parts = parts[:-1]
        else:
            module_parts = parts[:-1] + [Path(parts[-1]).stem]

        if not module_parts or any(not part.isidentifier() for part in module_parts):
            continue

        root_priority = 2
        if root.name == "src":
            root_priority = 0
        elif root.name in {"python", "lib"}:
            root_priority = 1

        candidates.append((root_priority, len(module_parts), ".".join(module_parts)))

    if candidates:
        _, _, best = min(candidates, key=lambda item: (item[0], item[1], item[2]))
        return best
    return resolved.stem


def scan_entities(paths: list[Path]) -> list[Entity]:
    """Collect entities with ranges and docstring metadata.

    Args:
        paths: Files/directories to scan.

    Returns:
        Entities for modules, classes, and functions.
    """
    entities: list[Entity] = []
    import_roots = build_import_roots(paths)

    for file_path in iter_python_files(paths):
        module = parse_python_file(file_path)
        if module is None:
            continue
        resolved_file_path = file_path.resolve()
        module_path = module_import_path(file_path, import_roots)

        module_end = max((getattr(node, "end_lineno", 1) or 1) for node in ast.walk(module))
        entities.append(
            Entity(
                file_path=resolved_file_path,
                import_path=module_path,
                kind="module",
                category="modules",
                start_line=1,
                end_line=module_end,
                documented=has_docstring(module),
            )
        )

        def visit_body(body: list[ast.stmt], qualname_parts: list[str]) -> None:
            for node in body:
                if isinstance(node, ast.ClassDef):
                    class_qualname = ".".join([*qualname_parts, node.name])
                    entities.append(
                        Entity(
                            file_path=resolved_file_path,
                            import_path=f"{module_path}.{class_qualname}",
                            kind="class",
                            category="classes",
                            start_line=node.lineno,
                            end_line=getattr(node, "end_lineno", node.lineno),
                            documented=has_docstring(node),
                        )
                    )
                    visit_body(node.body, [*qualname_parts, node.name])
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    function_qualname = ".".join([*qualname_parts, node.name])
                    category = (
                        "private_functions"
                        if node.name.startswith("_")
                        else "public_functions"
                    )
                    entities.append(
                        Entity(
                            file_path=resolved_file_path,
                            import_path=f"{module_path}.{function_qualname}",
                            kind="function",
                            category=category,
                            start_line=node.lineno,
                            end_line=getattr(node, "end_lineno", node.lineno),
                            documented=has_docstring(node),
                        )
                    )
                    visit_body(node.body, [*qualname_parts, node.name])

        visit_body(module.body, [])

    return entities


def run_google_style_ruff(files: list[Path]) -> list[dict]:
    """Run Ruff docstring checks in Google convention and return diagnostics.

    Args:
        files: Python files to inspect.

    Returns:
        Ruff diagnostics in JSON form.
    """
    if not files:
        return []

    cmd = [
        "ruff",
        "check",
        *[str(path) for path in files],
        "--select",
        "D",
        "--config",
        'lint.pydocstyle.convention="google"',
        "--output-format",
        "json",
        "--exit-zero",
    ]

    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        print(f"Skipping Google-style check (ruff not available): {exc}", file=sys.stderr)
        return []

    if result.returncode not in (0, 1):
        print(
            f"Skipping Google-style check (ruff failed): {result.stderr.strip()}",
            file=sys.stderr,
        )
        return []

    try:
        parsed = json.loads(result.stdout) if result.stdout.strip() else []
    except json.JSONDecodeError as exc:
        print(f"Skipping Google-style check (invalid ruff output): {exc}", file=sys.stderr)
        return []

    if not isinstance(parsed, list):
        return []
    return parsed


def build_style_analysis(paths: list[Path]) -> StyleAnalysis:
    """Compute Google-style compliance and function-level details.

    Args:
        paths: Files/directories to scan.

    Returns:
        Style coverage plus violating and undocumented function paths.
    """
    entities = scan_entities(paths)

    style = StyleCoverage(
        public_functions=Counter(),
        private_functions=Counter(),
        classes=Counter(),
        modules=Counter(),
    )

    documented_entities_by_category: dict[str, list[Entity]] = {
        "public_functions": [],
        "private_functions": [],
        "classes": [],
        "modules": [],
    }
    entities_by_file: dict[Path, list[Entity]] = {}
    function_entities: list[Entity] = []

    for entity in entities:
        entities_by_file.setdefault(entity.file_path, []).append(entity)
        if entity.kind == "function":
            function_entities.append(entity)
        if entity.documented:
            documented_entities_by_category[entity.category].append(entity)

    for category_key in documented_entities_by_category:
        counter = getattr(style, category_key)
        counter.total = len(documented_entities_by_category[category_key])

    diagnostics = run_google_style_ruff(iter_python_files(paths))

    violating_entities: set[Entity] = set()
    for diagnostic in diagnostics:
        code = str(diagnostic.get("code", ""))
        if code in MISSING_DOCSTRING_CODES:
            continue

        filename = diagnostic.get("filename")
        location = diagnostic.get("location", {})
        row = int(location.get("row", 0) or 0)
        if not filename or row <= 0:
            continue

        file_entities = entities_by_file.get(Path(filename).resolve(), [])
        if not file_entities:
            continue

        containing = [
            entity
            for entity in file_entities
            if entity.start_line <= row <= entity.end_line and entity.documented
        ]
        if not containing:
            continue

        # Pick the narrowest containing entity to map nested definitions correctly.
        target = min(containing, key=lambda item: (item.end_line - item.start_line, item.start_line))
        violating_entities.add(target)

    for category_label, category_key in CATEGORY_ORDER:
        del category_label  # avoid unused variable in future refactors
        documented_count = len(documented_entities_by_category[category_key])
        violating_count = sum(
            1 for entity in documented_entities_by_category[category_key] if entity in violating_entities
        )
        getattr(style, category_key).documented = documented_count - violating_count

    violating_function_paths = sorted(
        {
            entity.import_path
            for entity in violating_entities
            if entity.kind == "function"
        }
    )
    undocumented_function_paths = sorted(
        {
            entity.import_path
            for entity in function_entities
            if not entity.documented
        }
    )

    return StyleAnalysis(
        coverage=style,
        violating_function_paths=violating_function_paths,
        undocumented_function_paths=undocumented_function_paths,
    )


def format_ratio_table(rows: list[tuple[str, int, int]], value_header: str) -> str:
    """Render a plain-text table with `numerator / denominator` values.

    Args:
        rows: Table rows in `(label, numerator, denominator)` form.
        value_header: Header for the ratio column.

    Returns:
        A formatted plain-text table.
    """
    header = "metric"

    col1_width = max(len(header), *(len(name) for name, _, _ in rows))
    col2_width = max(
        len(value_header), *(len(f"{numerator} / {denominator}") for _, numerator, denominator in rows)
    )

    border = f"+-{'-' * col1_width}-+-{'-' * col2_width}-+"
    lines = [
        border,
        f"| {header.ljust(col1_width)} | {value_header.ljust(col2_width)} |",
        border,
    ]

    for name, numerator, denominator in rows:
        ratio = f"{numerator} / {denominator}"
        lines.append(f"| {name.ljust(col1_width)} | {ratio.ljust(col2_width)} |")

    lines.append(border)
    return "\n".join(lines)


def format_docstring_coverage_table(coverage: Coverage) -> str:
    """Render the docstring coverage table.

    Args:
        coverage: Coverage counters to display.

    Returns:
        A formatted plain-text table.
    """
    rows = [
        ("public functions", coverage.public_functions.documented, coverage.public_functions.total),
        ("private functions", coverage.private_functions.documented, coverage.private_functions.total),
        ("classes", coverage.classes.documented, coverage.classes.total),
        ("modules", coverage.modules.documented, coverage.modules.total),
    ]
    return format_ratio_table(rows, "documented / total")


def format_google_style_table(style: StyleCoverage) -> str:
    """Render the Google-style compliance table.

    Args:
        style: Compliance counters for documented entities.

    Returns:
        A formatted plain-text table.
    """
    rows = [
        ("public functions", style.public_functions.documented, style.public_functions.total),
        ("private functions", style.private_functions.documented, style.private_functions.total),
        ("classes", style.classes.documented, style.classes.total),
        ("modules", style.modules.documented, style.modules.total),
    ]
    return format_ratio_table(rows, "google-style / documented")


def format_path_list(title: str, paths: list[str]) -> str:
    """Render one titled list of dotted import paths.

    Args:
        title: Section title.
        paths: Import paths to render.

    Returns:
        A formatted section.
    """
    lines = [title]
    if not paths:
        lines.append("(none)")
    else:
        lines.extend(paths)
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for this script.

    Returns:
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Measure how many functions, classes, and modules have docstrings."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("src/satellit_sam")],
        help="Files or directories to scan (default: src/satellit_sam).",
    )
    return parser.parse_args()


def main() -> int:
    """Run the docstring coverage report CLI.

    Returns:
        Process exit code.
    """
    args = parse_args()
    coverage = scan(args.paths)
    style_analysis = build_style_analysis(args.paths)
    print(format_docstring_coverage_table(coverage))
    print()
    print(format_google_style_table(style_analysis.coverage))
    print()
    print(
        format_path_list(
            "Google-style violating functions (absolute import path):",
            style_analysis.violating_function_paths,
        )
    )
    print()
    print(
        format_path_list(
            "Undocumented functions (absolute import path):",
            style_analysis.undocumented_function_paths,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
