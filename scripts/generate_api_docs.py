#! /usr/bin/env -S pixi exec --spec ruff --spec python>=3.10 -- python

"""Generate Markdown API reference pages for satellit_sam.

This script parses Python source files with `ast` and writes Markdown pages to
`docs/content/api/` for the docmd static site.

Requires Python 3.10+ and `ruff` for signature formatting.

Arguments:
    --package-name: Python package name used in module paths (default: "satellit_sam").
    --source-dir: Directory containing Python package source files (default: "./satellit_sam/src/satellit_sam").
    --output-dir: Output directory for generated Markdown files (default: "./docs/content/api").
    --content-dir: Docmd content root directory used to compute site routes (default: "./docs/content").
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def is_public(name: str) -> bool:
    return not name.startswith("_")


def safe_unparse(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def format_arg(arg: ast.arg, default: ast.AST | None = None) -> str:
    text = arg.arg
    annotation = safe_unparse(arg.annotation)
    if annotation:
        text += f": {annotation}"
    if default is not None:
        text += f" = {safe_unparse(default)}"
    return text


def format_function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = node.args
    parts: list[str] = []

    pos_and_regular = args.posonlyargs + args.args
    positional_defaults: list[ast.AST | None] = [None] * (
        len(pos_and_regular) - len(args.defaults)
    ) + list(args.defaults)

    for idx, arg in enumerate(pos_and_regular):
        parts.append(format_arg(arg, positional_defaults[idx]))
        if args.posonlyargs and idx == len(args.posonlyargs) - 1:
            parts.append("/")

    if args.vararg is not None:
        vararg = args.vararg.arg
        vararg_ann = safe_unparse(args.vararg.annotation)
        if vararg_ann:
            parts.append(f"*{vararg}: {vararg_ann}")
        else:
            parts.append(f"*{vararg}")
    elif args.kwonlyargs:
        parts.append("*")

    for kwarg, kwdefault in zip(args.kwonlyargs, args.kw_defaults):
        parts.append(format_arg(kwarg, kwdefault))

    if args.kwarg is not None:
        kwarg_ann = safe_unparse(args.kwarg.annotation)
        if kwarg_ann:
            parts.append(f"**{args.kwarg.arg}: {kwarg_ann}")
        else:
            parts.append(f"**{args.kwarg.arg}")

    return_annotation = safe_unparse(node.returns)
    return_text = f" -> {return_annotation}" if return_annotation else ""
    prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    return f"{prefix}{node.name}({', '.join(parts)}){return_text}"


@dataclass
class FunctionDoc:
    name: str
    signature: str
    docstring: str


@dataclass
class ClassDoc:
    name: str
    docstring: str
    attributes: list[str]
    methods: list[FunctionDoc]


@dataclass
class ModuleDoc:
    name: str
    source_file: Path
    docstring: str
    functions: list[FunctionDoc]
    classes: list[ClassDoc]


@dataclass
class ParsedDocstring:
    description: str
    arguments: list[str]
    returns: list[str]
    exceptions: list[str]


SECTION_ALIASES = {
    "args": "arguments",
    "arg": "arguments",
    "arguments": "arguments",
    "parameters": "arguments",
    "params": "arguments",
    "returns": "returns",
    "return": "returns",
    "yields": "returns",
    "yield": "returns",
    "raises": "exceptions",
    "raise": "exceptions",
    "exceptions": "exceptions",
}


def canonical_section_name(raw_name: str) -> str | None:
    key = raw_name.strip().rstrip(":").strip().lower()
    return SECTION_ALIASES.get(key)


def normalize_section_items(items: list[str]) -> list[str]:
    normalized: list[str] = []
    for item in items:
        cleaned = re.sub(r"\s+", " ", item).strip(" -")
        if cleaned:
            normalized.append(cleaned)
    return normalized


def parse_docstring_sections(docstring: str) -> ParsedDocstring:
    if not docstring.strip():
        return ParsedDocstring(
            description="_No docstring._",
            arguments=[],
            returns=[],
            exceptions=[],
        )

    lines = docstring.expandtabs(4).splitlines()
    description_lines: list[str] = []
    sections: dict[str, list[str]] = {
        "arguments": [],
        "returns": [],
        "exceptions": [],
    }

    current_section: str | None = None
    current_item: str | None = None

    def flush_current_item() -> None:
        nonlocal current_item
        if current_section is not None and current_item:
            sections[current_section].append(current_item.strip())
        current_item = None

    idx = 0
    while idx < len(lines):
        raw = lines[idx]
        stripped = raw.strip()

        numpy_section = canonical_section_name(stripped)
        has_numpy_underline = idx + 1 < len(lines) and bool(
            re.fullmatch(r"-{3,}", lines[idx + 1].strip())
        )
        if numpy_section and has_numpy_underline:
            flush_current_item()
            current_section = numpy_section
            idx += 2
            continue

        google_section = None
        if stripped.endswith(":"):
            google_section = canonical_section_name(stripped[:-1])
        if google_section:
            flush_current_item()
            current_section = google_section
            idx += 1
            continue

        if current_section is None:
            description_lines.append(raw.rstrip())
            idx += 1
            continue

        if not stripped:
            flush_current_item()
            idx += 1
            continue

        if raw[:1].isspace():
            if current_item:
                current_item += " " + stripped
            else:
                current_item = stripped
        else:
            flush_current_item()
            current_item = stripped

        idx += 1

    flush_current_item()

    description = "\n".join(description_lines).strip()
    if not description:
        description = "_No docstring._"

    return ParsedDocstring(
        description=description,
        arguments=normalize_section_items(sections["arguments"]),
        returns=normalize_section_items(sections["returns"]),
        exceptions=normalize_section_items(sections["exceptions"]),
    )


def signature_for_code_block(signature: str) -> str:
    if signature.startswith("def ") or signature.startswith("async def "):
        return signature
    if signature.startswith("async "):
        return "async def " + signature[len("async ") :]
    return "def " + signature


def split_top_level_commas(text: str) -> list[str]:
    items: list[str] = []
    buffer: list[str] = []
    depth = 0
    quote: str | None = None
    escape = False

    for char in text:
        if quote is not None:
            buffer.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                quote = None
            continue

        if char in ("'", '"'):
            quote = char
            buffer.append(char)
            continue

        if char in "([{":
            depth += 1
            buffer.append(char)
            continue
        if char in ")]}":
            depth = max(depth - 1, 0)
            buffer.append(char)
            continue

        if char == "," and depth == 0:
            item = "".join(buffer).strip()
            if item:
                items.append(item)
            buffer = []
            continue

        buffer.append(char)

    tail = "".join(buffer).strip()
    if tail:
        items.append(tail)

    return items


def format_signature_without_ruff(
    signature: str, max_length: int = 88, indent: str = "    "
) -> list[str]:
    base = signature_for_code_block(signature)
    signature_body = base.removeprefix("async def ").removeprefix("def ")
    is_async = base.startswith("async def ")
    prefix = "async def " if is_async else "def "

    match = re.match(r"^([A-Za-z_]\w*)\((.*)\)(?:\s*->\s*(.*))?$", signature_body)
    if not match:
        return [base + ":"]

    name = match.group(1)
    args_text = match.group(2).strip()
    return_text = (match.group(3) or "").strip()

    one_line = f"{prefix}{name}({args_text})"
    if return_text:
        one_line += f" -> {return_text}"
    one_line += ":"
    if len(one_line) <= max_length:
        return [one_line]

    args = split_top_level_commas(args_text) if args_text else []
    if not args:
        close_line = ")"
        if return_text:
            close_line += f" -> {return_text}"
        close_line += ":"
        return [f"{prefix}{name}(", close_line]

    lines = [f"{prefix}{name}("]
    for arg in args:
        if arg in ("*", "/"):
            lines.append(f"{indent}{arg}")
        else:
            lines.append(f"{indent}{arg},")

    close_line = ")"
    if return_text:
        close_line += f" -> {return_text}"
    close_line += ":"
    lines.append(close_line)
    return lines


@lru_cache(maxsize=1024)
def format_signature_for_block(signature: str) -> list[str]:
    base = signature_for_code_block(signature)
    snippet = f"{base}:\n    pass\n"

    try:
        result = subprocess.run(
            ["ruff", "format", "--stdin-filename", "signature.py", "-"],
            input=snippet,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return format_signature_without_ruff(signature)

    if result.returncode != 0 or not result.stdout.strip():
        return format_signature_without_ruff(signature)

    lines = result.stdout.splitlines()
    pass_idx = next(
        (idx for idx, line in enumerate(lines) if line.strip() == "pass"), -1
    )
    if pass_idx <= 0:
        return format_signature_without_ruff(signature)

    signature_lines = lines[:pass_idx]
    while signature_lines and not signature_lines[-1].strip():
        signature_lines.pop()

    if not signature_lines:
        return format_signature_without_ruff(signature)

    if not signature_lines[0].lstrip().startswith(("def ", "async def ")):
        return format_signature_without_ruff(signature)

    return signature_lines


def render_callable_block(
    lines: list[str], callable_doc: FunctionDoc, heading_level: int
) -> None:
    parsed = parse_docstring_sections(callable_doc.docstring)
    heading = "#" * heading_level
    signature_lines = format_signature_for_block(callable_doc.signature)

    lines.append(f"{heading} {callable_doc.name}")
    lines.append("")
    lines.append("```python")
    lines.extend(signature_lines)
    lines.append("```")
    lines.append("")
    lines.append(parsed.description)
    lines.append("")

    if parsed.arguments or parsed.returns or parsed.exceptions:
        lines.append("<details>")
        lines.append("<summary>Arguments, Returns, and Exceptions</summary>")
        lines.append("")

        if parsed.arguments:
            lines.append("#### Arguments")
            lines.append("")
            for item in parsed.arguments:
                lines.append(f"- `{item}`")
            lines.append("")

        if parsed.returns:
            lines.append("#### Returns")
            lines.append("")
            for item in parsed.returns:
                lines.append(f"- `{item}`")
            lines.append("")

        if parsed.exceptions:
            lines.append("#### Exceptions")
            lines.append("")
            for item in parsed.exceptions:
                lines.append(f"- `{item}`")
            lines.append("")

        lines.append("</details>")
        lines.append("")



def parse_module(source_file: Path, module_name: str) -> ModuleDoc:
    tree = ast.parse(source_file.read_text(encoding="utf-8"))
    module_docstring = ast.get_docstring(tree) or ""

    functions: list[FunctionDoc] = []
    classes: list[ClassDoc] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and is_public(
            node.name
        ):
            functions.append(
                FunctionDoc(
                    name=node.name,
                    signature=format_function_signature(node),
                    docstring=ast.get_docstring(node) or "",
                )
            )
            continue

        if isinstance(node, ast.ClassDef) and is_public(node.name):
            attributes: list[str] = []
            methods: list[FunctionDoc] = []

            for class_node in node.body:
                if isinstance(class_node, ast.AnnAssign) and isinstance(
                    class_node.target, ast.Name
                ):
                    if is_public(class_node.target.id):
                        annotation = safe_unparse(class_node.annotation)
                        if annotation:
                            attributes.append(f"{class_node.target.id}: {annotation}")
                        else:
                            attributes.append(class_node.target.id)
                elif isinstance(class_node, ast.Assign):
                    for target in class_node.targets:
                        if isinstance(target, ast.Name) and is_public(target.id):
                            attributes.append(target.id)
                elif isinstance(class_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if is_public(class_node.name):
                        methods.append(
                            FunctionDoc(
                                name=class_node.name,
                                signature=format_function_signature(class_node),
                                docstring=ast.get_docstring(class_node) or "",
                            )
                        )

            classes.append(
                ClassDoc(
                    name=node.name,
                    docstring=ast.get_docstring(node) or "",
                    attributes=attributes,
                    methods=methods,
                )
            )

    return ModuleDoc(
        name=module_name,
        source_file=source_file,
        docstring=module_docstring,
        functions=functions,
        classes=classes,
    )


def relative_markdown_link(from_file: Path, to_file: Path) -> str:
    """Build a relative docmd link between two markdown files without extensions."""
    relative = Path(os.path.relpath(to_file.with_suffix(""), start=from_file.parent))
    link = relative.as_posix()
    if not link.startswith((".", "#")):
        return f"./{link}"
    return link


def write_module_page(module: ModuleDoc, output_file: Path, index_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    back_link = relative_markdown_link(from_file=output_file, to_file=index_file)
    cwd = Path.cwd().resolve()
    if module.source_file.is_relative_to(cwd):
        source_display = module.source_file.relative_to(cwd).as_posix()
    elif module.source_file.is_relative_to(cwd.parent):
        source_display = module.source_file.relative_to(cwd.parent).as_posix()
    else:
        source_display = module.source_file.as_posix()

    lines: list[str] = [
        "---",
        f'title: "{module.name}"',
        f'description: "Auto-generated API reference for {module.name}."',
        f'tags: ["{module.name}", "API"]',
        "---",
        "",
        f"# `{module.name}`",
        "",
        f"> Auto-generated from `{source_display}` by `satellit_sam/scripts/generate_api_docs.py`.",
        "",
        f"[Back to API index]({back_link})",
        "",
    ]

    lines.append("## Module Docstring")
    lines.append("")
    if module.docstring:
        lines.append(module.docstring)
    else:
        lines.append("_No module docstring._")
    lines.append("")

    lines.append("## Functions")
    lines.append("")
    if module.functions:
        for function in module.functions:
            render_callable_block(lines=lines, callable_doc=function, heading_level=3)
    else:
        lines.append("_No public module-level functions._")
        lines.append("")

    lines.append("## Classes")
    lines.append("")
    if module.classes:
        for klass in module.classes:
            lines.append(f"### `{klass.name}`")
            lines.append("")
            lines.append(klass.docstring or "_No class docstring._")
            lines.append("")

            lines.append("#### Attributes")
            lines.append("")
            if klass.attributes:
                for attribute in sorted(set(klass.attributes)):
                    lines.append(f"- `{attribute}`")
            else:
                lines.append("_No public class attributes detected._")
            lines.append("")

            lines.append("#### Methods")
            lines.append("")
            if klass.methods:
                for method in klass.methods:
                    render_callable_block(
                        lines=lines, callable_doc=method, heading_level=5
                    )
            else:
                lines.append("_No public methods detected._")
                lines.append("")
    else:
        lines.append("_No public classes._")
        lines.append("")

    output_file.write_text("\n".join(lines), encoding="utf-8")


def write_index(modules: list[ModuleDoc], output_dir: Path) -> Path:
    index_path = output_dir / "index.md"

    lines: list[str] = [
        "---",
        'title: "Python API Reference"',
        'description: "Auto-generated API pages from satellit_sam docstrings."',
        "---",
        "",
        "# Python API Reference",
        "",
        "These pages are auto-generated from `satellit_sam/src/satellit_sam`.",
        "",
        "## Modules",
        "",
    ]

    if modules:
        for module in modules:
            module_path = output_dir / (module.name.replace(".", "/") + ".md")
            module_link = relative_markdown_link(from_file=index_path, to_file=module_path)
            lines.append(f"- [`{module.name}`]({module_link})")
    else:
        lines.append("_No modules found._")

    lines.append("")
    index_path.write_text("\n".join(lines), encoding="utf-8")
    return index_path


def collect_modules(source_dir: Path, package_name: str) -> list[tuple[str, Path]]:
    modules: list[tuple[str, Path]] = []

    for py_file in sorted(source_dir.rglob("*.py")):
        if py_file.name == "__init__.py":
            continue
        rel = py_file.relative_to(source_dir).with_suffix("")
        module_name = ".".join([package_name, *rel.parts])
        modules.append((module_name, py_file))

    return modules


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Markdown API reference docs from Python source files."
    )
    parser.add_argument(
        "--package-name",
        default="satellit_sam",
        help="Python package name used in module paths.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("./satellit_sam/src/satellit_sam"),
        help="Directory containing Python package source files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./docs/content/api"),
        help="Output directory for generated Markdown files.",
    )
    parser.add_argument(
        "--content-dir",
        type=Path,
        default=Path("./docs/content"),
        help="Docmd content root directory used to compute site routes.",
    )
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not source_dir.exists():
        raise SystemExit(f"Source directory does not exist: {source_dir}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    module_locations = collect_modules(
        source_dir=source_dir, package_name=args.package_name
    )
    modules: list[ModuleDoc] = []
    for module_name, source_file in module_locations:
        modules.append(parse_module(source_file=source_file, module_name=module_name))

    modules.sort(key=lambda module: module.name)
    write_index(modules=modules, output_dir=output_dir)

    for module in modules:
        module_output = output_dir / (module.name.replace(".", "/") + ".md")
        write_module_page(
            module=module,
            output_file=module_output,
            index_file=output_dir / "index.md",
        )

    print(f"Generated {len(modules)} API module page(s) in {output_dir}")


if __name__ == "__main__":
    main()
