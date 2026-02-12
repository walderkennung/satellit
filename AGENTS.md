# AGENTS.md

This file gives coding agents practical defaults for working in this repository.
It has to be kept up to date with the current state of the repository and should be followed for all work done in this repository, including work done by human developers and coding agents.
If you find that something is missing or could be improved, please propose a change to this document.

IF THE WORKFLOW OR CONVENTIONS ARE CHANGED OR ADDED: UPDATE THIS DOCUMENT!

## Scope

- Applies to the entire repository rooted at this directory.
- Prefer small, focused changes over broad refactors.
- `with_sam` is a separate Pixi project.
- All work related to Segment Anything Model (SAM) must be placed in `with_sam`.

## Workflow

1. Read relevant files before editing.
2. Make minimal changes that solve the request.
3. Verify with available project commands where possible.
4. Summarize what changed and any limitations.

## Conventions

- Use `rg`/`rg --files` for fast search.
- Keep edits ASCII unless the file already uses Unicode.
- Do not revert unrelated local changes.
- Avoid destructive commands unless explicitly requested.

## Environment

- Use Pixi for all project-related commands in this repository.
- Do not run tools directly with global/system binaries (for example `python`, `pip`, `pytest`) when a Pixi-based invocation is possible.
- Run commands via `pixi run ...` from the relevant project directory.
- The only Pixi project currently in the repo is `with_sam` (SAM work).
- Install or sync dependencies with Pixi only.

## Documentation

- Store technical knowledge and learnings as concise, separate Markdown files in `docs/knowledge`.
- Cross-reference related knowledge documents with Markdown links.
- Keep detailed protocols for investigations and experiments in `docs/experimentation_notes`.
- Add one protocol document per investigation/experiment and keep it complete enough to reproduce the work.
- If you think that something is important to be documented propose a change to the documentation.

### Python Docstrings

- Use Google-style docstrings for Python code.
- Include type hints in function signatures where possible.
- Document parameters, return values, and exceptions in docstrings.
- For complex functions, include a brief example in the docstring.

## Validation

- Run the smallest meaningful check for the change.
- If no tests/checks exist, state that clearly.

## Notes

- `with_sam` has its own Pixi configuration and should be treated as the home for SAM-specific code and workflows.
- `with_sam` Pixi tasks (run from `with_sam/`):
- `pixi run sam-download` (download SAM checkpoints)
- `pixi run sam-test` (validate SAM setup)
- `pixi run predict-masks` (run mask prediction)
- `pixi run test` (run test suite with coverage)
- For CUDA on Linux: `pixi install -e cuda` and `pixi run -e cuda predict-masks`.
- If you add repeatable checks, document runnable commands here.
