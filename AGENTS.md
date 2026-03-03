# AGENTS.md

This file gives coding agents practical defaults for working in this repository.
It has to be kept up to date with the current state of the repository and should be followed for all work done in this repository, including work done by human developers and coding agents.
If you find that something is missing or could be improved, please propose a change to this document.

IF THE WORKFLOW OR CONVENTIONS ARE CHANGED OR ADDED: UPDATE THIS DOCUMENT!

## Scope

- Applies to the entire repository rooted at this directory.
- Prefer small, focused changes over broad refactors.
- `satellit_sam` is a separate Pixi project.
- All work related to Segment Anything Model (SAM) must be placed in `satellit_sam`.

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

- Use Pixi for SAM/project code commands in this repository.
- Do not run tools directly with global/system binaries (for example `python`, `pip`, `pytest`) when a Pixi-based invocation is possible.
- Run commands via `pixi run ...` from the relevant project directory.
- The only Pixi project currently in the repo is `satellit_sam` (SAM work).
- Install or sync dependencies with Pixi only.
- Documentation site commands are run from `docs/` using `bun`

## Documentation

- Store technical knowledge and learnings as concise, separate Markdown files in `docs/content/knowledgebase`.
- Cross-reference related knowledge documents with Markdown links.
- Keep detailed protocols for investigations and experiments in `docs/content/experimentation_notes`.
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

- `satellit_sam` has its own Pixi configuration and should be treated as the home for SAM-specific code and workflows.
- `satellit_sam` Pixi tasks (run from `satellit_sam/`):
- `pixi run satellit` (run main satellit CLI)
- `pixi run sam-download` (download SAM checkpoints)
- `pixi run sam-test` (validate SAM setup)
- `pixi run test` (run test suite with coverage)
- `pixi run docstring-coverage` (print docstring coverage counts plus Google-style compliance for modules, classes, and functions)
- For CUDA on Linux: `pixi install -e cuda` and `pixi run -e cuda predict-masks`.
- `satellit -- label weak` writes per-tree weak-label crown bounding boxes (`bbox_x1,bbox_y1,bbox_x2,bbox_y2`) into `labels_tiles.csv` and `labels_tiles.shp`.
- `satellit -- label by-bounding-boxes --weak-labels-csv <labels_tiles.csv>` consumes those stored tile-local bboxes as SAM box prompts.
- `satellit -- predict image-masks --image <image>` runs streamed-tile segmentation mask prediction from supported prompts and writes `image_masks_visualization.png` plus raw masks at `masks/image_masks.npz`.
- `satellit -- predict image-masks --model <sam3|sam2|dinov3>` selects the segmentation backend; default is `sam3` when omitted. `--text` is supported with `sam3` and `dinov3`, but not `sam2`. `dinov3` currently supports text prompts only. For current SAM3 processor support, point prompts are approximated as small box prompts.
- `satellit -- predict image-masks --tile-size <int> --tile-overlap <int> --merge-iou-threshold <float>` controls streamed prediction tiles and global cross-tile NMS merge.
- Documentation site tasks (run from `docs/`):
- `bun run gen:api` (generate API docs via Pixi task)
- `bun run dev` (generate API docs, then run docmd dev server)
- `bun run build` (generate API docs, then build static site)
- GitHub Pages deployment workflow: `.github/workflows/docs-pages.yml` builds and deploys docs on pushes to `main` that touch `docs/**`, `scripts/generate_api_docs.py`, or `satellit_sam/src/satellit_sam/**`.
- If you add repeatable checks, document runnable commands here.
