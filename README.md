# Walderkennung Satellit

This repository contains the `satellit_sam` CLI application and the project documentation site.

## Repository Layout

- `with_sam/`: SAM-based image processing and weak-label generation code (Pixi project)
- `docs/`: documentation site content and configuration (Bun + docmd)
- `data/`: local data inputs used during experiments/runs
- `output/`: generated outputs

## Quick Start

### CLI application (`with_sam/`)

```bash
cd with_sam
pixi install
pixi run sam-download
pixi run satellit -- --help
```

Main command groups:

- `image-processing`
- `label weak`

Detailed CLI docs:

- local docs page: `docs/content/cli/application.md`
- package README: `with_sam/README.md`

### Documentation site (`docs/`)

```bash
cd docs
bun install
bun run dev
```

Build static docs:

```bash
bun run build
```

## Notes

- Use Pixi commands for SAM/project code in this repository.
- The SAM-specific project lives under `with_sam/`.
