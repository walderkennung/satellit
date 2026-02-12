---
title: "Evaluation and Decision Matrix"
description: "Losses, metrics, risks, compute guidance, and promotion criteria."
---

# Evaluation and Decision Matrix

## Losses
- Primary: Dice plus BCE (or Focal for imbalance).
- Optional: boundary-aware loss for crown edges.

## Metrics
- mIoU
- Dice/F1
- AP50/AP75 for instances
- Precision and recall on small-object crowns
- Inference time per tile

## Baseline success criteria
- +8-15% Dice over zero-shot baseline on held-out regions.
- AP50 improvement without more than 20% inference slowdown.
- Stable performance across at least three geographic validation blocks.

## Compute planning
- Start with 1-2 NVIDIA GPUs (24 GB+ VRAM preferred for larger tiles).
- Use mixed precision (`bf16` or `fp16`) and gradient accumulation.
- Start with tile sizes of 512-1024.

Relative cost:
- Decoder-only < LoRA < RGB+height adapter < partial/full encoder fine-tune.

## Risk register
- Domain shift across regions and seasons:
  - Mitigation: spatially separated splits and diverse sampling.
- Label noise from manual polygons:
  - Mitigation: QA pass, confidence-weighted losses, uncertain-mask exclusion.
- Overfitting on limited data:
  - Mitigation: freeze-first strategy, LoRA-first trials, strict validation protocol.
- RGB/height misalignment:
  - Mitigation: co-registration checks before multimodal training.

## Production decision matrix
Weighted selection score:
- 50% segmentation quality (Dice + AP50/AP75)
- 25% robustness (variance across blocks and seasons)
- 15% runtime (tiles/sec and memory)
- 10% operational simplicity (integration and maintenance cost)

Promotion gate:
- Beat baseline on each geographic validation block.
- Stay within runtime budget for expected tiling volume.
- Stay stable in shadow-heavy and sparse-tree scenes.

See also:
- [Experiment Roadmap](./experiment_roadmap.md)
- [Data Requirements](./data_requirements.md)
- [Implementation Plan](./implementation_plan.md)
