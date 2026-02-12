---
title: "Goals and Scope"
description: "Objective and boundaries for SAM3 transfer learning in satellite forest segmentation."
---

# Goals and Scope

This documentation adapts `facebook/sam3` (used via `Sam3Model` and `Sam3Processor` in `satellit_sam/src/satellit_sam/sam3.py`) to satellite and orthophoto forest segmentation (tree crowns, canopy regions, optional species classes).

Primary objective:

- Improve segmentation quality over zero-shot prompting on local data.

Secondary objectives:

- Keep training practical on available GPU hardware.
- Preserve the current tiling pipeline and inference flow.

See also:

- [Transfer Learning vs Fine-Tuning](./fundamentals.md)
- [Phased Strategy](./phased_strategy.md)
- [Evaluation and Decision Matrix](./evaluation_decision.md)
