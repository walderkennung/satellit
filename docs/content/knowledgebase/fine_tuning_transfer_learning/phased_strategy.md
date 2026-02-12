---
title: "Phased Strategy"
description: "Low-risk-to-high-effort sequence for SAM3 adaptation."
---

# Phased Strategy

Use a staged strategy to reduce risk and control compute costs.

## Phase A: Baseline (no architecture change)
- Evaluate current zero-shot prompt workflow (`text=\"trees\"`) on labeled validation data.
- Record baseline metrics: mIoU, Dice/F1, AP@50.

## Phase B: Parameter-efficient transfer learning
- Freeze most of SAM3.
- Train either mask decoder only or LoRA adapters plus decoder.

## Phase C: Targeted architecture changes
- Add domain-specific modules.
- Train new modules first, then consider wider unfreezing.

## Phase D: Full or near-full fine-tuning
- Unfreeze larger parts of the image encoder.
- Use only if Phase B/C plateau and data plus compute are sufficient.

## Phase E: Domain-adaptive pretraining (optional)
- Continue self-supervised pretraining on large unlabeled local imagery.
- Then run Phase B/C fine-tuning on labeled data.

See also:
- [Transfer Learning Methods](./adaptation_methods.md)
- [SAM Architecture Variants](./sam_architecture_variants.md)
- [Experiment Roadmap](./experiment_roadmap.md)
