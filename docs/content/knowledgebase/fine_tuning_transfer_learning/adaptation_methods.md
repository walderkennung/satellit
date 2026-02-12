---
title: "Transfer Learning Methods"
description: "Method catalog from low-risk decoder tuning to pseudo-label self-training."
---

# Transfer Learning Methods

The SAM family has three core parts: image encoder, prompt encoder, and mask decoder. Prefer minimal changes unless there is a clear bottleneck.

## 1) Decoder-only fine-tuning
- No structural changes.
- Train mask decoder only.
- Freeze image encoder and most prompt encoder.

Why:
- Fast and often provides a strong first improvement.

## 2) LoRA or adapters in attention blocks
- Inject LoRA adapters in selected attention projections.
- Keep base weights mostly frozen.

Suggested start:
- LoRA rank: 8-16
- LoRA alpha: 16-32
- LoRA dropout: 0.05-0.1

## 3) Multimodal adapter for RGB + heightmap/DSM
- Add a small input adapter for a fourth channel (heightmap).

Practical designs:
1. `Conv1x1` projection from 4 channels to 3 channels before SAM3 preprocessing.
2. Dual-branch fusion with a small heightmap branch and feature fusion before decoder.

## 4) Multi-task head extension
- Keep SAM3 mask decoder for instance masks.
- Add a lightweight semantic head for canopy/background supervision.

## 5) Broad encoder unfreezing
- Progressively unfreeze upper encoder blocks.

## 6) Prompt tuning or learned prompts
- Keep encoder mostly frozen.
- Learn prompt tokens or prompt-encoder adapters.

## 7) BitFit or IA3
- BitFit: train selected bias terms.
- IA3: train multiplicative vectors in attention/MLP paths.

## 8) Progressive unfreezing with layer-wise LR decay
- Start from decoder or adapter tuning.
- Unfreeze top encoder blocks stepwise with smaller learning rates deeper in the encoder.

## 9) Teacher-student self-training
- Generate pseudo-labels with best current model.
- Filter by confidence and retrain student model.

## 10) Continual adaptation with replay
- Fine-tune on new region or season data.
- Replay representative old tiles to reduce forgetting.

See also:
- [Phased Strategy](./phased_strategy.md)
- [SAM Architecture Variants](./sam_architecture_variants.md)
- [Experiment Roadmap](./experiment_roadmap.md)
