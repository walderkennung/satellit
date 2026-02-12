---
title: "Implementation Plan"
description: "Repository-oriented implementation steps for training and integration."
---

# Implementation Plan

## Step 1: Training modules
Create modules in `with_sam/src/satellit_sam/training/`:
- `dataset.py`: COCO and tiling loader
- `augmentations.py`: train/validation transforms
- `modeling.py`: SAM3 wrapper with freeze/unfreeze and adapter hooks
- `losses.py`: Dice plus BCE/Focal
- `trainer.py`: training loop
- `evaluate.py`: evaluation and reports
- `config.py`: typed config objects

## Step 2: Config-driven experiments
Add experiment configs:
- `configs/exp_baseline_decoder_only.yaml`
- `configs/exp_lora_decoder_encoder.yaml`
- `configs/exp_rgb_height_adapter.yaml`

Track:
- checkpoint,
- frozen modules,
- adapter targets and rank,
- optimizer and schedule,
- batch and accumulation,
- augmentations,
- prompt strategy.

## Step 3: Data preprocessing pipeline
- Read orthophotos.
- Generate training tiles.
- Align labels.
- Export COCO JSON.
- Store provenance metadata (scene, date, sensor, region).

## Step 4: Training sequence
1. Decoder-only tuning.
2. LoRA tuning.
3. Backbone scale sweep with best parameter-efficient recipe.
4. RGB plus height adapter.
5. Progressive encoder unfreezing.
6. Optional pseudo-label self-training loop.

Use early stopping based on validation mIoU and Dice.

## Step 5: Inference integration
- Export best checkpoint and config.
- Update `with_sam/src/satellit_sam/sam3.py` to load fine-tuned weights optionally.
- Keep fallback to pretrained `facebook/sam3` when checkpoint is not provided.

## Step 6: Regression checks
- Model loads with and without fine-tuned checkpoint.
- Output shape consistency.
- Inference speed bounds on representative tile sizes.
- No CPU-path crash.

See also:
- [Data Requirements](./data_requirements.md)
- [Experiment Roadmap](./experiment_roadmap.md)
- [Evaluation and Decision Matrix](./evaluation_decision.md)
