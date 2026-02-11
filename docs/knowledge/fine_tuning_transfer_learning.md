# Plan: Fine-Tuning and Transfer Learning for SAM3 (Satellite Forest Segmentation)

## 1) Goals and Scope

This plan adapts `facebook/sam3` (already used via `Sam3Model` + `Sam3Processor` in `with_sam/src/satellit_sam/sam3.py`) to your domain: satellite/orthophoto forest segmentation (e.g., tree crowns, canopy regions, optional species classes).

Primary objective:
- Improve segmentation quality over zero-shot prompting on your local data.

Secondary objectives:
- Keep training practical on available GPU hardware.
- Preserve your current tiling pipeline and inference flow.

---

## 2) What Fine-Tuning and Transfer Learning Mean

### Transfer learning
Transfer learning means taking a pretrained model (SAM3) and adapting it to a new domain/task instead of training from scratch.

In practice here:
- Start from pretrained SAM3 weights.
- Train only selected parts (or small adapter modules) on your forest data.

### Fine-tuning
Fine-tuning is the training step used during transfer learning.

In practice here:
- Continue training SAM3 on labeled forest imagery.
- Use lower learning rates and domain-specific data.

Relationship:
- Transfer learning is the strategy.
- Fine-tuning is the mechanism.

---

## 3) Recommended Strategy (Phased)

Use a staged approach to reduce risk and cost.

### Phase A: Baseline (no architecture change)
- Evaluate current zero-shot SAM3 prompt workflow (`text="trees"`, etc.) on a labeled validation set.
- Record baseline metrics (mIoU, Dice/F1, AP@50 for instance masks).

### Phase B: Parameter-efficient transfer learning (recommended first)
- Freeze most of SAM3.
- Train either:
  - Mask decoder only, or
  - LoRA adapters in selected attention layers + decoder.
- This is usually the best speed/quality tradeoff.

### Phase C: Targeted architecture changes
- Add domain-specific modules for satellite data (details in section 4).
- Train adapters + new modules first; only then consider wider unfreezing.

### Phase D: Full/near-full fine-tuning (only if needed)
- Unfreeze large parts of the image encoder.
- Use only if Phase B/C plateau and enough data + compute are available.

---

## 4) Architecture Change Options (from low risk to high impact)

The SAM family generally has: image encoder, prompt encoder, mask decoder. Keep this structure unless a clear bottleneck appears.

### Option 1: Decoder-only fine-tuning (low risk)
What changes:
- No structural changes.
- Train mask decoder parameters only.
- Freeze image encoder and most prompt encoder.

Why:
- Fast to run, strong baseline improvement in many domain shifts.

### Option 2: LoRA/adapter tuning on attention blocks (recommended)
What changes:
- Inject LoRA adapters into selected attention projections of encoder/decoder.
- Keep base weights mostly frozen.

Why:
- Better adaptation than decoder-only with modest memory usage.

Suggested starting config:
- LoRA rank: 8-16
- LoRA alpha: 16-32
- LoRA dropout: 0.05-0.1

### Option 3: Multimodal adapter for RGB + heightmap/DSM (very relevant here)
What changes:
- Add a small input adapter for a 4th channel (heightmap), then feed into SAM3 image pipeline.

Two practical designs:
1. `Conv1x1` projection from 4 channels -> 3 channels before SAM3 processor.
2. Dual-branch fusion: separate tiny CNN for heightmap, fuse tokens/features before decoder.

Why:
- Your pipeline already creates heightmaps from LAS. This can materially improve crown boundary separation and reduce false positives.

### Option 4: Task head extension for semantic + instance output
What changes:
- Keep SAM3 mask decoder for instances.
- Add lightweight semantic segmentation head for coarse canopy/background supervision.

Why:
- Multi-task training stabilizes learning in noisy labels and improves generalization.

### Option 5: Broad encoder unfreezing (high compute)
What changes:
- Unfreeze upper encoder blocks progressively.

Why:
- Highest adaptation capacity, but strong overfitting/instability risk on smaller datasets.

---

## 5) Training Data Requirements

### Data types needed
- High-resolution orthophotos/satellite tiles (RGB at minimum).
- Optional but recommended: aligned height data (DSM/CHM from LAS), if available.
- Pixel-level masks:
  - Instance masks for tree crowns (primary target).
  - Optional semantic masks (forest/canopy/background).

### Label format
Use one canonical training format:
- COCO instance segmentation (`images`, `annotations`, `categories`), or
- Internal format converted to COCO during preprocessing.

Include:
- `image_id`, `category_id`, segmentation polygons/RLE, area, bbox.
- Quality flag per annotation (high/medium/low confidence).

### Dataset size guidance
- Pilot: 500-2,000 labeled tiles (e.g., 512-1024 px).
- Strong model: 5,000+ diverse tiles.
- Ideal: multiple geographies, seasons, sensor conditions.

### Split strategy (critical)
Prevent spatial leakage:
- Split by region/time, not random tile-level only.
- Suggested: 70% train, 15% val, 15% test by geographic blocks.

### Data balancing
Ensure representation across:
- Dense forest, sparse tree cover, urban trees, shadow-heavy scenes, mixed terrain.
- Different sun angles/seasons/resolutions.

### Augmentations
Use geospatial-safe augmentations:
- Horizontal/vertical flip, 90° rotations.
- Mild brightness/contrast/haze perturbation.
- Optional blur/noise to simulate sensor variation.

Avoid aggressive geometric warps that break geospatial realism.

---

## 6) Proposed Technology Stack

Fits your current repository (`pixi`, `torch`, `transformers`):

Core training:
- PyTorch
- Hugging Face Transformers (`Sam3Model`, `Sam3Processor`)
- PEFT (LoRA/adapters)
- Accelerate (multi-GPU/mixed precision orchestration)

Data + geospatial:
- Rasterio/GDAL (tile IO and georeferencing)
- NumPy
- pycocotools (mask handling + evaluation helpers)
- Albumentations (augmentation pipeline)

Experiment tracking:
- MLflow (local-first) or Weights & Biases (if cloud logging is acceptable)

Evaluation:
- TorchMetrics + custom mask metrics
- COCO-style AP metrics for instance segmentation

Packaging/execution:
- Keep Pixi environments (`default`, `cuda`)
- Add training/eval tasks in `with_sam/pixi.toml`

---

## 7) Implementation Plan in This Repository

### Step 1: Training module scaffolding
Add modules under `with_sam/src/satellit_sam/training/`:
- `dataset.py`: COCO + tiling dataset loader
- `augmentations.py`: train/val transform pipelines
- `modeling.py`: SAM3 wrapper with freeze/unfreeze + LoRA hooks
- `losses.py`: Dice + BCE/Focal (+ optional boundary loss)
- `trainer.py`: training loop (Accelerate)
- `evaluate.py`: metrics and report generation
- `config.py`: typed config dataclasses

### Step 2: Config-driven experiments
Create YAML/TOML experiment configs:
- `configs/exp_baseline_decoder_only.yaml`
- `configs/exp_lora_decoder_encoder.yaml`
- `configs/exp_rgb_height_adapter.yaml`

Track in each config:
- Model checkpoint
- Frozen modules
- LoRA targets + rank
- LR schedule
- Batch size/grad accumulation
- Augmentations
- Prompt strategy

### Step 3: Data pipeline
- Build a preprocessing script to:
  - read large orthophotos,
  - produce training tiles,
  - align labels,
  - export COCO JSON.
- Store provenance metadata (source scene, date, sensor, region id).

### Step 4: Train in sequence
1. Decoder-only tuning.
2. LoRA tuning.
3. RGB+height adapter (if heightmap data quality is sufficient).
4. Optional partial encoder unfreeze.

Use early stopping on validation mIoU/Dice.

### Step 5: Integrate inference
- Export best checkpoint and config.
- Update `with_sam/src/satellit_sam/sam3.py` to optionally load fine-tuned weights.
- Keep fallback to `facebook/sam3` if checkpoint not provided.

### Step 6: Regression checks
- Add tests that verify:
  - model loads with/without finetuned checkpoint,
  - output shape consistency,
  - inference speed bounds on representative tile sizes,
  - no crash on CPU path.

---

## 8) Losses, Metrics, and Success Criteria

### Losses
- Primary: Dice + BCE (or Focal for class imbalance).
- Optional: boundary-aware loss for crown edges.

### Metrics
- mIoU
- Dice/F1
- AP50/AP75 for instances
- Precision/Recall on small-object crowns
- Inference time per tile

### Success criteria (example)
- +8-15% Dice over zero-shot baseline on held-out regions.
- AP50 improvement without >20% inference slowdown.
- Stable performance across at least 3 distinct geographic validation blocks.

---

## 9) Compute Plan

Recommended starting setup:
- 1-2 NVIDIA GPUs (24GB+ VRAM preferred for larger tiles).
- Mixed precision (`bf16` or `fp16`) with gradient accumulation.
- Tile sizes: start at 512-1024 for training.

Order of cost:
- Decoder-only < LoRA < RGB+height adapter < partial/full encoder fine-tune.

---

## 10) Risks and Mitigations

- Domain shift across regions/seasons:
  - Mitigation: spatially separated splits + diverse sampling.

- Label noise from manual polygoning:
  - Mitigation: QA pass, confidence-weighted losses, ignore uncertain masks.

- Overfitting on limited data:
  - Mitigation: freeze strategy, LoRA first, stronger validation protocol.

- Geospatial misalignment (RGB vs heightmap):
  - Mitigation: strict co-registration checks before multimodal training.

---

## 11) Suggested 4-Week Execution Timeline

Week 1:
- Build dataset export + baseline evaluation harness.
- Collect baseline zero-shot metrics.

Week 2:
- Implement decoder-only + LoRA experiments.
- Run first ablations.

Week 3:
- Implement RGB+height adapter experiment.
- Compare against LoRA-only best model.

Week 4:
- Integrate best checkpoint into inference path.
- Add tests, produce final report and deployment checklist.

---

## 12) Concrete First Experiments to Run

1. `decoder_only`:
- Freeze encoder/prompt, train decoder, LR 1e-4, 20 epochs.

2. `lora_rank_8`:
- LoRA on attention in decoder + top encoder blocks, LR 2e-4, 20 epochs.

3. `lora_rank_16`:
- Same as above, compare quality vs memory/time.

4. `rgb_plus_height_adapter`:
- 4->3 adapter + LoRA decoder, LR 1e-4, 25 epochs.

Compare all on the same held-out geographic test set.

---

## 13) Inputs I Still Need From You

To finalize hyperparameters and implementation details, I need:
- GPU availability (type and VRAM).
- Approximate number of labeled masks/images currently available.
- Target output task:
  - only tree crowns,
  - forest vs non-forest,
  - or multi-class species/land-cover.
- Whether heightmap/DSM is reliably aligned for most training scenes.

If you share these, I can convert this into an exact training spec with initial config files and command templates.
