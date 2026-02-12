---
title: "Experiment Roadmap"
description: "Prioritized sequence of experiments and escalation rules."
---

# Experiment Roadmap

## Prioritized order
Use this order to maximize signal per GPU hour:

1. Base architecture, decoder-only tuning.
2. Base architecture, LoRA (`rank=8`, then `rank=16`).
3. Backbone scale sweep with best method from steps 1-2.
4. RGB plus height adapter with best current backbone.
5. Progressive unfreezing with layer-wise LR decay.
6. Pseudo-label self-training cycle.
7. Optional domain-adaptive pretraining.

## Stop and escalation rules
- If Dice gain is less than 2% and AP50 gain is less than 1% across two consecutive experiments, escalate to the next architecture class.
- If inference slowdown is greater than 25% for less than 2% quality gain, reject for production path.
- Accept only changes that improve all held-out geographic blocks, not just global average.

## Concrete first experiment set
1. `decoder_only`: freeze encoder/prompt, train decoder, `lr=1e-4`, 20 epochs.
2. `lora_rank_8`: LoRA on decoder and top encoder attention, `lr=2e-4`, 20 epochs.
3. `lora_rank_16`: same setup, compare quality versus memory/time.
4. `rgb_plus_height_adapter`: 4->3 adapter plus decoder LoRA, `lr=1e-4`, 25 epochs.
5. `bitfit_decoder_plus_top_encoder`: bias-only tuning, `lr=5e-4`, 20 epochs.
6. `progressive_unfreeze_llrd`: unfreeze top two encoder stages, continue 10-15 epochs.
7. `backbone_scale_sweep`: repeat best recipe on small/base/large candidates.
8. `pseudo_label_round_1`: confidence-filtered pseudo-label retraining.

## Timeline template (4 weeks)
- Week 1: baseline evaluation harness and zero-shot benchmark.
- Week 2: decoder-only and LoRA experiments.
- Week 3: RGB plus height adapter and variant comparisons.
- Week 4: checkpoint integration, regression checks, and report.

See also:
- [Phased Strategy](./phased_strategy.md)
- [Transfer Learning Methods](./adaptation_methods.md)
- [Evaluation and Decision Matrix](./evaluation_decision.md)
