---
title: "Transfer Learning vs Fine-Tuning"
description: "Core terminology and relationship between transfer learning and fine-tuning for SAM3."
---

# Transfer Learning vs Fine-Tuning

## Transfer learning

Transfer learning means taking a pretrained model (SAM3) and adapting it to a new domain or task instead of training from scratch.

In this project:
- Start from pretrained SAM3 weights.
- Train only selected parts (or small adapter modules) on forest data.

## Fine-tuning

Fine-tuning is the training step used during transfer learning.

In this project:
- Continue training SAM3 on labeled forest imagery.
- Use lower learning rates and domain-specific data.

Relationship:
- Transfer learning is the strategy.
- Fine-tuning is the mechanism.

See also:
- [Phased Strategy](./phased_strategy.md)
- [Transfer Learning Methods](./adaptation_methods.md)
