---
title: "Data Requirements"
description: "Data types, splits, annotation format, and augmentation guidance for SAM3 adaptation."
---

# Data Requirements

## Required data types
- High-resolution orthophotos or satellite tiles (RGB minimum).
- Optional but recommended: aligned DSM/CHM height data.
- Pixel masks:
  - Instance masks for tree crowns.
  - Optional semantic masks for canopy/background.

## Label format
Use one canonical format:
- COCO instance segmentation (`images`, `annotations`, `categories`), or
- Internal format converted to COCO during preprocessing.

Include:
- `image_id`, `category_id`, segmentation polygons or RLE, area, bbox.
- Annotation quality flag (`high`, `medium`, `low`).

## Dataset size guidance
- Pilot: 500-2,000 labeled tiles (512-1024 px).
- Strong model: 5,000+ diverse tiles.
- Ideal: multiple geographies, seasons, and sensor conditions.

## Split strategy
- Split by region/time, not random tile-only splitting.
- Suggested split: 70% train, 15% validation, 15% test by geographic blocks.

## Balancing and augmentation
- Balance across dense forest, sparse cover, urban trees, shadows, mixed terrain, and season variation.
- Use geospatial-safe augmentations:
  - horizontal and vertical flips,
  - 90-degree rotations,
  - mild brightness/contrast/haze,
  - optional blur and noise.
- Avoid heavy geometric warps that break geospatial realism.

See also:
- [SAM Architecture Variants](./sam_architecture_variants.md)
- [Implementation Plan](./implementation_plan.md)
- [Evaluation and Decision Matrix](./evaluation_decision.md)
