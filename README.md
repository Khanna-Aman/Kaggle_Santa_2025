# Kaggle Santa 2025 - Christmas Tree Packing Challenge

Optimization algorithms for the [Santa 2025 Kaggle Competition](https://www.kaggle.com/competitions/santa-2025).

## Challenge

Pack `n` Christmas trees (complex 15-vertex polygons) into the smallest square bounding box for groups n=1 to 200.

**Score** = sum of (side^2 / n) for all groups

**Target:** 69.13 | **Current Best:** 164.33

## Tree Geometry

Each tree is a 15-vertex polygon:
- 3-tier triangular canopy (widths: 0.7, 0.4, 0.25)
- Rectangular trunk (0.15 x 0.2)
- Total height: 1.0 unit

## Algorithms

| File | Strategy | Trials | Est. Score |
|------|----------|--------|------------|
| `mega_packer.py` | Multi-trial + checkpoints | 20 | ~155 |
| `ultra_packer.py` | **Best** - Simulated Annealing + Compaction | 30 | <100 target |

## Quick Start

```bash
pip install shapely pandas

# Run ultra packer (best, slower)
python ultra_packer.py

# Run mega packer (faster, good baseline)
python mega_packer.py

# Quick test (n=1-10 only)
python test_ultra.py
```

## Key Techniques

1. **Ray Marching** - Walk inward until collision, back out to find tight fit
2. **Weighted Angles** - Bias placement toward corners (sin^2 distribution)
3. **Multi-Trial** - Run N random trials per group, keep best
4. **Simulated Annealing** - Post-placement refinement with random perturbations
5. **Compaction** - Shift trees toward centroid to reduce bounding box
6. **Checkpointing** - Save progress every 5 groups (resume on crash)

## File Structure

```
mega_packer.py    # 20 trials + checkpoints
ultra_packer.py   # SA + compaction (best)
test_ultra.py     # Quick test n=1-10
```

## Progress Log

| Date | Score | Notes |
|------|-------|-------|
| Dec 27 | 166.76 | First valid submission |
| Dec 27 | 164.33 | Multi-trial approach |

## License

MIT

