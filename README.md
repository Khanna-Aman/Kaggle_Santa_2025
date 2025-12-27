# 🎄 Kaggle Santa 2025 - Christmas Tree Packing Challenge

Optimization algorithms for the [Santa 2025 Kaggle Competition](https://www.kaggle.com/competitions/santa-2025).

## 🎯 Challenge

Pack `n` Christmas trees (complex polygons) into the smallest square bounding box for groups n=1 to 200. Score = Σ(side² / n).

**Target Score:** 69.13 | **Current Best:** 164.33

## 🌲 Tree Geometry

Each tree is a 15-vertex polygon with:
- 3-tier triangular canopy (widths: 0.7, 0.4, 0.25)
- Rectangular trunk (0.15 × 0.2)
- Total height: 1.0 unit

## 📁 Algorithms

| File | Strategy | Trials | Score |
|------|----------|--------|-------|
| `fast_packer_full.py` | Greedy baseline, reuses n-1 | 1 | ~179 |
| `optimized_packer_v3.py` | Multi-trial random | 5 | ~164 |
| `optimized_packer_v4.py` | Multi-trial random | 10 | ~160 |
| `mega_packer.py` | **Best** - with checkpoints | 20 | ~155 |

## 🚀 Quick Start

```bash
# Install dependencies
pip install shapely pandas numpy

# Run the mega packer (recommended)
python mega_packer.py

# Or run a quick test first
python test_optimized_v3.py
```

## 🔧 Key Techniques

1. **Ray Marching** - Walk inward from far away until collision, back out to find tight fit
2. **Weighted Angles** - Bias placement directions toward corners (sin²)
3. **Multi-Trial** - Run N random trials per group, keep best
4. **Checkpointing** - Save progress every 5 groups (resume on crash)

## 📊 Algorithm Details

### Placement Strategy
```
For each new tree:
1. Try 10-12 random rotations (0-360°)
2. For each rotation, shoot rays from origin
3. Ray march: walk in until collision, back out until clear
4. Keep the position with smallest distance from origin
```

### Scoring
```
score(n) = side² / n
total = Σ score(n) for n ∈ [1, 200]
```

## 🏗️ File Structure

```
├── mega_packer.py          # Production packer (20 trials + checkpoints)
├── optimized_packer_v3.py  # 5 trials per group
├── optimized_packer_v4.py  # 10 trials per group
├── fast_packer_full.py     # Baseline greedy
├── test_optimized_*.py     # Quick test versions (n=1-10 optimized)
└── README.md
```

## 📈 Progress Log

| Date | Score | Change | Notes |
|------|-------|--------|-------|
| Dec 27 | 166.76 | - | First valid submission |
| Dec 27 | 164.33 | -2.43 | Multi-trial approach (v3) |

## 🔬 Hypothesis Tracking

- **Rotation Search**: 8-way fixed (0,45,90...) worse than random
- **Multi-Trial**: More trials = better scores (diminishing returns after 20)
- **Reusing n-1**: Faster but locks in suboptimal placements

## 📜 License

MIT

## 🙏 Acknowledgments

Kaggle Santa Competition Team for another fun optimization puzzle!

