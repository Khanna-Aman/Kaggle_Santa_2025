"""
OPTIMIZED PACKER v1 - Rotation Search + Area Sort
==================================================
Target: 69.13 | Current: 166 | Gap: 97
"""

import math
import random
import numpy as np
import pandas as pd
from decimal import Decimal, getcontext
from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree
import time

getcontext().prec = 25
SCALE = Decimal('1e15')

# Tree geometry
TRUNK_W, TRUNK_H = Decimal('0.15'), Decimal('0.2')
BASE_W, MID_W, TOP_W = Decimal('0.7'), Decimal('0.4'), Decimal('0.25')
TIP_Y, TIER_1_Y, TIER_2_Y, BASE_Y = Decimal('0.8'), Decimal('0.5'), Decimal('0.25'), Decimal('0.0')

# 8-way rotations
ROTATIONS = [0, 45, 90, 135, 180, 225, 270, 315]


def make_tree_poly(cx, cy, angle):
    """Create tree polygon at position with rotation."""
    cx, cy = Decimal(str(cx)), Decimal(str(cy))
    coords = [
        (Decimal('0') * SCALE, TIP_Y * SCALE),
        (TOP_W/2 * SCALE, TIER_1_Y * SCALE), (TOP_W/4 * SCALE, TIER_1_Y * SCALE),
        (MID_W/2 * SCALE, TIER_2_Y * SCALE), (MID_W/4 * SCALE, TIER_2_Y * SCALE),
        (BASE_W/2 * SCALE, BASE_Y * SCALE),
        (TRUNK_W/2 * SCALE, BASE_Y * SCALE), (TRUNK_W/2 * SCALE, -TRUNK_H * SCALE),
        (-TRUNK_W/2 * SCALE, -TRUNK_H * SCALE), (-TRUNK_W/2 * SCALE, BASE_Y * SCALE),
        (-BASE_W/2 * SCALE, BASE_Y * SCALE),
        (-MID_W/4 * SCALE, TIER_2_Y * SCALE), (-MID_W/2 * SCALE, TIER_2_Y * SCALE),
        (-TOP_W/4 * SCALE, TIER_1_Y * SCALE), (-TOP_W/2 * SCALE, TIER_1_Y * SCALE),
    ]
    poly = Polygon(coords)
    poly = affinity.rotate(poly, float(angle), origin=(0, 0))
    poly = affinity.translate(poly, xoff=float(cx * SCALE), yoff=float(cy * SCALE))
    return poly


def has_collision(poly, placed_polys, idx):
    """Check collision using STRtree spatial index."""
    for i in idx.query(poly):
        if poly.intersects(placed_polys[i]) and not poly.touches(placed_polys[i]):
            return True
    return False


def find_best_placement(base_polys_by_rot, placed_polys, idx, num_rays=24):
    """Find best placement across all rotations and directions."""
    best_r = float('inf')
    best_result = None
    
    for deg, base_poly in base_polys_by_rot.items():
        # Try multiple ray directions
        for ray_idx in range(num_rays):
            angle = 2 * math.pi * ray_idx / num_rays
            vx, vy = math.cos(angle), math.sin(angle)
            
            # Binary search for closest valid position
            r_min, r_max = 0.0, 15.0
            
            # First check if r_max is collision-free
            px, py = r_max * vx, r_max * vy
            cand = affinity.translate(base_poly, px * float(SCALE), py * float(SCALE))
            if has_collision(cand, placed_polys, idx):
                r_max = 30.0  # Expand search
            
            # Binary search
            for _ in range(20):
                r_mid = (r_min + r_max) / 2
                px, py = r_mid * vx, r_mid * vy
                cand = affinity.translate(base_poly, px * float(SCALE), py * float(SCALE))
                if has_collision(cand, placed_polys, idx):
                    r_min = r_mid
                else:
                    r_max = r_mid
            
            # Use r_max (guaranteed no collision)
            if r_max < best_r:
                best_r = r_max
                px, py = r_max * vx, r_max * vy
                final_poly = affinity.translate(base_poly, px * float(SCALE), py * float(SCALE))
                best_result = (Decimal(str(px)), Decimal(str(py)), Decimal(str(deg)), final_poly)
    
    return best_result


def pack_group(n):
    """Pack n trees from scratch - optimized for minimum bounding box."""
    if n == 1:
        poly = make_tree_poly(0, 0, 0)
        return [{'x': Decimal('0'), 'y': Decimal('0'), 'deg': Decimal('0'), 'poly': poly}]
    
    placed = []
    
    # First tree at origin
    poly = make_tree_poly(0, 0, 0)
    placed.append({'x': Decimal('0'), 'y': Decimal('0'), 'deg': Decimal('0'), 'poly': poly})
    
    while len(placed) < n:
        polys = [p['poly'] for p in placed]
        idx = STRtree(polys)
        
        # Pre-compute base polygons for all rotations
        base_polys = {deg: make_tree_poly(0, 0, deg) for deg in ROTATIONS}
        
        # Find best placement
        result = find_best_placement(base_polys, polys, idx)
        if result:
            px, py, deg, final_poly = result
            placed.append({'x': px, 'y': py, 'deg': deg, 'poly': final_poly})
    
    return placed


def get_score(placed):
    """Calculate score for a placement."""
    if not placed:
        return float('inf')
    bounds = unary_union([p['poly'] for p in placed]).bounds
    w = (bounds[2] - bounds[0]) / float(SCALE)
    h = (bounds[3] - bounds[1]) / float(SCALE)
    side = max(w, h)
    return side * side / len(placed)


def validate_no_overlap(placed):
    """Verify zero overlaps."""
    polys = [p['poly'] for p in placed]
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
                return False
    return True


def main():
    print("\n" + "=" * 60)
    print("⚡ OPTIMIZED PACKER v1 - 8-way Rotation + Binary Search")
    print("=" * 60)

    start = time.time()

    all_trees = {}
    total = 0.0

    # Hypothesis tracking
    hypothesis_log = {
        'rotation_benefit': [],
        'scores_by_n': []
    }

    for n in range(1, 201):
        placed = pack_group(n)
        all_trees[n] = placed

        score = get_score(placed)
        total += score
        hypothesis_log['scores_by_n'].append((n, score))

        if n <= 5 or n % 25 == 0:
            bounds = unary_union([p['poly'] for p in placed]).bounds
            side = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) / float(SCALE)
            print(f"   n={n:3d}: side={side:.3f}, score={score:.3f} | total={total:.2f}")

    elapsed = time.time() - start
    print(f"\n⏱️  Time: {elapsed:.1f}s")
    print(f"📊 TOTAL: {total:.2f}")

    # Validate group 003
    if validate_no_overlap(all_trees[3]):
        print("✅ Group 003 validated - no overlaps!")
    else:
        print("❌ OVERLAP DETECTED in group 003!")

    # Save CSV
    rows = []
    for n in range(1, 201):
        for i, t in enumerate(all_trees[n]):
            rows.append({'id': f'{n:03d}_{i}', 'x': f's{t["x"]}', 'y': f's{t["y"]}', 'deg': f's{t["deg"]}'})

    pd.DataFrame(rows).to_csv('submission_optimized_v1.csv', index=False)
    print(f"\n📄 Saved: submission_optimized_v1.csv")
    print(f"🎯 Target: 69.13 | Gap: {total - 69.13:.2f}")

    # Hypothesis log
    print(f"\n📈 HYPOTHESIS LOG:")
    print(f"   Rotation: 8-way fixed angles (0,45,90,...,315)")
    print(f"   Sort: None (sequential)")
    print(f"   Best n=200 score: {hypothesis_log['scores_by_n'][-1][1]:.3f}")


if __name__ == "__main__":
    main()

