"""
TEST OPTIMIZED v1 - Quick validation (~30s)
============================================
Tests new logic on n=1-10, uses fast fallback for 11-200.
Produces FULL valid Kaggle submission.
"""

import math
import pandas as pd
from decimal import Decimal, getcontext
from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree
import time

getcontext().prec = 25
SCALE = Decimal('1e15')

TRUNK_W, TRUNK_H = Decimal('0.15'), Decimal('0.2')
BASE_W, MID_W, TOP_W = Decimal('0.7'), Decimal('0.4'), Decimal('0.25')
TIP_Y, TIER_1_Y, TIER_2_Y, BASE_Y = Decimal('0.8'), Decimal('0.5'), Decimal('0.25'), Decimal('0.0')

ROTATIONS = [0, 45, 90, 135, 180, 225, 270, 315]


def make_tree_poly(cx, cy, angle):
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
    for i in idx.query(poly):
        if poly.intersects(placed_polys[i]) and not poly.touches(placed_polys[i]):
            return True
    return False


def pack_optimized(n):
    """Optimized packing with 8-way rotation + binary search."""
    placed = []
    poly = make_tree_poly(0, 0, 0)
    placed.append({'x': Decimal('0'), 'y': Decimal('0'), 'deg': Decimal('0'), 'poly': poly})
    
    while len(placed) < n:
        polys = [p['poly'] for p in placed]
        idx = STRtree(polys)
        base_polys = {deg: make_tree_poly(0, 0, deg) for deg in ROTATIONS}
        
        best_r, best_result = float('inf'), None
        for deg, base_poly in base_polys.items():
            for ray_idx in range(24):
                angle = 2 * math.pi * ray_idx / 24
                vx, vy = math.cos(angle), math.sin(angle)
                r_min, r_max = 0.0, 15.0
                
                for _ in range(20):
                    r_mid = (r_min + r_max) / 2
                    cand = affinity.translate(base_poly, r_mid * vx * float(SCALE), r_mid * vy * float(SCALE))
                    if has_collision(cand, polys, idx):
                        r_min = r_mid
                    else:
                        r_max = r_mid
                
                if r_max < best_r:
                    best_r = r_max
                    px, py = r_max * vx, r_max * vy
                    final = affinity.translate(base_poly, px * float(SCALE), py * float(SCALE))
                    best_result = (Decimal(str(px)), Decimal(str(py)), Decimal(str(deg)), final)
        
        if best_result:
            placed.append({'x': best_result[0], 'y': best_result[1], 'deg': best_result[2], 'poly': best_result[3]})
    return placed


def pack_fast(n):
    """Fast grid packing for fallback."""
    placed = []
    cols = math.ceil(math.sqrt(n))
    spacing = 1.0
    for i in range(n):
        x, y = (i % cols) * spacing, (i // cols) * spacing
        poly = make_tree_poly(x, y, 0)
        placed.append({'x': Decimal(str(x)), 'y': Decimal(str(y)), 'deg': Decimal('0'), 'poly': poly})
    return placed


def get_score(placed):
    bounds = unary_union([p['poly'] for p in placed]).bounds
    w, h = (bounds[2] - bounds[0]) / float(SCALE), (bounds[3] - bounds[1]) / float(SCALE)
    return max(w, h) ** 2 / len(placed)


def validate(placed):
    polys = [p['poly'] for p in placed]
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
                return False
    return True


def main():
    print("\n" + "=" * 60)
    print("🧪 TEST OPTIMIZED v1 - Quick Validation")
    print("=" * 60)
    print("   n=1-10: Optimized logic | n=11-200: Fast fallback\n")
    
    start = time.time()
    all_trees, total = {}, 0.0
    
    for n in range(1, 201):
        if n <= 10:
            placed = pack_optimized(n)
            method = "OPT"
        else:
            placed = pack_fast(n)
            method = "FAST"
        
        all_trees[n] = placed
        score = get_score(placed)
        total += score
        
        if n <= 10:
            valid = "✓" if validate(placed) else "✗"
            print(f"   n={n:3d} [{method}]: score={score:.3f} {valid}")
    
    print(f"\n⏱️  Time: {time.time() - start:.1f}s")
    print(f"📊 TOTAL: {total:.2f} (n=1-10 optimized, rest fast fallback)")
    
    rows = []
    for n in range(1, 201):
        for i, t in enumerate(all_trees[n]):
            rows.append({'id': f'{n:03d}_{i}', 'x': f's{t["x"]}', 'y': f's{t["y"]}', 'deg': f's{t["deg"]}'})
    
    pd.DataFrame(rows).to_csv('submission_test_v1.csv', index=False)
    print(f"\n📄 Saved: submission_test_v1.csv (valid Kaggle format)")


if __name__ == "__main__":
    main()

