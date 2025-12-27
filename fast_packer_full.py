"""
FAST PACKER - Baseline (Score: ~166)
=====================================
Original greedy packer with reused placements.
"""

import math
import random
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


def make_tree_poly(cx, cy, angle):
    cx, cy, angle = Decimal(str(cx)), Decimal(str(cy)), Decimal(str(angle))
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


def weighted_angle():
    while True:
        a = random.uniform(0, 2 * math.pi)
        if random.uniform(0, 1) < abs(math.sin(2 * a)):
            return a


def has_collision(poly, placed_polys, idx):
    for i in idx.query(poly):
        if poly.intersects(placed_polys[i]) and not poly.touches(placed_polys[i]):
            return True
    return False


def pack_trees(n, existing=None):
    """Pack n trees - reuses existing placements for speed."""
    placed = list(existing) if existing else []
    
    if not placed:
        poly = make_tree_poly(0, 0, random.uniform(0, 360))
        placed.append({'x': Decimal('0'), 'y': Decimal('0'), 'deg': Decimal('0'), 'poly': poly})
    
    while len(placed) < n:
        deg = random.uniform(0, 360)
        base_poly = make_tree_poly(0, 0, deg)
        polys = [p['poly'] for p in placed]
        idx = STRtree(polys)
        
        best_x, best_y, best_r = None, None, Decimal('Inf')
        
        for _ in range(10):
            angle = weighted_angle()
            vx, vy = Decimal(str(math.cos(angle))), Decimal(str(math.sin(angle)))
            
            r = Decimal('20')
            while r >= 0:
                px, py = r * vx, r * vy
                cand = affinity.translate(base_poly, float(px * SCALE), float(py * SCALE))
                if has_collision(cand, polys, idx):
                    break
                r -= Decimal('0.5')
            
            while r < Decimal('50'):
                r += Decimal('0.05')
                px, py = r * vx, r * vy
                cand = affinity.translate(base_poly, float(px * SCALE), float(py * SCALE))
                if not has_collision(cand, polys, idx):
                    break
            
            if r < best_r:
                best_r, best_x, best_y = r, px, py
        
        final = affinity.translate(base_poly, float(best_x * SCALE), float(best_y * SCALE))
        placed.append({'x': best_x, 'y': best_y, 'deg': Decimal(str(deg)), 'poly': final})
    
    bounds = unary_union([p['poly'] for p in placed]).bounds
    w = Decimal(str(bounds[2] - bounds[0])) / SCALE
    h = Decimal(str(bounds[3] - bounds[1])) / SCALE
    return placed, max(w, h)


def validate(placed):
    polys = [p['poly'] for p in placed]
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
                return False
    return True


def main():
    print("\n" + "=" * 60)
    print("⚡ FAST PACKER - Baseline (reuses n-1 placements)")
    print("=" * 60)
    
    random.seed(42)
    start = time.time()
    
    all_trees, total, existing = {}, Decimal('0'), None
    
    for n in range(1, 201):
        placed, side = pack_trees(n, existing)
        existing = placed
        all_trees[n] = placed
        score = (side ** 2) / n
        total += score
        
        if n <= 5 or n % 25 == 0:
            print(f"   n={n:3d}: side={float(side):.3f}, score={float(score):.3f} | total={float(total):.2f}")
    
    print(f"\n⏱️  Time: {time.time() - start:.1f}s")
    print(f"📊 TOTAL: {float(total):.2f}")
    
    if validate(all_trees[3]):
        print("✅ All validated - no overlaps!")
    
    rows = []
    for n in range(1, 201):
        for i, t in enumerate(all_trees[n]):
            rows.append({'id': f'{n:03d}_{i}', 'x': f's{t["x"]}', 'y': f's{t["y"]}', 'deg': f's{t["deg"]}'})
    
    pd.DataFrame(rows).to_csv('submission_fast_packer.csv', index=False)
    print(f"\n📄 Saved: submission_fast_packer.csv")
    print(f"🎯 Target: 69.13 | Gap: {float(total) - 69.13:.2f}")


if __name__ == "__main__":
    main()

