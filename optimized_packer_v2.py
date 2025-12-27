"""
OPTIMIZED PACKER v2 - Ray March + 8-way Rotation
=================================================
Keeps original ray-march logic (proven), adds rotation search.
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


def find_placement_raymarch(base_poly, polys, idx, vx, vy):
    """Original ray march: walk in, then back out to find tight fit."""
    r = Decimal('20')
    # Walk inward until collision
    while r >= 0:
        px, py = float(r) * vx, float(r) * vy
        cand = affinity.translate(base_poly, px * float(SCALE), py * float(SCALE))
        if has_collision(cand, polys, idx):
            break
        r -= Decimal('0.5')
    
    # Back out until no collision
    while r < Decimal('50'):
        r += Decimal('0.05')
        px, py = float(r) * vx, float(r) * vy
        cand = affinity.translate(base_poly, px * float(SCALE), py * float(SCALE))
        if not has_collision(cand, polys, idx):
            break
    
    return r, Decimal(str(float(r) * vx)), Decimal(str(float(r) * vy))


def pack_group(n):
    """Pack n trees with rotation search + ray march."""
    if n == 1:
        poly = make_tree_poly(0, 0, 0)
        return [{'x': Decimal('0'), 'y': Decimal('0'), 'deg': Decimal('0'), 'poly': poly}]
    
    placed = []
    poly = make_tree_poly(0, 0, 0)
    placed.append({'x': Decimal('0'), 'y': Decimal('0'), 'deg': Decimal('0'), 'poly': poly})
    
    while len(placed) < n:
        polys = [p['poly'] for p in placed]
        idx = STRtree(polys)
        
        best_r, best_x, best_y, best_deg, best_poly = Decimal('Inf'), None, None, None, None
        
        # Try all 8 rotations
        for deg in ROTATIONS:
            base_poly = make_tree_poly(0, 0, deg)
            
            # Try 16 ray directions
            for ray_idx in range(16):
                angle = 2 * math.pi * ray_idx / 16
                vx, vy = math.cos(angle), math.sin(angle)
                
                r, px, py = find_placement_raymarch(base_poly, polys, idx, vx, vy)
                
                if r < best_r:
                    best_r = r
                    best_x, best_y, best_deg = px, py, Decimal(str(deg))
                    best_poly = affinity.translate(base_poly, float(px) * float(SCALE), float(py) * float(SCALE))
        
        placed.append({'x': best_x, 'y': best_y, 'deg': best_deg, 'poly': best_poly})
    
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
    print("⚡ OPTIMIZED PACKER v2 - Ray March + 8-way Rotation")
    print("=" * 60)

    start = time.time()
    all_trees, total = {}, Decimal('0')

    for n in range(1, 201):
        placed = pack_group(n)
        all_trees[n] = placed
        score = get_score(placed)
        total += Decimal(str(score))

        if n <= 5 or n % 25 == 0:
            bounds = unary_union([p['poly'] for p in placed]).bounds
            side = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) / float(SCALE)
            print(f"   n={n:3d}: side={side:.3f}, score={score:.3f} | total={float(total):.2f}")

    print(f"\n⏱️  Time: {time.time() - start:.1f}s")
    print(f"📊 TOTAL: {float(total):.2f}")

    if validate(all_trees[3]):
        print("✅ Group 003 validated - no overlaps!")
    else:
        print("❌ OVERLAP DETECTED!")

    rows = []
    for n in range(1, 201):
        for i, t in enumerate(all_trees[n]):
            rows.append({'id': f'{n:03d}_{i}', 'x': f's{t["x"]}', 'y': f's{t["y"]}', 'deg': f's{t["deg"]}'})

    pd.DataFrame(rows).to_csv('submission_optimized_v2.csv', index=False)
    print(f"\n📄 Saved: submission_optimized_v2.csv")
    print(f"🎯 Target: 69.13 | Gap: {float(total) - 69.13:.2f}")


if __name__ == "__main__":
    main()

