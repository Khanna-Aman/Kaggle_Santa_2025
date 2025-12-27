"""
MEGA PACKER - Production Run with Checkpoints
==============================================
- 20 trials per group
- Saves progress after each group (resume on crash)
- Memory efficient
"""

import math
import random
import pandas as pd
import json
import os
from decimal import Decimal, getcontext
from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree
import time
import gc

getcontext().prec = 25
SCALE = Decimal('1e15')

TRUNK_W, TRUNK_H = Decimal('0.15'), Decimal('0.2')
BASE_W, MID_W, TOP_W = Decimal('0.7'), Decimal('0.4'), Decimal('0.25')
TIP_Y, TIER_1_Y, TIER_2_Y, BASE_Y = Decimal('0.8'), Decimal('0.5'), Decimal('0.25'), Decimal('0.0')

TRIALS_PER_GROUP = 20
CHECKPOINT_FILE = 'mega_checkpoint.json'
CSV_FILE = 'submission_mega.csv'


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


def pack_single_trial(n):
    placed = []
    deg0 = random.uniform(0, 360)
    poly = make_tree_poly(0, 0, deg0)
    placed.append({'x': Decimal('0'), 'y': Decimal('0'), 'deg': Decimal(str(deg0)), 'poly': poly})
    
    while len(placed) < n:
        deg = random.uniform(0, 360)
        base_poly = make_tree_poly(0, 0, deg)
        polys = [p['poly'] for p in placed]
        idx = STRtree(polys)
        
        best_r, best_x, best_y = Decimal('Inf'), None, None
        
        for _ in range(12):
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
    
    return placed


def get_score(placed):
    bounds = unary_union([p['poly'] for p in placed]).bounds
    w, h = (bounds[2] - bounds[0]) / float(SCALE), (bounds[3] - bounds[1]) / float(SCALE)
    return max(w, h) ** 2 / len(placed)


def pack_best_of_trials(n):
    best_placed, best_score = None, float('inf')
    for t in range(TRIALS_PER_GROUP):
        placed = pack_single_trial(n)
        score = get_score(placed)
        if score < best_score:
            best_score, best_placed = score, placed
    return best_placed, best_score


def save_checkpoint(results, last_n, total_score):
    data = {'last_n': last_n, 'total_score': total_score, 'results': results}
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(data, f)


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return None


def main():
    print("\n" + "=" * 60)
    print(f"MEGA PACKER - {TRIALS_PER_GROUP} Trials + Checkpoints")
    print("=" * 60)

    checkpoint = load_checkpoint()
    if checkpoint:
        print(f"[RESUME] from n={checkpoint['last_n']+1}, total={checkpoint['total_score']:.2f}")
        results = checkpoint['results']
        start_n = checkpoint['last_n'] + 1
        total = checkpoint['total_score']
    else:
        print("[NEW] Starting fresh run...")
        results = {}
        start_n = 1
        total = 0.0

    random.seed(42 + start_n)
    start_time = time.time()

    for n in range(start_n, 201):
        placed, score = pack_best_of_trials(n)
        total += score
        results[str(n)] = [{'x': str(t['x']), 'y': str(t['y']), 'deg': str(t['deg'])} for t in placed]

        if n % 5 == 0:
            save_checkpoint(results, n, total)
            gc.collect()

        if n <= 5 or n % 25 == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (n - start_n + 1)) * (200 - n) if n > start_n else 0
            print(f"   n={n:3d}: score={score:.3f} | total={total:.2f} | ETA: {eta/60:.1f}min")

    print(f"\nTime: {(time.time() - start_time)/60:.1f}min")
    print(f"TOTAL: {total:.2f}")

    rows = []
    for n in range(1, 201):
        for i, t in enumerate(results[str(n)]):
            rows.append({'id': f'{n:03d}_{i}', 'x': f's{t["x"]}', 'y': f's{t["y"]}', 'deg': f's{t["deg"]}'})

    pd.DataFrame(rows).to_csv(CSV_FILE, index=False)
    print(f"\nSaved: {CSV_FILE}")
    print(f"Target: 69.13 | Gap: {total - 69.13:.2f}")

    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("Checkpoint cleaned up")


if __name__ == "__main__":
    main()

