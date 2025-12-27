"""
ULTRA PACKER - Simulated Annealing + Compaction
================================================
Target: Break below 100
- 30 initial trials
- Simulated annealing refinement
- Centroid compaction
- Checkpointing
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

TRIALS_PER_GROUP = 30
SA_ITERATIONS = 100  # Simulated annealing iterations per group
CHECKPOINT_FILE = 'ultra_checkpoint.json'
CSV_FILE = 'submission_ultra.csv'


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


def has_collision(poly, other_polys):
    for other in other_polys:
        if poly.intersects(other) and not poly.touches(other):
            return True
    return False


def has_collision_idx(poly, placed_polys, idx):
    for i in idx.query(poly):
        if poly.intersects(placed_polys[i]) and not poly.touches(placed_polys[i]):
            return True
    return False


def get_bounding_side(polys):
    if not polys:
        return 0
    bounds = unary_union(polys).bounds
    w = (bounds[2] - bounds[0]) / float(SCALE)
    h = (bounds[3] - bounds[1]) / float(SCALE)
    return max(w, h)


def pack_single_trial(n):
    """Initial greedy placement."""
    placed = []
    deg0 = random.uniform(0, 360)
    poly = make_tree_poly(0, 0, deg0)
    placed.append({'x': 0.0, 'y': 0.0, 'deg': deg0, 'poly': poly})
    
    while len(placed) < n:
        deg = random.uniform(0, 360)
        base_poly = make_tree_poly(0, 0, deg)
        polys = [p['poly'] for p in placed]
        idx = STRtree(polys)
        
        best_r, best_x, best_y = float('inf'), 0, 0
        
        for _ in range(12):
            angle = weighted_angle()
            vx, vy = math.cos(angle), math.sin(angle)
            
            r = 20.0
            while r >= 0:
                px, py = r * vx, r * vy
                cand = affinity.translate(base_poly, px * float(SCALE), py * float(SCALE))
                if has_collision_idx(cand, polys, idx):
                    break
                r -= 0.5
            
            while r < 50:
                r += 0.05
                px, py = r * vx, r * vy
                cand = affinity.translate(base_poly, px * float(SCALE), py * float(SCALE))
                if not has_collision_idx(cand, polys, idx):
                    break
            
            if r < best_r:
                best_r, best_x, best_y = r, px, py
        
        final = affinity.translate(base_poly, best_x * float(SCALE), best_y * float(SCALE))
        placed.append({'x': best_x, 'y': best_y, 'deg': deg, 'poly': final})
    
    return placed


def try_move_tree(placed, idx, dx, dy, drot):
    """Try moving one tree, return new placement if valid."""
    t = placed[idx]
    new_x, new_y = t['x'] + dx, t['y'] + dy
    new_deg = t['deg'] + drot
    new_poly = make_tree_poly(new_x, new_y, new_deg)
    others = [p['poly'] for j, p in enumerate(placed) if j != idx]
    if has_collision(new_poly, others):
        return None
    new_placed = placed.copy()
    new_placed[idx] = {'x': new_x, 'y': new_y, 'deg': new_deg, 'poly': new_poly}
    return new_placed


def simulated_annealing(placed, iterations=SA_ITERATIONS):
    """Refine placement using simulated annealing."""
    n = len(placed)
    if n <= 1:
        return placed

    current = placed
    current_score = get_bounding_side([p['poly'] for p in current])
    best, best_score = current, current_score

    temp = 0.5
    cooling = 0.95

    for i in range(iterations):
        # Pick random tree (not first one to keep anchor)
        idx = random.randint(1, n - 1)

        # Random small move
        dx = random.gauss(0, 0.1)
        dy = random.gauss(0, 0.1)
        drot = random.gauss(0, 10)

        new_placed = try_move_tree(current, idx, dx, dy, drot)
        if new_placed is None:
            continue

        new_score = get_bounding_side([p['poly'] for p in new_placed])
        delta = new_score - current_score

        # Accept if better, or with probability based on temperature
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current, current_score = new_placed, new_score
            if new_score < best_score:
                best, best_score = new_placed, new_score

        temp *= cooling

    return best


def compact_toward_center(placed):
    """Try to move each tree toward centroid."""
    n = len(placed)
    if n <= 1:
        return placed

    # Find centroid
    cx = sum(p['x'] for p in placed) / n
    cy = sum(p['y'] for p in placed) / n

    improved = True
    while improved:
        improved = False
        for idx in range(n):
            t = placed[idx]
            dx = (cx - t['x']) * 0.1
            dy = (cy - t['y']) * 0.1

            new_placed = try_move_tree(placed, idx, dx, dy, 0)
            if new_placed:
                new_score = get_bounding_side([p['poly'] for p in new_placed])
                old_score = get_bounding_side([p['poly'] for p in placed])
                if new_score < old_score:
                    placed = new_placed
                    improved = True

    return placed


def pack_group(n):
    """Pack n trees with multi-trial + SA + compaction."""
    best_placed, best_score = None, float('inf')

    for trial in range(TRIALS_PER_GROUP):
        placed = pack_single_trial(n)
        placed = simulated_annealing(placed)
        placed = compact_toward_center(placed)

        score = get_bounding_side([p['poly'] for p in placed]) ** 2 / n
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
    print(f"ULTRA PACKER - SA + Compaction ({TRIALS_PER_GROUP} trials)")
    print("=" * 60)

    checkpoint = load_checkpoint()
    if checkpoint:
        print(f"[RESUME] from n={checkpoint['last_n']+1}")
        results = checkpoint['results']
        start_n = checkpoint['last_n'] + 1
        total = checkpoint['total_score']
    else:
        print("[NEW] Starting fresh...")
        results, start_n, total = {}, 1, 0.0

    random.seed(42 + start_n)
    start_time = time.time()

    for n in range(start_n, 201):
        placed, score = pack_group(n)
        total += score

        results[str(n)] = [{'x': str(t['x']), 'y': str(t['y']), 'deg': str(t['deg'])} for t in placed]

        if n % 5 == 0:
            save_checkpoint(results, n, total)
            gc.collect()

        if n <= 5 or n % 10 == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (n - start_n + 1)) * (200 - n) / 60 if n > start_n else 0
            print(f"   n={n:3d}: score={score:.3f} | total={total:.2f} | ETA: {eta:.1f}min")

    print(f"\nTime: {(time.time() - start_time)/60:.1f}min")
    print(f"TOTAL: {total:.2f}")

    rows = []
    for n in range(1, 201):
        for i, t in enumerate(results[str(n)]):
            rows.append({'id': f'{n:03d}_{i}', 'x': f's{t["x"]}', 'y': f's{t["y"]}', 'deg': f's{t["deg"]}'})

    pd.DataFrame(rows).to_csv(CSV_FILE, index=False)
    print(f"\nSaved: {CSV_FILE}")
    print(f"Target: <100 | Gap: {total - 100:.2f}")

    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)


if __name__ == "__main__":
    main()

