"""
Benchmark different configs on n=1-25 to find best for overnight run.
Generates CSVs for each config.
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
    for o in other_polys:
        if poly.intersects(o) and not poly.touches(o):
            return True
    return False


def has_collision_idx(poly, polys, idx):
    for i in idx.query(poly):
        if poly.intersects(polys[i]) and not poly.touches(polys[i]):
            return True
    return False


def get_side(polys):
    if not polys:
        return 0
    b = unary_union(polys).bounds
    return max((b[2]-b[0]), (b[3]-b[1])) / float(SCALE)


def pack_greedy(n):
    placed = []
    deg0 = random.uniform(0, 360)
    poly = make_tree_poly(0, 0, deg0)
    placed.append({'x': 0.0, 'y': 0.0, 'deg': deg0, 'poly': poly})
    
    while len(placed) < n:
        deg = random.uniform(0, 360)
        base = make_tree_poly(0, 0, deg)
        polys = [p['poly'] for p in placed]
        idx = STRtree(polys)
        
        best_r, best_x, best_y = float('inf'), 0, 0
        for _ in range(12):
            ang = weighted_angle()
            vx, vy = math.cos(ang), math.sin(ang)
            r = 20.0
            while r >= 0:
                px, py = r * vx, r * vy
                c = affinity.translate(base, px * float(SCALE), py * float(SCALE))
                if has_collision_idx(c, polys, idx):
                    break
                r -= 0.5
            while r < 50:
                r += 0.05
                px, py = r * vx, r * vy
                c = affinity.translate(base, px * float(SCALE), py * float(SCALE))
                if not has_collision_idx(c, polys, idx):
                    break
            if r < best_r:
                best_r, best_x, best_y = r, px, py
        
        final = affinity.translate(base, best_x * float(SCALE), best_y * float(SCALE))
        placed.append({'x': best_x, 'y': best_y, 'deg': deg, 'poly': final})
    return placed


def try_move(placed, idx, dx, dy, drot):
    t = placed[idx]
    nx, ny, nd = t['x'] + dx, t['y'] + dy, t['deg'] + drot
    np = make_tree_poly(nx, ny, nd)
    others = [p['poly'] for j, p in enumerate(placed) if j != idx]
    if has_collision(np, others):
        return None
    new = placed.copy()
    new[idx] = {'x': nx, 'y': ny, 'deg': nd, 'poly': np}
    return new


def simulated_annealing(placed, iters):
    n = len(placed)
    if n <= 1:
        return placed
    current = placed
    cur_score = get_side([p['poly'] for p in current])
    best, best_score = current, cur_score
    temp, cool = 0.5, 0.95
    
    for _ in range(iters):
        idx = random.randint(1, n-1)
        dx, dy = random.gauss(0, 0.1), random.gauss(0, 0.1)
        drot = random.gauss(0, 10)
        new = try_move(current, idx, dx, dy, drot)
        if new is None:
            continue
        new_score = get_side([p['poly'] for p in new])
        delta = new_score - cur_score
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current, cur_score = new, new_score
            if new_score < best_score:
                best, best_score = new, new_score
        temp *= cool
    return best


def compact(placed):
    n = len(placed)
    if n <= 1:
        return placed
    improved = True
    while improved:
        improved = False
        cx = sum(p['x'] for p in placed) / n
        cy = sum(p['y'] for p in placed) / n
        for idx in range(n):
            t = placed[idx]
            dx, dy = (cx - t['x']) * 0.1, (cy - t['y']) * 0.1
            new = try_move(placed, idx, dx, dy, 0)
            if new and get_side([p['poly'] for p in new]) < get_side([p['poly'] for p in placed]):
                placed = new
                improved = True
    return placed


def pack_with_config(n, trials, sa_iters):
    best, best_score = None, float('inf')
    for _ in range(trials):
        p = pack_greedy(n)
        if sa_iters > 0:
            p = simulated_annealing(p, sa_iters)
        p = compact(p)
        score = get_side([t['poly'] for t in p]) ** 2 / n
        if score < best_score:
            best, best_score = p, score
    return best, best_score


def run_config(name, trials, sa_iters, max_n=25):
    print(f"\n{'='*50}")
    print(f"CONFIG: {name} (trials={trials}, SA={sa_iters})")
    print('='*50)

    random.seed(42)
    start = time.time()
    results = {}
    total = 0

    for n in range(1, max_n + 1):
        placed, score = pack_with_config(n, trials, sa_iters)
        total += score
        results[n] = placed
        if n <= 5 or n % 5 == 0:
            print(f"   n={n:2d}: score={score:.3f} | total={total:.2f}")

    elapsed = time.time() - start
    rate = elapsed / max_n
    proj_time = rate * 200 / 3600
    proj_score = total / max_n * 200

    print(f"\nTime: {elapsed:.1f}s ({rate:.2f}s/group)")
    print(f"Total n=1-{max_n}: {total:.2f}")
    print(f"Projected n=1-200: ~{proj_score:.0f} score, ~{proj_time:.1f}h runtime")

    # Save CSV
    rows = []
    for n in range(1, max_n + 1):
        for i, t in enumerate(results[n]):
            rows.append({'id': f'{n:03d}_{i}', 'x': f's{t["x"]}', 'y': f's{t["y"]}', 'deg': f's{t["deg"]}'})

    csv_name = f'bench_{name}.csv'
    pd.DataFrame(rows).to_csv(csv_name, index=False)
    print(f"Saved: {csv_name}")

    return total, elapsed, proj_score, proj_time


if __name__ == "__main__":
    print("\n" + "="*60)
    print("BENCHMARK: Finding best config for overnight run")
    print("="*60)

    configs = [
        ("baseline", 10, 0),      # No SA
        ("light", 20, 50),        # Light SA
        ("medium", 30, 100),      # Current ultra
        ("heavy", 20, 200),       # More SA, fewer trials
    ]

    results = []
    for name, trials, sa in configs:
        total, elapsed, proj_score, proj_time = run_config(name, trials, sa, max_n=20)
        results.append((name, total, proj_score, proj_time))

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Config':<12} {'n=1-20':<10} {'Proj Score':<12} {'Proj Time':<10}")
    print("-"*44)
    for name, total, proj, time_h in results:
        print(f"{name:<12} {total:<10.2f} {proj:<12.0f} {time_h:<10.1f}h")

    best = min(results, key=lambda x: x[2])
    print(f"\nBEST: {best[0]} -> projected score ~{best[2]:.0f}")

