"""Quick test of ultra packer on n=1 to 10"""

from ultra_packer import pack_group, get_bounding_side
import time

print("\n" + "=" * 50)
print("ULTRA PACKER TEST (n=1 to 10)")
print("=" * 50)

start = time.time()
total = 0

for n in range(1, 11):
    placed, score = pack_group(n)
    total += score
    side = get_bounding_side([p['poly'] for p in placed])
    print(f"   n={n:2d}: side={side:.3f}, score={score:.3f} | total={total:.2f}")

print(f"\nTime: {time.time() - start:.1f}s")
print(f"Total for n=1-10: {total:.2f}")

