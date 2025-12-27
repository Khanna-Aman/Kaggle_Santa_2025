"""
QUICK VERIFICATION - Run in ~2-3 minutes
Optimizes ALL tree counts with minimal iterations and generates submission.csv
"""

from santa_2025_solution import (
    ChristmasTree,
    CollisionDetector,
    PackingOptimizer,
    MetricCalculator,
    SubmissionGenerator,
    TreeConfig
)
import time
import logging

logging.basicConfig(level=logging.WARNING)

# ============== CONFIGURATION ==============
MAX_ITERATIONS = 50  # Very low for quick test
OUTPUT_FILE = "submission.csv"
# ===========================================

print("\n" + "="*70)
print("⚡ QUICK VERIFICATION - Generates submission.csv")
print("="*70)

# Initialize
tree = ChristmasTree()
detector = CollisionDetector(tree)
optimizer = PackingOptimizer(tree, detector)
calculator = MetricCalculator(tree)
generator = SubmissionGenerator(tree)

print(f"\n📋 Tree area: {tree.base_polygon.area:.6f}")
print(f"📋 Iterations: {MAX_ITERATIONS}")
print(f"📋 Output: {OUTPUT_FILE}")

# Optimize ALL tree counts
print(f"\n🚀 Optimizing n=1 to 200...")
start_time = time.time()
all_configs = {}
total_score = 0.0

for n in range(1, 201):
    configs = optimizer.optimize_configuration(n, method='simulated_annealing', max_iterations=MAX_ITERATIONS)
    score = calculator.calculate_single_score(n, configs)
    total_score += score
    all_configs[n] = configs

    if n <= 5 or n % 20 == 0:
        print(f"   n={n:3d}: score={score:.4f} | Running total: {total_score:.4f}")

elapsed = time.time() - start_time

# Generate submission
generator.generate_submission(all_configs, OUTPUT_FILE)

print("\n" + "="*70)
print("✅ SUBMISSION GENERATED!")
print("="*70)
print(f"\n📊 Total Score: {total_score:.4f}")
print(f"⏱️  Time: {elapsed:.1f} seconds")
print(f"📤 Upload '{OUTPUT_FILE}' to Kaggle to validate!")

