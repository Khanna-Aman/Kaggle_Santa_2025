"""
SUBMISSION GENERATOR - Creates submission.csv for Kaggle
Run with: py run_submission.py
"""

from santa_2025_solution import (
    ChristmasTree,
    CollisionDetector,
    PackingOptimizer,
    MetricCalculator,
    SubmissionGenerator,
    TreeConfig
)
import logging
import time
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============== CONFIGURATION ==============
MAX_ITERATIONS = 100  # Increase for better score (2000-5000 for competition)
OUTPUT_FILE = "submission.csv"  # Always overwrites this file
# ===========================================

print("\n" + "="*70)
print("🎄 SANTA 2025 - CHRISTMAS TREE PACKING SUBMISSION GENERATOR 🎄")
print("="*70)

# Step 1: Verify geometry
print("\n📋 Step 1: Verifying tree geometry...")
tree = ChristmasTree()
detector = CollisionDetector(tree)

sample_configs = [
    TreeConfig(0.0, 0.0, 90.0),
    TreeConfig(0.202736, -0.511271, 90.0),
    TreeConfig(0.5206, 0.177413, 180.0),
]
collision = detector.check_collision(sample_configs)
if collision:
    print("❌ GEOMETRY ERROR - Sample submission has collision!")
    print("   Cannot generate valid submission. Fix geometry first.")
    sys.exit(1)
print("✅ Geometry matches Kaggle!")
print(f"   Tree area: {tree.base_polygon.area:.6f}")

# Step 2: Initialize
print(f"\n📋 Step 2: Initializing optimizer...")
print(f"   Iterations per tree count: {MAX_ITERATIONS}")
print(f"   Output file: {OUTPUT_FILE}")

optimizer = PackingOptimizer(tree, detector)
calculator = MetricCalculator(tree)
generator = SubmissionGenerator(tree)

# Step 3: Run optimization for all tree counts
print(f"\n🚀 Step 3: Optimizing all configurations (n=1 to 200)...")
print("   This will take some time. Progress shown below.\n")

start_time = time.time()
all_configs = {}
total_score = 0.0

for n in range(1, 201):
    iter_start = time.time()
    
    # Optimize this configuration
    configs = optimizer.optimize_configuration(
        n, 
        method='simulated_annealing', 
        max_iterations=MAX_ITERATIONS
    )
    
    # Calculate score
    score = calculator.calculate_single_score(n, configs)
    total_score += score
    all_configs[n] = configs
    
    iter_time = time.time() - iter_start
    
    # Progress output
    if n <= 10 or n % 10 == 0:
        print(f"   n={n:3d}: score={score:.4f} (time: {iter_time:.1f}s) | Running total: {total_score:.4f}")

elapsed = time.time() - start_time

# Step 4: Generate submission file
print(f"\n📋 Step 4: Generating submission file...")
generator.generate_submission(all_configs, OUTPUT_FILE)

# Step 5: Summary
print("\n" + "="*70)
print("✅ SUBMISSION COMPLETE!")
print("="*70)
print(f"\n📊 RESULTS:")
print(f"   Total Score: {total_score:.6f}")
print(f"   Time Elapsed: {elapsed/60:.1f} minutes")
print(f"   Submission File: {OUTPUT_FILE}")
print(f"\n🎯 LEADERBOARD COMPARISON:")
print(f"   1st place: 69.13")
print(f"   Your score: {total_score:.2f}")
print(f"   Gap: {total_score - 69.13:.2f}")
print(f"\n💡 TIP: Increase MAX_ITERATIONS in this script for better scores.")
print(f"   Current: {MAX_ITERATIONS} | Recommended: 2000-5000 for competition")
print("\n📤 Upload 'submission.csv' to Kaggle to submit!")

