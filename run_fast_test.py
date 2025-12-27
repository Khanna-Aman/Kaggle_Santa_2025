"""
FAST TEST - Run in 5-10 minutes with reduced iterations
Use this to verify the solution works before running full optimization
"""

from santa_2025_solution import (
    ChristmasTree,
    CollisionDetector,
    PackingOptimizer,
    MetricCalculator,
    SubmissionGenerator,
    SantaPackingSolver,
    TreeConfig
)
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("\n" + "="*70)
print("🚀 FAST TEST - Running with reduced iterations (5-10 minutes)")
print("="*70)

# First, verify sample submission doesn't have collisions
print("\n📋 Step 1: Verify sample submission geometry...")
tree = ChristmasTree()
detector = CollisionDetector(tree)

# Sample submission for n=3
sample_configs = [
    TreeConfig(0.0, 0.0, 90.0),
    TreeConfig(0.202736, -0.511271, 90.0),
    TreeConfig(0.5206, 0.177413, 180.0),
]
collision = detector.check_collision(sample_configs)
print(f"Sample submission n=3 collision: {collision}")
if collision:
    print("⚠️ WARNING: Sample submission shows collision - geometry may differ from Kaggle!")
    print("   Stopping test - geometry MUST be fixed first!")
    exit(1)
else:
    print("✅ Sample submission has no collision - geometry matches Kaggle!")
    print(f"   Tree area: {tree.base_polygon.area:.6f}")

# Configuration for FAST test
CONFIG = {
    'method': 'simulated_annealing',  # Faster than hybrid
    'max_iterations': 500,             # Reduced for speed (normally 10000)
    'use_parallel': False,             # Disabled - Windows multiprocessing issues
    'n_workers': None,
    'output_path': 'submission.csv',   # Always overwrite submission.csv
}

print(f"\n📋 Step 2: Running optimization with reduced iterations...")
print(f"   Method: {CONFIG['method']}")
print(f"   Iterations: {CONFIG['max_iterations']} (normally 10000)")
print(f"   Parallel: {CONFIG['use_parallel']}")

# Initialize solver
solver = SantaPackingSolver(
    use_parallel=CONFIG['use_parallel'],
    n_workers=CONFIG['n_workers']
)

start_time = time.time()

# Run optimization
print("\n🚀 Starting optimization...")
total_score = solver.run_full_pipeline(
    method=CONFIG['method'],
    max_iterations=CONFIG['max_iterations'],
    output_path=CONFIG['output_path'],
    visualize_samples=False  # Skip visualization for speed
)

elapsed_time = time.time() - start_time

print(f"\n" + "="*70)
print(f"✅ FAST TEST COMPLETE!")
print(f"="*70)
print(f"Total Score: {total_score:.6f}")
print(f"Elapsed Time: {elapsed_time/60:.2f} minutes")
print(f"Submission file: {CONFIG['output_path']}")
print(f"\n⚠️ NOTE: This was a fast test with only {CONFIG['max_iterations']} iterations.")
print(f"   For competition, use 10000-20000 iterations for better scores.")
print(f"\n🎯 Current 1st place: 69.13")
print(f"   Your score: {total_score:.2f}")
print(f"   Gap: {total_score - 69.13:.2f}")

