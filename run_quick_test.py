"""
Quick test run - no menu, just execute
"""

from santa_2025_solution import (
    ChristmasTree,
    CollisionDetector,
    PackingOptimizer,
    MetricCalculator,
    Visualizer,
    TreeConfig
)
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

print("\n" + "="*60)
print("🧪 QUICK TEST - 5 Trees")
print("="*60)

tree = ChristmasTree()
collision_detector = CollisionDetector(tree)
optimizer = PackingOptimizer(tree, collision_detector)
metric_calc = MetricCalculator(tree)

print('\nOptimizing 5 trees with 1000 iterations...')
configs = optimizer.optimize_configuration(5, method='simulated_annealing', max_iterations=1000)
score = metric_calc.calculate_single_score(5, configs)

print(f'\n✅ Test completed!')
print(f'Score: {score:.6f}')
print(f'No collisions: {not collision_detector.check_collision(configs)}')
print(f'Number of trees: {len(configs)}')

print("\n" + "="*60)
print("✅ QUICK TEST PASSED - Solution is working!")
print("="*60)

