"""
Santa 2025 - Local Execution Script
Run this locally instead of using Colab
"""

from santa_2025_solution import (
    ChristmasTree,
    CollisionDetector,
    PackingOptimizer,
    MetricCalculator,
    Visualizer,
    SubmissionGenerator,
    SantaPackingSolver,
    TreeConfig
)
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def quick_test():
    """Quick test with 5 trees"""
    print("\n" + "="*60)
    print("🧪 QUICK TEST - 5 Trees")
    print("="*60)
    
    tree = ChristmasTree()
    collision_detector = CollisionDetector(tree)
    optimizer = PackingOptimizer(tree, collision_detector)
    visualizer = Visualizer(tree)
    metric_calc = MetricCalculator(tree)
    
    print('Optimizing 5 trees...')
    configs = optimizer.optimize_configuration(5, method='simulated_annealing', max_iterations=1000)
    score = metric_calc.calculate_single_score(5, configs)
    
    print(f'\n✅ Test completed!')
    print(f'Score: {score:.6f}')
    print(f'No collisions: {not collision_detector.check_collision(configs)}')
    
    visualizer.plot_configuration(configs, 5, score)
    
    return score

def run_full_optimization():
    """Run full optimization for all 200 configurations"""
    print("\n" + "="*60)
    print("🚀 FULL OPTIMIZATION - 1 to 200 Trees")
    print("="*60)
    
    # Configuration
    CONFIG = {
        'method': 'hybrid',           # simulated_annealing, genetic, or hybrid
        'max_iterations': 10000,      # Increase for better results (try 20000)
        'use_parallel': True,         # Set False if memory issues
        'n_workers': None,            # None = auto-detect CPU count
        'output_path': 'submission.csv',
        'visualize': True
    }
    
    print('\nConfiguration:')
    for key, value in CONFIG.items():
        print(f'  {key}: {value}')
    
    # Initialize solver
    print('\n✅ Initializing solver...')
    solver = SantaPackingSolver(
        use_parallel=CONFIG['use_parallel'],
        n_workers=CONFIG['n_workers']
    )
    
    # Run full pipeline
    print('\n🚀 Starting optimization...')
    print('This will take 30-60 minutes with parallel processing')
    print('Or 2-4 hours without parallel processing\n')
    
    start_time = time.time()
    
    total_score = solver.run_full_pipeline(
        method=CONFIG['method'],
        max_iterations=CONFIG['max_iterations'],
        output_path=CONFIG['output_path'],
        visualize_samples=CONFIG['visualize']
    )
    
    elapsed_time = time.time() - start_time
    
    print(f'\n🎉 OPTIMIZATION COMPLETE!')
    print(f'Total Score: {total_score:.6f}')
    print(f'Elapsed Time: {elapsed_time/60:.2f} minutes')
    print(f'Submission file: {CONFIG["output_path"]}')
    
    return total_score

def validate_submission(filepath='submission.csv'):
    """Validate submission file format"""
    print("\n" + "="*60)
    print("📊 VALIDATING SUBMISSION")
    print("="*60)
    
    import pandas as pd
    
    df = pd.read_csv(filepath)
    print(f'Submission shape: {df.shape}')
    print(f'\nFirst 10 rows:')
    print(df.head(10))
    
    print(f'\n✅ Format validation:')
    print(f'  All x values start with s: {df["x"].str.startswith("s").all()}')
    print(f'  All y values start with s: {df["y"].str.startswith("s").all()}')
    print(f'  All deg values start with s: {df["deg"].str.startswith("s").all()}')
    print(f'  Total rows: {len(df)} (expected: 20100)')
    print(f'  Correct count: {len(df) == 20100}')
    
    return df

def main():
    """Main execution"""
    print("\n" + "="*80)
    print("🎄 SANTA 2025 - CHRISTMAS TREE PACKING CHALLENGE")
    print("="*80)
    
    # Menu
    print("\nSelect mode:")
    print("1. Quick test (5 trees)")
    print("2. Full optimization (1-200 trees)")
    print("3. Validate existing submission")
    print("4. Run all (test + full optimization + validation)")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        quick_test()
    elif choice == '2':
        run_full_optimization()
    elif choice == '3':
        validate_submission()
    elif choice == '4':
        print("\n📋 Running complete pipeline...")
        quick_test()
        score = run_full_optimization()
        validate_submission()
        print(f"\n🏆 Final Score: {score:.6f}")
    else:
        print("Invalid choice. Running full optimization by default...")
        run_full_optimization()

if __name__ == '__main__':
    main()

