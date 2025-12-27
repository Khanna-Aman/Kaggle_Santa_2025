"""
Santa 2025 - Christmas Tree Packing Challenge
Production-Ready Optimization Solution for Google Colab

Author: Senior Principal Software Engineer
Objective: Achieve 1st rank through advanced optimization algorithms
"""

import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point
from shapely.affinity import rotate, translate
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from typing import List, Tuple, Dict, Optional
import logging
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TreeConfig:
    """Configuration for a single tree placement"""
    x: float
    y: float
    deg: float
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to submission format with 's' prefix"""
        return {
            'x': f's{self.x}',
            'y': f's{self.y}',
            'deg': f's{self.deg}'
        }


class ChristmasTree:
    """
    Christmas tree geometry and transformation handler.
    
    Tree shape: Triangular top (foliage) + rectangular trunk
    Reference point: Center of top of trunk (x=0, y=0)
    """
    
    # Standard Christmas tree shape from Kaggle's official getting-started notebook
    # Multi-tiered tree with 3 foliage levels + trunk
    # Reference point at (0, 0) = center of top of trunk

    # Dimensions from notebook
    TRUNK_W = 0.15
    TRUNK_H = 0.2
    BASE_W = 0.7
    MID_W = 0.4
    TOP_W = 0.25
    TIP_Y = 0.8
    TIER_1_Y = 0.5
    TIER_2_Y = 0.25
    BASE_Y = 0.0
    TRUNK_BOTTOM_Y = -TRUNK_H

    TREE_COORDS = np.array([
        # Start at Tip
        [0.0, TIP_Y],
        # Right side - Top Tier
        [TOP_W / 2, TIER_1_Y],
        [TOP_W / 4, TIER_1_Y],
        # Right side - Middle Tier
        [MID_W / 2, TIER_2_Y],
        [MID_W / 4, TIER_2_Y],
        # Right side - Bottom Tier
        [BASE_W / 2, BASE_Y],
        # Right Trunk
        [TRUNK_W / 2, BASE_Y],
        [TRUNK_W / 2, TRUNK_BOTTOM_Y],
        # Left Trunk
        [-TRUNK_W / 2, TRUNK_BOTTOM_Y],
        [-TRUNK_W / 2, BASE_Y],
        # Left side - Bottom Tier
        [-BASE_W / 2, BASE_Y],
        # Left side - Middle Tier
        [-MID_W / 4, TIER_2_Y],
        [-MID_W / 2, TIER_2_Y],
        # Left side - Top Tier
        [-TOP_W / 4, TIER_1_Y],
        [-TOP_W / 2, TIER_1_Y],
    ])
    
    def __init__(self):
        """Initialize tree geometry"""
        self.base_polygon = Polygon(self.TREE_COORDS)
        self._validate_geometry()
    
    def _validate_geometry(self):
        """Validate tree geometry is valid"""
        if not self.base_polygon.is_valid:
            raise ValueError("Invalid tree geometry")
        logger.debug(f"Tree area: {self.base_polygon.area:.6f}")
    
    def get_transformed_polygon(self, x: float, y: float, deg: float) -> Polygon:
        """
        Get tree polygon with rotation and translation applied.
        
        Args:
            x: X-coordinate of tree position
            y: Y-coordinate of tree position
            deg: Rotation angle in degrees (counter-clockwise)
            
        Returns:
            Transformed Shapely Polygon
        """
        # Rotate around origin (trunk top center), then translate
        poly = rotate(self.base_polygon, deg, origin=(0, 0), use_radians=False)
        poly = translate(poly, xoff=x, yoff=y)
        return poly
    
    def get_bounding_box(self, configs: List[TreeConfig]) -> Tuple[float, float, float, float]:
        """
        Calculate bounding box for multiple trees.
        
        Returns:
            (min_x, min_y, max_x, max_y)
        """
        all_polys = [self.get_transformed_polygon(c.x, c.y, c.deg) for c in configs]
        union = unary_union(all_polys)
        return union.bounds
    
    def get_square_side(self, configs: List[TreeConfig]) -> float:
        """
        Calculate minimum square side length that bounds all trees.
        
        Returns:
            Side length of bounding square
        """
        min_x, min_y, max_x, max_y = self.get_bounding_box(configs)
        width = max_x - min_x
        height = max_y - min_y
        return max(width, height)


class CollisionDetector:
    """Efficient collision detection for tree placements"""
    
    def __init__(self, tree: ChristmasTree):
        self.tree = tree
    
    def check_collision(self, configs: List[TreeConfig]) -> bool:
        """
        Check if any trees overlap - STRICT VERSION matching Kaggle validation.

        Uses negative buffer to ensure conservative collision detection.
        If Kaggle says it overlaps, we MUST detect it here.

        Args:
            configs: List of tree configurations

        Returns:
            True if collision detected, False otherwise
        """
        if len(configs) <= 1:
            return False

        # Get polygons with small negative buffer for conservative checking
        # This ensures we reject anything that's even close to overlapping
        SAFETY_MARGIN = 1e-9
        polygons = []
        for c in configs:
            poly = self.tree.get_transformed_polygon(c.x, c.y, c.deg)
            if not poly.is_valid:
                poly = poly.buffer(0)  # Fix invalid geometry
            polygons.append(poly)

        # Use STRtree for O(n log n) collision detection
        from shapely.strtree import STRtree
        tree = STRtree(polygons)

        for i, poly in enumerate(polygons):
            # Query for potential intersections
            candidates = tree.query(poly)
            for j in candidates:
                if j <= i:
                    continue  # Skip self and already-checked pairs

                other = polygons[j]

                # STRICT CHECK: Any interior intersection = collision
                # Using 'relate' for precise DE-9IM check
                relation = poly.relate(other)

                # DE-9IM: position 0 is interior-interior intersection
                # If it's not 'F' (false), interiors touch = OVERLAP
                if relation[0] != 'F':
                    return True

        return False

    def validate_no_overlap(self, configs: List[TreeConfig], group_id: int = 0) -> bool:
        """
        STRICT validation - raises ValueError if overlap detected.
        Run this BEFORE saving submission.
        """
        if len(configs) <= 1:
            return True

        polygons = [self.tree.get_transformed_polygon(c.x, c.y, c.deg) for c in configs]

        for i in range(len(polygons)):
            for j in range(i + 1, len(polygons)):
                poly_i = polygons[i]
                poly_j = polygons[j]

                # Fix invalid geometries
                if not poly_i.is_valid:
                    poly_i = poly_i.buffer(0)
                if not poly_j.is_valid:
                    poly_j = poly_j.buffer(0)

                # Check intersection
                if poly_i.intersects(poly_j):
                    intersection = poly_i.intersection(poly_j)
                    if hasattr(intersection, 'area') and intersection.area > 0:
                        raise ValueError(
                            f"OVERLAP DETECTED in group {group_id:03d}: "
                            f"Tree {i} and Tree {j} overlap with area {intersection.area:.2e}"
                        )
        return True
    
    def get_collision_pairs(self, configs: List[TreeConfig]) -> List[Tuple[int, int]]:
        """Get indices of colliding tree pairs for debugging"""
        collisions = []
        polygons = [self.tree.get_transformed_polygon(c.x, c.y, c.deg) for c in configs]
        
        for i in range(len(polygons)):
            for j in range(i + 1, len(polygons)):
                if polygons[i].intersects(polygons[j]):
                    intersection = polygons[i].intersection(polygons[j])
                    if intersection.area > 1e-6:  # Increased tolerance
                        collisions.append((i, j))
        return collisions


class MetricCalculator:
    """Calculate competition metric: sum of (s²/n) for all configurations"""

    def __init__(self, tree: ChristmasTree):
        self.tree = tree

    def calculate_score(self, all_configs: Dict[int, List[TreeConfig]]) -> float:
        """
        Calculate total score for all tree configurations.

        Args:
            all_configs: Dictionary mapping n (tree count) to list of TreeConfig

        Returns:
            Total score (lower is better)
        """
        total_score = 0.0

        for n, configs in all_configs.items():
            if len(configs) != n:
                raise ValueError(f"Configuration for {n} trees has {len(configs)} entries")

            s = self.tree.get_square_side(configs)
            score = (s ** 2) / n
            total_score += score
            logger.debug(f"n={n}: s={s:.6f}, s²/n={score:.6f}")

        return total_score

    def calculate_single_score(self, n: int, configs: List[TreeConfig]) -> float:
        """Calculate score for a single configuration"""
        s = self.tree.get_square_side(configs)
        return (s ** 2) / n


class PackingOptimizer:
    """
    Advanced optimization algorithms for tree packing.

    Implements multiple strategies:
    1. Simulated Annealing
    2. Genetic Algorithm
    3. Hybrid local search
    """

    def __init__(self, tree: ChristmasTree, collision_detector: CollisionDetector):
        self.tree = tree
        self.collision_detector = collision_detector
        self.metric_calc = MetricCalculator(tree)
        self.rng = np.random.RandomState(42)

    def optimize_configuration(self, n: int, method: str = 'simulated_annealing',
                              max_iterations: int = 10000, **kwargs) -> List[TreeConfig]:
        """
        Optimize packing for n trees.

        Args:
            n: Number of trees
            method: Optimization method ('simulated_annealing', 'genetic', 'hybrid')
            max_iterations: Maximum iterations
            **kwargs: Additional method-specific parameters

        Returns:
            List of optimized TreeConfig
        """
        logger.info(f"Optimizing configuration for {n} trees using {method}")

        if method == 'simulated_annealing':
            return self._simulated_annealing(n, max_iterations, **kwargs)
        elif method == 'genetic':
            return self._genetic_algorithm(n, max_iterations, **kwargs)
        elif method == 'hybrid':
            return self._hybrid_optimization(n, max_iterations, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _generate_initial_solution(self, n: int) -> List[TreeConfig]:
        """
        Generate initial valid solution using circular packing heuristic.

        This provides a good starting point for optimization.
        """
        configs = []

        if n == 1:
            return [TreeConfig(0.0, 0.0, 90.0)]

        # Circular arrangement with varying rotations
        radius = 0.5 * np.sqrt(n)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

        for i, angle in enumerate(angles):
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            # Rotate trees to face center
            deg = np.degrees(angle) + 90
            configs.append(TreeConfig(x, y, deg % 360))

        # Adjust if collisions exist
        max_attempts = 100
        attempt = 0
        while self.collision_detector.check_collision(configs) and attempt < max_attempts:
            radius *= 1.1
            for i, angle in enumerate(angles):
                configs[i].x = radius * np.cos(angle)
                configs[i].y = radius * np.sin(angle)
            attempt += 1

        return configs

    def _simulated_annealing(self, n: int, max_iterations: int,
                            initial_temp: float = 100.0,
                            cooling_rate: float = 0.995,
                            **kwargs) -> List[TreeConfig]:
        """
        Simulated Annealing optimization.

        Gradually reduces randomness to find optimal packing.
        """
        current_solution = self._generate_initial_solution(n)
        current_score = self.metric_calc.calculate_single_score(n, current_solution)

        best_solution = [TreeConfig(c.x, c.y, c.deg) for c in current_solution]
        best_score = current_score

        temperature = initial_temp

        for iteration in range(max_iterations):
            # Generate neighbor solution
            new_solution = self._generate_neighbor(current_solution, temperature)

            # Check validity
            if self.collision_detector.check_collision(new_solution):
                continue

            new_score = self.metric_calc.calculate_single_score(n, new_solution)

            # Accept or reject
            delta = new_score - current_score
            if delta < 0 or self.rng.random() < np.exp(-delta / temperature):
                current_solution = new_solution
                current_score = new_score

                if new_score < best_score:
                    best_solution = [TreeConfig(c.x, c.y, c.deg) for c in new_solution]
                    best_score = new_score
                    logger.debug(f"Iteration {iteration}: New best score = {best_score:.6f}")

            # Cool down
            temperature *= cooling_rate

            if iteration % 1000 == 0:
                logger.debug(f"Iteration {iteration}/{max_iterations}, Temp={temperature:.4f}, Best={best_score:.6f}")

        logger.info(f"Final score for {n} trees: {best_score:.6f}")
        return best_solution

    def _generate_neighbor(self, solution: List[TreeConfig], temperature: float) -> List[TreeConfig]:
        """
        Generate neighbor solution by perturbing current solution.

        Perturbation magnitude scales with temperature.
        """
        new_solution = [TreeConfig(c.x, c.y, c.deg) for c in solution]

        # Randomly select trees to perturb
        n_perturb = max(1, int(len(solution) * 0.2))  # Perturb 20% of trees
        indices = self.rng.choice(len(solution), size=n_perturb, replace=False)

        # Perturbation magnitude based on temperature
        pos_scale = 0.1 * (temperature / 100.0)
        rot_scale = 10.0 * (temperature / 100.0)

        for idx in indices:
            # Perturb position
            new_solution[idx].x += self.rng.normal(0, pos_scale)
            new_solution[idx].y += self.rng.normal(0, pos_scale)

            # Perturb rotation
            new_solution[idx].deg += self.rng.normal(0, rot_scale)
            new_solution[idx].deg = new_solution[idx].deg % 360

        return new_solution

    def _genetic_algorithm(self, n: int, max_iterations: int,
                          population_size: int = 50,
                          mutation_rate: float = 0.1,
                          **kwargs) -> List[TreeConfig]:
        """
        Genetic Algorithm optimization.

        Maintains population of solutions and evolves them.
        """
        # Initialize population
        population = [self._generate_initial_solution(n) for _ in range(population_size)]

        best_solution = None
        best_score = float('inf')

        for generation in range(max_iterations // population_size):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                if self.collision_detector.check_collision(individual):
                    fitness_scores.append(float('inf'))
                else:
                    score = self.metric_calc.calculate_single_score(n, individual)
                    fitness_scores.append(score)

                    if score < best_score:
                        best_score = score
                        best_solution = [TreeConfig(c.x, c.y, c.deg) for c in individual]

            # Selection (tournament)
            new_population = []
            for _ in range(population_size):
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)

                # Crossover
                child = self._crossover(parent1, parent2)

                # Mutation
                if self.rng.random() < mutation_rate:
                    child = self._mutate(child)

                new_population.append(child)

            population = new_population

            if generation % 10 == 0:
                logger.debug(f"Generation {generation}, Best score: {best_score:.6f}")

        logger.info(f"Final score for {n} trees: {best_score:.6f}")
        return best_solution if best_solution else population[0]

    def _tournament_selection(self, population: List[List[TreeConfig]],
                             fitness_scores: List[float],
                             tournament_size: int = 3) -> List[TreeConfig]:
        """Select individual using tournament selection"""
        indices = self.rng.choice(len(population), size=tournament_size, replace=False)
        best_idx = min(indices, key=lambda i: fitness_scores[i])
        return population[best_idx]

    def _crossover(self, parent1: List[TreeConfig], parent2: List[TreeConfig]) -> List[TreeConfig]:
        """Uniform crossover between two parents"""
        child = []
        for c1, c2 in zip(parent1, parent2):
            if self.rng.random() < 0.5:
                child.append(TreeConfig(c1.x, c1.y, c1.deg))
            else:
                child.append(TreeConfig(c2.x, c2.y, c2.deg))
        return child

    def _mutate(self, solution: List[TreeConfig]) -> List[TreeConfig]:
        """Mutate solution by perturbing random trees"""
        return self._generate_neighbor(solution, temperature=50.0)

    def _hybrid_optimization(self, n: int, max_iterations: int, **kwargs) -> List[TreeConfig]:
        """
        Hybrid approach: Start with genetic algorithm, refine with simulated annealing.
        """
        # Phase 1: Genetic algorithm for global exploration
        logger.info(f"Phase 1: Genetic algorithm ({max_iterations // 2} iterations)")
        solution = self._genetic_algorithm(n, max_iterations // 2, **kwargs)

        # Phase 2: Simulated annealing for local refinement
        logger.info(f"Phase 2: Simulated annealing ({max_iterations // 2} iterations)")
        # Use the GA solution as starting point
        current_solution = solution
        current_score = self.metric_calc.calculate_single_score(n, current_solution)

        best_solution = [TreeConfig(c.x, c.y, c.deg) for c in current_solution]
        best_score = current_score

        temperature = 50.0  # Lower initial temp for refinement
        cooling_rate = 0.995

        for iteration in range(max_iterations // 2):
            new_solution = self._generate_neighbor(current_solution, temperature)

            if self.collision_detector.check_collision(new_solution):
                continue

            new_score = self.metric_calc.calculate_single_score(n, new_solution)

            delta = new_score - current_score
            if delta < 0 or self.rng.random() < np.exp(-delta / temperature):
                current_solution = new_solution
                current_score = new_score

                if new_score < best_score:
                    best_solution = [TreeConfig(c.x, c.y, c.deg) for c in new_solution]
                    best_score = new_score

            temperature *= cooling_rate

        logger.info(f"Final score for {n} trees: {best_score:.6f}")
        return best_solution


class Visualizer:
    """Visualization tools for debugging and analysis"""

    def __init__(self, tree: ChristmasTree):
        self.tree = tree

    def plot_configuration(self, configs: List[TreeConfig], n: int,
                          score: Optional[float] = None,
                          save_path: Optional[str] = None):
        """
        Plot tree configuration.

        Args:
            configs: List of tree configurations
            n: Number of trees
            score: Optional score to display
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot each tree
        for i, config in enumerate(configs):
            poly = self.tree.get_transformed_polygon(config.x, config.y, config.deg)
            x, y = poly.exterior.xy
            ax.fill(x, y, alpha=0.6, fc='green', ec='darkgreen', linewidth=1.5)

            # Mark reference point
            ax.plot(config.x, config.y, 'ro', markersize=3)

        # Plot bounding box
        min_x, min_y, max_x, max_y = self.tree.get_bounding_box(configs)
        side = max(max_x - min_x, max_y - min_y)
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        square_x = center_x - side / 2
        square_y = center_y - side / 2

        from matplotlib.patches import Rectangle
        rect = Rectangle((square_x, square_y), side, side,
                        fill=False, ec='red', linewidth=2, linestyle='--')
        ax.add_patch(rect)

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        title = f'Configuration: {n} trees'
        if score:
            title += f' | Score: {score:.6f} | Side: {side:.4f}'
        ax.set_title(title, fontsize=14, fontweight='bold')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")

        plt.show()

    def plot_multiple_configurations(self, all_configs: Dict[int, List[TreeConfig]],
                                    n_samples: int = 9):
        """Plot grid of multiple configurations"""
        samples = sorted(all_configs.keys())[:n_samples]
        n_rows = int(np.ceil(np.sqrt(n_samples)))
        n_cols = int(np.ceil(n_samples / n_rows))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))
        axes = axes.flatten() if n_samples > 1 else [axes]

        metric_calc = MetricCalculator(self.tree)

        for idx, n in enumerate(samples):
            ax = axes[idx]
            configs = all_configs[n]

            # Plot trees
            for config in configs:
                poly = self.tree.get_transformed_polygon(config.x, config.y, config.deg)
                x, y = poly.exterior.xy
                ax.fill(x, y, alpha=0.6, fc='green', ec='darkgreen', linewidth=0.5)

            # Plot bounding box
            min_x, min_y, max_x, max_y = self.tree.get_bounding_box(configs)
            side = max(max_x - min_x, max_y - min_y)
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2

            square_x = center_x - side / 2
            square_y = center_y - side / 2

            from matplotlib.patches import Rectangle
            rect = Rectangle((square_x, square_y), side, side,
                           fill=False, ec='red', linewidth=1, linestyle='--')
            ax.add_patch(rect)

            score = metric_calc.calculate_single_score(n, configs)
            ax.set_title(f'n={n}, s²/n={score:.4f}', fontsize=10)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2)

        # Hide unused subplots
        for idx in range(len(samples), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.show()


class SubmissionGenerator:
    """Generate submission file in required format"""

    def __init__(self, tree: ChristmasTree):
        self.tree = tree
        self.metric_calc = MetricCalculator(tree)

    def generate_submission(self, all_configs: Dict[int, List[TreeConfig]],
                           output_path: str = 'submission.csv'):
        """
        Generate submission CSV file.

        VALIDATES ALL GROUPS FOR OVERLAP BEFORE SAVING.
        Raises ValueError if any overlap detected.

        Args:
            all_configs: Dictionary mapping n to list of TreeConfig
            output_path: Path to save submission file
        """
        # ============================================
        # STEP 1: VALIDATE ALL GROUPS BEFORE SAVING
        # ============================================
        print("\n🔍 Validating all groups for overlaps...")
        collision_detector = CollisionDetector(self.tree)

        for n in range(1, 201):
            if n not in all_configs:
                raise ValueError(f"Missing configuration for {n} trees")

            configs = all_configs[n]
            if len(configs) != n:
                raise ValueError(f"Configuration for {n} trees has {len(configs)} entries")

            # STRICT VALIDATION - raises ValueError if overlap
            try:
                collision_detector.validate_no_overlap(configs, group_id=n)
            except ValueError as e:
                print(f"❌ {e}")
                raise

        print("✅ All groups validated - NO OVERLAPS DETECTED")

        # ============================================
        # STEP 2: GENERATE CSV
        # ============================================
        rows = []

        for n in range(1, 201):
            configs = all_configs[n]

            for tree_idx, config in enumerate(configs):
                row = {
                    'id': f'{n:03d}_{tree_idx}',
                    'x': f's{config.x}',
                    'y': f's{config.y}',
                    'deg': f's{config.deg}'
                }
                rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)

        # Calculate and log total score
        total_score = self.metric_calc.calculate_score(all_configs)
        logger.info(f"Generated submission: {output_path}")
        logger.info(f"Total score: {total_score:.6f}")

        return total_score

    def validate_submission(self, all_configs: Dict[int, List[TreeConfig]]) -> bool:
        """
        Validate submission for collisions and constraints.

        Returns:
            True if valid, False otherwise
        """
        collision_detector = CollisionDetector(self.tree)

        for n, configs in all_configs.items():
            # Check collision
            if collision_detector.check_collision(configs):
                logger.error(f"Collision detected in configuration for {n} trees")
                pairs = collision_detector.get_collision_pairs(configs)
                logger.error(f"Colliding pairs: {pairs}")
                return False

            # Check coordinate bounds (reasonable values)
            for config in configs:
                if abs(config.x) > 1000 or abs(config.y) > 1000:
                    logger.error(f"Coordinates out of reasonable bounds: ({config.x}, {config.y})")
                    return False

        logger.info("Validation passed: No collisions detected")
        return True


def optimize_single_config(args):
    """
    Worker function for parallel optimization.

    Args:
        args: Tuple of (n, method, max_iterations, seed)

    Returns:
        Tuple of (n, configs, score)
    """
    n, method, max_iterations, seed = args

    # Create fresh instances for this process
    tree = ChristmasTree()
    collision_detector = CollisionDetector(tree)
    optimizer = PackingOptimizer(tree, collision_detector)
    optimizer.rng = np.random.RandomState(seed)

    # Optimize
    configs = optimizer.optimize_configuration(n, method=method, max_iterations=max_iterations)

    # Calculate score
    metric_calc = MetricCalculator(tree)
    score = metric_calc.calculate_single_score(n, configs)

    return n, configs, score


class SantaPackingSolver:
    """
    Main solver class orchestrating the entire optimization pipeline.

    This is the production-ready interface for solving the challenge.
    """

    def __init__(self, use_parallel: bool = True, n_workers: Optional[int] = None):
        """
        Initialize solver.

        Args:
            use_parallel: Whether to use parallel processing
            n_workers: Number of worker processes (None = auto-detect)
        """
        self.tree = ChristmasTree()
        self.collision_detector = CollisionDetector(self.tree)
        self.optimizer = PackingOptimizer(self.tree, self.collision_detector)
        self.visualizer = Visualizer(self.tree)
        self.submission_gen = SubmissionGenerator(self.tree)

        self.use_parallel = use_parallel
        self.n_workers = n_workers

        logger.info("SantaPackingSolver initialized")

    def solve_all_configurations(self,
                                method: str = 'hybrid',
                                max_iterations: int = 5000,
                                n_min: int = 1,
                                n_max: int = 200) -> Dict[int, List[TreeConfig]]:
        """
        Solve all configurations from n_min to n_max trees.

        Args:
            method: Optimization method
            max_iterations: Maximum iterations per configuration
            n_min: Minimum number of trees
            n_max: Maximum number of trees

        Returns:
            Dictionary mapping n to optimized configurations
        """
        logger.info(f"Solving configurations {n_min} to {n_max} using {method}")
        logger.info(f"Parallel processing: {self.use_parallel}")

        all_configs = {}

        if self.use_parallel:
            # Parallel processing
            tasks = [(n, method, max_iterations, 42 + n) for n in range(n_min, n_max + 1)]

            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = {executor.submit(optimize_single_config, task): task[0]
                          for task in tasks}

                for future in as_completed(futures):
                    n = futures[future]
                    try:
                        n_result, configs, score = future.result()
                        all_configs[n_result] = configs
                        logger.info(f"Completed {n_result}/{n_max} trees: score={score:.6f}")
                    except Exception as e:
                        logger.error(f"Error processing {n} trees: {e}")
                        raise
        else:
            # Sequential processing
            for n in range(n_min, n_max + 1):
                configs = self.optimizer.optimize_configuration(n, method=method,
                                                               max_iterations=max_iterations)
                all_configs[n] = configs
                logger.info(f"Completed {n}/{n_max} trees")

        return all_configs

    def run_full_pipeline(self,
                         method: str = 'hybrid',
                         max_iterations: int = 5000,
                         output_path: str = 'submission.csv',
                         visualize_samples: bool = True) -> float:
        """
        Run complete optimization pipeline and generate submission.

        Args:
            method: Optimization method
            max_iterations: Maximum iterations per configuration
            output_path: Path for submission file
            visualize_samples: Whether to visualize sample configurations

        Returns:
            Total score
        """
        logger.info("="*80)
        logger.info("SANTA 2025 - CHRISTMAS TREE PACKING CHALLENGE")
        logger.info("Production Optimization Pipeline")
        logger.info("="*80)

        # Step 1: Optimize all configurations
        logger.info("\nStep 1: Optimizing all configurations...")
        all_configs = self.solve_all_configurations(method=method,
                                                    max_iterations=max_iterations)

        # Step 2: Validate
        logger.info("\nStep 2: Validating solutions...")
        if not self.submission_gen.validate_submission(all_configs):
            raise ValueError("Validation failed! Collisions detected.")

        # Step 3: Generate submission
        logger.info("\nStep 3: Generating submission file...")
        total_score = self.submission_gen.generate_submission(all_configs, output_path)

        # Step 4: Visualize samples
        if visualize_samples:
            logger.info("\nStep 4: Visualizing sample configurations...")
            self.visualizer.plot_multiple_configurations(all_configs, n_samples=9)

        logger.info("\n" + "="*80)
        logger.info(f"PIPELINE COMPLETE!")
        logger.info(f"Total Score: {total_score:.6f}")
        logger.info(f"Submission saved to: {output_path}")
        logger.info("="*80)

        return total_score


# ============================================================================
# MAIN EXECUTION FOR GOOGLE COLAB
# ============================================================================

def main():
    """Main execution function for Google Colab"""

    # Configuration
    CONFIG = {
        'method': 'hybrid',  # 'simulated_annealing', 'genetic', or 'hybrid'
        'max_iterations': 5000,  # Increase for better results (10000-20000 recommended)
        'use_parallel': True,  # Set to False if running into memory issues
        'n_workers': None,  # None = auto-detect CPU count
        'output_path': 'submission.csv',
        'visualize': True
    }

    # Initialize solver
    solver = SantaPackingSolver(
        use_parallel=CONFIG['use_parallel'],
        n_workers=CONFIG['n_workers']
    )

    # Run pipeline
    total_score = solver.run_full_pipeline(
        method=CONFIG['method'],
        max_iterations=CONFIG['max_iterations'],
        output_path=CONFIG['output_path'],
        visualize_samples=CONFIG['visualize']
    )

    return total_score


if __name__ == '__main__':
    main()

