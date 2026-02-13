"""Benchmark runner for alignment in simulation."""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import transforms3d as t3d
import draccus

from alignit.config import InferConfig
from alignit.models.alignnet import AlignNet
from alignit.utils.zhou import sixd_se3
from alignit.utils.tfs import are_tfs_close
from alignit.robots.xarmsim import XarmSim
from benchmark.metrics import (
    compute_pose_error,
    check_convergence,
    compute_statistics,
)


@draccus.wrap()
def run_benchmark(cfg: InferConfig, num_trials: int = 5, max_iterations: int = 50):
    """
    Run alignment benchmark in simulation.
    
    Args:
        cfg: InferConfig with model settings
        num_trials: Number of random alignment trials to run
        max_iterations: Max iterations per trial
    
    Returns:
        dict with benchmark results
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    net = AlignNet(
        backbone_name=cfg.model.backbone,
        backbone_weights=cfg.model.backbone_weights,
        use_vector_input=cfg.model.use_vector_input,
        fc_layers=cfg.model.fc_layers,
        vector_hidden_dim=cfg.model.vector_hidden_dim,
        output_dim=cfg.model.output_dim,
        feature_agg=cfg.model.feature_agg,
        use_depth_input=cfg.model.use_depth_input,
    )
    net.load_state_dict(torch.load(cfg.model.path, map_location=device))
    net.to(device)
    net.eval()
    
    robot = XarmSim()
    
    # Initialize results
    results = {
        "timestamp": datetime.now().isoformat(),
        "model_path": cfg.model.path,
        "num_trials": num_trials,
        "max_iterations": max_iterations,
        "trials": [],
        "summary": {},
    }
    
    # Run benchmark trials
    print(f"\n{'='*60}")
    print(f"Running {num_trials} alignment trials...")
    print(f"{'='*60}\n")
    
    all_final_errors = []
    convergence_count = 0
    
    for trial_idx in range(num_trials):
        print(f"\n[Trial {trial_idx + 1}/{num_trials}]")
        
        # Random start pose
        start_pose = t3d.affines.compose(
            [np.random.uniform(0.15, 0.30),
             np.random.uniform(-0.15, 0.15),
             np.random.uniform(0.20, 0.35)],
            t3d.euler.euler2mat(np.pi + np.random.uniform(-0.3, 0.3),
                               np.random.uniform(-0.3, 0.3),
                               np.random.uniform(-np.pi, np.pi)),
            [1, 1, 1]
        )
        
        # Random target pose (within reasonable bounds)
        target_pose = t3d.affines.compose(
            [np.random.uniform(0.15, 0.30),
             np.random.uniform(-0.15, 0.15),
             np.random.uniform(0.20, 0.35)],
            t3d.euler.euler2mat(np.pi + np.random.uniform(-0.3, 0.3),
                               np.random.uniform(-0.3, 0.3),
                               np.random.uniform(-np.pi, np.pi)),
            [1, 1, 1]
        )
        
        robot.servo_to_pose(start_pose, lin_tol=1e-2, ang_tol=0.1)
        
        trial_data = {
            "start_pose": start_pose.tolist(),
            "target_pose": target_pose.tolist(),
            "iterations": [],
            "converged": False,
        }
        
        iteration = 0
        converged = False
        
        try:
            while iteration < max_iterations:
                observation = robot.get_observation()
                rgb_np = observation["rgb"].astype(np.float32) / 255.0
                
                # Ensure 3 channels
                if rgb_np.ndim == 2:
                    rgb_np = np.expand_dims(rgb_np, axis=-1)
                if rgb_np.shape[-1] == 1:
                    rgb_np = np.repeat(rgb_np, 3, axis=-1)
                
                rgb_images_batch = (
                    torch.from_numpy(rgb_np)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(device)
                )
                
                with torch.no_grad():
                    relative_action = net(rgb_images_batch)
                
                relative_action = relative_action.squeeze(0).cpu().numpy()
                relative_action = sixd_se3(relative_action)
                
                # Apply rotation multiplier for more stable convergence
                relative_action[:3, :3] = np.linalg.matrix_power(
                    relative_action[:3, :3], cfg.rotation_matrix_multiplier
                )
                
                # Compute error before this step
                current_pose = robot.pose()
                error = compute_pose_error(current_pose, target_pose)
                
                # Check convergence
                is_converged = check_convergence(
                    error,
                    trans_tol_mm=cfg.lin_tolerance * 1000,
                    rot_tol_deg=np.degrees(cfg.ang_tolerance)
                )
                
                trial_data["iterations"].append({
                    "iteration": iteration,
                    "translation_error_mm": error["translation_mm"],
                    "rotation_error_deg": error["rotation_deg"],
                    "converged": is_converged,
                })
                
                print(f"  Iter {iteration + 1:2d}: "
                      f"Trans={error['translation_mm']:6.2f}mm, "
                      f"Rot={error['rotation_deg']:6.2f}°, "
                      f"Converged={is_converged}")
                
                if is_converged:
                    converged = True
                    trial_data["converged"] = True
                    convergence_count += 1
                    break
                
                # Execute action
                target_pose_iter = robot.pose() @ relative_action
                robot.servo_to_pose(pose=target_pose_iter, lin_tol=1e-3, ang_tol=1e-2)
                iteration += 1
        
        except Exception as e:
            print(f"  Error during trial: {e}")
        
        # Get final error
        final_pose = robot.pose()
        final_error = compute_pose_error(final_pose, target_pose)
        trial_data["final_error"] = {
            "translation_mm": final_error["translation_mm"],
            "rotation_deg": final_error["rotation_deg"],
        }
        trial_data["num_iterations"] = iteration + 1
        
        all_final_errors.append(final_error)
        results["trials"].append(trial_data)
        
        print(f"  Final: Trans={final_error['translation_mm']:.2f}mm, "
              f"Rot={final_error['rotation_deg']:.2f}°, "
              f"Iters={trial_data['num_iterations']}")
    
    # Compute summary statistics
    stats = compute_statistics(all_final_errors)
    results["summary"] = {
        "convergence_rate": convergence_count / num_trials,
        "convergence_count": convergence_count,
        "translation": stats["translation"],
        "rotation": stats["rotation"],
        "avg_iterations": np.mean([t["num_iterations"] for t in results["trials"]]),
    }
    
    robot.disconnect()
    
    return results


def save_results(results, output_dir: str = "./benchmark/results"):
    """Save benchmark results to JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_path / f"benchmark_{timestamp}.json"
    
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    
    return result_file


def print_summary(results):
    """Print benchmark summary to console."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK SUMMARY")
    print(f"{'='*60}\n")
    
    summary = results["summary"]
    
    print(f"Convergence Rate: {summary['convergence_rate']*100:.1f}% "
          f"({summary['convergence_count']}/{results['num_trials']})")
    print(f"Average Iterations: {summary['avg_iterations']:.1f}")
    
    print(f"\nTranslation Error (mm):")
    print(f"  Mean: {summary['translation']['mean_mm']:.2f} ± {summary['translation']['std_mm']:.2f}")
    print(f"  Range: [{summary['translation']['min_mm']:.2f}, {summary['translation']['max_mm']:.2f}]")
    
    print(f"\nRotation Error (degrees):")
    print(f"  Mean: {summary['rotation']['mean_deg']:.2f} ± {summary['rotation']['std_deg']:.2f}")
    print(f"  Range: [{summary['rotation']['min_deg']:.2f}, {summary['rotation']['max_deg']:.2f}]")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    import sys
    
    # Parse arguments
    num_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    
    # Run benchmark
    results = run_benchmark(num_trials=num_trials, max_iterations=50)
    
    # Save and print results
    result_file = save_results(results)
    print(f"Results saved to: {result_file}")
    
    print_summary(results)
