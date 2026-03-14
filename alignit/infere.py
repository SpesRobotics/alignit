import time
import torch
import transforms3d as t3d
import numpy as np
import draccus

from alignit.config import InferConfig
from alignit.models.alignnet import AlignNet
from alignit.utils.zhou import sixd_se3
from alignit.utils.tfs import print_pose, are_tfs_close
from alignit.robots.xarmsim import XarmSim


Xarm = None
try:
    from alignit.robots.xarm import Xarm
except ImportError:
    pass

@draccus.wrap()
def main(cfg: InferConfig):
    """Run inference/alignment using configuration parameters."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    robot = Xarm()
    
    num_alignments = getattr(cfg, 'num_alignments', 5)
    ang_tol_rad = np.deg2rad(cfg.ang_tolerance)
    alignment_results = []
    
    MAX_TOTAL_STEPS = 1000

    print(f"\nRunning {num_alignments} alignment trials...\n")

    for alignment_trial in range(num_alignments):
        print(f"\n{'='*60}")
        print(f"Alignment Trial {alignment_trial + 1}/{num_alignments}")
        print(f"{'='*60}")
        
        start_pose = t3d.affines.compose(
            [0.225, 0.0, 0.275],
            t3d.euler.euler2mat(np.pi, 0.0, 0.0),
            [1, 1, 1]
        )
        robot.servo_to_pose(start_pose, lin_tol=1e-2, ang_tol=0.1)
        
        iteration = 0
        iterations_within_tolerance = 0
        trial_data = []
        
        try:
            while True:
                
                observation = robot.get_observation()
                rgb_np = observation["rgb"].astype(np.float32) / 255.0

                if rgb_np.ndim == 2:
                    rgb_np = np.expand_dims(rgb_np, axis=-1)
                if rgb_np.shape[-1] == 1:
                    rgb_np = np.repeat(rgb_np, 3, axis=-1)

                rgb_images_batch = (
                    torch.from_numpy(rgb_np)
                    .permute(2, 0, 1)
                    .unsqueeze(0).unsqueeze(0)
                    .to(device)
                )

                with torch.no_grad():
                    raw_model_output = net(rgb_images_batch)

                relative_action_np = raw_model_output.squeeze(0).cpu().numpy()
                relative_action = sixd_se3(relative_action_np)

                rot_mat = np.array(relative_action[:3, :3], dtype=np.float64)
                try:
                    euler_rad = t3d.euler.mat2euler(rot_mat)
                    euler_deg = np.degrees(euler_rad)
                except Exception:
                    euler_deg = [0.0, 0.0, 0.0]

                trans = relative_action[:3, 3]

                print(f"\n--- Model Prediction (Relative to EE) ---")
                print(f"Translation (m):  X: {trans[0]:.4f}, Y: {trans[1]:.4f}, Z: {trans[2]:.4f}")
                print(f"Rotation (deg):   R: {euler_deg[0]:.2f}, P: {euler_deg[1]:.2f}, Y: {euler_deg[2]:.2f}")
                print(f"-----------------------------------------\n")

                error_magnitude = np.linalg.norm(relative_action[:3, 3])
                
                if are_tfs_close(
                    relative_action, lin_tol=cfg.lin_tolerance, ang_tol=ang_tol_rad
                ):
                    iterations_within_tolerance += 1
                    print(f"Step {iteration}: Within Tol ({iterations_within_tolerance}/{cfg.debouncing_count}) [error: {error_magnitude:.6f}]")
                else:
                    iterations_within_tolerance = 0
                    print(f"Step {iteration}: Adjusting... [error: {error_magnitude:.6f}]")

                scaled_action = relative_action.copy()
                translation_mult = getattr(cfg, 'translation_multiplier', 1.0)
                scaled_action[:3, 3] *= translation_mult
                scaled_action[:3, :3] = np.linalg.matrix_power(
                    scaled_action[:3, :3], int(cfg.rotation_matrix_multiplier)
                )
                
                current_pose = robot.pose()
                target_pose = current_pose @ scaled_action
                iteration += 1
                
                input("Hold ENTER to move robot (release to continue)...")
                robot.send_action({"pose": target_pose, "gripper.pos": 1.0})
                
                
                if iterations_within_tolerance >= cfg.max_iterations:
                    print(f"✓ Converged after {iteration} total steps.")
                    
                    gripper_z_offset = np.array([
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, cfg.manual_height],
                        [0, 0, 0, 1],
                    ])
                    robot.servo_to_pose(pose=robot.pose() @ gripper_z_offset)
                    
                    alignment_results.append({
                        "trial": alignment_trial + 1,
                        "success": True,
                        "iterations": iteration,
                    })
                    break
                
                if iteration >= MAX_TOTAL_STEPS:
                    print(f"✗ Failed: Timeout reached ({MAX_TOTAL_STEPS} steps).")
                    alignment_results.append({
                        "trial": alignment_trial + 1,
                        "success": False,
                        "iterations": iteration,
                    })
                    break

        except KeyboardInterrupt:
            print("\nTrial interrupted by user.")
            break
    
    print(f"\n{'='*60}")
    print(f"INFERENCE SUMMARY")
    print(f"{'='*60}")
    successful = sum(1 for r in alignment_results if r["success"])
    print(f"Success Rate: {successful}/{len(alignment_results)} ({successful*100//max(1, len(alignment_results))}%)")
    if alignment_results:
        print(f"Avg Steps to Converge: {np.mean([r['iterations'] for r in alignment_results]):.1f}")
    
    robot.disconnect()

if __name__ == "__main__":
    main()
