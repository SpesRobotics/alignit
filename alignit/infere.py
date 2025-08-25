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
from alignit.robots.xarm import Xarm
import matplotlib.pyplot as plt
import csv

def rot_angle_deg(R: np.ndarray) -> float:
    # Robust acos for rotation angle in [0, pi]
    c = (np.trace(R) - 1.0) * 0.5
    c = np.clip(c, -1.0, 1.0)
    return np.degrees(np.arccos(c))


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

    robot = XarmSim()

    start_pose = t3d.affines.compose(
        [0.23, 0, 0.25], t3d.euler.euler2mat(np.pi, 0, 0), [1, 1, 1]
    )
    robot.servo_to_pose(start_pose, lin_tol=1e-2)
    iteration = 0
    iterations_within_tolerance = 0
    ang_tol_rad = np.deg2rad(cfg.ang_tolerance)
    # ---- Convergence logging + live plot setup ----
    t0 = time.perf_counter()
    times_s = []
    trans_m = []
    rot_deg = []

    enable_live_plot = True
    save_plot_path = "convergence.png"
    save_csv_path = "convergence_log.csv"
    if enable_live_plot:
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 6))
        ln_trans, = ax1.plot([], [], linewidth=2)
        ax1.set_ylabel("Translation error [m]")
        ax1.grid(True, linestyle=":")

        ln_rot, = ax2.plot([], [], linewidth=2)
        ax2.set_ylabel("Rotation error [deg]")
        ax2.set_xlabel("Time [s]")
        ax2.grid(True, linestyle=":")
        fig.tight_layout()
    try:
        while True:
            observation = robot.get_observation()
            rgb_image = observation["rgb"].astype(np.float32) / 255.0
            depth_image = observation["depth"].astype(np.float32)
            print(
                "Min/Max depth,mean (raw):",
                observation["depth"].min(),
                observation["depth"].max(),
                observation["depth"].mean(),
            )
            print(
                "Min/Max depth,mean (scaled):",
                depth_image.min(),
                depth_image.max(),
                depth_image.mean(),
            )
            rgb_image_tensor = (
                torch.from_numpy(np.array(rgb_image))
                .permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
                .unsqueeze(0)
                .to(device)
            )

            depth_image_tensor = (
                torch.from_numpy(np.array(depth_image))
                .unsqueeze(0)  # Add channel dimension: (1, H, W)
                .unsqueeze(0)  # Add batch dimension: (1, 1, H, W)
                .to(device)
            )
            rgb_images_batch = rgb_image_tensor.unsqueeze(1)
            depth_images_batch = depth_image_tensor.unsqueeze(1)

            with torch.no_grad():
                relative_action = net(rgb_images_batch, depth_images=depth_images_batch)
            relative_action = relative_action.squeeze(0).cpu().numpy()
            relative_action = sixd_se3(relative_action)

            if cfg.debug_output:
                print_pose(relative_action)

            relative_action[:3, :3] = np.linalg.matrix_power(
                relative_action[:3, :3], cfg.rotation_matrix_multiplier
            )

            elapsed = time.perf_counter() - t0
            t_err = float(np.linalg.norm(relative_action[:3, 3]))
            r_err = float(rot_angle_deg(relative_action[:3, :3]))

            times_s.append(elapsed)
            trans_m.append(t_err)
            rot_deg.append(r_err)

            # ---- Live plot update ----
            if enable_live_plot:
                ln_trans.set_data(times_s, trans_m)
                ln_rot.set_data(times_s, rot_deg)
                ax1.relim(); ax1.autoscale_view()
                ax2.relim(); ax2.autoscale_view()
                # Optionally annotate current values in the title:
                ax1.set_title(f"Convergence (t={elapsed:.2f}s, ‖t‖={t_err:.3f} m, θ={r_err:.1f}°)")
                plt.pause(0.001)

            if are_tfs_close(
                relative_action, lin_tol=cfg.lin_tolerance, ang_tol=ang_tol_rad
            ):
                iterations_within_tolerance += 1
            else:
                iterations_within_tolerance = 0

            print(relative_action)
            if np.linalg.norm(relative_action[2, 3]) < 0.02:
               relative_action[:3, 3] = relative_action[:3, 3] / 5.0

            target_pose = robot.pose() @ relative_action
            iteration += 1
            action = {
                "pose": target_pose,
                "gripper.pos": 1.0,
            }
            robot.send_action(action)
            if iterations_within_tolerance >= cfg.max_iterations:
                print(f"Reached maximum iterations ({cfg.max_iterations}) - stopping.")
                print("Moving robot to final pose.")
                # time.sleep(10.0)
                # current_pose = robot.pose()
                # gripper_z_offset = np.array(
                #     [
                #         [1, 0, 0, 0],
                #         [0, 1, 0, 0],
                #         [0, 0, 1, cfg.manual_height],
                #         [0, 0, 0, 1],
                #     ]
                # )
                # offset_pose = current_pose @ gripper_z_offset
                # robot.servo_to_pose(pose=offset_pose)
                # robot.close_gripper()
                # robot.gripper_off()
                # stop_time = time.perf_counter() - t0
                # print(f"Total time to stop: {stop_time:.3f} s")
                # if enable_live_plot:
                #     # Draw a vertical line at stop
                #     for ax in (ax1, ax2):
                #         ax.axvline(stop_time, linestyle="--")
                #     fig.canvas.draw()
                #     fig.canvas.flush_events()
                break
    except KeyboardInterrupt:
        print("\nExiting...")


    if enable_live_plot:
        plt.ioff()
    if save_plot_path:
        fig.savefig(save_plot_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_plot_path}")

    if save_csv_path and len(times_s) > 0:
        with open(save_csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "trans_error_m", "rot_error_deg"])
            w.writerows(zip(times_s, trans_m, rot_deg))
        print(f"Saved log to {save_csv_path}")   
    
    robot.disconnect()


if __name__ == "__main__":
    main()