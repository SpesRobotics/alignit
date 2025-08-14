import transforms3d as t3d
import numpy as np
import time
import draccus
from alignit.config import InferConfig
import torch
from alignit.models.alignnet import AlignNet
from alignit.utils.zhou import sixd_se3
from alignit.utils.tfs import print_pose, are_tfs_close
from alignit.robots.xarmsim import XarmSim
from alignit.robots.xarm import Xarm


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

    start_pose = t3d.affines.compose(
        [0.23, 0, 0.25], t3d.euler.euler2mat(np.pi, 0, 0), [1, 1, 1]
    )
    robot.servo_to_pose(start_pose, lin_tol=1e-2)
    iteration = 0
    iterations_within_tolerance = 0
    ang_tol_rad = np.deg2rad(cfg.ang_tolerance)
    print("beforee")
    try:
        while True:
            observation = robot.get_observation()
            rgb_image = observation["rgb"].astype(np.float32) / 255.0
            depth_image_unscaled = observation["depth"].astype(np.float32)
            depth_image_unscaled = np.clip(depth_image_unscaled, a_min=None, a_max=1000)

            depth_image = depth_image_unscaled / 1000.0  # Scale depth to meters
            print(
                "Min/Max depth (raw):",
                observation["depth"].min(),
                observation["depth"].max(),
            )
            print("Min/Max depth (scaled):", depth_image.min(), depth_image.max())
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
            if are_tfs_close(
                relative_action, lin_tol=cfg.lin_tolerance, ang_tol=ang_tol_rad
            ):
                iterations_within_tolerance += 1
            else:
                iterations_within_tolerance = 0

            print(relative_action)
            target_pose = robot.pose() @ relative_action
            iteration += 1
            action = {
                "pose": target_pose,
                "gripper.pos": 1.0,
            }
            robot.send_action(action)
            # Check max iterations
            if cfg.max_iterations and iteration >= cfg.max_iterations:
                print(f"Reached maximum iterations ({cfg.max_iterations}) - stopping.")
                break

        time.sleep(10.0)
    except KeyboardInterrupt:
        print("\nExiting...")

    robot.disconnect()


if __name__ == "__main__":
    main()
