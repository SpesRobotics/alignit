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

    start_pose = t3d.affines.compose(
        [0.23, 0, 0.25], t3d.euler.euler2mat(np.pi, 0, 0), [1, 1, 1]
    )
    robot.servo_to_pose(start_pose, lin_tol=1e-2)
    iteration = 0
    iterations_within_tolerance = 0
    ang_tol_rad = np.deg2rad(cfg.ang_tolerance)
    
    try:
        while True:
            observation = robot.get_observation()
            # 1. Capture the raw numpy array from LeRobot's async_read
            rgb_np = observation["rgb"].astype(np.float32) / 255.0

            # 2. Safety check: Ensure it has 3 dimensions (H, W, C)
            # If the camera returned (480, 640), turn it into (480, 640, 1)
            if rgb_np.ndim == 2:
                rgb_np = np.expand_dims(rgb_np, axis=-1)

            # 3. Backbone requirement: EfficientNet needs 3 channels
            # If it's grayscale (1 channel), broadcast it to 3 channels
            if rgb_np.shape[-1] == 1:
                rgb_np = np.repeat(rgb_np, 3, axis=-1)

            # 4. Transform to (Batch, Sequence, Channel, Height, Width)
            # This converts (480, 640, 3) -> (3, 480, 640) -> (1, 1, 3, 480, 640)
            rgb_images_batch = (
                torch.from_numpy(rgb_np)
                .permute(2, 0, 1)    # Move channels to front
                .unsqueeze(0)        # Add Batch dimension
                .unsqueeze(0)        # Add Sequence dimension
                .to(device)
            )

            with torch.no_grad():
                if cfg.model.use_depth_input:
                    relative_action = net(rgb_images_batch, depth_images=depth_images_batch)
                else:
                    relative_action = net(rgb_images_batch)

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
            if iterations_within_tolerance >= cfg.max_iterations:
                print(f"Reached maximum iterations ({cfg.max_iterations}) - stopping.")
                print("Moving robot to final pose.")
                current_pose = robot.pose()
                gripper_z_offset = np.array(
                    [
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, cfg.manual_height],
                        [0, 0, 0, 1],
                    ]
                )
                offset_pose = current_pose @ gripper_z_offset
                robot.servo_to_pose(pose=offset_pose)
                robot.close_gripper()
                robot.gripper_off()

                break

        time.sleep(10.0)
    except KeyboardInterrupt:
        print("\nExiting...")

    robot.disconnect()


if __name__ == "__main__":
    main()