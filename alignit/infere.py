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

    # load model from file
    net = AlignNet(
        backbone_name=cfg.model.backbone,
        backbone_weights=cfg.model.backbone_weights,
        use_vector_input=cfg.model.use_vector_input,
        fc_layers=cfg.model.fc_layers,
        vector_hidden_dim=cfg.model.vector_hidden_dim,
        output_dim=cfg.model.output_dim,
        feature_agg=cfg.model.feature_agg,
    )
    net.load_state_dict(torch.load(cfg.model.path, map_location=device))
    net.to(device)
    net.eval()

    robot = XarmSim()

    # Set initial pose from config
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
            images = [observation["camera.rgb"].astype(np.float32) / 255.0]

            # Convert images to tensor and reshape from HWC to CHW format
            images_tensor = (
                torch.from_numpy(np.array(images))
                .permute(0, 3, 1, 2)
                .unsqueeze(0)
                .to(device)
            )

            if cfg.debug_output:
                print(f"Max pixel value: {torch.max(images_tensor)}")

            start = time.time()
            with torch.no_grad():
                relative_action = net(images_tensor)
            relative_action = relative_action.squeeze(0).cpu().numpy()
            relative_action = sixd_se3(relative_action)

            if cfg.debug_output:
                print_pose(relative_action)

            # Check convergence
            if are_tfs_close(
                relative_action, lin_tol=cfg.lin_tolerance, ang_tol=ang_tol_rad
            ):
                iterations_within_tolerance += 1
            else:
                iterations_within_tolerance = 0

            if iterations_within_tolerance >= cfg.debouncing_count:
                print("Alignment achieved - stopping.")
                break

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
