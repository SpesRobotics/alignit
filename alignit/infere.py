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
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    net = AlignNet(
        output_dim=9,
        use_vector_input=False,
        use_depth_input=True,
    )
    net.load_state_dict(torch.load("alignnet_model.pth", map_location=device))
    net.to(device)
    net.eval()

    robot = Xarm()

    start_pose = t3d.affines.compose(
        [0.23, 0, 0.25], t3d.euler.euler2mat(np.pi, 0, 0), [1, 1, 1]
    )
    robot.servo_to_pose(start_pose, lin_tol=1e-2)
    total = 0
    tick = 0
    try:
        while True:
            start_capture = time.time()
            observation = robot.get_observation()
            print(f"Observation time: {time.time() - start_capture:.3f}s")

            rgb_image = observation["rgb"].astype(np.float32) / 255.0
            depth_image = observation["depth"].astype(np.float32) / 1000.0
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


            start = time.time()
            with torch.no_grad():
                relative_action = net(rgb_images_batch, depth_images=depth_images_batch)
            relative_action = relative_action.squeeze(0).cpu().numpy()
            relative_action = sixd_se3(relative_action)
            relative_action [:3,:3] = relative_action[:3,:3] @ relative_action[:3,:3] @ relative_action[:3,:3]
            print_pose(relative_action)

            action = robot.pose() @ relative_action
            elapsed = time.time() - start
            total = total + elapsed
            tick += 1
            avg = total / tick
            action = {
                "pose": action,
                "gripper.pos": 1.0,
            }
            robot.send_action(action)
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()
