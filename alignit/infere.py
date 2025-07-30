import torch
from alignit.models.alignnet import AlignNet
from alignit.utils.zhou import sixd_se3
from alignit.utils.tfs import print_pose
from alignit.robots.xarmsim import XarmSim
import transforms3d as t3d
import numpy as np
import time


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load model from file
    net = AlignNet(
        output_dim=9,
        use_vector_input=False,
    )
    net.load_state_dict(torch.load("alignnet_model.pth", map_location=device))
    net.to(device)
    net.eval()

    robot = XarmSim()

    start_pose = t3d.affines.compose(
        [0.33, 0, 0.35], t3d.euler.euler2mat(np.pi, 0, 0), [1, 1, 1]
    )
    robot.servo_to_pose(start_pose, lin_tol=1e-2)
    total = 0
    tick = 0
    try:
        while True:
            observation = robot.get_observation()
            images = [observation["camera.rgb"].astype(np.float32) / 255.0]

            # Convert images to tensor and reshape from HWC to CHW format
            images_tensor = (
                torch.tensor(images, dtype=torch.float32)
                .permute(0, 3, 1, 2)
                .unsqueeze(0)
                .to(device)
            )
            print(torch.max(images_tensor))
            start = time.time()
            with torch.no_grad():
                relative_action = net(images_tensor)
            relative_action = relative_action.squeeze(0).cpu().numpy()
            relative_action = sixd_se3(relative_action)
            print_pose(relative_action)

            action = robot.pose() @ relative_action
            elapsed = time.time() - start
            total = total + elapsed
            tick += 1
            avg = total / tick
            print(avg)

            robot.send_action(action)
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()
