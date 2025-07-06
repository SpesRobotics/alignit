import torch
from alignit.models.alignnet import AlignNet
from alignit.utils.zhou import sixd_se3
from alignit.robots.bullet import Bullet
from alignit.utils.tfs import print_pose
import transforms3d as t3d
import numpy as np


def main():
    # load model from file
    net = AlignNet(
        output_dim=9,
        use_vector_input=False,
    )
    net.load_state_dict(torch.load("alignnet_model.pth"))
    net.eval()

    robot = Bullet()

    start_pose = t3d.affines.compose(
        [0.5, 0.1, 0.2], t3d.euler.euler2mat(np.pi, 0, 0), [1, 1, 1]
    )
    robot.servo_to_pose(start_pose)

    while True:
        observation = robot.get_observation()
        images = [ observation["camera.rgb"] ]

        # Convert images to tensor and reshape from HWC to CHW format
        images_tensor = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2).unsqueeze(0)
        with torch.no_grad():
            relative_action = net(images_tensor)
        relative_action = relative_action.squeeze(0).numpy()
        relative_action = sixd_se3(relative_action)
        print_pose(relative_action)

        action = robot.pose() @ relative_action
        robot.send_action(action)

if __name__ == "__main__":
    main()
