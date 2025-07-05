import torch
from alignit.models.alignnet import AlignNet
from alignit.utils.zhou import sixd_se3
from alignit.robots.bullet import Bullet


def main():
    # load model from file
    net = AlignNet(
        output_dim=9,  # 3 for translation + 6 for rotation in sixd format
        use_vector_input=False,  # Disable vector input since we're not using it
    )
    net.load_state_dict(torch.load("alignnet_model.pth"))
    net.eval()

    robot = Bullet()
    observation = robot.get_observation()
    images = [ observation["camera.rgb"] ]

    # Convert images to tensor and reshape from HWC to CHW format
    images_tensor = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2).unsqueeze(0)
    # Keep only RGB channels (first 3) if image has alpha channel
    images_tensor = images_tensor[:, :, :3, :, :]
    with torch.no_grad():
        action = net(images_tensor)
    action = action.squeeze(0).numpy()
    action = sixd_se3(action)
    print(f"Predicted action: {action}")

if __name__ == "__main__":
    main()
