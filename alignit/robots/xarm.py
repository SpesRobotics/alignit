from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
from lerobot_xarm.xarm import Xarm as LeXarm
from lerobot_xarm.config import XarmConfig


class Xarm:
    def __init__(self):
        config = RealSenseCameraConfig(
            serial_number_or_name="021222071076",
        )
        self.camera = RealSenseCamera(config)

        robot_config = XarmConfig()
        self.robot = LeXarm(robot_config)

    def connect(self):
        self.camera.connect()
        self.robot.connect()


    def send_action(self, pose):
        action = {
            "pose": pose,
            "gripper.pos": 0.0,
        }
        self.robot.send_action(action)

    def get_observation(self):
        rgb_image = self.camera.async_read()

        return {
            "rgb": rgb_image,
        }

    def disconnect(self):
        self.camera.disconnect()


if __name__ == "__main__":
    xarm = Xarm()
    xarm.connect()
    obs = xarm.get_observation()

    print("Observation:")
    print("RGB Image:", obs["rgb"])

    xarm.disconnect()
