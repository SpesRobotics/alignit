from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
from lerobot_xarm.xarm import Xarm as LeXarm
from lerobot_xarm.config import XarmConfig
from alignit.robots.robot import Robot
import numpy as np
import transforms3d as t3d
import time
from alignit.utils.tfs import are_tfs_close


class Xarm(Robot):
    def __init__(self):
        config = RealSenseCameraConfig(
            serial_number_or_name="233522070823",
            fps=60,
            width=640,
            height=480,
            use_depth=True
        )
        self.camera = RealSenseCamera(config)

        robot_config = XarmConfig()
        self.robot = LeXarm(robot_config)
        self._connect()

    def _connect(self):
        self.camera.connect()
        self.robot.connect()

    def send_action(self, action):
        self.robot.send_action(action)

    def get_observation(self):
        rgb_image = self.camera.read()

        return {
            "rgb": rgb_image,
        }

    def disconnect(self):
        self.camera.disconnect()

    def servo_to_pose(self, pose, lin_tol=1e-3, ang_tol=1e-2):
        while not are_tfs_close(self.pose(), pose, lin_tol, ang_tol):
            action = {
                "pose": pose,
                "gripper.pos": 1.0,  # Optional: set gripper state (0.0=closed, 1.0=open)
            }
            self.send_action(action)
            time.sleep(1.0 / 60.0)  # Adjust frequency as needed

    def reset(self):
        """
        Reset routine:
        1. Allows manual movement of the arm
        2. Waits for user input (Enter key)
        3. Applies gripper-frame Z offset
        4. Applies world-frame Z offset
        5. Returns to normal operation

        Args:
            manual_height: Height above surface to maintain during manual movement (meters)
            world_z_offset: Additional Z offset in world frame after manual positioning (meters)
        """
        manual_height = -0.05
        world_z_offset = -0.02
        self.robot.disconnect()
        input("Press Enter after positioning the arm...")
        self.robot.connect()
        current_pose = self.pose()
        gripper_z_offset = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, manual_height], [0, 0, 0, 1]]
        )
        offset_pose = current_pose @ gripper_z_offset
        self.servo_to_pose(pose=offset_pose)

        world_z_offset_mat = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, world_z_offset], [0, 0, 0, 1]]
        )
        final_pose = offset_pose @ world_z_offset_mat
        self.servo_to_pose(pose=final_pose)

        pose_start = current_pose @ t3d.affines.compose(
            [0, 0, -0.090], t3d.euler.euler2mat(0, 0, 0), [1, 1, 1]
        )
        pose_alignment_target = current_pose @ t3d.affines.compose(
            [0, 0, -0.1], t3d.euler.euler2mat(0, 0, 0), [1, 1, 1]
        )
        _, (position, _, _) = self.robot._arm.get_joint_states()
        for i in range(6):  
            joint_name = f"joint{i+1}"  
            self.robot._jacobi.set_joint_position(joint_name, position[i])


        return pose_start, pose_alignment_target

    def pose(self):
        return self.robot._jacobi.get_ee_pose()


if __name__ == "__main__":
    xarm = Xarm()
    for i in range(10):
        obs = xarm.get_observation()
        frames = []
        frame = {"images": [obs["rgb"]]}
        frames.append(frame)

    pose_matrix = np.eye(4)
    translation = [0.23, 0, 0.1]
    rotation = t3d.euler.euler2mat(np.pi, 0, 0)
    pose_matrix = t3d.affines.compose(translation, rotation, [1, 1, 1])

    xarm.servo_to_pose(pose=pose_matrix, lin_tol=1e-3, ang_tol=1e-2)

    print("Observation:")
    print("RGB Image:", obs["rgb"])

    xarm.disconnect()
