import numpy as np
from xarm.wrapper import XArmAPI
import transforms3d as t3d


class Xarm:
    def __init__(self, ip):
        self.ip = ip
        self.arm = None

    def connect(self):
        self.arm = XArmAPI(self.ip, is_radian=True)
        self.arm.connect()
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(1)
        self.arm.set_state(state=0)

    def joint_positions(self):
        code, (joint_angles, joint_velocities, joint_efforts) = (
            self._arm.get_joint_states()
        )
        return np.array(joint_angles)

    def pose(self):
        ok, pose = self.arm.get_position()
        if ok != 0:
            raise RuntimeError(f"Failed to get arm position: {ok}")

        translation = np.array(pose[:3]) / 1000
        eulers = np.array(pose[3:])
        rotation = t3d.euler.euler2mat(eulers[0], eulers[1], eulers[2], "sxyz")
        pose = t3d.affines.compose(translation, rotation, np.ones(3))
        return pose

    def servoj(self, joint_positions):
        self.arm.set_servo_angle_j(joint_positions)

    def servo(self, pose):
        x = pose[0, 3] * 1000
        y = pose[1, 3] * 1000
        z = pose[2, 3] * 1000
        roll, pitch, yaw = t3d.euler.mat2euler(pose[:3, :3])
        error = self.arm.set_servo_cartesian(
            [x, y, z, roll, pitch, yaw], speed=100, mvacc=100
        )
        return error
