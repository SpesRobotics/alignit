import cv2
import numpy as np
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation

from alignit.robots.xarm import Xarm
from alignit.config import RecordConfig

class AprilTagBenchmark:
    def __init__(self, tag_size=0.06):
        camera_params = robot.get_intrinsics()
        self.detector = Detector(families="tag36h11")
        self.tag_size = tag_size
        self.camera_params = camera_params

    def detect_pose(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        tags = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=self.camera_params,
            tag_size=self.tag_size,
        )

        if len(tags) == 0:
            print("No tags detected")
            return None

        tag = tags[0]
        pose = np.eye(4)
        pose[:3, :3] = tag.pose_R
        pose[:3, 3] = tag.pose_t.flatten()
        return pose

    def pose_difference(self, T_ideal, T_current):
        """
        Compute difference between two 4x4 homogeneous transforms.
        Returns (dx, dy, dz, droll, dpitch, dyaw).
        """
        T_ideal = np.asarray(T_ideal)
        T_current = np.asarray(T_current)

        # Translation difference
        p_ideal = T_ideal[:3, 3]
        p_current = T_current[:3, 3]
        dp = p_current - p_ideal

        # Rotation difference
        R_ideal = T_ideal[:3, :3]
        R_current = T_current[:3, :3]

        # Relative rotation
        R_diff = R_current @ R_ideal.T
        
        # Convert rotation matrix to Euler angles (roll, pitch, yaw)
        rotation = Rotation.from_matrix(R_diff)
        d_rpy = rotation.as_euler('xyz')  # or 'zyx' depending on your convention

        return np.concatenate((dp, d_rpy))

    def pose_in_tag_frame(tag_pose_world, robot_pose_world):
        """Convert robot pose from world frame to AprilTag's frame."""
        tag_pose_inv = np.linalg.inv(tag_pose_world)
        return tag_pose_inv @ robot_pose_world

if __name__ == "__main__":
    robot = Xarm()
    detector = AprilTagBenchmark()
    while True:
        observation = robot.get_observation()
        curr_pose = robot.pose()
        rgb_image = observation["rgb"]
        T_after_inference = detector.detect_pose(rgb_image)
        T_ideal_tag = np.array(
            [
                [ 0.99143166,  0.13028123,  0.00949008,  0.02653058],
                [-0.12631724,  0.97469879, -0.18440775, -0.05716378],
                [-0.03327483,  0.18162892,  0.98280401,  0.25267621],
                [ 0. ,         0.   ,       0.   ,       1.        ],
            ],
            dtype=np.float32,
        )

        result = detector.pose_difference(T_ideal_tag, T_after_inference)
        dx, dy, dz, droll, dpitch, dyaw = result
        droll_deg = np.degrees(droll)
        dpitch_deg = np.degrees(dpitch)
        dyaw_deg = np.degrees(dyaw)
        print(f"Translation diff: ({dx:.3f}, {dy:.3f}, {dz:.3f})")
        print(f"Rotation diff (RPY): ({droll_deg:.3f}, {dpitch_deg:.3f}, {dyaw_deg:.3f})")
