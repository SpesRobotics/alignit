import cv2
import numpy as np
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation as R

from alignit.robots.xarm import Xarm


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

    def pose_difference(self, T_ideal, T_current, max_pos_distance=1.0):

        try:
            # Convert and validate inputs
            T1 = np.array(T_ideal, dtype=np.float32)
            T2 = np.array(T_current, dtype=np.float32)

            if T1.shape != (4, 4) or T2.shape != (4, 4):
                raise ValueError("Input matrices must be 4x4")

            # Extract components
            p1 = T1[:3, 3] * 1000  # Ideal position
            p2 = T2[:3, 3] * 1000  # Current position
            R1 = T1[:3, :3]  # Ideal rotation
            R2 = T2[:3, :3]  # Current rotation

            # Position difference (Euclidean distance)
            pos_distance = np.linalg.norm(p2 - p1)  # Convert to millimeters
            pos_diff_pct = min(pos_distance / max_pos_distance, 1.0) * 100

            # Rotation difference (angle in radians)
            rel_rot = R1.T @ R2
            angle_rad = np.arccos(np.clip((np.trace(rel_rot) - 1) / 2, -1, 1))
            rot_diff_pct = (angle_rad / np.pi) * 100

            # Euler angle differences
            rpy1 = R.from_matrix(R1).as_euler("xyz")
            rpy2 = R.from_matrix(R2).as_euler("xyz")
            rpy_diff = np.abs(rpy2 - rpy1)

            return {
                "position_diff_milimeters": pos_distance,
                "position_diff%": pos_diff_pct,
                "rotation_diff_rad": angle_rad,
                "rotation_diff%": rot_diff_pct,
                "rpy_diff_rad": rpy_diff,
                "xyz_diff_milimeters": (p2 - p1).tolist(),
                "combined_diff%": 0.5 * pos_diff_pct + 0.5 * rot_diff_pct,
            }

        except Exception as e:
            print(f"Pose comparison failed: {str(e)}")
            return None


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
        apriltag_pose = detector.detect_pose(rgb_image)
        T_ideal_tag = np.array(
            [
                [0.92529235, 0.37684695, 0.04266663, 0.25978898],
                [0.37511194, -0.89279238, -0.24942494, 0.06284318],
                [-0.05590259, 0.24679575, -0.96745375, -0.25993736],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        T_after_inference = pose_in_tag_frame(apriltag_pose, curr_pose)
        result = detector.pose_difference(T_ideal_tag, T_after_inference)
        if result:
            print("xyz_diff_milimeters:", result["xyz_diff_milimeters"])
            print("position_diff_milimeters:", result["position_diff_milimeters"])
            print("rotation_diff_rad:", result["rotation_diff_rad"])
            print("rpy_diff_rad:", result["rpy_diff_rad"])
            print("combined_diff%:", result["combined_diff%"])
        else:
            print("Error: Failed to get current pose estimate")
