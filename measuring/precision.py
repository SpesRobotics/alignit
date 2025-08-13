import cv2
import numpy as np
from pupil_apriltags import Detector
from alignit.robots.xarm import Xarm


class AprilTagDetector:
    def __init__(self, tag_size=0.08):
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
            return None

        tag = tags[0]
        pose = np.eye(4)
        pose[:3, :3] = tag.pose_R
        pose[:3, 3] = tag.pose_t.flatten()
        return pose


if __name__ == "__main__":
    robot = Xarm()
    detector = AprilTagDetector()

    while True:
        observation = robot.get_observation()
        rgb_image = observation["rgb"]
        pose = detector.detect_pose(rgb_image)
        print("Detected pose:", pose)
