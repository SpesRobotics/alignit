import numpy as np
import transforms3d as t3d
from alignit.robots.bullet import Bullet
from alignit.utils.tfs import are_tfs_close


def main():
    robot = Bullet()
    # pose = np.array([[0, 1, 0, 0.5], [1, 0, 0, 0], [0, 0, -1, 0.3], [0, 0, 0, 1]])
    pose = t3d.affines.compose(
        [0.5, 0.1, 0.18], t3d.euler.euler2mat(np.pi, 0, 0), [1, 1, 1]
    )

    for t in range(100000000):
        robot.send_action(pose)
        observation = robot.get_observation()

        current_pose = robot.pose()
        if are_tfs_close(current_pose, pose):
            print("Pose is close enough!")
            break

        print(observation)
    robot.close()


if __name__ == "__main__":
    main()
