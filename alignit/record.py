import numpy as np
from alignit.robots.bullet import Bullet


def main():
    robot = Bullet()
    for t in range(100000000):
        pose = np.array([[0, 1, 0, 0.5], [1, 0, 0, 0], [0, 0, -1, 0.3], [0, 0, 0, 1]])
        robot.send_action(pose)
        observation = robot.get_observation()
        print(observation)
    robot.close()


if __name__ == "__main__":
    main()
