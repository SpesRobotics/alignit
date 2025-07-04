import time
import numpy as np
import transforms3d as t3d
from alignit.robots.bullet import Bullet
from alignit.utils.tfs import are_tfs_close


def generate_spiral_trajectory(
    start_pose, z_step=0.001, radius_step=0.001, num_steps=100
):
    """
    Generate a spiral trajectory in 3D space, moving backwards along the gripper's local z-axis.
    The gripper keeps its original orientation from start_pose.

    Args:
        start_pose (np.ndarray): 4x4 pose matrix of the gripper at the starting position.
        z_step (float): Spiral step size in the negative local z-direction.
        radius_step (float): How much the spiral radius grows per step.
        num_steps (int): Number of steps to generate.

    Returns:
        list of np.ndarray: Each is a 4x4 pose matrix.
    """
    trajectory = []
    R = start_pose[:3, :3]
    t = start_pose[:3, 3]

    for i in range(num_steps):
        radius = radius_step * i
        angle = 2 * np.pi * i / 10  # tweak this to control spiral tightness

        # Local offset in gripper frame
        local_offset = np.array(
            [
                radius * np.cos(angle),  # x
                radius * np.sin(angle),  # y
                -z_step * i,  # negative z
            ]
        )

        # Convert to world offset
        world_offset = R @ local_offset
        position = t + world_offset

        # Build new pose
        T = np.eye(4)
        T[:3, :3] = R  # same orientation as start
        T[:3, 3] = position
        trajectory.append(T)

    return trajectory


def servo_to_pose(robot, pose, lin_tol=1e-1, ang_tol=1e-1):
    while not are_tfs_close(robot.pose(), pose, lin_tol, ang_tol):
        robot.send_action(pose)


def main():
    robot = Bullet()
    pose_final_target = t3d.affines.compose(
        [0.5, 0.1, 0.18], t3d.euler.euler2mat(np.pi, 0, 0), [1, 1, 1]
    )
    pose_alignment_target = pose_final_target @ t3d.affines.compose(
        [0, 0, -0.15], t3d.euler.euler2mat(0, 0, 0), [1, 1, 1]
    )

    servo_to_pose(robot, pose_final_target)
    servo_to_pose(robot, pose_alignment_target)

    trajectory = generate_spiral_trajectory(
        pose_alignment_target, z_step=0.002, radius_step=0.002, num_steps=50
    )

    for pose in trajectory:
        servo_to_pose(robot, pose, lin_tol=0.01, ang_tol=0.01)
        time.sleep(0.05)

    robot.close()


if __name__ == "__main__":
    main()
