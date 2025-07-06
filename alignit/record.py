import time
import numpy as np
import transforms3d as t3d
from alignit.robots.bullet import Bullet
from alignit.utils.tfs import are_tfs_close
from datasets import Dataset, Features, Sequence, Value, Image, load_from_disk, concatenate_datasets
from alignit.utils.zhou import se3_sixd
import shutil
import os


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


def main():
    robot = Bullet()
    features = Features({
        "images": Sequence(Image()),
        "action": Sequence(Value("float32"))
    })

    pose_final_target = t3d.affines.compose(
        [0.5, 0.1, 0.18], t3d.euler.euler2mat(np.pi, 0, 0), [1, 1, 1]
    )
    pose_alignment_target = pose_final_target @ t3d.affines.compose(
        [0, 0, -0.15], t3d.euler.euler2mat(0, 0, 0), [1, 1, 1]
    )
    pose_record_start = pose_alignment_target @ t3d.affines.compose(
        [0, 0, 0.05], t3d.euler.euler2mat(0, 0, 0), [1, 1, 1]
    )

    robot.servo_to_pose(pose_final_target)

    for i in range(10):
        pose_record_start_episode = pose_record_start.copy()
        pose_record_start_episode[:3, 3] += np.random.uniform(
            low=[-0.03, -0.03, -0.1],
            high=[0.03, 0.03, 0.01]
        )
        robot.servo_to_pose(pose_record_start_episode)

        trajectory = generate_spiral_trajectory(
            pose_record_start_episode, z_step=0.001, radius_step=0.001, num_steps=70
        )

        frames = []
        for pose in trajectory:
            robot.servo_to_pose(pose, lin_tol=0.01, ang_tol=0.01)
            current_pose = robot.pose()
            action_pose = np.linalg.inv(pose_alignment_target) @ current_pose
            action_sixd = se3_sixd(action_pose)

            observation = robot.get_observation()
            frame = {
                "images": [ observation["camera.rgb"]],
                "action": action_sixd
            }
            frames.append(frame)
        
        print(f"Episode {i+1} completed with {len(frames)} frames.")
        episode_dataset = Dataset.from_list(frames, features=features)
        
        if i == 0:
            # First episode, save directly
            combined_dataset = episode_dataset
        else:
            previous_dataset = load_from_disk("data/duck")
            previous_dataset = previous_dataset.cast(features)
            combined_dataset = concatenate_datasets([previous_dataset, episode_dataset])
            del previous_dataset

        temp_path = "data/duck_temp"
        combined_dataset.save_to_disk(temp_path)
        
        # Remove old dataset and move temp to final location
        if os.path.exists("data/duck"):
            shutil.rmtree("data/duck")
        shutil.move(temp_path, "data/duck")

    robot.close()


if __name__ == "__main__":
    main()
