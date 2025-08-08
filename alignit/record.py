import numpy as np
import transforms3d as t3d
from alignit.robots.xarmsim import XarmSim
from alignit.robots.xarm import Xarm
from datasets import (
    Dataset,
    Features,
    Sequence,
    Value,
    Image,
    load_from_disk,
    concatenate_datasets,
)
import os
import shutil
from alignit.utils.zhou import se3_sixd
import time


import numpy as np
from scipy.spatial.transform import Rotation as R


def generate_spiral_trajectory(
    start_pose,
    z_step=0.1,
    radius_step=0.001,
    num_steps=50,
    lift_height_before_spiral=0.0005,
    max_random_rotation_angle=10.0,  # in degrees
):
    trajectory = []
    R_start = start_pose[:3, :3]
    t_start_initial = start_pose[:3, 3]

    object_z_axis = R_start[:, 2]

    lift_offset_world = object_z_axis * lift_height_before_spiral
    t_start_spiral = t_start_initial + lift_offset_world

    for i in range(num_steps):
        radius = radius_step * i
        angle = 2 * np.pi * i / 10

        local_offset = np.array(
            [radius * np.cos(angle), radius * np.sin(angle), -z_step * i]
        )

        world_offset = R_start @ local_offset
        base_position = t_start_spiral + world_offset
        x_rot = np.random.uniform(-10, 10)
        y_rot = np.random.uniform(-10, 10)
        z_rot = np.random.uniform(-10, 10)

        # random_angles = np.radians(np.random.uniform(-max_random_rotation_angle, max_random_rotation_angle, 3))
        random_angles = np.radians([x_rot, y_rot, z_rot])
        random_rotation = R.from_euler("xyz", random_angles).as_matrix()

        randomized_rotation = R_start @ random_rotation

        T = np.eye(4)
        T[:3, :3] = randomized_rotation
        T[:3, 3] = base_position
        trajectory.append(T)

    return trajectory


def main():
    robot = Xarm()
    features = Features(
        {"images": Sequence(Image()), "action": Sequence(Value("float32"))}
    )

    for episode in range(15):
        pose_start, pose_alignment_target = robot.reset()

        # robot.servo_to_pose(pose_alignment_target, lin_tol=0.015, ang_tol=0.015)

        trajectory = generate_spiral_trajectory(
            pose_start,
            z_step=0.0007,
            radius_step=0.001,
            num_steps=50,
        )
        frames = []
        for pose in trajectory:
            robot.servo_to_pose(pose, lin_tol=3e-3, ang_tol=0.03)
            current_pose = robot.pose()

            action_pose = np.linalg.inv(current_pose) @ pose_alignment_target
            action_sixd = se3_sixd(action_pose)

            observation = robot.get_observation()
            frame = {"images": [observation["rgb"].copy()], "action": action_sixd}
            frames.append(frame)

        print(f"Episode {episode+1} completed with {len(frames)} frames.")

        episode_dataset = Dataset.from_list(frames, features=features)
        if episode == 0:
            combined_dataset = episode_dataset
        else:
            previous_dataset = load_from_disk("data/duck")
            previous_dataset = previous_dataset.cast(features)
            combined_dataset = concatenate_datasets([previous_dataset, episode_dataset])
            del previous_dataset

        temp_path = "data/duck_temp"
        combined_dataset.save_to_disk(temp_path)
        if os.path.exists("data/duck"):
            shutil.rmtree("data/duck")
        shutil.move(temp_path, "data/duck")

    robot.disconnect()


if __name__ == "__main__":
    main()
