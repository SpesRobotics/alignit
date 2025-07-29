import numpy as np
import transforms3d as t3d
from alignit.robots.xarmsim import XarmSim
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


def generate_spiral_trajectory(
    start_pose,
    z_step=0.1,
    radius_step=0.001,
    num_steps=100,
    cone_angle=45,
    visible_sweep=180,
    viewing_angle_offset=0,
    angular_resolution=5,
    include_cone_poses=True,
    lift_height_before_spiral=0.01,
):
    trajectory = []
    R_start = start_pose[:3, :3]
    t_start_initial = start_pose[:3, 3]

    cone_angle_rad = np.deg2rad(cone_angle)

    object_z_axis = R_start[:, 2]

    lift_offset_world = object_z_axis * lift_height_before_spiral
    t_start_spiral = t_start_initial + lift_offset_world

    start_angle = -visible_sweep / 2 + viewing_angle_offset
    end_angle = visible_sweep / 2 + viewing_angle_offset

    for i in range(num_steps):
        radius = radius_step * i
        angle = 2 * np.pi * i / 10

        local_offset = np.array(
            [radius * np.cos(angle), radius * np.sin(angle), -z_step * i]
        )

        world_offset = R_start @ local_offset
        base_position = t_start_spiral + world_offset

        T = np.eye(4)
        T[:3, :3] = R_start
        T[:3, 3] = base_position
        trajectory.append(T)

        if include_cone_poses:
            for deg in np.arange(start_angle, end_angle, angular_resolution):
                theta = np.deg2rad(deg)

                tilt = t3d.euler.euler2mat(cone_angle_rad, 0, 0)
                spin = t3d.euler.euler2mat(0, 0, theta)
                R_cone = R_start @ spin @ tilt

                T_cone = np.eye(4)
                T_cone[:3, :3] = R_cone
                T_cone[:3, 3] = base_position
                trajectory.append(T_cone)

    return trajectory


def main():
    robot = XarmSim()
    features = Features(
        {"images": Sequence(Image()), "action": Sequence(Value("float32"))}
    )
    for episode in range(20):

        pose_start, pose_alignment_target = robot.reset()

        robot.servo_to_pose(pose_alignment_target, lin_tol=0.015, ang_tol=0.015)

        trajectory = generate_spiral_trajectory(
            pose_start,
            z_step=0.0007,
            radius_step=0.001,
            num_steps=100,
            cone_angle=30,
            visible_sweep=60,
            viewing_angle_offset=-120,
            angular_resolution=10,
            include_cone_poses=False,
        )
        frames = []
        for pose in trajectory:
            robot.servo_to_pose(pose, lin_tol=0.05, ang_tol=0.05)
            current_pose = robot.pose()

            action_pose = np.linalg.inv(current_pose) @ pose_alignment_target
            action_sixd = se3_sixd(action_pose)

            observation = robot.get_observation()
            frame = {"images": [observation["camera.rgb"]], "action": action_sixd}
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

    robot.close()


if __name__ == "__main__":
    main()
