import os
import shutil

import transforms3d as t3d
import numpy as np
from scipy.spatial.transform import Rotation as R
from datasets import (
    Dataset,
    Features,
    Sequence,
    Value,
    Image,
    load_from_disk,
    concatenate_datasets,
)

from alignit.robots.xarmsim import XarmSim
from alignit.robots.xarm import Xarm
from alignit.utils.zhou import se3_sixd
import draccus
from alignit.config import RecordConfig


def generate_spiral_trajectory(start_pose, cfg):
    """Generate spiral trajectory using configuration parameters."""
    trajectory = []
    R_start = start_pose[:3, :3]
    t_start_initial = start_pose[:3, 3]

    cone_angle_rad = np.deg2rad(cfg.cone_angle)

    object_z_axis = R_start[:, 2]

    lift_offset_world = object_z_axis * cfg.lift_height_before_spiral
    t_start_spiral = t_start_initial + lift_offset_world

    start_angle = -cfg.visible_sweep / 2 + cfg.viewing_angle_offset
    end_angle = cfg.visible_sweep / 2 + cfg.viewing_angle_offset

    for i in range(cfg.num_steps):
        radius = cfg.radius_step * i
        angle = 2 * np.pi * i / 10

        local_offset = np.array(
            [radius * np.cos(angle), radius * np.sin(angle), -cfg.z_step * i]
        )

        world_offset = R_start @ local_offset
        base_position = t_start_spiral + world_offset
        x_rot = np.random.uniform(-10, 10)
        y_rot = np.random.uniform(-10, 10)
        z_rot = np.random.uniform(-10, 10)

        random_angles = np.radians([x_rot, y_rot, z_rot])
        random_rotation = R.from_euler("xyz", random_angles).as_matrix()

        randomized_rotation = R_start @ random_rotation

        T = np.eye(4)
        T[:3, :3] = randomized_rotation
        T[:3, 3] = base_position
        trajectory.append(T)

        if cfg.include_cone_poses:
            for deg in np.arange(start_angle, end_angle, cfg.angular_resolution):
                theta = np.deg2rad(deg)

                tilt = t3d.euler.euler2mat(cone_angle_rad, 0, 0)
                spin = t3d.euler.euler2mat(0, 0, theta)
                R_cone = R_start @ spin @ tilt

                T_cone = np.eye(4)
                T_cone[:3, :3] = R_cone
                T_cone[:3, 3] = base_position
                trajectory.append(T_cone)

    return trajectory


@draccus.wrap()
def main(cfg: RecordConfig):
    """Record alignment dataset using configuration parameters."""
    robot = XarmSim()
    features = Features(
        {"images": Sequence(Image()), "action": Sequence(Value("float32"))}
    )

    for episode in range(cfg.episodes):
        pose_start, pose_alignment_target = robot.reset()

        robot.servo_to_pose(pose_alignment_target, lin_tol=0.015, ang_tol=0.015)

        robot.servo_to_pose(
            pose_alignment_target,
            lin_tol=cfg.lin_tol_alignment,
            ang_tol=cfg.ang_tol_alignment,
        )

        trajectory = generate_spiral_trajectory(pose_start, cfg.trajectory)

        frames = []
        for pose in trajectory:
            robot.servo_to_pose(
                pose, lin_tol=cfg.lin_tol_trajectory, ang_tol=cfg.ang_tol_trajectory
            )
            current_pose = robot.pose()

            action_pose = np.linalg.inv(current_pose) @ pose_alignment_target
            action_sixd = se3_sixd(action_pose)

            observation = robot.get_observation()
            frame = {
                "images": [observation["camera.rgb"].copy()],
                "action": action_sixd,
            }
            frames.append(frame)

        print(f"Episode {episode+1} completed with {len(frames)} frames.")

        episode_dataset = Dataset.from_list(frames, features=features)
        if episode == 0:
            combined_dataset = episode_dataset
        else:
            previous_dataset = load_from_disk(cfg.dataset.path)
            previous_dataset = previous_dataset.cast(features)
            combined_dataset = concatenate_datasets([previous_dataset, episode_dataset])
            del previous_dataset

        temp_path = f"{cfg.dataset.path}_temp"
        combined_dataset.save_to_disk(temp_path)
        if os.path.exists(cfg.dataset.path):
            shutil.rmtree(cfg.dataset.path)
        shutil.move(temp_path, cfg.dataset.path)

    robot.disconnect()


if __name__ == "__main__":
    main()
