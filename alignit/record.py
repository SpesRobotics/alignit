import os
import shutil
import transforms3d as t3d
import numpy as np
from scipy.spatial.transform import Rotation as R
from alignit.robots.xarm import Xarm
from alignit.utils.zhou import se3_sixd
import argparse  # Added for command line arguments
from datasets import (
    Dataset,
    Features,
    Sequence,
    Value,
    Image,
    load_from_disk,
    concatenate_datasets,
)

def generate_lift_trajectory_with_random_rpy(
    start_pose,
    lift_distance=0.1,          
    num_steps=100,              
    max_rotation_angle=15.0,   
    initial_lift=0.005,        
):
    
    trajectory = []
    R_start = start_pose[:3, :3]
    t_start = start_pose[:3, 3]
    
    object_z_axis = R_start[:, 2]
    
    t_current = t_start + object_z_axis * initial_lift
    
    for i in range(num_steps):
        lift_progress = i / (num_steps - 1)  # 0 to 1
        current_lift = t_start + object_z_axis * -(initial_lift + lift_distance * lift_progress)
        
        r_rand = np.random.uniform(-max_rotation_angle, max_rotation_angle)
        p_rand = np.random.uniform(-max_rotation_angle, max_rotation_angle)
        y_rand = np.random.uniform(-max_rotation_angle, max_rotation_angle)
        
        random_rotation = R.from_euler('xyz', 
                                     [np.radians(r_rand), 
                                      np.radians(p_rand), 
                                      np.radians(y_rand)]).as_matrix()
        
        R_current = R_start @ random_rotation
        
        T = np.eye(4)
        T[:3, :3] = R_current
        T[:3, 3] = t_start
        
        trajectory.append(T)
    
    return trajectory

def generate_spiral_trajectory(
    start_pose,
    z_step=0.1,
    radius_step=0.001,
    num_steps=50,
    cone_angle=45,
    visible_sweep=180,
    viewing_angle_offset=0,
    angular_resolution=5,
    include_cone_poses=False,
    lift_height_before_spiral=0.0005,
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

def load_and_merge_datasets(dataset_paths, output_path, features):
    """Load and merge multiple datasets from disk"""
    if not dataset_paths:
        return None
        
    datasets = []
    for path in dataset_paths:
        if os.path.exists(path):
            try:
                ds = load_from_disk(path)
                ds = ds.cast(features)  # Ensure consistent features
                datasets.append(ds)
                print(f"Loaded dataset from {path}")
            except Exception as e:
                print(f"Error loading dataset from {path}: {e}")
        else:
            print(f"Dataset path not found: {path}")
    
    if not datasets:
        return None
    
    merged_dataset = concatenate_datasets(datasets)
    
    # Save the merged dataset
    temp_path = output_path + "_temp"
    merged_dataset.save_to_disk(temp_path)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    shutil.move(temp_path, output_path)
    
    return merged_dataset

def main(args):
    robot = Xarm()
    features = Features(
        {"images": Sequence(Image()), "action": Sequence(Value("float32")), "depth": Sequence(Image())}
    )

    if args.merge_datasets:
        print(f"Merging datasets: {args.merge_datasets}")
        merged_dataset = load_and_merge_datasets(
            args.merge_datasets, 
            args.output_path, 
            features
        )
        if merged_dataset is not None:
            print(f"Merged dataset created with {len(merged_dataset)} samples at {args.output_path}")
        else:
            print("No valid datasets found to merge")
    max_rot = 15 

    for episode in range(args.num_episodes):
        pose_start, pose_alignment_target = robot.reset()
        max_rot = max_rot + 2  
        trajectory = generate_lift_trajectory_with_random_rpy(
            pose_start,
            lift_distance=0.1,
            num_steps=100,
            max_rotation_angle=max_rot,
            initial_lift=0.005,
        )
        print(f"Generated trajectory with {len(trajectory)} poses for episode {episode+1}")
        frames = []
        for pose in trajectory:
            robot.servo_to_pose(pose, lin_tol=5e-3, ang_tol=0.09)
            current_pose = robot.pose()
            
            action_pose = np.linalg.inv(current_pose) @ pose_alignment_target
            action_sixd = se3_sixd(action_pose)

            observation = robot.get_observation()
            print(observation.keys())
            frame = {"images": [observation["rgb"].copy()],"depth":  [observation["depth"].copy()],"action": action_sixd}
            frames.append(frame)

        print(f"Episode {episode+1} completed with {len(frames)} frames.")

        episode_dataset = Dataset.from_list(frames, features=features)
        if episode == 0 and not os.path.exists(args.output_path):
            combined_dataset = episode_dataset
        else:
            previous_dataset = load_from_disk(args.output_path)
            previous_dataset = previous_dataset.cast(features)
            combined_dataset = concatenate_datasets([previous_dataset, episode_dataset])
            del previous_dataset

        temp_path = args.output_path + "_temp"
        combined_dataset.save_to_disk(temp_path)
        if os.path.exists(args.output_path):
            shutil.rmtree(args.output_path)
        shutil.move(temp_path, args.output_path)

    robot.disconnect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Robot data collection and dataset merging')
    parser.add_argument('--output-path', type=str, default="data/duck",
                       help='Path to save the output dataset')
    parser.add_argument('--num-episodes', type=int, default=15,
                       help='Number of episodes to run')
    parser.add_argument('--merge-datasets', type=str, nargs='+', default=None,
                       help='List of dataset paths to merge before collection')
    
    args = parser.parse_args()
    main(args)
