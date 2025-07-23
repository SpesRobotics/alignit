import time
import numpy as np
import transforms3d as t3d
from alignit.robots.bullet import Bullet
from alignit.robots.mujoco import MuJoCoRobot
from alignit.utils.tfs import are_tfs_close
from datasets import Dataset, Features, Sequence, Value, Image, load_from_disk, concatenate_datasets
from alignit.utils.zhou import se3_sixd
from alignit.robots.robot import Robot
import shutil
import os
from scipy.spatial.transform import Rotation as R
import xml.etree.ElementTree as ET
def generate_spiral_trajectory(
    start_pose,
    z_step=0.001,
    radius_step=0.001,
    num_steps=100,
    cone_angle=45,
    visible_sweep=180,
    viewing_angle_offset=0,
    angular_resolution=5,
    include_cone_poses=True  # New: toggle for cone poses
):
    """
    Generate optimized spiral trajectory with adjustable rotation speed.
    
    Args:
        viewing_angle_offset: Rotates the visible window (0°=forward, 90°=left)
        angular_resolution: Degrees between cone poses (larger=faster rotation)
        include_cone_poses: If False, generates only the spiral without cone poses
    """
    trajectory = []
    R_start = start_pose[:3, :3]
    t_start = start_pose[:3, 3]
    cone_angle_rad = np.deg2rad(cone_angle)
    
    # Calculate angular range with offset
    start_angle = -visible_sweep/2 + viewing_angle_offset
    end_angle = visible_sweep/2 + viewing_angle_offset
    
    for i in range(num_steps):
        # Spiral motion
        radius = radius_step * i
        angle = 2 * np.pi * i / 10
        
        local_offset = np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            -z_step * i
        ])
        world_offset = R_start @ local_offset
        base_position = t_start + world_offset
        
        # Add initial pose with same orientation as start pose
        T_initial = np.eye(4)
        T_initial[:3, :3] = R_start
        T_initial[:3, 3] = base_position
        trajectory.append(T_initial)
        
        if include_cone_poses:
            # Generate visible poses with adjustable resolution
            for deg in np.arange(start_angle, end_angle, angular_resolution):
                theta = np.deg2rad(deg)
                
                tilt = t3d.euler.euler2mat(cone_angle_rad, 0, 0)
                spin = t3d.euler.euler2mat(0, 0, theta)
                R_cone = R_start @ spin @ tilt
                
                T = np.eye(4)
                T[:3, :3] = R_cone
                T[:3, 3] = base_position
                trajectory.append(T)
            
    return trajectory

def main():
    robot = MuJoCoRobot()
    features = Features({
        "images": Sequence(Image()),
        "action": Sequence(Value("float32"))
    })
    robot.gripper_close()
    time.sleep(1)
    obj_pose = robot.get_object_pose("pickup_object")
    initial_pose=robot.pose()
    initial_rot = initial_pose[:3,:3]

    obj_pos = obj_pose[:3, 3]
    obj_rot = obj_pose[:3,:3]
    approach_pos = obj_pos + np.array([0, 0, 0.1])
    approach_rot =  initial_rot # Match object orientation
    approach_pose = t3d.affines.compose(approach_pos, approach_rot, [1, 1, 1])

    pose_alignment_target = approach_pose

    pose_record_start = pose_alignment_target @ t3d.affines.compose(
        [0, 0, -0.08], t3d.euler.euler2mat(0, 0, 0), [1, 1, 1]
    )
    print(obj_pose)
    # Move to initial position
    robot.servo_to_pose(approach_pose,lin_tol=0.005)
    time.sleep(1)
    off_rot = t3d.euler.euler2mat(0, 0, np.pi/2)
    current_pos= approach_pose[:3,3] + np.array([-0.030, 0, -0.01])
    new_rot = obj_rot @ off_rot
    rotated_pose = t3d.affines.compose(current_pos, new_rot, [1, 1, 1])
    robot.servo_to_pose(rotated_pose,lin_tol=0.008)
    curr = robot.pose()
    time.sleep(1)
    time.sleep(1)
    curr_rot = curr[:3,:3]
    current_pos= curr[:3,3] + np.array([0.008, 0, 0]) # -0.035
    new_rot = curr_rot @ off_rot
    rotated_pose = t3d.affines.compose(current_pos, curr_rot, [1, 1, 1])
    robot.servo_to_pose(rotated_pose,lin_tol=0.005)
    print("done")
    curr = robot.pose()
    time.sleep(2)
    time.sleep(2)
    curr_rot = curr[:3,:3]
    current_pos= curr[:3,3] + np.array([0, 0, -0.02]) # -0.035
    new_rot = curr_rot @ off_rot
    rotated_pose = t3d.affines.compose(current_pos, curr_rot, [1, 1, 1])
    robot.gripper_open()
    time.sleep(1)
    robot.servo_to_pose(rotated_pose,lin_tol=0.001) 
    curr_rot = curr[:3,:3]
    current_pos= curr[:3,3] + np.array([0, 0, 0.1]) # -0.035
    new_rot = curr_rot @ off_rot
    rotated_pose = t3d.affines.compose(current_pos, curr_rot, [1, 1, 1])
    robot.servo_to_pose(rotated_pose,lin_tol=0.005)
    print("moved")
    time.sleep(2)
    print("generating random rpy")
    rotation =  t3d.euler.euler2mat(np.pi/1.4, np.pi/0.9, np.pi/2.3)
    print(rotation)
    curr = robot.pose()
    curr_rot = curr[:3,:3]
    current_pos= curr[:3,3] # -0.035
    new_rot = curr_rot @ rotation
    rotated_pose = t3d.affines.compose(current_pos, new_rot, [1, 1, 1])
    print(f"moving to {rotated_pose}")
    robot.servo_to_pose(rotated_pose,lin_tol=0.005)
    robot.gripper_close()
    time.sleep(1)
    print(f"moved to {rotated_pose}")

    time.sleep(100)
   

    # for episode in range(30):
    #     # Randomize starting position slightly
    #     pose_episode_start = pose_record_start.copy()
    #     pose_episode_start[:3, 3] += np.random.uniform(
    #         low=[-0.03, -0.03, -0.1],
    #         high=[0.03, 0.03, 0.01]
    #     )
    #     robot.servo_to_pose(pose_episode_start)

    #     # Generate trajectory with conical rotations
    #     trajectory = generate_spiral_trajectory(
    #         pose_episode_start,
    #         z_step=0.001,
    #         radius_step=0.001,
    #         num_steps=100,
    #         cone_angle=30,
    #         visible_sweep=60,
    #         viewing_angle_offset=-120,
    #         angular_resolution=10,
    #         include_cone_poses=False
    #     )

    #     frames = []
    #     for pose in trajectory:
    #         # Execute motion
    #         robot.servo_to_pose(pose, lin_tol=0.01, ang_tol=0.01)
    #         current_pose = robot.pose()

    #         # Calculate action (relative to alignment target)
    #         action_pose = np.linalg.inv(current_pose) @ pose_alignment_target
    #         action_sixd = se3_sixd(action_pose)

    #         # Capture observation
    #         observation = robot.get_observation()
    #         frame = {
    #             "images": [observation["camera.rgb"]],
    #             "action": action_sixd
    #         }
    #         frames.append(frame)
        
    #     print(f"Episode {episode+1} completed with {len(frames)} frames.")
        
    #     # Save dataset
    #     episode_dataset = Dataset.from_list(frames, features=features)
    #     if episode == 0:
    #         combined_dataset = episode_dataset
    #     else:
    #         previous_dataset = load_from_disk("data/duck")
    #         previous_dataset = previous_dataset.cast(features)
    #         combined_dataset = concatenate_datasets([previous_dataset, episode_dataset])
    #         del previous_dataset

    #     # Atomic write operation
    #     temp_path = "data/duck_temp"
    #     combined_dataset.save_to_disk(temp_path)
    #     if os.path.exists("data/duck"):
    #         shutil.rmtree("data/duck")
    #     shutil.move(temp_path, "data/duck")


if __name__ == "__main__":
    main()