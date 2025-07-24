import time
import numpy as np
import transforms3d as t3d
from alignit.robots.mujoco import MuJoCoRobot
from datasets import Dataset, Features, Sequence, Value, Image, load_from_disk, concatenate_datasets
from scipy.spatial.transform import Rotation as R
import random
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
    robot.reset()
    robot.reset()

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