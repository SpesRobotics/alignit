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
def generate_spiral_trajectory(
    start_pose,
    z_step=0.001,
    radius_step=0.001,
    num_steps=100,
    cone_angle=45,
    visible_sweep=180,
    viewing_angle_offset=0,
    angular_resolution=5,
    include_cone_poses=True
):
    """
    Generate spiral trajectory that maintains the initial object rotation throughout.
    
    Args:
        start_pose: Initial pose (should match pickup object's rotation)
        z_step: Step size in z-axis (depth)
        radius_step: Step size for spiral radius
        num_steps: Number of steps in spiral
        cone_angle: Angle of cone for viewing (if include_cone_poses is True)
        visible_sweep: Angular sweep for viewing (if include_cone_poses is True)
        viewing_angle_offset: Offset for viewing angle (if include_cone_poses is True)
        angular_resolution: Resolution for viewing angles (if include_cone_poses is True)
        include_cone_poses: Whether to include viewing cone poses
    """
    trajectory = []
    R_start = start_pose[:3, :3]  # This maintains the object's initial rotation
    t_start = start_pose[:3, 3]
    cone_angle_rad = np.deg2rad(cone_angle)
    
    # Calculate angular range with offset
    start_angle = -visible_sweep/2 + viewing_angle_offset
    end_angle = visible_sweep/2 + viewing_angle_offset
    
    for i in range(num_steps):
        # Spiral motion - only translate, keep rotation constant
        radius = radius_step * i
        angle = 2 * np.pi * i / 10  # This controls the spiral pattern
        
        # Calculate position in local frame
        local_offset = np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            -z_step * i
        ])
        
        # Transform to world frame using initial rotation
        world_offset = R_start @ local_offset
        base_position = t_start + world_offset
        
        # Create pose with initial rotation and new position
        T = np.eye(4)
        T[:3, :3] = R_start  # Maintain the same rotation throughout
        T[:3, 3] = base_position
        trajectory.append(T)
        
        if include_cone_poses:
            # Generate visible poses with adjustable resolution
            for deg in np.arange(start_angle, end_angle, angular_resolution):
                theta = np.deg2rad(deg)
                
                # Apply cone rotations relative to initial orientation
                tilt = t3d.euler.euler2mat(cone_angle_rad, 0, 0)
                spin = t3d.euler.euler2mat(0, 0, theta)
                R_cone = R_start @ spin @ tilt  # Rotate relative to initial orientation
                
                T_cone = np.eye(4)
                T_cone[:3, :3] = R_cone
                T_cone[:3, 3] = base_position
                trajectory.append(T_cone)
            
    return trajectory

def main():
    robot = MuJoCoRobot()
    features = Features({
        "images": Sequence(Image()),
        "action": Sequence(Value("float32"))
    }) # Setup target poses
    
    pose1 = robot.get_object_pose()
    transl = pose1[:3,3] + np.array([0,0,0.1])
    rot = pose1[:3,:3]

    pose_final_target = t3d.affines.compose(
       transl, rot, [1, 1, 1]
    )

    pose_alignment_target = pose_final_target @ t3d.affines.compose(
        [0., 0, -0.15], t3d.euler.euler2mat(0, 0, 0), [1, 1, 1]
    )
    pose_record_start = pose_alignment_target @ t3d.affines.compose(
        [0, 0, 0.1], t3d.euler.euler2mat(0, 0, 0), [1, 1, 1]
    )
    print("Current robot pose: ")
    # Move to initial position
    for episode in range(30):
        
        random_pos = [
        0.25 + np.random.uniform(-0.01, 0.01),  
        0.0 + np.random.uniform(-0.01, 0.01),    
        0.08   
        ]
        roll = np.pi  
        pitch = np.random.uniform(-np.pi/4, np.pi/4)   
        yaw = np.random.uniform(-np.pi, np.pi)        
    
        pose = t3d.affines.compose(
            random_pos,
            t3d.euler.euler2mat(roll, pitch, yaw),  # Convert RPY to rotation matrix
            [1, 1, 1]  # Scale (keep as 1)
        )
        
        robot.set_object_pose("pickup_object", pose)
        
        pose1 = robot.get_object_pose()
        object_z_axis = pose1[:3, 2]  # The Z-axis of the object's coordinate frame

        # Offset along the object's Z-axis by 0.15 meters
        offset_distance = -0.15
        offset_vector = object_z_axis * offset_distance

        transl = pose1[:3, 3] + offset_vector  # Add offset in object's Z direction
        rot = pose1[:3, :3]

        pose_final_target = t3d.affines.compose(
            transl, rot, [1, 1, 1]
        )

        # Randomize starting position slightly
        robot.servo_to_pose(pose_final_target, lin_tol=0.015,ang_tol=0.015)
        print("sleeping")
        robot.groff()
        # Generate trajectory with conical rotations
        trajectory = generate_spiral_trajectory(
            pose_final_target,
            z_step=0.001,        # Increased from 0.0005
            radius_step=0.001,   # Increased from 0.0005
            num_steps=100,       # Reduced from 200 since steps are larger
            cone_angle=30,
            visible_sweep=60,
            viewing_angle_offset=-120,
            angular_resolution=10,
            include_cone_poses=False
        )

        frames = []
        i = 0
        for pose in trajectory:
            # Execute motion
            robot.servo_to_pose(pose, lin_tol=0.05,ang_tol=0.05)
            current_pose = robot.pose()

            # Calculate action (relative to alignment target)
            action_pose = np.linalg.inv(current_pose) @ pose_alignment_target
            action_sixd = se3_sixd(action_pose)

            # Capture observation
            observation = robot.get_observation()
            frame = {
                "images": [observation["camera.rgb"]],
                "action": action_sixd
            }
            frames.append(frame)
            print(f"moved to {pose}")
            i = i +1
            print(f"Completed : {i} %")
        
        print(f"Episode {episode+1} completed with {len(frames)} frames.")
        
        # episode_dataset = Dataset.from_list(frames, features=features)
        # if episode == 0:
        #     combined_dataset = episode_dataset
        # else:
        #     previous_dataset = load_from_disk("data/duck")
        #     previous_dataset = previous_dataset.cast(features)
        #     combined_dataset = concatenate_datasets([previous_dataset, episode_dataset])
        #     del previous_dataset

        # temp_path = "data/duck_temp"
        # combined_dataset.save_to_disk(temp_path)
        # if os.path.exists("data/duck"):
        #     shutil.rmtree("data/duck")
        # shutil.move(temp_path, "data/duck")
        #robot.reset()

    robot.close()



if __name__ == "__main__":
    main()