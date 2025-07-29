import time
from pathlib import Path
import numpy as np
import transforms3d as t3d
from teleop.utils.jacobi_robot import JacobiRobot
from alignit.robots.robot import Robot
import cv2
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI
from transforms3d import euler
from transforms3d import affines 
from transforms3d.euler import euler2mat 
import math

class XarmTable(Robot):
    def __init__(self, ip_address="192.168.1.184"):
        """
        Initialize the real xArm Lite6 robot with RealSense camera.
        
        Args:
            ip_address: IP address of the xArm controller
        """
        self.ROBOT_IP = ip_address
        print("Initializing UFactory xArm...")
        print(f"with the ip {self.ROBOT_IP}")
        self.arm = XArmAPI(self.ROBOT_IP)
        self.arm.connect()
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0) # Position control mode
        self.arm.set_state(state=0) # Ready state
        print("xArm Initialized.")
        print("Initializing Intel RealSense Camera...")
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config)
        
        self.camera_intrinsics = None
        self._setup_camera()
        
        self.base_position = np.array([0, 0, 0])  # Adjust based on your setup
        self.base_rotation = np.eye(3)
        
        profile = self.pipeline.get_active_profile()
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        color_intrinsics = color_profile.get_intrinsics()
        
        self.camera_intrinsics = {
            'width': color_intrinsics.width,
            'height': color_intrinsics.height,
            'fx': color_intrinsics.fx,
            'fy': color_intrinsics.fy,
            'ppx': color_intrinsics.ppx,
            'ppy': color_intrinsics.ppy,
            'model': color_intrinsics.model,
            'coeffs': color_intrinsics.coeffs
        }        
        
    def gripper_close(self):
        self.arm.close_lite6_gripper()
    def reset(self, manual_height=0.05, world_z_offset=0.02):
        """
        Reset routine:
        1. Allows manual movement of the arm
        2. Waits for user input (Enter key)
        3. Applies gripper-frame Z offset
        4. Applies world-frame Z offset
        5. Returns to normal operation
        
        Args:
            manual_height: Height above surface to maintain during manual movement (meters)
            world_z_offset: Additional Z offset in world frame after manual positioning (meters)
        """
        
        self.arm.set_mode(1)  # Servo mode
        self.arm.set_state(0)  # Start
        input("Press Enter after positioning the arm...")
        self.arm.set_mode(0)  # Position control mode
        self.arm.set_state(0)  # Start
        current_pose = self.pose()
        gripper_z_offset = np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, manual_height],
                                    [0, 0, 0, 1]])
        offset_pose = current_pose @ gripper_z_offset
        self.send_action(offset_pose)
        
        world_z_offset_mat = np.array([[1, 0, 0, 0],
                                        [0, 1, 0, 0],
                                    [0, 0, 1, world_z_offset],
                                    [0, 0, 0, 1]])
        final_pose = offset_pose @ world_z_offset_mat
        self.send_action(final_pose)
        return final_pose
  
    def gripper_open(self):
        self.arm.open_lite6_gripper()        
    def send_action(self, pose: np.ndarray, speed: int = None):
        if speed is None:
            speed = 80
        print(f"Moving robot to pose: {pose}")
        x = pose[0, 3] * 1000
        y = pose[1, 3] * 1000
        z = pose[2, 3] * 1000
        roll, pitch, yaw = euler.mat2euler(pose[:3, :3])
        roll = math.degrees(roll)
        pitch = math.degrees(pitch)
        yaw = math.degrees(yaw)
        error = self.arm.set_position(
            x=x,
            y=y,
            z=z,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            speed=speed,
            wait=True,
        )
        time.sleep(1)
        
    def pose(self):
        ok, pose = self.arm.get_position()
        if ok != 0:
            return None

        translation = np.array(pose[:3]) / 1000
        eulers = np.array(pose[3:]) * math.pi / 180
        rotation = euler.euler2mat(
            eulers[0], eulers[1], eulers[2], 'sxyz')
        pose = affines.compose(translation, rotation, np.ones(3))
        print("Retrieved robot pose.")
        return pose
        
    def get_observation(self):
        obs = {
            'joint_positions': self.arm.get_servo_angle()[1],
            'eef_pose': self.pose(),
            'gripper_position': self.current_gripper_pos
        }
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        if color_frame:
            color_image = np.asanyarray(color_frame.get_data())
            obs['camera.color'] = color_image
            
        if depth_frame:
            depth_image = np.asanyarray(depth_frame.get_data())
            obs['camera.depth'] = depth_image
            
        return obs
        
    def stop(self):
        self.arm.stop()
        
    def close(self):
        self.arm.disconnect()
        self.pipeline.stop()
        
    def update_sim(self):
        pass
        
    def groff(self):
        pass
        
    def get_object_pose(self, object_name=None):
        return None


if __name__ == "__main__":
    robot = XarmTable()
    
    current_pose = robot.pose()
    print("Current pose:", current_pose)
    
    target_pose = current_pose.copy()
    target_pose[2, 3] += 0.1  # 10cm up
    robot.send_action(target_pose)
    
    robot.gripper_open()
    time.sleep(1)
    robot.gripper_close()
    
    robot.close()