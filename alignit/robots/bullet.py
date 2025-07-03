import pybullet as p
import pybullet_data
import time
import numpy as np

class SimpleBulletSim:
    def __init__(self):
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.plane = p.loadURDF("plane.urdf")
        self.robot = self.load_robot()
        self.cube = self.load_cube()
        self.camera_link = self.get_gripper_link()

    def load_robot(self):
        # Use KUKA iiwa as a simple 7DOF arm
        robot_urdf = "kuka_iiwa/model.urdf"
        start_pos = [0, 0, 0]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        robot = p.loadURDF(robot_urdf, start_pos, start_orientation, useFixedBase=True)
        return robot

    def load_cube(self):
        cube_start_pos = [0.6, 0, 0.05]
        cube_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        cube = p.loadURDF("cube_small.urdf", cube_start_pos, cube_start_orientation)
        return cube

    def get_gripper_link(self):
        # For KUKA iiwa, link 6 is the end effector
        return 6

    def step(self):
        p.stepSimulation()
        time.sleep(1./240.)

    def get_joint_info(self):
        num_joints = p.getNumJoints(self.robot)
        joint_infos = []
        for i in range(num_joints):
            info = p.getJointInfo(self.robot, i)
            joint_infos.append(info)
        return joint_infos

    def move_ee_to(self, target_pos, target_orn=None):
        if target_orn is None:
            # Default: keep orientation pointing down
            target_orn = p.getQuaternionFromEuler([0, np.pi, 0])
        joint_poses = p.calculateInverseKinematics(
            self.robot, self.camera_link, target_pos, target_orn)
        for i in range(p.getNumJoints(self.robot)):
            p.setJointMotorControl2(
                bodyIndex=self.robot,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_poses[i],
                force=500)

    def render_camera(self):
        # Camera mounted on gripper
        link_state = p.getLinkState(self.robot, self.camera_link)
        cam_pos = link_state[0]
        cam_orn = p.getMatrixFromQuaternion(link_state[1])
        forward = [cam_orn[0], cam_orn[3], cam_orn[6]]
        up = [cam_orn[2], cam_orn[5], cam_orn[8]]
        target = [cam_pos[0] + forward[0], cam_pos[1] + forward[1], cam_pos[2] + forward[2]]
        view_matrix = p.computeViewMatrix(cam_pos, target, up)
        proj_matrix = p.computeProjectionMatrixFOV(60, 1, 0.01, 2)
        _, _, px, _, _ = p.getCameraImage(320, 320, view_matrix, proj_matrix)
        return px

    def close(self):
        p.disconnect()

if __name__ == "__main__":
    sim = SimpleBulletSim()
    # Move end effector in a small circle above the cube
    for t in range(1000):
        x = 0.6 + 0.05 * np.cos(2 * np.pi * t / 240)
        y = 0.05 * np.sin(2 * np.pi * t / 240)
        z = 0.25
        sim.move_ee_to([x, y, z])
        sim.step()
        if t % 120 == 0:
            sim.render_camera()
    sim.close()
