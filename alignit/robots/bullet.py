import pybullet as p
import pybullet_data
import time
import numpy as np
import transforms3d as t3d


class Bullet:
    def __init__(self):
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.plane = p.loadURDF("plane.urdf")
        self.robot = self._load_robot()
        self.cube = self._load_object()
        self.camera_link = self._get_gripper_link()

    def _load_robot(self):
        robot_urdf = "kuka_iiwa/model.urdf"
        start_pos = [0, 0, 0]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        robot = p.loadURDF(robot_urdf, start_pos, start_orientation, useFixedBase=True)
        return robot

    def _load_object(self):
        mesh_path = "duck_vhacd.obj"
        mesh_start_pos = [0.6, 0, 0.05]
        mesh_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=mesh_path,
            meshScale=[0.1, 0.1, 0.1],
            rgbaColor=[1, 1, 0, 1],
        )
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=mesh_path,
            meshScale=[0.1, 0.1, 0.1],
        )
        mesh = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=mesh_start_pos,
            baseOrientation=mesh_start_orientation,
        )
        return mesh

    def _get_gripper_link(self):
        return 6

    def send_action(self, action):
        self._servo(action)
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    def _servo(self, pose):
        target_pos = pose[:3, 3]
        target_rot = pose[:3, :3]
        q = t3d.quaternions.mat2quat(target_rot)
        target_orn = [q[1], q[2], q[3], q[0]]

        joint_poses = p.calculateInverseKinematics(
            self.robot, self.camera_link, target_pos, target_orn
        )
        for i in range(p.getNumJoints(self.robot)):
            p.setJointMotorControl2(
                bodyIndex=self.robot,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_poses[i],
                force=500,
            )

    def pose(self):
        link_state = p.getLinkState(self.robot, self.camera_link)
        pos = link_state[4]
        q = link_state[5]
        return t3d.affines.compose(
            pos,
            t3d.quaternions.quat2mat([q[3], q[0], q[1], q[2]]),
            [1, 1, 1],
        )

    def get_observation(self):
        # Camera mounted on gripper
        link_state = p.getLinkState(self.robot, self.camera_link)
        cam_pos = link_state[0]
        cam_pos = [cam_pos[0] + 0.1, cam_pos[1], cam_pos[2]]
        cam_orn = p.getMatrixFromQuaternion(link_state[1])
        up = [cam_orn[0], cam_orn[3], cam_orn[6]]
        forward = [cam_orn[2], cam_orn[5], cam_orn[8]]
        target = [
            cam_pos[0] + forward[0],
            cam_pos[1] + forward[1],
            cam_pos[2] + forward[2],
        ]
        view_matrix = p.computeViewMatrix(cam_pos, target, up)
        proj_matrix = p.computeProjectionMatrixFOV(60, 1, 0.01, 2)
        _, _, px, _, _ = p.getCameraImage(640, 480, view_matrix, proj_matrix)

        # make sure px 3 channels, rgb, not rgba
        if len(px.shape) == 2:
            px = np.stack([px, px, px], axis=-1)
        elif px.shape[2] == 4:
            px = px[:, :, :3]

        return {
            "camera.rgb": px,
        }

    def close(self):
        p.disconnect()


if __name__ == "__main__":
    sim = Bullet()
    for t in range(100000000):
        pose = np.array([[0, 1, 0, 0.5], [1, 0, 0, 0], [0, 0, -1, 0.3], [0, 0, 0, 1]])
        sim.send_action(pose)
        observation = sim.get_observation()
    sim.close()
