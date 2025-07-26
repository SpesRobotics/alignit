import time
from pathlib import Path

import mujoco as mj
import mujoco.viewer
import numpy as np
import transforms3d as t3d
from teleop.utils.jacobi_robot import JacobiRobot
from alignit.robots.robot import Robot


class XarmSim(Robot):
    def __init__(self):
        self.gripper_open_pos = 0.008
        self.gripper_close_pos = -0.008
        self.current_gripper_pos = 0.0
        self.this_dir = Path(__file__).parent
        mjcf_path = self.this_dir / "ufactory_lite6" / "scene.xml"
        urdf_path_jacobi = self.this_dir / "lite6.urdf"
        end_effector_frame_name_jacobi = "link6"

        try:
            self.model = mj.MjModel.from_xml_path(str(mjcf_path))
            self.data = mj.MjData(self.model)

            self.model.opt.timestep = 0.005
            self.model.opt.iterations = 50
            self.model.opt.tolerance = 1e-8
            self.model.opt.solver = mj.mjtSolver.mjSOL_NEWTON

            for i in range(self.model.nv):
                self.model.dof_damping[i] = 0.5

        except Exception as e:
            print(f"Failed to load MuJoCo model from {mjcf_path}: {e}")
            raise RuntimeError(f"Failed to load MuJoCo model from {mjcf_path}: {e}")

        self.renderer = mujoco.Renderer(self.model, 240, 320)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.robot_jacobi = JacobiRobot(
            str(urdf_path_jacobi), ee_link=end_effector_frame_name_jacobi
        )

        self.mujoco_actuator_ids = []
        self.mujoco_actuator_to_jacobi_joint_idx = []
        self.mujoco_qpos_indices_for_actuators = []
        self.gripper_ctrl_id = None

        jacobi_joint_name_to_idx_map = {}
        jacobi_joint_names = self.robot_jacobi.get_joint_names()
        for joint_idx, joint_name in enumerate(jacobi_joint_names):
            jacobi_joint_name_to_idx_map[joint_name] = joint_idx

        for i in range(self.model.nu):
            actuator_name = self.model.actuator(i).name
            mujoco_joint_id = self.model.actuator_trnid[i, 0]
            mujoco_joint_name = self.model.joint(mujoco_joint_id).name

            if actuator_name == "gripper":
                self.gripper_ctrl_id = i
                print(f"Found gripper actuator at index {i}")
                continue  # Skip adding to Jacobi mapping

            if mujoco_joint_name in jacobi_joint_name_to_idx_map:
                self.mujoco_actuator_ids.append(self.model.actuator(i).id)
                self.mujoco_actuator_to_jacobi_joint_idx.append(
                    jacobi_joint_name_to_idx_map[mujoco_joint_name]
                )
                self.mujoco_qpos_indices_for_actuators.append(
                    self.model.joint(mujoco_joint_id).qposadr[0]
                )
            else:
                print(
                    f"MuJoCo joint '{mujoco_joint_name}' (actuator '{actuator_name}' at index {i}) has no JacobiRobot counterpart"
                )

        mj.mj_resetData(self.model, self.data)
        mj.mj_forward(self.model, self.data)
        self.viewer.sync()
        time.sleep(1.0)

    def stop_object_movement(self, object_name="pickup_object"):
        while self.data.body("pickup_object").xpos[2] > 0.08:
            mj.mj_step(self.model, self.data)
            self.viewer.sync()
        self.model.opt.gravity[:] = [0, 0, 0]
        obj_id = self.model.body(object_name).id
        joint_id = self.model.body_jntadr[obj_id]
        if joint_id != -1:
            qvel_adr = self.model.jnt_dofadr[joint_id]
            self.data.qvel[qvel_adr : qvel_adr + 6] = 0
            mj.mj_forward(self.model, self.data)

    def groff(self):
        self.model.opt.gravity[:] = [0, 0, 0]
        mj.mj_forward(self.model, self.data)

    def set_object_pose(self, object_name, pose_matrix):
        body_id = self.model.body(object_name).id
        self.data.xpos[body_id] = pose_matrix[:3, 3]
        quat = t3d.quaternions.mat2quat(pose_matrix[:3, :3])
        self.data.xquat[body_id] = quat

        joint_id = self.model.body_jntadr[body_id]
        if joint_id >= 0:  # If body has a joint
            jnt_type = self.model.jnt_type[joint_id]
            qpos_adr = self.model.jnt_qposadr[joint_id]

            if jnt_type == 0:  # mjJNT_FREE
                self.data.qpos[qpos_adr : qpos_adr + 3] = pose_matrix[:3, 3]
                self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat
                qvel_adr = self.model.jnt_dofadr[joint_id]
                self.data.qvel[qvel_adr : qvel_adr + 6] = 0
        mj.mj_forward(self.model, self.data)

    def reset_auto(self):
        self.gripper_close()
        self.model.opt.gravity[:] = [0, 0, 0]
        obj_pose = self.get_object_pose("pickup_object")
        initial_pose = self.pose()
        initial_rot = initial_pose[:3, :3]
        obj_pos = obj_pose[:3, 3]
        obj_rot = obj_pose[:3, :3]
        local_offset1 = np.array([0, 0, -0.15])
        world_offset1 = obj_rot @ local_offset1
        approach_pos = obj_pos + world_offset1
        approach_pose = t3d.affines.compose(approach_pos, obj_rot, [1, 1, 1])
        self.servo_to_pose(approach_pose, lin_tol=0.003, ang_tol=0.1)

        print("Approaching")
        local_offset1 = np.array([-0.030, 0, 0.015])  # In object frame
        world_offset1 = obj_rot @ local_offset1  # Transform to world frame
        current_pos = approach_pose[:3, 3] + world_offset1
        off_rot = t3d.euler.euler2mat(0, 0, np.pi / 2)
        new_rot = obj_rot @ off_rot
        rotated_pose = t3d.affines.compose(current_pos, new_rot, [1, 1, 1])
        self.servo_to_pose(rotated_pose, lin_tol=0.003, ang_tol=0.1)

        curr_pose = self.pose()
        local_offset2 = np.array([0.013, 0, 0])  # In gripper frame
        world_offset2 = curr_pose[:3, :3] @ local_offset2
        current_pos = curr_pose[:3, 3] + world_offset2
        rotated_pose = t3d.affines.compose(current_pos, curr_pose[:3, :3], [1, 1, 1])
        self.servo_to_pose(rotated_pose, lin_tol=0.0015, ang_tol=0.1)

        curr_pose = self.pose()
        local_offset3 = np.array([0, 0, 0.07])  # In gripper frame
        world_offset3 = curr_pose[:3, :3] @ local_offset3
        current_pos = curr_pose[:3, 3] + world_offset3
        rotated_pose = t3d.affines.compose(current_pos, curr_pose[:3, :3], [1, 1, 1])
        self.servo_to_pose(rotated_pose, lin_tol=0.0015, ang_tol=0.1)

        self.gripper_open()

        current_pos = curr_pose[:3, 3] + np.array([0, 0, 0.1])  # World frame
        rotated_pose = t3d.affines.compose(current_pos, curr_pose[:3, :3], [1, 1, 1])
        self.servo_to_pose(rotated_pose, lin_tol=0.003, ang_tol=0.1)

        angle_x = np.random.uniform(-0.3, 0.3)
        angle_y = np.random.uniform(-0.3, 0.3)
        angle_z = np.random.uniform(-np.pi, np.pi)
        random_rot = t3d.euler.euler2mat(
            np.pi / 8 + angle_x, np.pi / 8 + angle_y, np.pi / 8 + angle_z
        )
        const_rot = np.array(
            [
                [9.99996083e-01, -5.31966273e-07, -2.79879023e-03],
                [-5.40655149e-07, -1.00000000e00, -3.10375475e-06],
                [-2.79879023e-03, 3.10525578e-06, -9.99996083e-01],
            ]
        )
        curr_pose = self.pose()
        new_rot = const_rot @ random_rot
        trans = np.array([0.25, np.random.uniform(-0.01, 0.01), 0.2])

        print(f"Moved to {trans}")
        rotated_pose = t3d.affines.compose(trans, new_rot, [1, 1, 1])
        self.servo_to_pose(rotated_pose, lin_tol=0.003, ang_tol=0.1)

        # Final grasp and lift
        self.gripper_close()
        self.model.opt.gravity[:] = [0, 0, -9.81]
        mj.mj_forward(self.model, self.data)
        self.stop_object_movement()

        curr_obj_pose = self.get_object_pose()
        current_pos = curr_obj_pose[:3, 3] + np.array([0, 0, 0.2])  # World Z
        rotated_pose = t3d.affines.compose(current_pos, initial_rot, [1, 1, 1])
        self.servo_to_pose(rotated_pose, lin_tol=0.003, ang_tol=0.1)

    def update_sim(self):
        mj.mj_forward(self.model, self.data)
        self.viewer.sync()

    def gripper_close(self):
        self._set_gripper_position(self.gripper_close_pos)

    def gripper_open(self):
        self._set_gripper_position(self.gripper_open_pos)

    def _set_gripper_position(self, pos, tolerance=1e-3, max_sim_steps=2000):
        target_pos = np.clip(pos, self.gripper_close_pos, self.gripper_open_pos)
        self.data.ctrl[self.gripper_ctrl_id] = target_pos
        steps_taken = 0
        while (
            abs(self.data.qpos[self.gripper_ctrl_id] - target_pos) > tolerance
            and steps_taken < max_sim_steps
        ):
            mj.mj_step(self.model, self.data)
            self.viewer.sync()
            steps_taken += 1
            self.current_gripper_pos = target_pos

    def send_action(self, target_pose_matrix):
        try:
            base_pos = self.data.xpos[self.model.body("link_base").id]
            base_rot = self.data.xmat[self.model.body("link_base").id].reshape(3, 3)
            world_to_base = t3d.affines.compose(base_pos, base_rot, [1, 1, 1])
        except KeyError:
            world_to_base = np.eye(4)
        base_target_pose = np.linalg.inv(world_to_base) @ target_pose_matrix
        servo_dt = self.model.opt.timestep
        self.robot_jacobi.servo_to_pose(base_target_pose, dt=servo_dt)

        full_jacobi_q = self.robot_jacobi.q
        target_joint_qpos_for_mujoco = np.array(
            [full_jacobi_q[idx] for idx in self.mujoco_actuator_to_jacobi_joint_idx]
        )
        self.data.ctrl[self.mujoco_actuator_ids] = target_joint_qpos_for_mujoco

        mj.mj_step(self.model, self.data)

        self.viewer.sync()
        return True

    def get_object_pose(self, object_name="pickup_object"):
        try:
            obj_id = self.model.body(object_name).id

            pos = self.data.body(obj_id).xpos
            rot = self.data.body(obj_id).xmat.reshape(3, 3)

            return t3d.affines.compose(pos, rot, [1, 1, 1])
        except Exception as e:
            return None

    def pose(self):
        eef_id = self.model.body("link6").id
        pos = self.data.xpos[eef_id]
        rot_mat = self.data.xmat[eef_id].reshape(3, 3)
        return t3d.affines.compose(pos, rot_mat, [1, 1, 1])

    def get_observation(self):
        obs = {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "eef_pose": self.pose(),
        }

        for i in range(self.model.ncam):
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, i
            )
            self.renderer.update_scene(self.data, camera=name)
            image = self.renderer.render()
            obs["camera." + name] = image[:, :, ::-1]

        return obs

    def close(self):
        print("Closing MuJoCo resources.")
        if self.viewer:
            self.viewer.close()
            self.viewer = None


if __name__ == "__main__":
    sim = XarmSim()
    pose = sim.pose()

    pose[0, 3] += 0.1
    sim.servo_to_pose(pose)

    sim.close()
