import mujoco as mj
import mujoco.viewer
import numpy as np
import transforms3d as t3d
import time
import pinocchio as pin
from pathlib import Path
import os
os.environ['MUJOCO_GL'] = 'osmesa'  # Force software rendering
import mujoco as mj

class MuJoCoRobot:
    def __init__(self, mjcf_path, urdf_path_pinocchio, end_effector_frame_name_pinocchio):
        # MuJoCo setup
        try:
            self.model = mj.MjModel.from_xml_path(str(mjcf_path))
            self.data = mj.MjData(self.model)
            
            # Configure simulation for stability
            self.model.opt.timestep = 0.002  # Smaller timestep
            self.model.opt.iterations = 100    # More solver iterations
            self.model.opt.tolerance = 1e-10   # Solver tolerance
            
            # Add damping to all joints
            for i in range(self.model.nv):
                self.model.dof_damping[i] = 0.1
                
            print(f"DEBUG: Successfully loaded MuJoCo model from: {mjcf_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load MuJoCo model from {mjcf_path}: {e}")

        # Initialize passive viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        print("DEBUG: MuJoCo passive viewer launched.")

        # Offscreen rendering setup (lazy initialization)
        self.scn = None
        self.cam = None
        self.vopt = None
        self.pert = None
        self.mjr_context = None
        self._offscreen_initialized = False

        # End-effector site
        self.eef_site_id = self.model.site("gripper_site").id
        
        # Pinocchio setup
        try:
            self.robot_pin = pin.RobotWrapper.BuildFromURDF(urdf_path_pinocchio)
            self.end_effector_frame_id_pin = self.robot_pin.model.getFrameId(end_effector_frame_name_pinocchio)
            if self.end_effector_frame_id_pin == self.robot_pin.model.nframes:
                raise ValueError(f"End-effector frame '{end_effector_frame_name_pinocchio}' not found")
            print(f"DEBUG: Pinocchio model loaded from: {urdf_path_pinocchio}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Pinocchio: {e}")

        # Initialize in neutral configuration
        self.current_q_pin = pin.neutral(self.robot_pin.model)
        
        # Map actuators
        self.mujoco_actuator_ids = []
        for i in range(1, self.robot_pin.model.nq + 1):
            actuator_name = f"joint{i}_ctrl"
            try:
                self.mujoco_actuator_ids.append(self.model.actuator(actuator_name).id)
            except KeyError:
                raise ValueError(f"Actuator '{actuator_name}' not found")

        # Warmup simulation
        for _ in range(10):
            mj.mj_step(self.model, self.data)
            self.data.ctrl[:] = 0

    def _initialize_offscreen_rendering(self):
        """Lazy initialization of offscreen rendering"""
        if self._offscreen_initialized:
            return
            
        try:
            self.scn = mj.MjvScene(self.model, maxgeom=1000)
            self.cam = mj.MjvCamera()
            mj.mjv_defaultCamera(self.cam)
            self.vopt = mj.MjvOption()
            self.pert = mj.MjvPerturb()
            self.mjr_context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150)
            self._offscreen_initialized = True
        except Exception as e:
            print(f"WARNING: Failed offscreen rendering init: {e}")

    def _calculate_ik_pinocchio(self, target_pose_matrix):
        """Stable IK solver with joint limit enforcement"""
        target_se3 = pin.SE3(target_pose_matrix[:3, :3], target_pose_matrix[:3, 3])
        q = self.current_q_pin.copy()
        
        # IK parameters
        eps = 1e-6
        IT_MAX = 100
        DT = 1e-2
        damp = 1e-3  # Increased damping for stability

        for i in range(IT_MAX):
            pin.forwardKinematics(self.robot_pin.model, self.robot_pin.data, q)
            pin.updateFramePlacements(self.robot_pin.model, self.robot_pin.data)
            
            current_se3 = self.robot_pin.data.oMf[self.end_effector_frame_id_pin]
            err = pin.log6(current_se3.inverse() * target_se3).vector
            
            if np.linalg.norm(err) < eps:
                break

            J = pin.computeFrameJacobian(
                self.robot_pin.model, self.robot_pin.data, q,
                self.end_effector_frame_id_pin, 
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            
            # Damped least-squares solution
            dq = J.T @ np.linalg.solve(J @ J.T + damp * np.eye(6), err)
            q = pin.integrate(self.robot_pin.model, q, dq * DT)
            
            # Enforce joint limits during iteration
            for j in range(self.robot_pin.model.nq):
                if self.model.jnt_limited[j]:
                    q[j] = np.clip(q[j], 
                                  self.model.jnt_range[j, 0], 
                                  self.model.jnt_range[j, 1])

        self.current_q_pin = q
        return q[self.robot_pin.nq-self.model.nu:]

    def send_action(self, action_pose_matrix):
        """Send action with joint limit checking"""
        target_joint_poses = self._calculate_ik_pinocchio(action_pose_matrix)
        
        # Validate joint limits
        for i, q in enumerate(target_joint_poses):
            if self.model.jnt_limited[i]:
                if not (self.model.jnt_range[i, 0] <= q <= self.model.jnt_range[i, 1]):
                    print(f"WARNING: Joint {i} target {q} exceeds limits")
                    return False  # Skip this action if limits violated

        self.data.ctrl[self.mujoco_actuator_ids] = target_joint_poses
        mj.mj_step(self.model, self.data)
        
        if self.viewer.is_running():
            self.viewer.sync()
        return True

    def pose(self):
        """Get current end-effector pose"""
        pos = self.data.site_xpos[self.eef_site_id]
        rot_mat = self.data.site_xmat[self.eef_site_id].reshape(3, 3)
        return t3d.affines.compose(pos, rot_mat, [1, 1, 1])

    def get_observation(self, camera_name: str = "front_camera"):
        """Get observation with optional camera image"""
        if self.viewer.is_running():
            return {
                "qpos": self.data.qpos.copy(),
                "qvel": self.data.qvel.copy(),
                "eef_pose": self.pose()
            }
            
        self._initialize_offscreen_rendering()
        if not self._offscreen_initialized:
            return {
                "qpos": self.data.qpos.copy(),
                "qvel": self.data.qvel.copy(),
                "eef_pose": self.pose()
            }
            
        try:
            # Camera setup
            if camera_name in [self.model.camera(i).name for i in range(self.model.ncam)]:
                self.cam.type = mj.mjtCamera.mjCAMERA_FIXED
                self.cam.fixedcamid = self.model.camera(camera_name).id
            else:
                self.cam.type = mj.mjtCamera.mjCAMERA_FREE
                mj.mjv_defaultCamera(self.cam)
                
            # Render
            mj.mjv_updateScene(self.model, self.data, self.vopt, self.pert, self.cam, 0, self.scn)
            rgb_data = np.zeros((240, 320, 3), dtype=np.uint8)
            depth_data = np.zeros((240, 320), dtype=np.float32)
            viewport = mj.MjrRect(0, 0, 320, 240)
            mj.mjr_render(viewport, self.scn, self.mjr_context)
            mj.mjr_readPixels(rgb_data, depth_data, viewport, self.mjr_context)
            
            return {
                "qpos": self.data.qpos.copy(),
                "qvel": self.data.qvel.copy(),
                "eef_pose": self.pose(),
                "camera.rgb": np.flipud(rgb_data)
            }
        except Exception as e:
            print(f"WARNING: Camera render failed: {e}")
            return {
                "qpos": self.data.qpos.copy(),
                "qvel": self.data.qvel.copy(),
                "eef_pose": self.pose()
            }

    def close(self):
        """Cleanup resources"""
        if self.viewer and self.viewer.is_running():
            self.viewer.close()
        if hasattr(self, 'mjr_context') and self.mjr_context:
            try:
                mj.mjr_freeContext(self.mjr_context)
            except:
                pass

if __name__ == "__main__":
    # Configuration
    mjcf_file = Path("/home/nikola/code/alignit/alignit/lite6mjcf.xml")
    urdf_file_pinocchio = Path("/home/nikola/code/alignit/alignit/lite6.urdf")
    end_effector_link_name = "link6"

    # Initialize
    sim = None
    try:
        sim = MuJoCoRobot(mjcf_file, urdf_file_pinocchio, end_effector_link_name)
        
        # Safer initial motion
        for t_step in range(1000):
            # Simple straight-line motion first
            x = 0.3 + 0.001 * t_step if t_step < 500 else 0.8 - 0.001 * (t_step-500)
            target_pos = np.array([x, 0.0, 0.4])
            target_rot = np.eye(3)  # Neutral orientation
            
            pose = t3d.affines.compose(target_pos, target_rot, [1, 1, 1])
            if not sim.send_action(pose):
                break  # Stop if limits violated
                
            if not sim.viewer.is_running():
                break

    finally:
        if sim:
            sim.close()

if __name__ == "__main__1":
    # Configuration
    mjcf_file = Path("/home/nikola/code/alignit/alignit/lite6mjcf.xml")
    urdf_file_pinocchio = Path("/home/nikola/code/alignit/alignit/lite6.urdf")
    end_effector_link_name = "link6"

    # Initialize
    sim = None
    try:
        sim = MuJoCoRobot(mjcf_file, urdf_file_pinocchio, end_effector_link_name)
        
        # Set all controls to zero to keep robot stationary
        sim.data.ctrl[:] = 0
        
        # Keep viewer open without moving
        print("Robot standing still - close viewer to exit")
        while sim.viewer.is_running():
            # Just step the simulation with zero controls
            mj.mj_step(sim.model, sim.data)
            sim.viewer.sync()
            time.sleep(0.01)  # Small delay to reduce CPU usage

    except KeyboardInterrupt:
        print("Simulation stopped by user")
    finally:
        if sim:
            sim.close()