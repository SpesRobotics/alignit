from alignit.utils.tfs import are_tfs_close


class Robot:
    def send_action(self, action):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def pose(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def get_observation(self) -> dict:
        raise NotImplementedError("This method should be overridden by subclasses.")
        
    def servo_to_pose(self, pose, lin_tol=1e-1, ang_tol=1e-1):
        while not are_tfs_close(self.pose(), pose, lin_tol, ang_tol):
            self.send_action(pose)
