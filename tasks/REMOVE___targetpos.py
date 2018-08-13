import numpy as np
from physics_sim import PhysicsSim

from tasks.task import Task


class TargetPos(Task):
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, init_angle_velocities=None, runtime=5., target_pos=None):
        super().__init__(init_pose, init_velocities, init_angle_velocities, runtime, target_pos)

        
        
        
    #Then, the reward is computed from get_reward(). The episode is considered done if the time limit has been exceeded, 
    # or the quadcopter has travelled outside of the bounds of the simulation.    
    def get_reward(self):
        """Uses current pose of sim to return reward."""
        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        #reward = np.tanh(1 - 0.003*(abs(self.sim.pose[:3] - self.target_pos))).sum() #TESTING
        target_pos = self.target_pos
        current_pos = self.sim.pose[:3]
        
        reward = np.tanh(1 - (abs(current_pos - target_pos))).sum() 

        return reward


    