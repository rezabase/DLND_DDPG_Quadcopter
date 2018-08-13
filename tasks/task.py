import numpy as np
from physics_sim import PhysicsSim

#Example tasks for quadcopter:
#   takeoff, 
#   hover in place, 
#   land softly, 
#   or reach a target pose



class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        # The simulator is initialized as an instance of the PhysicsSim class (from physics_sim.py).
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        
        #Inspired by the methodology in the original DDPG paper, we make use of action repeats. For each timestep of the agent, we step the simulation action_repeats timesteps. 
        # If you are not familiar with action repeats, please read the Results section in the DDPG paper.
        self.action_repeat = 3

        #We set the number of elements in the state vector. For the sample task, we only work with the 6-dimensional pose information. T
        # To set the size of the state (state_size), we must take action repeats into account.
        self.state_size = self.action_repeat * 6
        
        #The environment will always have a 4-dimensional action space, with one entry for each rotor (action_size=4). 
        # You can set the minimum (action_low) and maximum (action_high) values of each entry here.
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        #TARGET/GOAL: The sample task in this provided file is for the agent to reach a target position. We specify that target position as a variable.
        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

        
    #Then, the reward is computed from get_reward(). The episode is considered done if the time limit has been exceeded, 
    # or the quadcopter has travelled outside of the bounds of the simulation.    
    def get_reward(self):
        raise NotImplementedError("{} must override get_reward()".format(self.__class__.__name__))

    
    #The step() method is perhaps the most important. It accepts the agent's choice of action rotor_speeds, which is used to prepare the next state to pass on to the agent
    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    
    #The reset() method resets the simulator. The agent should call this method every time the episode ends.
    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state