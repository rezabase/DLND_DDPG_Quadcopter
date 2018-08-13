import numpy as np
from physics_sim import PhysicsSim

from tasks.task import Task



#import math
class Vector3:
    def __init__(self, x=0, y=0, z=0):
        self.update(x, y, z)
    
      # Used for debugging. This method is called when you print an instance  
    def __str__(self):
        return "(" + str(self.value[0]) + ", " + str(self.value[1]) + ", " + str(self.value[2]) + ")"
    
    def update(self, x=0, y=0, z=0):
        self.value = np.array([x, y, z])
        
    def distance(self, target=None):
        if target == None:
            return np.linalg.norm(self.value) #distance to 0,0,0
        else:
            return np.linalg.norm(self.value - target.value)
        
        
        
class FlightData():
    def __init__(self):
        self.position = Vector3()
        self.euler_angle = Vector3()
        self.velocity = Vector3()
        self.angular_velocity = Vector3()  
        self.linear_accel = Vector3()
        self.angular_accels = Vector3()        
        self.distance_vect = Vector3()
        self.distance = 100000
        
    def update(self, simulator, targetpos):
        ###self.time.append(task.sim.time)
        self.position.update(simulator.pose[0], simulator.pose[1], simulator.pose[2])
        self.euler_angle.update(simulator.pose[3], simulator.pose[4], simulator.pose[5])
        self.velocity.value = simulator.v
        self.angular_velocity.value = simulator.angular_v 
        self.linear_accel.value = simulator.linear_accel
        self.angular_accels.value = simulator.angular_accels
        
        #summarise the total distance of x,y,z. close to 0 if reaching destination.
        self.target_distance = self.position.distance(targetpos)

        
        
        
#Use (LAMBDA * reward) to make a small number between -1 to +1 
#LAMBDA =  0.001
LAMBDA =  0.01  #use this one when using linear calcs.

class TaskController(Task):
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, init_angle_velocities=None, runtime = 5, target_pos = None):
        super().__init__(init_pose, init_velocities, init_angle_velocities, runtime, target_pos)
        
        
        self.flightdata = FlightData()
        self.vtargetpos = Vector3()
        
        self.operation = 'takeoff'
        
        self.operation_rewards = {
            'takeoff': lambda r, d: self.reward_takeoff(r, d),
            'land': lambda r, d: self.reward_landing(r, d),
            'hover': lambda r, d: self.reward_hover(r, d),
            'goto': lambda r, d: self.reward_goto(r, d),
        }
        
        self.one = 1/LAMBDA #Becouse I'm using lambda to make a small number of the total reward, I need sometime a number that is +1 as a reward. SoI'm using the variable one to calculate that number based on thew size of LAMBDA
        

        
    def setOperation(self, ops):
        self.operation = ops
    
    def reward_takeoff(self, reward, done):
        return reward, done
    
    def reward_hover(self, reward, done):
        return reward, done
    
    def reward_landing(self, reward, done):
        return reward, done    
    
    '''
    def euler_angle_reward(self, reward, done):
        angle = self.flightdata.euler_angle.np_array()
        reward -= abs(angle - np.array([0,0,0])).sum()
        return reward, done
    '''
    

    def reward_goto(self, reward, done):
        dist = self.flightdata.target_distance
        if dist < 2:  # agent is within bonus region? significantly increase reward tanh_distance < 
            reward += 1.0  
            done = True
        return reward, done
      
    #def trajectory_reward(self, reward, done):        
    #    reward -= self.flightdata.target_distance **1.2
    #    reward = max(reward, -self.one) #Floors the top distance value if needed to -1.
    #    return reward, done
    
    def trajectory_reward(self, reward, done):        
        reward -= np.tanh(self.flightdata.target_distance * LAMBDA)
        return reward, done
    
    #Then, the reward is computed from get_reward(). The episode is considered done if the time limit has been exceeded, 
    # or the quadcopter has travelled outside of the bounds of the simulation.    
    def calc_rewards(self, done):
        self.vtargetpos.update(self.target_pos[0], self.target_pos[1], self.target_pos[2])
        self.flightdata.update(self.sim, self.vtargetpos) #Uptade all flight positioning info
        
        reward = 0
        reward, done = self.trajectory_reward(reward, done)
        ###reward, done = self.euler_angle_reward(reward, done) #Removing it for Test 4
        # reward = LAMBDA * reward / self.action_repeat #create a small number out of reward. 
        
        reward, done = self.operation_rewards[self.operation](reward, done) #Operations specific reward  
        reward = reward / self.action_repeat #create a small number out of reward. 
        return reward, done


    


    #The step() method is perhaps the most important. It accepts the agent's choice of action rotor_speeds, which is used to prepare the next state to pass on to the agent
    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            r, done = self.calc_rewards(done)
            reward += r
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done
    