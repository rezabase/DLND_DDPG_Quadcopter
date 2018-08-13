import json
from tasks.task import Task


def myfilename(name):
    return './data/' + name + '.txt'


''' Base class of all below: Manages building a JSON data structur '''
class mydata:
    """Manages the data file read and writes.

        Params
        ======
            name (str): filenames
        """
    def __init__(self, name):
        self.name = name
        
    def toJSON(self, indent=0):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=False, indent=indent)
    
    def fromJSON(self, json_text):
        self.__dict__ = json.loads(json_text)
    
    def fromDictionary(self, json_dict):
        self.__dict__ = json_dict
        
    def getDictionary(self):
        return self.__dict__
        
    def isSet(self, variablename):
        if variablename in self.__dict__:
            return True
        else:
            return False    
        
    def loadData(self, payload_dict):
        self.fromDictionary(payload_dict) #NOTE: This is a dictionary 
        return self
    
    def savedata(self):
        try:
            with open(myfilename(self.name), 'w') as outfile:
                outfile.write(self.toJSON())
        except Exception as error:
            return error
    
    def loaddata(self):
        try:
            with open(myfilename(self.name)) as data_file:
                self.__dict__ = json.load(data_file)
        except Exception as error:
            return error
        
        
        
class myResults(mydata):
    def __init__(self, name, loadfile=False):
        super().__init__(name)
        self.episodes = []
        self.rewards = []        
        if loadfile:
            self.loaddata()

            
class mySimulationResults(mydata):
    def __init__(self, name, loadfile=False):
        super().__init__(name)
        self.episodes = []
        
        if loadfile:
            self.loaddata()
            self.episodes = json.loads(json.dumps(self.episodes))
            
    def get_simulationdata(self, i_episode):
        ep = simulationdata()
        ep.fromDictionary(self.episodes[i_episode])
        return ep
            
class simulationdata(mydata):
    def __init__(self):
        super().__init__(name='')
        
        self.time = []
        
        self.x_pos = []
        self.y_pos = []
        self.z_pos = []
        
        self.x_euler_angle = []
        self.y_euler_angle = []
        self.z_euler_angle = []
        
        #self.phi = []
        #self.theta = []
        #self.psi = []
        
        self.x_velocity = []
        self.y_velocity = []
        self.z_velocity = []
        
        #self.phi_velocity = []
        #self.theta_velocity = []
        #self.psi_velocity = []
        
        self.x_angular_velocity = []
        self.y_angular_velocity = []
        self.z_angular_velocity = []
        
        #self.linear_accel = []
        #self.angular_accels = []
        #self.prop_wind_speed = []
        
        self.rotor_speed1 = []
        self.rotor_speed2 = []
        self.rotor_speed3 = []
        self.rotor_speed4 = []

        
    def add(self, task, rotor_speeds):
        self.time.append(task.sim.time)
        
        
        self.x_pos.append(task.sim.pose[0])        
        self.y_pos.append(task.sim.pose[1])
        self.z_pos.append(task.sim.pose[2])

        self.x_euler_angle.append(task.sim.pose[3])
        self.y_euler_angle.append(task.sim.pose[4])
        self.z_euler_angle.append(task.sim.pose[5])
        
        #self.phi.append(task.sim.v[0])
        #self.theta.append(task.sim.v[1])
        #self.psi.append(task.sim.v[2])
        
        self.x_velocity.append(task.sim.v[0])
        self.y_velocity.append(task.sim.v[1])
        self.z_velocity.append(task.sim.v[2])
        
        #self.phi_velocity.append()
        #self.theta_velocity.append()
        #self.psi_velocity.append()
        
        self.x_angular_velocity.append(task.sim.angular_v[0])
        self.y_angular_velocity.append(task.sim.angular_v[1])
        self.z_angular_velocity.append(task.sim.angular_v[2])
        
        #self.linear_accel.append(task.sim.linear_accel.tolist()) ### MUST CHANGE TO X,Y,Z
        #self.angular_accels.append(task.sim.angular_accels.tolist())
        #self.prop_wind_speed.append(task.sim.prop_wind_speed.tolist())
        
        self.rotor_speed1.append(rotor_speeds[0])
        self.rotor_speed2.append(rotor_speeds[1])
        self.rotor_speed3.append(rotor_speeds[2])
        self.rotor_speed4.append(rotor_speeds[3])

        
        