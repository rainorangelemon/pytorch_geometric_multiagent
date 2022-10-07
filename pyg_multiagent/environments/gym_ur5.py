import numpy as np
import pybullet as p
from time import sleep, time
import pybullet_data
import pickle
from environment.gym_abstract import AbstractState, AbstractEnv
from torch_geometric.data import Data, HeteroData
from scipy.spatial.distance import cdist
import torch
import math
from environment.utils import save_gif
from time import sleep

AGENT_TOP_K = 6
OBSTACLE_TOP_K = 1
AGENT_OBS_RADIUS = 4.0
OBSTACLE_OBS_RADIUS = 4.0
AGENT_DISTANCE_THRESHOLD = 0.3
OBSTACLE_DISTANCE_THRESHOLD = 0.3
GOAL_THRESHOLD = 0.3
n_candidates = 2000   

class UR5State():
    def __init__(self, num_agents, arm_ids=None, **kwargs):
        
        self.agents = np.array([[-1.17792784, -0.34144958, 0.57453987, 0.49696033, 0.02230923],
                        [-1.27181467, -1.83372885, -2.32752448, -2.33945606, -1.14477833],
                        [-0.88225001, -1.51438693, 1.81583661, -0.96966697, 0.55916841],
                        [-2.54433012, -2.20229549, -0.32797754, -1.34072494, 0.26237901]])
        
#         self.agents = np.array([[-1.62115719, -1.80431687,  1.38434103, -0.24863413, -0.13774741],
#                         [-0.80383085, -0.81837049, -0.29196595, -2.16015494, -1.3827381 ],
#                         [-0.88225001, -1.51438693, 1.81583661, -0.96966697, 0.55916841],
#                         [-3.54433012, -2.20229549, -0.32797754, -1.34072494, 0.26237901]])
        
        self.agent_goals = np.array([[-1.62044076, -1.80115857, 1.38368296, -0.25042694, -0.13860791],
                        #[-0.9210396, 0.00231394, -0.81820546, -2.1713121, -1.35159425],
                                     [-0.803686, -0.81500896, -0.29771421, -2.16319469, -1.38464397],
                        [-1.29073086, -8.66180575e-01, -6.16697935e-01, 1.94443816e-01, -4.09509211e-02],
                        [-0.41040391, -2.25299177, 2.22303545, -3.54345599e-01, 2.44911166e-02]])        
        
        self.obstacles = np.array([[[0.1, 0.1, 0.1], [0.3, 0.2, 0.5]],
                    [[0.1, 0.1, 0.05], [-0.3, 0.7, 0.4]],
                    [[0.05, 0.05, 0.05],[-0.65, 0.2, 0.6]],
                    [[0.05, 0.05, 0.3],[-0.3, -0.15, 0.2]]])
        
        if arm_ids is None:
            if num_agents != 4:
                arm_ids = np.random.choice(4, size=num_agents, replace=False)
            else:
                arm_ids = np.random.choice(4, size=num_agents, replace=False)
        
        self.num_agents = num_agents = len(arm_ids)        
        # assert len(arm_ids)==num_agents
        
        self.arm_ids = arm_ids
        self.agents = self.agents[arm_ids, :]
        self.agent_goals = self.agent_goals[arm_ids, :]
        self.obstacles = self.obstacles
        
        self.reset_env(arm_ids)
        self.obs_ids = []
        for halfExtents, basePosition in self.obstacles:
            self.obs_ids.append(self.create_voxel(halfExtents, basePosition, [0, 0, 0, 1]))        
    
    def reset_env(self, arm_ids):
        p.resetSimulation()
        
        base_positions = [[0, 0, 0], [0, 0.7, 0], [-0.7, 0.7, 0], [-0.7, 0, 0]]
        orientations = [[0, 0, 0, 1], p.getQuaternionFromEuler([0,0,np.pi/2]), 
                        p.getQuaternionFromEuler([0,0,np.pi]), p.getQuaternionFromEuler([0,0,-np.pi/2])]
        self.ur5s = []
        self.arms_ids = arm_ids
        for arm_id in arm_ids:
            ur5 = p.loadURDF("environment/ur5/ur5.urdf", base_positions[arm_id], 
                             orientations[arm_id], useFixedBase=True)        
            self.ur5s.append(ur5)
            
        # if self.num_agents==2:
        #     self.ur5 = p.loadURDF("environment/ur5/ur5.urdf", [0, 0, 0], [0, 0, 0, 1], useFixedBase=True)        
        #     orien = p.getQuaternionFromEuler([0,0,np.pi/2])
        #     self.ur5_second = p.loadURDF("environment/ur5/ur5.urdf", [0, 0.7, 0], orien, useFixedBase=True)
        #     self.ur5s = [self.ur5, self.ur5_second]
        # elif self.num_agents==4:
        #     self.ur5 = p.loadURDF("environment/ur5/ur5.urdf", [0, 0, 0], [0, 0, 0, 1], useFixedBase=True)        
        #     orien = p.getQuaternionFromEuler([0,0,np.pi/2])
        #     self.ur5_second = p.loadURDF("environment/ur5/ur5.urdf", [0, 0.7, 0], orien, useFixedBase=True)
        #     orien = p.getQuaternionFromEuler([0,0,np.pi])
        #     self.ur5_third = p.loadURDF("environment/ur5/ur5.urdf", [-0.7, 0.7, 0], orien, useFixedBase=True)
        #     orien = p.getQuaternionFromEuler([0,0,-np.pi/2])
        #     self.ur5_fourth = p.loadURDF("environment/ur5/ur5.urdf", [-0.7, 0, 0], orien, useFixedBase=True)
        #     self.ur5s = [self.ur5, self.ur5_second, self.ur5_third, self.ur5_fourth]            
        # else:
        #     assert False

#         plane = p.createCollisionShape(p.GEOM_PLANE)
#         self.plane = p.createMultiBody(0, plane)

#         for ur5 in self.ur5s:
#             p.setCollisionFilterPair(ur5, self.plane, 1, -1, 0)

        self.ur5 = self.ur5s[0]
        n_joints = p.getNumJoints(self.ur5)
        joints = [p.getJointInfo(self.ur5, i) for i in range(n_joints)]
        self.joints = [j[0] for j in joints if j[2] == p.JOINT_REVOLUTE]
        self.pose_range = np.array([(p.getJointInfo(self.ur5, jointId)[8], p.getJointInfo(self.ur5, jointId)[9]) for jointId in
                           self.joints])
        self.config_dim = len(self.joints)
        self.bound = self.pose_range.T.reshape(-1)

        _link_name_to_index = {p.getBodyInfo(self.ur5)[0].decode('UTF-8'): -1, }
        for _id in range(p.getNumJoints(self.ur5)):
            _name = p.getJointInfo(self.ur5, _id)[12].decode('UTF-8')
            _link_name_to_index[_name] = _id
        self.tip_index = _link_name_to_index['ee_link']
    
    def sample_agents(self, n_agents, prob=0.1):
        assert False
        
    def create_voxel(self, halfExtents, basePosition, baseOrientation, color='random'):
        groundColId = p.createCollisionShape(p.GEOM_BOX, halfExtents=halfExtents)
        if color == 'random':
            groundVisID = p.createVisualShape(shapeType=p.GEOM_BOX,
                                              rgbaColor=np.random.uniform(0, 1, size=3).tolist() + [0.8],
                                              # specularColor=[0.4, .4, 0],
                                              halfExtents=halfExtents)
        else:
            groundVisID = p.createVisualShape(shapeType=p.GEOM_BOX,
                                              rgbaColor=color,
                                              # specularColor=[0.4, .4, 0],
                                              halfExtents=halfExtents)
        groundId = p.createMultiBody(baseMass=0,
                                     baseCollisionShapeIndex=groundColId,
                                     baseVisualShapeIndex=groundVisID,
                                     basePosition=basePosition,
                                     baseOrientation=baseOrientation)
        return groundId        
        
    def get_status(self):
        status = []
        agents = np.array(self.agents)
        dist2goal = np.linalg.norm(np.array(self.agents)-np.array(self.agent_goals), axis=-1)

        p.performCollisionDetection()
        contact_pairs = [set([t[2] for t in p.getContactPoints(ur5)]) for ur5 in self.ur5s]

        status = ['' for _ in range(self.num_agents)]
        status = [s+'danger_agent' if len(set(self.ur5s) & pairs) else s for s, pairs in zip(status, contact_pairs)]
        status = [s+'danger_obstacle' if len(pairs - set(self.ur5s)) else s for s, pairs in zip(status, contact_pairs)]
        status = [s+'done' if d<GOAL_THRESHOLD else s for s, d in zip(status, dist2goal)]
        status = [s+'free' if ('danger' not in s) else s for s in status]

        return status  
    
    def done(self, status=None):
        if status is None:
            status = self.get_status()
        return np.sum(['done' in s for s in status])==len(self.agents)    


class UR5Env:

    def __init__(self, num_agents, GUI=False, 
                 action_dim=5, space_dim=5, state_dim=5, angle_dim=0, 
                 save_file=None,
                 arm_ids=None,
                 randomize=False,
                 **kwargs):

        
        self.space_dim = space_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.angle_dim = angle_dim

        try:
            p.disconnect(0)
        except:
            pass
        
        if save_file is not None:
            save_file = f' --mp4={save_file}' + ' --width=2160 --height=1440 --mp4fps=15'
        else:
            save_file = ''
        
        if GUI:
            p.connect(p.GUI, options='--background_color_red=1.0 --background_color_green=1.0 --background_color_blue=1.0'+save_file)
        else:
            p.connect(p.DIRECT)

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, lightPosition=[0, 0, 100])
        p.resetDebugVisualizerCamera(
            cameraDistance=1.1,
            cameraYaw=12.040756225585938,
            cameraPitch=-37.56093978881836,
            cameraTargetPosition=[0, 0, 0.7])
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.world = UR5State(num_agents=num_agents, arm_ids=arm_ids)
        self.ur5s = self.world.ur5s
        self.pose_range = self.world.pose_range
        self.joints = self.world.joints
        self.num_agents = len(self.ur5s)
        
        if not randomize:
            self.set_config(self.world.agents)
        else:
#             while True:
#                 random_config = np.random.uniform(self.pose_range[:, 0], self.pose_range[:, 1], size=self.world.agents.shape)
#                 self.set_config(random_config)
#                 status = self.world.get_status()
#                 if np.all(['danger' not in s for s in status]):
#                     self.world.agent_goals = random_config
#                     break            
            
            while True:
                random_config = np.random.uniform(-3, 3, size=self.world.agents.shape)
                self.set_config(random_config)
                status = self.world.get_status()
                if np.all(['danger' not in s for s in status]):
                    self.world.agents = random_config
                    break
            self.set_config(self.world.agents)
        # print(self.get_config(), self.world.agents, self.world.arm_ids)
        # assert False

    def set_config(self, configs, ur5s=None):
        if ur5s is None:
            ur5s = self.ur5s
        for ur5, c in zip(ur5s, configs):
            for i, value in zip(self.joints, c):
                p.resetJointState(ur5, i, value)
        p.performCollisionDetection()

    def get_config(self, ur5s=None):
        if ur5s is None:
            ur5s = self.ur5s
        return np.array([[p.getJointState(ur5, i)[0] for i in self.joints] for ur5 in ur5s])

    def dynamic(self, agents, actions):
        next_pos = agents + actions*0.1
        next_pos = np.clip(next_pos, self.world.pose_range[:, 0], self.world.pose_range[:, 1])
        return next_pos  

    @staticmethod
    def dynamic_torch(pos, action):
        pos = pos + action*0.1
        new_pos = pos.clone().clip(-2*math.pi, 2*math.pi)
        return new_pos

    def _get_obs(self, **kwargs):
        configs = self.get_config()
        one_hot = torch.zeros(self.num_agents, 4)
        one_hot[torch.arange(self.num_agents), self.world.arm_ids] = 1
        data = HeteroData()
        data['agent'].pos = torch.FloatTensor(configs)
        configs = torch.cat((one_hot, torch.FloatTensor(configs)), dim=-1)
        data['agent'].x = torch.zeros(len(configs), 1) # torch.LongTensor(self.world.arm_ids)
        
        # make a complete graph
        a2a_index = torch.ones(self.num_agents, self.num_agents)
        a2a_index = a2a_index - torch.eye(self.num_agents)
        a2a_index = torch.vstack(torch.where(a2a_index))    
        data['agent', 'a_near_a', 'agent'].edge_index = a2a_index
        data['agent', 'a_near_a', 'agent'].edge_attr = torch.cat((configs[a2a_index[0, :]], configs[a2a_index[1, :]]), dim=-1)
        return data
    
    def obs_from_pos(self, data, agent_pos, **kwargs):
        edge_index = data['agent', 'a_near_a', 'agent'].edge_index
        data['agent', 'a_near_a', 'agent'].edge_attr[:,4:(4+agent_pos.shape[1])] = agent_pos[edge_index[0, :]]
        data['agent', 'a_near_a', 'agent'].edge_attr[:,(8+agent_pos.shape[1]):((8+2*agent_pos.shape[1]))] = agent_pos[edge_index[1, :]]
        return data        

    # Returns an observation of an agent
    def _get_obs_random_k(self, **kwargs):
        
        data = self._get_obs()
        data['agent'].n_id = torch.arange(self.num_agents)
        
        a_edge_index = data['a_near_a'].edge_index
        neighbors_a = [(a_edge_index[1,:]==k) for k in range(self.num_agents)]
        a_candidates = [np.arange(a_edge_index.shape[1])[neighbors_a[k].data.numpy()] for k in range(self.num_agents)]
        cover_a_bool = torch.BoolTensor([0]*a_edge_index.shape[1])

        while True:

            sub_data = data.clone()
            edge_index_a_current = [np.random.choice(a, 1, replace=False) for a in a_candidates]
            edges_a = np.concatenate(edge_index_a_current)

            sub_data['a_near_a'].edge_index = data['a_near_a'].edge_index[:, edges_a]
            sub_data['a_near_a'].edge_attr = data['a_near_a'].edge_attr[edges_a, :]  
            cover_a_bool[edges_a] = True
            cover_a = cover_a_bool.all()

            yield sub_data

            if cover_a:
                break    

    # Executes an action by an agent
    def step(self, action_input, **kwargs):
        assert len(action_input) == self.num_agents, 'Action input should be a tuple with the form (num_agents, action_dim)'
        assert len(action_input[0]) == self.action_dim, 'Action input should be a tuple with the form (num_agents, action_dim)'
        
        prev_status = self.world.get_status()
        prev_o = self._get_obs()
        
        dist = np.linalg.norm(self.world.agents[:, :self.space_dim]-self.world.agent_goals[:, :self.space_dim], axis=-1)
        next_pos = self.dynamic(self.world.agents, action_input)
        self.set_config(next_pos)
        self.world.agents = next_pos
        next_dist = np.linalg.norm(next_pos[:, :self.space_dim]-self.world.agent_goals[:, :self.space_dim], axis=-1)
        displacement = dist - next_dist        
        
        # Perform observation
        next_o = self._get_obs() 

        # Done?
        next_status = self.world.get_status()
        done = self.world.done(status=next_status)
        
        rewards = [-10*('danger_agent' in s)-10*('danger_obstacle' in s) for s in next_status]
        rewards = np.array(rewards) + 10*displacement + 10*done - 0.1        
        
        info = {
                "action": action_input,
                "prev_free": ['free' in s for s in prev_status],
                "prev_danger": ['free' not in s for s in prev_status],
                "next_goal": ['done' in s for s in next_status],
                "next_free": ['free' in s for s in next_status],
                "next_danger": ['free' not in s for s in next_status],
                "prev_obstacle": ['obstacle' in s for s in prev_status],
                "prev_agent": ['agent' in s for s in prev_status],
                "meet_obstacle": ['obstacle' in s for s in next_status],
                "meet_agent": ['agent' in s for s in next_status],
                "rewards": rewards,
                }
        for key, value in info.items():
            prev_o[key] = torch.FloatTensor(value)

        return next_o, rewards, done, prev_o

    def potential_field(self, actions, K1, K2, ignore_agent=False):
        
        # size of actions: (num_agents, n_candidates, action_dim)
        
        assert actions.shape[0]==self.num_agents
        assert actions.shape[-1]==self.action_dim
        n_candidates = actions.shape[1]
        
        origin_pos = np.copy(self.world.agents)
        next_pos = np.expand_dims(origin_pos, 1)
        next_pos = np.tile(next_pos, (1, n_candidates, 1)) # num_agents x n_candidates x state_dim
        
        next_pos = self.dynamic(next_pos.reshape(-1, self.state_dim), actions.reshape(-1, self.action_dim))
        next_pos = next_pos.reshape((self.num_agents, n_candidates, self.state_dim))
        
        goal_dim = len(self.world.agent_goals[0])
        goal_force = ((next_pos[:, :, :goal_dim] - np.expand_dims(self.world.agent_goals, axis=1)[:, :, :goal_dim])**2).sum(axis=-1)  # num_agents x n_candidates
        score = -K2 * goal_force.reshape(-1)
        score = score.reshape(self.num_agents, n_candidates)
        return score

    def save_fig(self, agents, goals, obstacles, filename):
        agents = np.array(agents)
        self.world.reset_env(self.world.arm_ids)
        self.ur5s = self.world.ur5s
        obstacle_colors = [[85/256,231/256,211/256, 1.0], [93/256,153/256,163/256, 1.0], 
                           [143/256,248/256, 52/256, 1.0], [192/256, 50/256, 130/256, 1.0]]
        obstacle_id = 0
        for halfExtents, basePosition in obstacles:
            self.world.create_voxel(halfExtents, basePosition, [0, 0, 0, 1], obstacle_colors[obstacle_id])
            obstacle_id += 1
            
        print(self.world.arm_ids)
        print(self.ur5s)
        print(agents)

        # ur5 = p.loadURDF("environment/ur5/ur5.urdf", [0, 0, 0], [0, 0, 0, 1], useFixedBase=True, flags=p.URDF_IGNORE_COLLISION_SHAPES)
        # orien = p.getQuaternionFromEuler([0,0,np.pi/2])
        # ur5_second = p.loadURDF("environment/ur5/ur5.urdf", [0, 0.7, 0], orien, useFixedBase=True, flags=p.URDF_IGNORE_COLLISION_SHAPES)
        # new_ur5s = [ur5, ur5_second]
        # self.set_config(agents[0], new_ur5s)

        # if make_gif:
        #     for _ in range(100):
        #         p.stepSimulation()
        #         sleep(0.1)

        self.set_config(agents[0], self.ur5s)
        prev_poses = [p.getLinkState(ur5, self.world.tip_index)[0] for ur5 in self.ur5s]
        self.set_config(goals, self.ur5s)

        # for new_ur5 in self.ur5s:
        #     for data in p.getVisualShapeData(new_ur5):
        #         color = list(data[-1])
        #         color[-1] = 0.2
        #         p.changeVisualShape(new_ur5, data[1], rgbaColor=color)        

        gifs = []
        current_state_idx = 0

        # ur5 = p.loadURDF("environment/ur5/ur5.urdf", [0, 0, 0], [0, 0, 0, 1], useFixedBase=True, flags=p.URDF_IGNORE_COLLISION_SHAPES)
        # orien = p.getQuaternionFromEuler([0,0,np.pi/2])
        # ur5_second = p.loadURDF("environment/ur5/ur5.urdf", [0, 0.7, 0], orien, useFixedBase=True, flags=p.URDF_IGNORE_COLLISION_SHAPES)
        # new_ur5s = [ur5, ur5_second]

        # gifs.append(p.getCameraImage(width=1080, height=720, lightDirection=[0, 0, -1], shadow=0,
        #                                  renderer=p.ER_BULLET_HARDWARE_OPENGL)[2])                
               
        line_colors = [[0,0,1],[0,9,0],[1.0,0.95,0.0],[1,0,0]]
            
        while True:
            current_state = agents[current_state_idx]
            disp = agents[current_state_idx + 1] - agents[current_state_idx]

            d = np.linalg.norm(agents[current_state_idx]-agents[current_state_idx + 1], axis=-1)

            K = int(np.ceil((d / 0.3).max()))
            for k in range(0, K):

                c = agents[current_state_idx] + k * 1. / K * disp
                self.set_config(c, self.ur5s)
                new_poses = [p.getLinkState(new_ur5, self.world.tip_index)[0] for new_ur5 in self.ur5s]
                for line_color, prev_pos, new_pos in zip(line_colors, prev_poses, new_poses):
                    p.addUserDebugLine(prev_pos, new_pos, line_color, 5, 0)
                    # prev_pos = new_pos
#                     b = p.loadURDF("sphere2red.urdf", new_pos, globalScaling=0.01, flags=p.URDF_IGNORE_COLLISION_SHAPES)
                prev_poses = new_poses
                # if k==0:
                #     p.changeVisualShape(b, -1, rgbaColor=[0, 0, 0.7, 0.7])
                # gifs.append(p.getCameraImage(width=1080, height=720, lightDirection=[0, 0, -1], shadow=0,
                #                          renderer=p.ER_BULLET_HARDWARE_OPENGL)[2])
            current_state_idx += 1

            # gifs.append(p.getCameraImage(width=1000, height=800, lightDirection=[1, 1, 1], shadow=1,
            #                              renderer=p.ER_BULLET_HARDWARE_OPENGL)[2])
            if current_state_idx == len(agents) - 1:
                self.set_config(agents[-1], self.ur5s)
                final_poses = [p.getLinkState(new_ur5, self.world.tip_index)[0] for new_ur5 in self.ur5s]
                for line_color, prev_pos, final_pos in zip(line_colors, prev_poses, final_poses):
                    p.addUserDebugLine(prev_pos, final_pos, line_color, 5, 0)
#                     p.loadURDF("sphere2red.urdf", final_pos, globalScaling=0.01, flags=p.URDF_IGNORE_COLLISION_SHAPES)
                gifs.append(p.getCameraImage(width=1080, height=720, lightDirection=[0, 0, -1], shadow=0,
                                         renderer=p.ER_BULLET_HARDWARE_OPENGL)[2])
                break

        # save_gif(gifs, name=filename)       
        
        for _ in range(1000):
            p.stepSimulation()
            sleep(0.1)
            

    def _valid_state(self, state):
        return (state >= self.pose_range[:, 0]).all() and \
               (state <= self.pose_range[:, 1]).all()

    def _state_fp(self, state):
        if state is not None:
            if not self._valid_state(state):
                return False
            self.set_config(state)
        p.performCollisionDetection()
        if np.any([len(p.getContactPoints(ur5)) for ur5 in self.ur5s]): #and np.max([len(p.getClosestPoints(self.ur5, obs, distance=0.09)) for obs in self.obs_ids]) == 0:
            return False
        else:
            return True

if __name__ == '__main__':
    env = UR5Env(GUI=True)
    env.init_new_problem_with_config()
    env.set_config(env.goal_state)
    print(env.ur5, env.ur5_second)
    # gifs = env.plot([env.init_state, env.goal_state], make_gif=True)
    # print(gifs)
    while True:
        p.stepSimulation()
        sleep(0.1)
        print(env._state_fp(env.get_config()))
        print(env.get_config())
