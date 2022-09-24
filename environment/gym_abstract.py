from abc import ABC, abstractmethod
import sys
from matplotlib.colors import hsv_to_rgb
import random
import math
import copy
import numpy as np
from scipy.spatial.distance import cdist
import torch
from torch_cluster import radius, radius_graph, knn_graph, knn
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data, HeteroData
from scipy.linalg import block_diag
from torch_sparse import SparseTensor
# from torch_geometric.utils import index_to_mask
from functools import reduce
from environment.utils import less_or_equal

neighbor_sample = torch.ops.torch_sparse.neighbor_sample


AGENT_TOP_K = 6
OBSTACLE_TOP_K = 2
AGENT_OBS_RADIUS = 2.0
OBSTACLE_OBS_RADIUS = 2.0
AGENT_DISTANCE_THRESHOLD = 0.3
OBSTACLE_DISTANCE_THRESHOLD = 0.3
GOAL_THRESHOLD = 0.45


class AbstractState(ABC):
    def __init__(self, world0, goals, space_dim, state_dim, 
                 obstacle_threshold, agent_threshold, goal_threshold,
                 num_agents=1, prob=0.,keep_sample_obs=False):
        assert(len(world0.shape) == 2 and world0.shape==goals.shape)
        self.state = world0.copy()
        self.goals = goals.copy()
        self.prob = prob
        self.num_agents = num_agents
        self.space_dim = space_dim
        self.state_dim = state_dim
        self.obstacle_threshold, self.agent_threshold, self.goal_threshold = obstacle_threshold, agent_threshold, goal_threshold
        self.keep_sample_obs = keep_sample_obs
        self.obstacles, self.agents, self.agent_goals = self.scanForAgents()
        self.obstacles = np.array(self.obstacles).astype(float)
        self.agent_goals = np.array(self.agent_goals).astype(float)
        self.agents = np.array(self.agents).astype(float)
        assert(self.agents.shape == (num_agents, state_dim))

    @abstractmethod
    def scanForAgents(self):
        pass
    
    def get_status(self):
        status = []
        agents = np.array(self.agents)
        obstacles = self.obstacles
        if len(agents) > 1:
            distance = cdist(agents[:, :self.space_dim], agents[:, :self.space_dim])
            distance_nearest_agent = np.sort(distance, axis=-1)[:, 1]    
        else:
            distance_nearest_agent = 100*np.ones((len(self.agents),))
        if len(obstacles) > 0:
            distance_obs = cdist(agents[:, :2], np.array(obstacles)[:, :2])
            distance_nearest_obs = np.min(distance_obs, axis=-1)
        else:
            distance_nearest_obs = 100*np.ones((len(self.agents),))
        dist2goal = np.linalg.norm(np.array(self.agents[:, :self.space_dim])-np.array(self.agent_goals[:, :self.space_dim]), axis=-1)

        status = ['' for _ in range(self.num_agents)]
        status = [s+'danger_obstacle' if less_or_equal(d, self.obstacle_threshold) else s for s, d in zip(status, distance_nearest_obs)]
        status = [s+'danger_agent' if less_or_equal(d, self.agent_threshold) else s for s, d in zip(status, distance_nearest_agent)]
        status = [s+'safe' if (less_or_equal(0.1+self.obstacle_threshold, d1) and less_or_equal(0.1+self.agent_threshold, d2)) else s for s, d1, d2 in zip(status, distance_nearest_obs, distance_nearest_agent)]
        status = [s+'done' if less_or_equal(d, self.goal_threshold) else s for s, d in zip(status, dist2goal)]
        status = [s+'free' if ('danger' not in s) else s for s in status]
        
        return status

    @abstractmethod
    def sample_agents(self, n_agents, prob=0.1):
        pass

    def done(self, status=None):
        if status is None:
            status = self.get_status()
        return np.sum(['done' in s for s in status])==len(self.agents)


class AbstractEnv(ABC):
    
    # Initialize env
    def __init__(self, absState, action_dim, state_dim, space_dim, angle_dim, 
                 num_agents=1, connected=False, 
                 SIZE=(10,40), PROB=(0,.5), simple=False,
                 agent_top_k=None,obstacle_top_k=None, angle_embed=False,
                 obstacle_threshold=None, agent_threshold=None, 
                 goal_threshold=None, agent_obs_radius=None, obstacle_obs_radius=None,
                 min_dist=None, max_dist=None, hetero=True,
                 keep_sample_obs=False,):
        """
        Args:
            SIZE: size of a side of the square grid
            PROB: range of probabilities that a given block is an obstacle
        """
        # Initialize member variables
        if agent_top_k is None:
            agent_top_k = AGENT_TOP_K
        if obstacle_top_k is None:
            obstacle_top_k = OBSTACLE_TOP_K
        if obstacle_threshold is None:
            obstacle_threshold = OBSTACLE_DISTANCE_THRESHOLD
        if agent_threshold is None:
            agent_threshold = AGENT_DISTANCE_THRESHOLD
        if goal_threshold is None:
            goal_threshold = GOAL_THRESHOLD
        if agent_obs_radius is None:
            agent_obs_radius = AGENT_OBS_RADIUS
        if obstacle_obs_radius is None:
            obstacle_obs_radius = OBSTACLE_OBS_RADIUS
        
        self.agent_top_k = agent_top_k
        self.obstacle_top_k = obstacle_top_k
        self.angle_embed = angle_embed
        self.num_agents        = num_agents 
        self.SIZE              = SIZE
        self.PROB              = PROB
        self.simple = simple
        self.connected         = connected
        self.finished          = False
        self.absState = absState
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.space_dim = space_dim    
        self.angle_dim = angle_dim
        self.obstacle_threshold = obstacle_threshold
        self.agent_threshold = agent_threshold
        self.goal_threshold = goal_threshold
        self.agent_obs_radius = agent_obs_radius
        self.obstacle_obs_radius = obstacle_obs_radius
        self.hetero = hetero
        self.keep_sample_obs = keep_sample_obs
        
        if min_dist is None:
            self.min_dist = float('-inf')
        else:
            self.min_dist = min_dist
        if max_dist is None:
            self.max_dist = float('inf')
        else:
            self.max_dist = max_dist
        
        # Initialize data structures
        self._setWorld()

    def isConnected(self,world0):
        sys.setrecursionlimit(10000)
        world0 = world0.copy()

        def firstFree(world0):
            for x in range(world0.shape[0]):
                for y in range(world0.shape[1]):
                    if world0[x,y]==0:
                        return x,y
        def floodfill(world,i,j):
            sx,sy=world.shape[0],world.shape[1]
            if(i<0 or i>=sx or j<0 or j>=sy):#out of bounds, return
                return
            if(world[i,j]==-1):return
            world[i,j] = -1
            floodfill(world,i+1,j)
            floodfill(world,i,j+1)
            floodfill(world,i-1,j)
            floodfill(world,i,j-1)

        i,j = firstFree(world0)
        floodfill(world0,i,j)
        if np.any(world0==0):
            return False
        else:
            return True

    def getObstacleMap(self):
        return (self.world.state==-1).astype(int)
    
    def _setWorld(self):
        def getConnectedRegion(world,regions_dict,x,y):
            sys.setrecursionlimit(1000000)
            '''returns a list of tuples of connected squares to the given tile
            this is memoized with a dict'''
            if (x,y) in regions_dict:
                return regions_dict[(x,y)]
            visited=set()
            sx,sy=world.shape[0],world.shape[1]
            work_list=[(x,y)]
            while len(work_list)>0:
                (i,j)=work_list.pop()
                if(i<0 or i>=sx or j<0 or j>=sy):#out of bounds, return
                    continue
                if(world[i,j]==-1):
                    continue#crashes
                if world[i,j]>0:
                    regions_dict[(i,j)]=visited
                if (i,j) in visited:continue
                visited.add((i,j))
                work_list.append((i+1,j))
                work_list.append((i,j+1))
                work_list.append((i-1,j))
                work_list.append((i,j-1))
            regions_dict[(x,y)]=visited
            return visited

        #RANDOMIZE THE POSITIONS OF AGENTS
        x = np.random.rand(10, 10)
        positions = np.where(x<0.5)
        idx = np.random.choice(len(positions[0]), size=(10,), replace=False)
        x[np.vstack(positions)[:,idx][0], np.vstack(positions)[:,idx][1]] = 0        
        
        size=np.random.choice(np.arange(self.SIZE[0], self.SIZE[1]+1))
        world = np.zeros(shape=(int(size),int(size))).astype(int)
        
        positions = np.where(world==0)
        idx = np.random.choice(len(positions[0]), size=(self.num_agents,), replace=False)
        world[np.vstack(positions)[:,idx][0], np.vstack(positions)[:,idx][1]] = np.arange(1,1+self.num_agents)
        
        #RANDOMIZE THE GOALS OF AGENTS
        goals = np.zeros(world.shape).astype(int)
        if self.connected:
            goal_counter = 1
            agent_regions=dict()     
            while goal_counter<=self.num_agents:
                agent_pos=agent_locations[goal_counter-1]
                valid_tiles=getConnectedRegion(world,agent_regions,agent_pos[0],agent_pos[1])
                x,y  = random.choice(list(valid_tiles))
                if(goals[x,y]==0 and world[x,y]!=-1):
                    goals[x,y]    = goal_counter
                    goal_counter += 1
        else:  
            for agent_id in range(1,self.num_agents+1):
                
                positions = np.where((world!=-1) & (goals==0))
                agent_pos=np.vstack(np.where(world==agent_id)).reshape(-1,1)
                valid_tiles = np.vstack(positions)
                dist2s = np.linalg.norm(valid_tiles-agent_pos, axis=0)
                good_tiles = (dist2s <= self.max_dist) & (dist2s >= self.min_dist)
                if good_tiles.sum() > 0:
                    x, y = random.choice(list(valid_tiles[:,good_tiles].T))
                    goals[x,y] = agent_id
                else:
                    # choose the nearest goal
                    idx = np.argsort(dist2s)[0]
                    x, y = valid_tiles[:,idx]
                    goals[x,y] = agent_id

            
            # idx = np.random.choice(len(positions[0]), size=(self.num_agents,), replace=False)
            # goals[np.vstack(positions)[:,idx][0], np.vstack(positions)[:,idx][1]] = np.arange(1,1+self.num_agents)
          
        try:
            prob=np.random.uniform(self.PROB[0],self.PROB[1])
        except:
            prob=self.PROB[0]        
        
        #RANDOMIZE THE STATIC OBSTACLES
        obs_world = -(np.random.rand(int(size),int(size))<prob).astype(int)
        if self.simple:
            obs_world[np.meshgrid(np.arange(start=0,stop=int(size),step=2),np.arange(start=1,stop=int(size),step=2))] = 0
            obs_world[np.meshgrid(np.arange(start=1,stop=int(size),step=2),np.arange(start=0,stop=int(size),step=2))] = 0
        obs_world[world!=0] = 0
        obs_world[goals!=0] = 0
        world = world + obs_world              
            
        self.initial_world = world
        self.initial_goals = goals
        self.world = self.absState(world,goals,self.space_dim,self.state_dim,
                                   self.obstacle_threshold,self.agent_threshold,self.goal_threshold,
                                   prob=prob,num_agents=self.num_agents,keep_sample_obs=self.keep_sample_obs)

        
    def _get_obs_lidar(self):
        # agents and obstacles are just be viewed as obstacles
        pass
        
        
    # Returns an observation of an agent
    def _get_obs(self, loop=False, clip=True, has_goal=False, share_weight=False, 
                       rgraph_a=False, rgraph_o=False, lidar=False):
        
        if lidar:
            return self._get_obs_lidar()
        
        a2a_index, a2a_attr, o2a_index, o2a_attr, g2a_index, g2a_attr = None, None, None, None, None, None
        agent_pos = torch.FloatTensor(self.world.agents)
        agent_origin_pos = torch.FloatTensor(self.world.agents)
        
        if self.angle_embed:
            agent_angle = agent_pos[:,-self.angle_dim:]
            if self.angle_dim ==0:
                pass
            elif self.angle_dim==1:
                agent_pos = torch.cat((agent_pos[:,:-self.angle_dim], torch.sin(agent_angle), torch.cos(agent_angle)), dim=-1)
            elif self.angle_dim==3:
                alpha = agent_angle[:,[0]]
                beta = agent_angle[:,[1]]
                gamma = agent_angle[:,[2]]
                agent_pos = torch.cat((agent_pos[:,:-self.angle_dim], 
                                       torch.sin(alpha), torch.cos(alpha), torch.sin(beta), torch.cos(beta)), dim=-1)
                                       
                                               # torch.cos(beta)*torch.cos(gamma),
                                               # torch.cos(beta)*torch.sin(gamma),
                                               # -torch.sin(beta),
                                               # torch.sin(alpha)*torch.sin(beta)*torch.cos(gamma)-torch.cos(alpha)*torch.sin(gamma),
                                               # torch.sin(alpha)*torch.sin(beta)*torch.sin(gamma)+torch.cos(alpha)*torch.cos(gamma),
                                               # torch.sin(alpha)*torch.cos(beta)), dim=-1)
            else:
                assert False
                
        feature_len_max = 3 + self.space_dim + (agent_pos.shape[1]-self.space_dim)*2        
        
        if rgraph_a:
            a2a_index = radius_graph(agent_pos[:,:self.space_dim], r=self.agent_obs_radius, loop=loop)
        else:
            a2a_index = knn_graph(agent_pos[:,:self.space_dim], self.agent_top_k, loop=loop)
            pairs = torch.FloatTensor(self.world.agents[:,:self.space_dim])[a2a_index]
            distance = (pairs[0] - pairs[1]).norm(dim=-1)
            a2a_index = a2a_index[:, distance<=self.agent_obs_radius]
        a2a_attr = (agent_pos[a2a_index[0,:]]-agent_pos[a2a_index[1,:]])[:,:self.space_dim]
        a2a_attr = torch.cat((agent_pos[a2a_index[0,:]][:,self.space_dim:], agent_pos[a2a_index[1,:]][:,self.space_dim:], a2a_attr), dim=-1)

        if share_weight:
            one_hot = np.array([[1,0,0]]*len(a2a_attr))
            a2a_attr = torch.cat((torch.FloatTensor(one_hot), a2a_attr), dim=-1)

        if len(self.world.obstacles) != 0:
            obstacle_pos = torch.FloatTensor(self.world.obstacles)
            if rgraph_o:
                indexes = radius(agent_pos[:,:2], obstacle_pos[:,:2], r=self.obstacle_obs_radius)
            else:
                indexes = knn(obstacle_pos[:,:2], agent_pos[:,:2], self.obstacle_top_k).flip(0)
                distance = (obstacle_pos[indexes[0],:2] - agent_pos[indexes[1],:2]).norm(dim=-1)
                indexes = indexes[:, distance<=self.obstacle_obs_radius]
            o2a_index = indexes
            o2a_attr = obstacle_pos[o2a_index[0,:]][:,:2]-agent_pos[o2a_index[1,:]][:,:2]
            if self.space_dim > 2:
                o2a_attr = torch.cat((o2a_attr, torch.zeros(len(agent_pos[o2a_index[1,:]]), self.space_dim-2),
                                      agent_pos[o2a_index[1,:]][:,self.space_dim:]), dim=-1)
            else:
                o2a_attr = torch.cat((agent_pos[o2a_index[1,:]][:,self.space_dim:], 
                                  o2a_attr), dim=-1)
            
            if share_weight:
                one_hot = np.array([[0,1,0]]*len(o2a_attr))
                o2a_attr = torch.cat((torch.FloatTensor(one_hot), o2a_attr), dim=-1)

        else:
            obstacle_pos = torch.zeros(0,2)
            o2a_index = torch.zeros(2,0).long()
            o2a_attr = torch.zeros(0,3+2+(agent_pos.shape[1]-self.space_dim))
            
        
        goals = self.world.agent_goals.copy()
        if clip:
            goals[:,:self.space_dim] = (goals[:,:self.space_dim]-self.world.agents[:,:self.space_dim]).clip(-6, 6)+self.world.agents[:,:self.space_dim]
        
        goal_pos = torch.FloatTensor(goals)
        g2a_index = torch.arange(self.num_agents).unsqueeze(0).repeat(2, 1).long()
        g2a_attr = goal_pos[g2a_index[0,:]][:,:self.space_dim]-agent_pos[g2a_index[1,:]][:,:self.space_dim]
        g2a_attr = torch.cat((agent_pos[g2a_index[1,:]][:,self.space_dim:], g2a_attr), dim=-1)

        if share_weight:
            one_hot = np.array([[0,0,1]]*self.num_agents)
            g2a_attr = torch.cat((torch.FloatTensor(one_hot), g2a_attr), dim=-1)
        
        if share_weight:
            # padding zero
            for attr_str in ['a2a_attr', 'o2a_attr', 'g2a_attr']:
                attr = eval(attr_str)
                if (attr is not None) and (attr.shape[1] < feature_len_max):
                    out = torch.cat((attr, torch.zeros(len(attr), feature_len_max-attr.shape[1])), dim=-1)
                    if attr_str=='a2a_attr':
                        a2a_attr = out
                    elif attr_str=='o2a_attr':
                        o2a_attr = out
                    elif attr_str=='g2a_attr':
                        g2a_attr = out
        
        # assign label to agents
        agent_x = torch.zeros(self.num_agents, 3)
        agent_x[:, 0] = 1
        obstacle_x = torch.zeros(len(self.world.obstacles), 3)
        obstacle_x[:, 1] = 1
        goal_x = torch.zeros(self.num_agents, 3)
        goal_x[:, 2] = 1
        
        if self.hetero:
            data = HeteroData()
            data['agent'].x, data['goal'].x = agent_x, goal_x
            data['agent'].pos, data['goal'].pos = agent_origin_pos, goal_pos
            data['agent', 'a_near_a', 'agent'].edge_index = a2a_index
            data['agent', 'a_near_a', 'agent'].edge_attr = a2a_attr
            data['goal', 'toward', 'agent'].edge_index = g2a_index
            data['goal', 'toward', 'agent'].edge_attr = g2a_attr              
            
            data['obstacle'].x = obstacle_x
            data['obstacle'].pos = obstacle_pos
            data['obstacle', 'o_near_a', 'agent'].edge_index = o2a_index
            data['obstacle', 'o_near_a', 'agent'].edge_attr = o2a_attr          
        else:
            data = Data()
            if has_goal:
                data.x = torch.cat((agent_x, obstacle_x, goal_x), dim=0)
                o2a_index[0, :] = o2a_index[0, :] + len(agent_x)
                g2a_index[0, :] = g2a_index[0, :] + len(agent_x) + len(obstacle_x)
                data.edge_index = torch.cat((a2a_index, o2a_index, g2a_index), dim=1)
                data.edge_attr = torch.cat((a2a_attr, o2a_attr, g2a_attr), dim=0)
            else:
                data.x = torch.cat((agent_x, obstacle_x), dim=0)
                o2a_index[0, :] = o2a_index[0, :] + len(agent_x)
                data.edge_index = torch.cat((a2a_index, o2a_index), dim=1)
                data.edge_attr = torch.cat((a2a_attr, o2a_attr), dim=0)
        
        return data
    

    def obs_from_pos(self, data, agent_pos, loop=False, clip=True, has_goal=False, share_weight=False, rgraph_a=False, rgraph_o=False):
        
        assert self.hetero
        a2a_index, a2a_attr, o2a_index, o2a_attr, g2a_index, g2a_attr = None, None, None, None, None, None
        
        data = data.to('cpu')
        agent_pos = agent_pos.to('cpu')
        agent_origin_pos = agent_pos.clone()
        if self.angle_embed:
            agent_angle = agent_pos[:,-self.angle_dim:]
            if self.angle_dim ==0:
                pass
            elif self.angle_dim==1:
                agent_pos = torch.cat((agent_pos[:,:-self.angle_dim], torch.sin(agent_angle), torch.cos(agent_angle)), dim=-1)
            elif self.angle_dim==3:
                alpha = agent_angle[:,[0]]
                beta = agent_angle[:,[1]]
                gamma = agent_angle[:,[2]]
                agent_pos = torch.cat((agent_pos[:,:-self.angle_dim], 
                                       torch.sin(alpha), torch.cos(alpha), torch.sin(beta), torch.cos(beta)), dim=-1)
            else:
                assert False
                
        feature_len_max = 3 + self.space_dim + (agent_pos.shape[1]-self.space_dim)*2        
                
        a2a_index = data['agent', 'a_near_a', 'agent'].edge_index
        a2a_attr = (agent_pos[a2a_index[0,:]]-agent_pos[a2a_index[1,:]])[:,:self.space_dim]
        a2a_attr = torch.cat((agent_pos[a2a_index[0,:]][:,self.space_dim:], agent_pos[a2a_index[1,:]][:,self.space_dim:], a2a_attr), dim=-1)
        if share_weight:
            one_hot = np.array([[1,0,0]]*len(a2a_attr))
            a2a_attr = torch.cat((torch.FloatTensor(one_hot), a2a_attr), dim=-1)

        if 'edge_index' in data['o_near_a']:
            o2a_index = data['obstacle', 'o_near_a', 'agent'].edge_index
            obstacle_pos = data['obstacle'].pos
            o2a_attr = obstacle_pos[o2a_index[0,:]][:,:2]-agent_pos[o2a_index[1,:]][:,:2]
            if self.space_dim > 2:
                o2a_attr = torch.cat((o2a_attr, torch.zeros(len(agent_pos[o2a_index[1,:]]), self.space_dim-2),
                                      agent_pos[o2a_index[1,:]][:,self.space_dim:]), dim=-1)
            else:
                o2a_attr = torch.cat((agent_pos[o2a_index[1,:]][:,self.space_dim:], 
                                  o2a_attr), dim=-1)
            if share_weight:
                one_hot = np.array([[0,1,0]]*len(o2a_attr))
                o2a_attr = torch.cat((torch.FloatTensor(one_hot), o2a_attr), dim=-1)
        
        goal_pos = data['goal'].pos
        g2a_index = data['goal', 'toward', 'agent'].edge_index
        g2a_attr = goal_pos[g2a_index[0,:]][:,:self.space_dim]-agent_pos[g2a_index[1,:]][:,:self.space_dim]
        g2a_attr = torch.cat((agent_pos[g2a_index[1,:]][:,self.space_dim:], g2a_attr), dim=-1)
        if share_weight:
            one_hot = np.array([[0,0,1]]*len(g2a_attr))
            g2a_attr = torch.cat((torch.FloatTensor(one_hot), g2a_attr), dim=-1)
        
        if share_weight:
            # padding zero
            for attr_str in ['a2a_attr', 'o2a_attr', 'g2a_attr']:
                attr = eval(attr_str)
                if (attr is not None) and (attr.shape[1] < feature_len_max):
                    out = torch.cat((attr, torch.zeros(len(attr), feature_len_max-attr.shape[1])), dim=-1)
                    if attr_str=='a2a_attr':
                        a2a_attr = out
                    elif attr_str=='o2a_attr':
                        o2a_attr = out
                    elif attr_str=='g2a_attr':
                        g2a_attr = out 
        
        if self.hetero:
            data['agent'].pos, data['goal'].pos = agent_origin_pos, goal_pos
            data['agent', 'a_near_a', 'agent'].edge_attr = a2a_attr
            data['goal', 'toward', 'agent'].edge_attr = g2a_attr              
            if 'edge_index' in data['o_near_a']:
                data['obstacle'].pos = obstacle_pos
                data['obstacle', 'o_near_a', 'agent'].edge_attr = o2a_attr          
        else:
            assert False
        
        return data.to('cuda:0')
    
    def sample_edge(self, edge_index, k, target_node_size):
        adj = SparseTensor.from_edge_index(edge_index)
        colptr, row, _ = adj.csc()
        _, row, col, output_edge = neighbor_sample(colptr, row, torch.arange(target_node_size), [k], False, True)
        return torch.stack([row, col], dim=0)
    
    # Returns an observation of an agent
    def _get_obs_random_k(self, loop=False, clip=True, has_goal=False, share_weight=True, rgraph_a=True, rgraph_o=True,
                                n_sub_o=None, n_sub_a=None, iteration=None, **kwargs):
        if n_sub_o is None:
            n_sub_o = (2,2)
        
        if n_sub_a is None:
            n_sub_a = (2,2)
            
        if iteration is None:
            iteration = 1
        
        data = self._get_obs(loop=loop, clip=clip, share_weight=share_weight, rgraph_a=rgraph_a, rgraph_o=rgraph_o)
        data['agent'].n_id = torch.arange(self.num_agents)
        
        for _ in range(iteration):
        
            if 'edge_index' in data['a_near_a']:
                a_edge_index = data['a_near_a'].edge_index
                neighbors_a = [(a_edge_index[1,:]==k) for k in range(self.num_agents)]
                a_candidates = [np.arange(a_edge_index.shape[1])[neighbors_a[k].data.numpy()] for k in range(self.num_agents)]
                cover_a_bool = torch.BoolTensor([0]*a_edge_index.shape[1])
            else:
                neighbors_a = None

            if 'edge_index' in data['o_near_a']:
                o_edge_index = data['o_near_a'].edge_index
                neighbors_o = [(o_edge_index[1,:]==k) for k in range(self.num_agents)]
                o_candidates = [np.arange(o_edge_index.shape[1])[neighbors_o[k].data.numpy()] for k in range(self.num_agents)]
                cover_o_bool = torch.BoolTensor([0]*o_edge_index.shape[1])
            else:
                neighbors_o = None

            while True:
                
                sub_data = data.clone()
                if neighbors_a is not None:
                    edge_index_a_current = [np.random.choice(a, min(len(a), np.random.randint(n_sub_a[0], 1+n_sub_a[1])), replace=False) for a in a_candidates]
                    edges_a = np.concatenate(edge_index_a_current)
                
                    sub_data['a_near_a'].edge_index = data['a_near_a'].edge_index[:, edges_a]
                    sub_data['a_near_a'].edge_attr = data['a_near_a'].edge_attr[edges_a, :]  
                    cover_a_bool[edges_a] = True
                    cover_a = cover_a_bool.all()                
                else:
                    cover_a = True
                
                if neighbors_o is not None:
                    edge_index_o_current = [np.random.choice(o, min(len(o), np.random.randint(n_sub_o[0], 1+n_sub_o[1])), replace=False) for o in o_candidates]
                    edges_o = np.concatenate(edge_index_o_current)
                    
                    sub_data['o_near_a'].edge_index = data['o_near_a'].edge_index[:, edges_o]
                    sub_data['o_near_a'].edge_attr = data['o_near_a'].edge_attr[edges_o, :]  
                    cover_o_bool[edges_o] = True
                    cover_o = cover_o_bool.all()                       
                else:
                    cover_o = True

                yield sub_data

                if cover_a and cover_o:
                    break
    
    
#     # Returns an observation of an agent
#     def _get_obs_group_k(self, preferred_cluster, loop=False, clip=True, rgraph_a=True, rgraph_o=False, **kwargs):
#         data = self._get_obs(loop=loop, clip=clip, rgraph_a=rgraph_a, rgraph_o=rgraph_o)
#         k_data = self._get_obs(loop=loop, clip=clip)
        
#         def merge(arg1, arg2):
#             nodes1, edge1, edge_attr1, center1, edge_mask1 = arg1
#             nodes2, edge2, edge_attr2, center2, edge_mask2 = arg2
#             if nodes1 & nodes2:
#                 if np.random.rand() < 0.5:
#                     return nodes1, edge1, edge_attr1, center1, edge_mask1
#                 else:
#                     return nodes2, edge2, edge_attr2, center2, edge_mask2
#             else:
#                 return nodes1 | nodes2, torch.cat((edge1, edge2), dim=-1), torch.cat((edge_attr1, edge_attr2), dim=0), center1 | center2, edge_mask1 | edge_mask2
        
#         def find_cluster(node_id, k_edge_index, edge_index, edge_attr):
#             edges = k_edge_index[:, k_edge_index[1,:]==node_id]
#             nodes = torch.unique(edges, sorted=False)
#             node_mask = index_to_mask(nodes, size=self.num_agents)
#             edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
#             sub_edge_index = edge_index[:, edge_mask]
#             sub_edge_attr = edge_attr[edge_mask, :]
#             return set(nodes.data.numpy()), sub_edge_index, sub_edge_attr, {node_id}, edge_mask
        
#         initial_cluster = map(find_cluster, preferred_cluster, [k_data['a_near_a'].edge_index]*len(preferred_cluster), [data['a_near_a'].edge_index]*len(preferred_cluster), [data['a_near_a'].edge_attr]*len(preferred_cluster))
        
#         affected_nodes, edge_index_a, edge_attr_a, prefer_nodes, edge_mask = reduce(merge, initial_cluster)
        
#         data['agent', 'a_near_a', 'agent'].edge_index = edge_index_a
#         data['agent', 'a_near_a', 'agent'].edge_attr = edge_attr_a
#         return data, affected_nodes, prefer_nodes, edge_mask

    # Resets environment
    def _reset(self):
        self.finished = False

        # Initialize data structures
        self._setWorld()

    # Executes an action by an agent
    def step(self, action_input, obs_config=None, bound=False):
        assert len(action_input) == self.num_agents, 'Action input should be a tuple with the form (num_agents, action_dim)'
        assert len(action_input[0]) == self.action_dim, 'Action input should be a tuple with the form (num_agents, action_dim)'
        
        if obs_config is None:
            obs_config = {}
        
        prev_status = self.world.get_status()
        prev_o = self._get_obs(**obs_config)

        # Check action input

        next_pos = self.dynamic(self.world.agents, action_input)
        dist = np.linalg.norm(self.world.agents[:, :self.space_dim]-self.world.agent_goals[:, :self.space_dim], axis=-1)
        next_dist = np.linalg.norm(next_pos[:, :self.space_dim]-self.world.agent_goals[:, :self.space_dim], axis=-1)
        displacement = dist - next_dist
        
        self.world.agents = next_pos
        
        if bound:
            self.world.agents[:, :2] = np.clip(self.world.agents[:, :2], 0, len(self.world.state))

        # Perform observation
        next_o = self._get_obs(**obs_config) 

        # Done?
        next_status = self.world.get_status()
        done = self.world.done(status=next_status)
        
        self.finished |= done
        
        rewards = [-10*('danger_agent' in s)-10*('danger_obstacle' in s) for s in next_status]
        rewards = np.array(rewards) + 10*displacement + 10*done - 0.1

        info = {
                "action": action_input,
                "prev_free": ['free' in s for s in prev_status],
                "prev_safe": ['safe' in s for s in prev_status],
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
        
        if K1==0:
            score = -K2 * goal_force.reshape(-1)
            score = score.reshape(self.num_agents, n_candidates)
            return score
        
        if (len(self.world.obstacles)!=0):
            dist2obs = cdist(self.world.obstacles, next_pos.reshape(-1, self.state_dim)[:, :2]).min(axis=0)  # (num_agents x n_candidates)
        else:
            dist2obs = 100 * np.ones((self.num_agents * n_candidates))
        
        if ignore_agent:
            dist = dist2obs
        else:
            dist = cdist(next_pos.reshape(-1, self.state_dim)[:, :self.space_dim], next_pos.reshape(-1, self.state_dim)[:, :self.space_dim]) # (num_agents x n_candidates, num_agents x n_candidates)
            A = np.ones((n_candidates, n_candidates))
            eye = block_diag(*[A for _ in range(self.num_agents)])
            dist = dist * (1-eye) + 1000 * eye
            dist = dist.reshape(n_candidates*self.num_agents, self.num_agents, n_candidates)
            dist = dist.min(axis=-1)  # (num_agents x n_candidates) x num_agents
            dist = dist.min(axis=-1)  # (num_agents x n_candidates)
            dist = np.minimum(dist2obs, dist)
        dist = dist*(dist>0.1)+0.1*(dist<0.1)  # numerical stability
        D = self.obstacle_threshold + 0.1
        obs_force = (1.0 / dist - 1.0 / D)**2
        obs_force = (obs_force)*(dist < D)
        
        score = K1 * obs_force - K2 * goal_force.reshape(-1)
        score = score.reshape(self.num_agents, n_candidates)
        return score
    
    
    def initColors(self):
        c={a+1:hsv_to_rgb(np.array([a/float(self.num_agents),1,1])) for a in range(self.num_agents)}
        return c    
    
    @abstractmethod
    def _render(self):
        pass
    
    @abstractmethod
    def _render_with_contour(self, xys, values):
        pass
    
    @abstractmethod
    def dynamic(self, pos, action):
        pass