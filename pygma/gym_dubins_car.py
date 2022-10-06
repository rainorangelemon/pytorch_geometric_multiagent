import numpy as np
from collections import OrderedDict
import sys
from matplotlib.colors import hsv_to_rgb
import random
import math
import copy
import numpy as np
from scipy.spatial.distance import cdist
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch_cluster import radius_graph, radius, knn_graph
from collections import defaultdict
from torch_sparse import coalesce
from torch_geometric.utils import remove_self_loops
from torch_geometric.data import HeteroData
from scipy.linalg import block_diag

TOP_K = 5
AGENT_OBS_RADIUS = 3.0
OBSTACLE_OBS_RADIUS = 2.0
AGENT_DISTANCE_THRESHOLD = 0.3
OBSTACLE_DISTANCE_THRESHOLD = 0.3
GOAL_THRESHOLD = 0.45
STEER = 2 * np.pi / 3
n_candidates = 2000
    
def find_path(grid, start, goal):
    openSet = []
    cameFrom = dict()

    gScore = defaultdict(lambda:float('inf'))
    gScore[start] = 0

    fScore = defaultdict(lambda:float('inf'))
    fScore[start] = np.linalg.norm(np.array(start)-np.array(goal))
    
    heapq.heappush(openSet, (fScore[start], start))

    while len(openSet)!=0:
        fscore, current = heapq.heappop(openSet)
        if current == goal:
            # construct path
            path = [current]
            father = current
            while father != start:
                father = cameFrom[father]
                path.append(father)
            return path[::-1]

        for dir_ in [[0,1],[0,-1],[-1,0],[1,0]]:
            neighbor = (current[0]+dir_[0], current[1]+dir_[1])
            if (len(grid)>neighbor[0]>=0) and (len(grid[0])>neighbor[1]>=0) and (grid[neighbor[0],neighbor[1]]!=1):
                tentative_gScore = gScore[current] + 1
                if tentative_gScore < gScore[neighbor]:
                    cameFrom[neighbor] = current
                    gScore[neighbor] = tentative_gScore
                    fScore[neighbor] = tentative_gScore + np.linalg.norm(np.array(neighbor)-np.array(goal))
                    for idx in range(len(openSet)):
                        if openSet[idx][1]==neighbor:
                            openSet.pop(idx)
                            break
                    heapq.heappush(openSet, (fScore[neighbor], neighbor))

    return []



def get_astar_action(env):
    try:
        actions = []
        for agent in range(env.num_agents):
            if env.world.getPos(agent)[0]>=len(env.getObstacleMap()):
                actions.append([-1,0])
            elif env.world.getPos(agent)[1]>=len(env.getObstacleMap()[0]):
                actions.append([0,-1])
            elif env.world.getPos(agent)[0]<0:
                actions.append([1,0])
            elif env.world.getPos(agent)[1]<0:
                actions.append([0,1])
            elif env.getObstacleMap()[env.world.getPos(agent)]==1:
                raise NoSolutionError                
            else:
                path = find_path(env.getObstacleMap(),tuple(env.world.getPos(agent)),tuple(env.world.getGoal(agent)))
                path = np.array(path).reshape((-1, 2))
                if len(path)==1:
                    actions.append([0,0])
                else:
                    actions.append(path[1]-path[0])
        return np.array(actions)
    except Exception as e:
        return get_simple_direction(env)


    
def get_simple_direction(env):
    direction = np.array(env.world.agent_goals[:,:2])-np.array(env.world.agents[:,:2])
    direction = direction.astype(float)
    theta = np.arctan2(direction[:,1], direction[:,0])
#     length = np.linalg.norm(direction)
#     theta = np.zeros(theta.shape)*(length<1)+theta*(length>=1)
    theta1 = (((theta<0)*theta+(theta>=0)*(theta-2*np.pi))- env.world.agents[:,2])/2
    theta2 = (((theta<0)*(theta+2*np.pi)+(theta>=0)*theta)- env.world.agents[:,2])/2
    dtheta = (np.abs(theta1)<np.abs(theta2))*theta1+(np.abs(theta1)>=np.abs(theta2))*theta2
    return dtheta.reshape((-1,1))      

'''
    Observation: (position maps of current agent, current goal, other agents, other goals, obstacles)
        
    Action space: (Tuple)
        agent_id: positive integer
        action: {0:STILL, 1:MOVE_NORTH, 2:MOVE_EAST, 3:MOVE_SOUTH, 4:MOVE_WEST,
        5:NE, 6:SE, 7:SW, 8:NW}
    Reward: ACTION_COST for each action, GOAL_REWARD when robot arrives at target
'''
ACTION_COST, IDLE_COST, GOAL_REWARD, COLLISION_REWARD,FINISH_REWARD,BLOCKING_COST = -0.3, -.5, 0.0, -2, 20.,-1.
opposite_actions = {0: -1, 1: 3, 2: 4, 3: 1, 4: 2, 5: 7, 6: 8, 7: 5, 8: 6}
JOINT = False # True for joint estimation of rewards for closeby agents
dirDict = {0:(0,0),1:(0,1),2:(1,0),3:(0,-1),4:(-1,0),5:(1,1),6:(1,-1),7:(-1,-1),8:(-1,1)}
actionDict={v:k for k,v in dirDict.items()}
class State(object):
    '''
    State.
    Implemented as 2 2d numpy arrays.
    first one "state":
        static obstacle: -1
        empty: 0
        agent = positive integer (agent_id)
    second one "goals":
        agent goal = positive int(agent_id)
    third one "agents":
        the exact position of agents
    '''
    def __init__(self, world0, goals, num_agents=1):
        assert(len(world0.shape) == 2 and world0.shape==goals.shape)
        self.state                    = world0.copy()
        self.goals                    = goals.copy()
        self.num_agents               = num_agents
        self.obstacles, self.agents, self.agent_goals = self.scanForAgents()
        self.obstacles = np.array(self.obstacles).astype(float)
        self.agent_goals = np.array(self.agent_goals).astype(float)
        self.agents = np.array(self.agents).astype(float)        
        assert(self.agents.shape == (num_agents, 3))

    def scanForAgents(self):
        obstacles = []
        agents = [(-1,-1,0) for i in range(self.num_agents)]     
        agent_goals = [(-1,-1) for i in range(self.num_agents)]        
        for i in range(self.state.shape[0]):
            for j in range(self.state.shape[1]):
                if(self.state[i,j]>0):
                    agents[self.state[i,j]-1] = (i+0.5,j+0.5,0.)
                if(self.goals[i,j]>0):
                    agent_goals[self.goals[i,j]-1] = (i+0.5,j+0.5,0.)
                if(self.state[i,j]==-1):
                    obstacles.append((i+0.5,j+0.5))
        return obstacles, agents, agent_goals

    def act(self, dtheta, agent_id):
        ax, ay, theta = self.agents[agent_id][0], self.agents[agent_id][1], self.agents[agent_id][2]
        
        dx, dy = 0.05*np.cos(theta), 0.05*np.sin(theta)
        theta = theta + STEER*dtheta
        if theta > np.pi:
            theta = theta - 2*np.pi
        elif theta < -np.pi:
            theta = theta + 2*np.pi
        new_pos = [ax+dx, ay+dy, theta]
        
        self.agents[agent_id] = new_pos


    def getPos(self, agent_id):
        return tuple(np.floor(self.agents[agent_id, :2]).astype(int))

    def getGoal(self, agent_id):
        return tuple(np.floor(self.agent_goals[agent_id, :2]).astype(int))
    
    def getDir(self,action):
        return dirDict[action]
    def getAction(self,direction):
        return actionDict[direction]
    
    def get_status(self):
        status = []
        agents = np.array(self.agents)
        obstacles = self.obstacles
        if len(agents) > 1:
            distance = cdist(agents[:, :2], agents[:, :2])
            distance_nearest_agent = np.sort(distance, axis=-1)[:, 1]    
        else:
            distance_nearest_agent = 100*np.ones((len(self.agents),))
        if len(obstacles) > 0:
            distance_obs = cdist(agents[:, :2], np.array(obstacles)[:, :2])
            distance_nearest_obs = np.sort(distance_obs, axis=-1)[:, 0]
        else:
            distance_nearest_obs = 100*np.ones((len(self.agents),))
        dist2goal = np.linalg.norm(np.array(self.agents[:, :2])-np.array(self.agent_goals[:, :2]), axis=-1)

        status = ['' for _ in range(self.num_agents)]
        status = [s+'danger_obstacle' if d<OBSTACLE_DISTANCE_THRESHOLD else s for s, d in zip(status, distance_nearest_obs)]
        status = [s+'danger_agent' if d<AGENT_DISTANCE_THRESHOLD else s for s, d in zip(status, distance_nearest_agent)]
        status = [s+'done' if d<GOAL_THRESHOLD else s for s, d in zip(status, dist2goal)]
        status = [s+'free' if ('danger' not in s) else s for s in status]
        
        return status
    
    def sample_agents(self, n_agents, prob=0.1):
        if np.random.uniform() < prob:
            agents = np.random.uniform(-0.3, 0.3, size=(n_agents, 3))
            agents = agents + self.agent_goals
            agents[:, :2] = agents[:, :2].clip(0, len(self.state))
            agents[:, 2] = agents[:, 2].clip(-np.pi, np.pi)
            return agents
        else:
            agents = np.random.uniform(0, len(self.state), size=(n_agents, 3))
            agents[:, 2] = np.random.uniform(-np.pi, np.pi, size=(n_agents,))
            return agents
    

    # Compare with a plan to determine job completion
    def done(self, status=None):
        if status is None:
            status = self.get_status()
        return np.sum(['done' in s for s in status])==len(self.agents)


class DubinsCarEnv:
    
    action_dim=1
    space_dim = 2
    state_dim=3
    
    def getFinishReward(self):
        return FINISH_REWARD

    # Initialize env
    def __init__(self, num_agents=1, observation_size=6,mode='barrier',world0=None, goals0=None, SIZE=(10,40), PROB=(0,.5), FULL_HELP=False,blank_world=False):
        """
        Args:
            SIZE: size of a side of the square grid
            PROB: range of probabilities that a given block is an obstacle
            FULL_HELP
        """
        # Initialize member variables
        self.num_agents        = num_agents 
        #a way of doing joint rewards
        self.observation_size  = observation_size
        self.SIZE              = SIZE
        self.PROB              = PROB
        self.FULL_HELP         = FULL_HELP
        self.finished          = False

        # Initialize data structures
        self._setWorld(world0,goals0,blank_world=blank_world)
        self.action_dim = 1
        self.state_dim = 3
        self.space_dim = 2
        self.viewer           = None
        self.mode = mode
        assert self.mode=='lyapunov' or self.mode=='barrier'

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
    
    def _setWorld(self, world0=None, goals0=None,blank_world=False):
        #blank_world is a flag indicating that the world given has no agent or goal positions 
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
        #defines the State object, which includes initializing goals and agents
        #sets the world to world0 and goals, or if they are None randomizes world
        if not (world0 is None):
            if goals0 is None and not blank_world:
                raise Exception("you gave a world with no goals!")
            if blank_world:
                #RANDOMIZE THE POSITIONS OF AGENTS
                agent_counter = 1
                agent_locations=[]
                while agent_counter<=self.num_agents:
                    x,y       = np.random.randint(0,world0.shape[0]),np.random.randint(0,world0.shape[1])
                    if(world0[x,y] == 0):
                        world0[x,y]=agent_counter
                        agent_locations.append((x,y))
                        agent_counter += 1   
                #RANDOMIZE THE GOALS OF AGENTS
                goals0 = np.zeros(world0.shape).astype(int)
                goal_counter = 1
                agent_regions=dict()  
                while goal_counter<=self.num_agents:
                    agent_pos=agent_locations[goal_counter-1]
                    valid_tiles=getConnectedRegion(world0,agent_regions,agent_pos[0],agent_pos[1])#crashes
                    x,y  = random.choice(list(valid_tiles))
                    if(goals0[x,y]==0 and world0[x,y]!=-1):
                        goals0[x,y]    = goal_counter
                        goal_counter += 1
                self.initial_world = world0.copy()
                self.initial_goals = goals0.copy()
                self.world = State(self.initial_world,self.initial_goals,self.num_agents)
                return
            self.initial_world = world0
            self.initial_goals = goals0
            self.world = State(world0,goals0,self.num_agents)
            return

        #otherwise we have to randomize the world
        #RANDOMIZE THE STATIC OBSTACLES
        size=np.random.choice([self.SIZE[0],self.SIZE[0]*.5+self.SIZE[1]*.5,self.SIZE[1]],p=[.5,.25,.25])
        while True:
            try:
                prob=np.random.triangular(self.PROB[0],.33*self.PROB[0]+.66*self.PROB[1],self.PROB[1])
            except:
                prob=self.PROB[0]
            world     = -(np.random.rand(int(size),int(size))<prob).astype(int)
            if (size*size+world.sum()) >= self.num_agents:
                break

        #RANDOMIZE THE POSITIONS OF AGENTS
        agent_counter = 1
        agent_locations=[]
        while agent_counter<=self.num_agents:
            x,y       = np.random.randint(0,world.shape[0]),np.random.randint(0,world.shape[1])
            if(world[x,y] == 0):
                world[x,y]=agent_counter
                agent_locations.append((x,y))
                agent_counter += 1        
        
        #RANDOMIZE THE GOALS OF AGENTS
        goals = np.zeros(world.shape).astype(int)
        goal_counter = 1
        agent_regions=dict()     
        while goal_counter<=self.num_agents:
            agent_pos=agent_locations[goal_counter-1]
            valid_tiles=getConnectedRegion(world,agent_regions,agent_pos[0],agent_pos[1])
            x,y  = random.choice(list(valid_tiles))
            if(goals[x,y]==0 and world[x,y]!=-1):
                goals[x,y]    = goal_counter
                goal_counter += 1
        self.initial_world = world
        self.initial_goals = goals
        self.world = State(world,goals,num_agents=self.num_agents)

    # Returns an observation of an agent
    def _get_obs(self, mode=None, loop=False, clip=False):
        data = HeteroData()
        data['agent'].x = torch.FloatTensor(self.world.agents)
        agent_to_agent = data['agent', 'a_near_a', 'agent'].edge_index = knn_graph(data['agent'].x[:,:self.space_dim], TOP_K, loop=loop)
        pairs = torch.FloatTensor(self.world.agents[:,:self.space_dim])[agent_to_agent]
        distance = (pairs[0] - pairs[1]).norm(dim=-1)
        agent_to_agent = agent_to_agent[:, distance<AGENT_OBS_RADIUS]
        edges = data['agent', 'a_near_a', 'agent'].edge_index = agent_to_agent
        data['agent', 'a_near_a', 'agent'].edge_attr = (data['agent'].x[edges[0,:]]-data['agent'].x[edges[1,:]])[:,:self.space_dim]
        data['agent', 'a_near_a', 'agent'].edge_attr = torch.cat((data['agent'].x[edges[0,:]][:,self.space_dim:], data['agent'].x[edges[1,:]][:,self.space_dim:], data['agent', 'a_near_a', 'agent'].edge_attr), dim=-1)

        if len(self.world.obstacles) != 0:
            data['obstacle'].x = torch.FloatTensor(self.world.obstacles)
            edges = data['obstacle', 'o_near_a', 'agent'].edge_index = radius(data['agent'].x[:,:self.space_dim], data['obstacle'].x[:,:self.space_dim], r=OBSTACLE_OBS_RADIUS)
            data['obstacle', 'o_near_a', 'agent'].edge_attr = data['obstacle'].x[edges[0,:]][:,:self.space_dim]-data['agent'].x[edges[1,:]][:,:self.space_dim]
            data['obstacle', 'o_near_a', 'agent'].edge_attr = torch.cat((data['agent'].x[edges[1,:]][:,self.space_dim:], data['obstacle', 'o_near_a', 'agent'].edge_attr), dim=-1)
        else:
            data['obstacle'].x = torch.FloatTensor([[100,100]])
            
        
        goals = self.world.agent_goals
        clip_goal = (goals-self.world.agents).clip(-6, 6)+self.world.agents
        data['goal'].x = torch.FloatTensor(self.world.agent_goals)
        edges = data['goal', 'toward', 'agent'].edge_index = torch.arange(self.num_agents).unsqueeze(0).repeat(2, 1).long()
        data['goal', 'toward', 'agent'].edge_attr = data['goal'].x[edges[0,:]][:,:self.space_dim]-data['agent'].x[edges[1,:]][:,:self.space_dim]
        
        data['agent'].x = torch.FloatTensor([[1]]*self.num_agents)
        if len(self.world.obstacles) != 0:
            data['obstacle'].x = torch.FloatTensor([[1]]*len(self.world.obstacles))
        else:
            data['obstacle'].x = torch.FloatTensor([[1]])
        data['goal'].x = torch.FloatTensor([[1]]*self.num_agents)
        
        return data
        

#     # Returns an observation of an agent
#     def _get_obs(self, loop=True, mode=None, clip=True):
#         if mode==None:
#             mode = self.mode
#         else:
#             assert mode=='lyapunov' or mode=='barrier'
#             pass
        
#         agents = np.array(self.world.agents)
#         obstacles = np.array(self.world.obstacles)
        
#         # add obstacles nodes
#         if (mode=='barrier') and (len(obstacles)!=0):
#             pad_obstacles = np.zeros((len(obstacles), self.state_dim))
#             pad_obstacles[:, :len(obstacles[0])] = obstacles
#             nodes = np.vstack((pad_obstacles, agents))
#         else:
#             obstacles = np.array([])
#             nodes = agents
#         labels = [[1,0,0]] * len(obstacles) + [[0,1,0]] * len(agents)        
       
#         edges = []
#         agent_infos = []
        
#         obs_to_agent = agent_to_agent = torch.arange(self.num_agents).reshape((1,-1)).repeat(2,1)+len(obstacles)
#         if len(obstacles) != 0:
#             new_obs_to_agent = radius(torch.FloatTensor(agents[:,:2]), torch.FloatTensor(obstacles[:,:2]), r=OBSTACLE_OBS_RADIUS)
#             new_obs_to_agent[1, :] = new_obs_to_agent[1, :]+len(obstacles)
#             obs_to_agent = torch.cat((obs_to_agent, new_obs_to_agent), dim=-1)
#         if len(agents) > 1:
#             agent_to_agent = knn_graph(torch.FloatTensor(agents[:,:2]), TOP_K, loop=loop)
#             # agent_to_agent = radius_graph(torch.FloatTensor(agents), r=2, loop=True)
#             # ↓ filter out distant agents
#             pairs = torch.FloatTensor(agents[:,:2])[agent_to_agent]
#             distance = (pairs[0] - pairs[1]).norm(dim=-1)
#             agent_to_agent = agent_to_agent[:, distance<AGENT_OBS_RADIUS]
#             agent_to_agent = agent_to_agent+len(obstacles)        
        
#         # nodes_ = obstacles if mode=='lyapunov' else nodes
#         # if len(nodes_) != 0:
#         #     for agent in range(self.num_agents):
#         #         idxs = np.where(np.all(((self.observation_size // 2 - self.observation_size) < ([agents[agent, :2]] - nodes_[:, :2]),
#         #            ([agents[agent, :2]] - nodes_[:, :2]) <= (self.observation_size // 2)), axis=(0,2)))[0]
#         #         edges.extend([[idx, agent+len(obstacles)] for idx in idxs])
#         #         edges.append([agent+len(obstacles), agent+len(obstacles)])
#         # else:  # add self-loop
#         #     for agent in range(self.num_agents):
#         #         edges.append([agent+len(obstacles), agent+len(obstacles)])
        
#         if mode == 'lyapunov':
#             edge_index = torch.arange(self.num_agents).reshape((1,-1)).repeat(2,1)
#         elif mode == 'barrier':
#             edge_index = torch.cat((obs_to_agent, agent_to_agent), dim=-1)
#         edges = edge_index.T.numpy().tolist()
        
#         if mode=='lyapunov':
#             for agent in range(self.num_agents):
#                 agent_id = agent
#                 dx=self.world.agents[agent_id][0]-self.world.agent_goals[agent_id][0]
#                 dy=self.world.agents[agent_id][1]-self.world.agent_goals[agent_id][1]
#                 mag=(dx**2+dy**2)**.5
#                 if mag!=0:
#                     dx=dx/mag
#                     dy=dy/mag
#                 goal = self.world.agent_goals[agent]
#                 agent_cur = self.world.agents[agent]
#                 diff = self.world.agent_goals[agent]-self.world.agents[agent]
#                 if clip:
#                     nodes = np.vstack((nodes, agent_cur+diff.clip(-7, 7)))                 
#                 else:
#                     nodes = np.vstack((nodes, self.world.agent_goals[agent]))                 
#                 edges.append([len(nodes)-1,agent+len(obstacles)])
#                 labels.append([0,0,1])
#                 agent_infos.append([dx,dy,mag])
            
#         labels = np.array(labels, dtype=float)
#         edges = np.array(edges).T
#         edges = torch.LongTensor(edges)
#         try:
#             edges, _ = coalesce(edges, None, len(nodes), len(nodes))
#         except:
#             print(edges)
#             raise
        
#         return  {"x": torch.FloatTensor(nodes),
#                  "edge_index": torch.LongTensor(edges),
#                  "label": torch.FloatTensor(labels),
#                  "goal": torch.FloatTensor(self.world.agent_goals)}

    # Resets environment
    def _reset(self,world0=None,goals0=None):
        self.finished = False

        # Initialize data structures
        self._setWorld(world0, goals0)

    # def _complete(self):
    #     return self.world.done()

        
    # Executes an action by an agent
    def step(self, action_input, bound=False):
        assert len(action_input) == self.num_agents, 'Action input should be a tuple with the form (num_agents, 1)'
        assert len(action_input[0]) == 1, 'Action input should be a tuple with the form (num_agents, 1)'
        
#         next_targets = self.world.agents + get_simple_direction(self)
        
        prev_status = self.world.get_status()
        prev_o = self._get_obs()

        # Check action input

        next_pos = dynamic(self.world.agents, action_input)
        self.world.agents = next_pos
        
        if bound:
            self.world.agents[:, :2] = np.clip(self.world.agents[:, :2], 0, len(self.world.state))
            
        # Perform observation
        next_o = self._get_obs() 

        # Done?
        next_status = self.world.get_status()
        done = self.world.done(status=next_status)
        
        self.finished |= done
        
        rewards = [-2*('danger_agent' in s)-2*('danger_obstacle' in s) + 1*('done' in s) for s in next_status]
        
        info = {
                "action": action_input,
                "prev_free": ['free' in s for s in prev_status],
                "prev_danger": ['free' not in s for s in prev_status],
                "next_goal": ['done' in s for s in prev_status],
                "next_free": ['free' in s for s in next_status],
                "next_danger": ['free' not in s for s in next_status],
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
        
        next_pos = dynamic(next_pos.reshape(-1, self.state_dim), actions.reshape(-1, self.action_dim))
        next_pos = next_pos.reshape((self.num_agents, n_candidates, self.state_dim))
        
        
        if len(self.world.obstacles)!=0:
            dist2obs = cdist(self.world.obstacles, next_pos.reshape(-1, self.state_dim)[:, :2]).min(axis=0)  # (num_agents x n_candidates)
        else:
            dist2obs = 100 * np.ones((self.num_agents * n_candidates))
        
        if ignore_agent:
            dist = dist2obs
        else:
            dist = cdist(next_pos.reshape(-1, self.state_dim)[:, :2], next_pos.reshape(-1, self.state_dim)[:, :2]) # (num_agents x n_candidates, num_agents x n_candidates)
            A = np.ones((n_candidates, n_candidates))
            eye = block_diag(*[A for _ in range(self.num_agents)])
            dist = dist * (1-eye) + 1000 * eye
            dist = dist.reshape(n_candidates*self.num_agents, self.num_agents, n_candidates)
            dist = dist.min(axis=-1)  # (num_agents x n_candidates) x num_agents
            dist = dist.min(axis=-1)  # (num_agents x n_candidates)
            dist = np.minimum(dist2obs, dist)
        dist = dist*(dist>0.1)+0.1*(dist<0.1)  # numerical stability
        D = OBSTACLE_DISTANCE_THRESHOLD + 0.1
        obs_force = (1.0 / dist - 1.0 / D)**2
        obs_force = (obs_force)*(dist < D)
        goal_force = ((next_pos[:, :, :2] - np.expand_dims(self.world.agent_goals, axis=1)[:, :, :2])**2).sum(axis=-1)  # num_agents x n_candidates
        score = K1 * obs_force - K2 * goal_force.reshape(-1)
        score = score.reshape(self.num_agents, n_candidates)
        return score
    
    
    def initColors(self):
        c={a+1:hsv_to_rgb(np.array([a/float(self.num_agents),1,1])) for a in range(self.num_agents)}
        return c    
    
    def _render(self, mode='human',close=False,screen_width=800,screen_height=800,action_probs=None):
        env = self
        
        plt.clf()
        plt.close('all')
        plt.figure(figsize=(10, 10))
        environment_map = env.world.state
        map_x, map_y = env.world.state.shape
        # rect = patches.Rectangle((0.0, 0.0), 2.0, 2.0 * map_y / map_x, linewidth=1, edgecolor='black', facecolor='none')
        # plt.gca().add_patch(rect)

        map_width = env.world.state.shape
        d_x = 2.0 / map_width[0]
        d_y = 2.0 / map_width[0]
        for obstacle in env.world.obstacles:
            circle = patches.Circle((obstacle[0] * d_x, obstacle[1] * d_y), 0.15*d_x, edgecolor='black', facecolor='#253494')
            plt.gca().add_patch(circle)

        colors=env.initColors()
        for color, agent, goal in zip(colors.values(), env.world.agents, env.world.agent_goals):
            circle = patches.Circle((agent[0] * d_x, agent[1] * d_y), 0.15*d_x, edgecolor=color, facecolor=color)
            plt.gca().add_patch(circle)
            circle = patches.Circle((goal[0] * d_x, goal[1] * d_y), 0.3*d_x, zorder=-1, linewidth=2, edgecolor='black', facecolor=color)
            plt.gca().add_patch(circle)  
            arrow = patches.Arrow(agent[0]*d_x-np.cos(agent[2])*(0.15*d_x), agent[1]*d_y-np.sin(agent[2])*(0.15*d_y), np.cos(agent[2])*0.3*d_x, np.sin(agent[2])*0.3*d_y, width=0.3*d_x, edgecolor='black', facecolor='black')
            plt.gca().add_patch(arrow)

        plt.xlim(0, d_x * map_width[0])
        plt.ylim(0, d_y * map_width[1])
        # plt.show()

        fig = plt.gcf()
        fig.canvas.draw()

        # convert canvas to image using numpy
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return img 
    
    
    def _render_with_contour(self, xys, values):
        env = self
        
        plt.clf()
        plt.close('all')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
#         ax1.set_aspect(1)
#         ax1.set_adjustable("box")
#         ax2.set_aspect(1)
#         ax2.set_adjustable("box")        
        

        environment_map = env.world.state
        map_x, map_y = env.world.state.shape
        # rect = patches.Rectangle((0.0, 0.0), 2.0, 2.0 * map_y / map_x, linewidth=1, edgecolor='black', facecolor='none')
        # plt.gca().add_patch(rect)

        map_width = env.world.state.shape
        d_x = 2.0 / map_width[0]
        d_y = 2.0 / map_width[0]
        for obstacle in env.world.obstacles:
            circle = patches.Circle((obstacle[0] * d_x, obstacle[1] * d_y), 0.25*d_x, edgecolor='black', facecolor='#253494')
            plt.gca().add_patch(circle)

        colors=env.initColors()
        for agent_id, color, agent, goal, xy, value in zip(np.arange(env.num_agents), colors.values(), env.world.agents, env.world.agent_goals, xys, values):
            circle = patches.Circle((agent[0] * d_x, agent[1] * d_y), 0.25*d_x, edgecolor=color, facecolor=color)
            ax1.add_patch(circle)           
            
            rectan = patches.Rectangle((goal[0] * d_x - 0.3 * d_x, goal[1] * d_y - 0.3 * d_y), 0.6 * d_x, 0.6 * d_y, linewidth=1, edgecolor=color, facecolor=color)
            ax1.add_patch(rectan)
            
            arrow = patches.Arrow(agent[0]*d_x-np.cos(agent[2])*(0.25*d_x), agent[1]*d_y-np.sin(agent[2])*(0.25*d_y), np.cos(agent[2])*0.5*d_x, np.sin(agent[2])*0.5*d_y, width=0.5*d_x, edgecolor='black', facecolor='black')
            ax1.add_patch(arrow)            

        ax1.scatter(env.world.agents[0][0] * d_x, env.world.agents[0][1] * d_y, s=320, marker='*', color='black', zorder=3)             
            
        ax1.set_xlim(0, d_x * map_width[0])
        ax1.set_ylim(0, d_y * map_width[1])
        # plt.show()
        
        for agent_id, color, xy, value in zip(np.arange(env.num_agents), colors.values(), xys, values):
            if agent_id == 0:
                x, y = xyZ
                contours = ax2.contour(x, y, value, colors=[color]*len(x.reshape(-1)))
                ax2.clabel(contours, inline=1, fontsize=10)        
        
        fig.tight_layout()
        fig.canvas.draw()     

        # convert canvas to image using numpy
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return img
    
    
    
def dynamic(pos, action):
    next_pos = pos.copy()
    next_pos[:, 2] = next_pos[:, 2] + STEER * action.copy().squeeze(-1)
    next_pos[:, 0] = next_pos[:, 0] + 0.05 * np.cos(next_pos[:, 2])
    next_pos[:, 1] = next_pos[:, 1] + 0.05 * np.sin(next_pos[:, 2])
    next_pos[:, 2] = next_pos[:, 2]%(2*math.pi)
    return next_pos
    
    