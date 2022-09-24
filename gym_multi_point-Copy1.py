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
import heapq
from torch_geometric.utils import remove_self_loops
from torch_geometric.data import HeteroData

TOP_K = 8
AGENT_OBS_RADIUS = 3.0
OBSTACLE_OBS_RADIUS = 2.0
DISTANCE_THRESHOLD = 0.4


def get_oracle_action(env):
    world=env.getObstacleMap()
    start_positions=tuple(env.getPositions())
    goals=tuple(env.getGoals())
    mstar_path = None
    try:
        mstar_path=cpp_mstar.find_path(world,start_positions,goals,2,5)
    except OutOfTimeError:
        #M* timed out 
#         print("timeout",episode_count)
        pass
    except NoSolutionError:
#         print("nosol????",episode_count,start_positions)
        pass
    if (mstar_path is not None) and (len(mstar_path)!=0):
        mstar_path = np.array(mstar_path)
        return mstar_path
    else:
        return []    
    
    
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
    direction = np.array(env.world.agent_goals)-np.array(env.world.agents)
    direction = direction.astype(float)
    direction_length = np.linalg.norm(direction, axis=-1)
    need_clamp = np.where(direction_length>1)[0]
    direction[need_clamp,:] = direction[need_clamp] / (2 * direction_length[need_clamp].reshape(-1,1))
    return direction      

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
        assert(len(self.agents) == num_agents)

    def scanForAgents(self):
        obstacles = []
        agents = [(-1,-1) for i in range(self.num_agents)]     
        agent_goals = [(-1,-1) for i in range(self.num_agents)]        
        for i in range(self.state.shape[0]):
            for j in range(self.state.shape[1]):
                if(self.state[i,j]>0):
                    agents[self.state[i,j]-1] = (i+0.5,j+0.5)
                if(self.goals[i,j]>0):
                    agent_goals[self.goals[i,j]-1] = (i+0.5,j+0.5)
                if(self.state[i,j]==-1):
                    obstacles.append((i+0.5,j+0.5))
        return obstacles, agents, agent_goals

    def getPos(self, agent_id):
        return tuple(np.floor(self.agents[agent_id]).astype(int))

    def getGoal(self, agent_id):
        return tuple(np.floor(self.agent_goals[agent_id]).astype(int))

#     #try to move agent and return the status
#     def moveAgent(self, direction, agent_id):
#         ax, ay=self.agents[agent_id][0], self.agents[agent_id][1]
#         dx,dy = direction[0], direction[1]
        
#         xprev, yprev = np.floor([ax, ay]).astype(int)
#         new_pos = [ax+dx, ay+dy]
#         xnext, ynext = np.floor(new_pos).astype(int)
#         status = 0
        
#         self.agents[agent_id] = (ax+dx,ay+dy)
        
#         if(ax+dx>=self.state.shape[0] or ax+dx<0 or ay+dy>=self.state.shape[1] or ay+dy<0): #out of bounds
#             self.agents[agent_id] = (max(min(ax+dx, self.state.shape[0]-1e-9), 0.),
#                                      max(min(ay+dy, self.state.shape[1]-1e-9), 0.))
#             return -1
        
#         if (len(self.obstacles)!=0) and cdist(self.obstacles, np.array([new_pos])).min() < 0.5:  #collide with static obstacle
#             return -2
#         # TODO: it is possible for two agents to overlap at the same cell. think more.
        
#         if np.linalg.norm(np.array(self.agents[agent_id])-np.array(self.agent_goals[agent_id])) < 1.0:
#             return 1
# #         elif self.goals[xnext,ynext]!=agent_id and self.goals[xprev,yprev]==agent_id:
# #             return 2
#         else:
#             return 0

#     # try to execture action and return whether action was executed or not and why
#     #returns:
#     #     2: action executed and left goal
#     #     1: action executed and reached goal (or stayed on)
#     #     0: action executed
#     #    -1: out of bounds
#     #    -2: collision with wall
#     #    -3: collision with robot
#     def act(self, action, agent_id):
#         # action: [dx, dy]
#         moved = self.moveAgent(action, agent_id)
#         return moved

    def getDir(self,action):
        return dirDict[action]
    def getAction(self,direction):
        return actionDict[direction]
    
    def get_status(self):
        status = []
        agents = np.array(self.agents)
        if len(agents) > 1:
            distance = cdist(agents, agents)
            distance_nearest = np.sort(distance, axis=-1)[:, 1]        

        for agent_id, agent in enumerate(self.agents):
            s = ''
            if (len(self.obstacles)!=0) and cdist(self.obstacles, np.array([agent])).min() < DISTANCE_THRESHOLD:
                s = s+'danger_obstacle'
            if (len(agents) > 1) and (distance_nearest[agent_id] < DISTANCE_THRESHOLD):
                s = s+'danger_agent'
            # if (len(agents) > 1) and (distance_nearest[agent_id] > 0.7):
            #     s = s+'safe'
            if np.linalg.norm(np.array(self.agents[agent_id])-np.array(self.agent_goals[agent_id])) < 0.5:
                s = s+'done'
            if 'danger' not in s:
                s = s+'free'
            status.append(s)
        return status    
    
    def sample_agents(self, n_agents, prob=0.1):
        agents = np.random.uniform(0, len(self.state), size=(n_agents, 2))
        return agents    

    # Compare with a plan to determine job completion
    def done(self):
        numComplete = 0
        for i in range(len(self.agents)):
            if np.linalg.norm(np.array(self.agents[i])-np.array(self.agent_goals[i])) < 1.0:
                numComplete += 1
        return numComplete==len(self.agents) #, numComplete/float(len(self.agents))


class MultiPointEnv():
    action_dim = 2
    state_dim = 2
    space_dim = 2
    
    def getFinishReward(self):
        return FINISH_REWARD

    # Initialize env
    def __init__(self,num_agents=1,observation_size=4,mode='barrier',world0=None, goals0=None, SIZE=(10,40), PROB=(0,.5), FULL_HELP=False,blank_world=False):
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
        self.fresh             = True
        self.FULL_HELP         = FULL_HELP
        self.finished          = False
        
        assert mode=='lyapunov' or mode=='barrier'
        self.mode = mode
        

        # Initialize data structures
        self._setWorld(world0,goals0,blank_world=blank_world)
        self.viewer           = None
        self.action_dim = 2
        self.state_dim = 2        

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
    
    def getGoals(self):
        result=[]
        for i in range(self.num_agents):
            result.append(self.world.getGoal(i))
        return result
    
    def getPositions(self):
        result=[]
        for i in range(self.num_agents):
            result.append(self.world.getPos(i))
        return result
    
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
        try:
            prob=np.random.triangular(self.PROB[0],.33*self.PROB[0]+.66*self.PROB[1],self.PROB[1])
        except Exception:
            prob=self.PROB[0]
        size=np.random.choice([self.SIZE[0],self.SIZE[0]*.5+self.SIZE[1]*.5,self.SIZE[1]],p=[.5,.25,.25])
        world     = -(np.random.rand(int(size),int(size))<prob).astype(int)

        #RANDOMIZE THE POSITIONS OF AGENTS
        agent_counter = 1
        agent_locations=[]
        while agent_counter<=self.num_agents:
            x,y = np.random.randint(0,world.shape[0]),np.random.randint(0,world.shape[1])
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
        data['agent', 'a_near_a', 'agent'].edge_attr = data['agent'].x[edges[0, :]]-data['agent'].x[edges[1, :]]

        if len(self.world.obstacles) != 0:
            data['obstacle'].x = torch.FloatTensor(self.world.obstacles)
            edges = data['obstacle', 'o_near_a', 'agent'].edge_index = radius(data['agent'].x[:,:self.space_dim], data['obstacle'].x[:,:self.space_dim], r=OBSTACLE_OBS_RADIUS)
            data['obstacle', 'o_near_a', 'agent'].edge_attr = data['obstacle'].x[edges[0, :]]-data['agent'].x[edges[1, :]]
        else:
            data['obstacle'].x = torch.FloatTensor([[100, 100]])
            
        
        goals = self.world.agent_goals
        clip_goal = (goals-self.world.agents).clip(-6, 6)+self.world.agents
        data['goal'].x = torch.FloatTensor(self.world.agent_goals)
        edges = data['goal', 'toward', 'agent'].edge_index = torch.arange(self.num_agents).unsqueeze(0).repeat(2, 1).long()
        data['goal', 'toward', 'agent'].edge_attr = data['goal'].x[edges[0, :]]-data['agent'].x[edges[1, :]]
        
        data['agent'].x = torch.FloatTensor([[1]]*self.num_agents)
        if len(self.world.obstacles) != 0:
            data['obstacle'].x = torch.FloatTensor([[1]]*len(self.world.obstacles))
        else:
            data['obstacle'].x = torch.FloatTensor([[1]])
        data['goal'].x = torch.FloatTensor([[1]]*self.num_agents)

#         if mode==None:
#             mode = self.mode
#         else:
#             assert mode=='lyapunov' or mode=='barrier'
#             pass
        
#         agents = np.array(self.world.agents)
#         obstacles = np.array(self.world.obstacles)
        
#         # add obstacles nodes
#         if (mode=='barrier') and (len(obstacles)!=0):
#             nodes = np.vstack((obstacles, agents))
#         else:
#             obstacles = np.array([])
#             nodes = agents
#         labels = [[1,0,0]] * len(obstacles) + [[0,1,0]] * len(agents)

#         edges = []
#         agent_infos = []

#         obs_to_agent = agent_to_agent = torch.arange(self.num_agents).reshape((1,-1)).repeat(2,1)+len(obstacles)
#         if len(obstacles) != 0:
#             new_obs_to_agent = radius(torch.FloatTensor(agents), torch.FloatTensor(obstacles), r=OBSTACLE_OBS_RADIUS)
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
        
#         if mode == 'lyapunov':
#             edge_index = torch.arange(self.num_agents).reshape((1,-1)).repeat(2,1)
#         elif mode == 'barrier':
#             edge_index = torch.cat((obs_to_agent, agent_to_agent), dim=-1)
#         edges = edge_index.T.numpy().tolist()
        
#         if (mode=='lyapunov'):
#             for agent in range(self.num_agents):
#                 agent_id = agent
#                 distance=lambda x1,y1,x2,y2:((x2-x1)**2+(y2-y1)**2)**.5
#                 dx=self.world.agents[agent_id][0]-self.world.agent_goals[agent_id][0]
#                 dy=self.world.agents[agent_id][1]-self.world.agent_goals[agent_id][1]
#                 mag=(dx**2+dy**2)**.5
#                 if mag!=0:
#                     dx=dx/mag
#                     dy=dy/mag
#                 goal = self.world.agent_goals[agent]
#                 if clip:
#                     nodes = np.vstack((nodes, (goal-self.world.agents[agent]).clip(-6, 6)+self.world.agents[agent]))    
#                 else:
#                     nodes = np.vstack((nodes, goal))    
#                 edges.append([len(nodes)-1,agent+len(obstacles)])
#                 labels.append([0,0,1])
        
#         labels = np.array(labels, dtype=float)
#         edges = np.array(edges).T
#         edges = torch.LongTensor(edges)
#         edges, _ = coalesce(edges, None, len(nodes), len(nodes))
#         edges, _ = remove_self_loops(edges)
        
        return data
                # {"x": torch.FloatTensor(nodes),
                #  "edge_index": edges,
                #  "label": torch.FloatTensor(labels),
                #  "goal": torch.FloatTensor(self.world.agent_goals)}

    # Resets environment
    def _reset(self,world0=None,goals0=None):
        self.finished = False

        # Initialize data structures
        self._setWorld(world0, goals0)
        self.fresh = True

    def _complete(self):
        return self.world.done()

        
    # Executes an action by an agent
    def step(self, action_input, bound=False):
        origin_pos = np.copy(self.world.agents)
        prev_o = self._get_obs() 
        prev_status = self.world.get_status()
#         next_targets = self.world.agents + get_simple_direction(self)
        
        self.fresh = False
        rewards = [0]*self.num_agents

        # Check action input
        assert len(action_input) == self.num_agents, 'Action input should be a tuple with the form (num_agents, 2)'

        s = self.world.agents + 0.3 * action_input
        
        if bound:
            s[:, :2] = np.clip(s[:, :2], 0, len(self.world.state))
            
        self.world.agents = s
        
        status = self.world.get_status()
        rewards = [-1 if 'danger' in s else 0 for s in status]

        # Perform observation
        next_o = self._get_obs() 

        # Done?
        done = self.world.done()
        
        self.finished |= done
        next_status = self.world.get_status()
        
        
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
            circle = patches.Circle((obstacle[0] * d_x, obstacle[1] * d_y), 0.2*d_x, edgecolor='black', facecolor='#253494')
            plt.gca().add_patch(circle)

        colors=env.initColors()
        for color, agent, goal in zip(colors.values(), env.world.agents, env.world.agent_goals):
            circle = patches.Circle((agent[0] * d_x, agent[1] * d_y), 0.2*d_x, edgecolor=color, facecolor=color)
            plt.gca().add_patch(circle)
            # circle = patches.Rectangle((goal[0] * d_x - 0.3 * d_x, goal[1] * d_y - 0.3 * d_y), 0.6 * d_x, 0.6 * d_y, linewidth=1, edgecolor=color, facecolor=color)
            circle = patches.Circle((goal[0] * d_x, goal[1] * d_y), 0.3*d_x, zorder=-1, linewidth=2, edgecolor='black', facecolor=color)
            plt.gca().add_patch(circle)    

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
        for i in range(map_width[0]):
            for j in range(map_width[1]):
                if environment_map[i, j] == -1:
                    rect = patches.Rectangle((d_x * i, d_y * j), d_x, d_y, linewidth=1, edgecolor='#253494',
                                                facecolor='#253494')
                    ax1.add_patch(rect)

        colors=env.initColors()
        for agent_id, color, agent, goal, xy, value in zip(np.arange(env.num_agents), colors.values(), env.world.agents, env.world.agent_goals, xys, values):
            circle = patches.Circle((agent[0] * d_x, agent[1] * d_y), 0.25*d_x, edgecolor=color, facecolor=color)
            ax1.add_patch(circle)           
            
            rectan = patches.Rectangle((goal[0] * d_x - 0.3 * d_x, goal[1] * d_y - 0.3 * d_y), 0.6 * d_x, 0.6 * d_y, linewidth=1, edgecolor=color, facecolor=color)
            ax1.add_patch(rectan)         

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

    