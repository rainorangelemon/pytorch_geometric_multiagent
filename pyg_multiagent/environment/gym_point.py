import numpy as np
import math
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter, PillowWriter
from matplotlib.collections import PatchCollection, EllipseCollection
from environment.gym_abstract import AbstractState, AbstractEnv
from torch_geometric.data import Data, HeteroData
from scipy.spatial.distance import cdist
import torch

AGENT_TOP_K = 6
OBSTACLE_TOP_K = 1
AGENT_OBS_RADIUS = 4.0
OBSTACLE_OBS_RADIUS = 4.0
AGENT_DISTANCE_THRESHOLD = 0.3
OBSTACLE_DISTANCE_THRESHOLD = 0.3
GOAL_THRESHOLD = 0.3
n_candidates = 2000     



def intersects(circle, rect):
    r = 0.15
    circle_x, circle_y = circle
    rect_x, rect_y, half_rect_width, half_rect_height = rect
    
    circle_distance_x = abs(circle_x - rect_x)
    circle_distance_y = abs(circle_y - rect_y)

    if (circle_distance_x > (half_rect_width + r)):
        return False
    if (circle_distance_y > (half_rect_height + r)):
        return False

    if (circle_distance_x <= (half_rect_width)):
        return True
    if (circle_distance_y <= (half_rect_height)):
        return True

    cornerDistance_sq = (circle_distance_x - half_rect_width)**2 + (circle_distance_y - half_rect_height)**2

    return (cornerDistance_sq <= (r**2))


class PointState(AbstractState):

    def scanForAgents(self):
        obstacles = [(0, 0, 0.25, 0.5), 
                     (-1.5, 0, 0.4, 2), 
                     (1.5, 0, 0.4, 2)]
        agents = [(0,-1.7)]     
        agent_goals = [(0,1.7)]
        return obstacles, agents, agent_goals
    
    def sample_agents(self, n_agents, prob=0.1):
        if np.random.uniform() < prob:
            agents = np.random.uniform(-0.3, 0.3, size=(n_agents, 2))
            agents = agents + self.agent_goals
            agents[:, :2] = agents[:, :2].clip(0, len(self.state))
            return agents
        else:
            agents = np.random.uniform(0, len(self.state), size=(n_agents, 2))
            return agents
        
    def get_status(self):
        status = []
        agents = np.array(self.agents)
        obstacles = self.obstacles
        dist2goal = np.linalg.norm(np.array(self.agents[:, :self.space_dim])-np.array(self.agent_goals[:, :self.space_dim]), axis=-1)

        danger_obstacle = np.any([intersects(self.agents[0], o) for o in obstacles])
        status = ['' for _ in range(self.num_agents)]
        status = [s+'danger_obstacle' if danger_obstacle else s for s in status]
        status = [s+'done' if d<self.goal_threshold else s for s, d in zip(status, dist2goal)]
        status = [s+'free' if ('danger' not in s) else s for s in status]
        
        return status


class PointEnv(AbstractEnv):

    def __init__(self, agent_top_k=None, obstacle_top_k=None,
                 agent_obs_radius=None, obstacle_obs_radius=None, **kwargs):
        if agent_top_k is None:
            agent_top_k = AGENT_TOP_K
            
        if obstacle_top_k is None:
            obstacle_top_k = OBSTACLE_TOP_K
            
        if agent_obs_radius is None:
            agent_obs_radius = AGENT_OBS_RADIUS
            
        if obstacle_obs_radius is None:
            obstacle_obs_radius = OBSTACLE_OBS_RADIUS

        super().__init__(**kwargs, absState=PointState, 
                         action_dim=2, space_dim=2, state_dim=2, 
                         angle_dim=0,
                            agent_top_k=agent_top_k, obstacle_top_k=obstacle_top_k,
                            obstacle_threshold=OBSTACLE_DISTANCE_THRESHOLD, 
                            agent_threshold=AGENT_DISTANCE_THRESHOLD, 
                            goal_threshold=GOAL_THRESHOLD,
                            agent_obs_radius=agent_obs_radius,
                            obstacle_obs_radius=obstacle_obs_radius,) 
        
    def _get_obs(self, **kwargs):
        data = HeteroData()
        data['agent'].x = torch.FloatTensor(self.world.agents)
        return data
    
    # Executes an action by an agent
    def step(self, action_input, obs_config=None, bound=False):
        assert len(action_input) == self.num_agents, 'Action input should be a tuple with the form (num_agents, action_dim)'
        assert len(action_input[0]) == self.action_dim, 'Action input should be a tuple with the form (num_agents, action_dim)'
        
        if obs_config is None:
            obs_config = {}
        
        prev_status = self.world.get_status()
        prev_o = self._get_obs()

        # Check action input

        next_pos = self.dynamic(self.world.agents, action_input)
        self.world.agents = next_pos
        
        if bound:
            self.world.agents[:, :2] = np.clip(self.world.agents[:, :2], 0, len(self.world.state))
            
        # Perform observation
        next_o = self._get_obs() 

        # Done?
        next_status = self.world.get_status()
        done = self.world.done(status=next_status)
        
        self.finished |= done
        
        rewards = [-0.1-2*('danger_agent' in s)-2*('danger_obstacle' in s) + 1*('done' in s) for s in next_status]
        
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
    
    def _render(self):
        env = self
        plt.clf()
        plt.close('all')
        fig = plt.figure(figsize=(10, 10))
        fig.tight_layout()
        environment_map = env.world.state
        map_x, map_y = env.world.state.shape

        for obstacle in env.world.obstacles:
            center_x, center_y, dx, dy = obstacle
            rectangle = patches.Rectangle((center_x-dx, center_y-dy), 2*dx, 2*dy, edgecolor='black', facecolor='#253494')
            plt.gca().add_patch(rectangle)

        colors=env.initColors()
        for color, agent, goal in zip(colors.values(), env.world.agents, env.world.agent_goals):
            circle = patches.Circle((agent[0], agent[1]), 0.15, edgecolor=color, facecolor=color)
            plt.gca().add_patch(circle)
            
            plt.scatter(goal[0], goal[1], s=1280, color=color, linewidths=2, marker=(5, 1), edgecolors='black')  # zorder=-1,
            # circle = patches.Circle((), 0.3, zorder=-1, linewidth=2, edgecolor='black', facecolor=color)
            plt.gca().add_patch(circle)

        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.axis('off')
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
    
    
    def dynamic(self, pos, action):
        next_pos = pos.copy()
        next_pos[:, 0] = next_pos[:, 0] + 0.05 * action[:, 0]
        next_pos[:, 1] = next_pos[:, 1] + 0.05 * action[:, 1]
        return next_pos

    
    def save_fig(self, agents, goals, obstacles, filename):

        env = self
        plt.clf()
        plt.close('all')
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)        

        colors=env.initColors()
        colors_v = np.array(list(colors.values()))

        num_agents = self.num_agents

        for obstacle in env.world.obstacles:
            center_x, center_y, dx, dy = obstacle
            rectangle = patches.Rectangle((center_x-dx, center_y-dy), 2*dx, 2*dy, edgecolor='black', facecolor='#253494')
            ax.add_patch(rectangle)

        colors=env.initColors()
        for color, agent, goal in zip(colors.values(), env.world.agents, env.world.agent_goals):
            self.circle = patches.Circle((agent[0], agent[1]), 0.15, edgecolor=color, facecolor=color)
            ax.add_patch(self.circle)
            
            ax.scatter(goal[0], goal[1], s=1280, color=color, linewidths=2, marker=(5, 1), edgecolors='black')   

        ax.set(xlim=(-3, 3))
        ax.set(ylim=(-3, 3))
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        plt.tight_layout()

        def update(frame_number):
            curr_agents = agents[frame_number]
            agent = curr_agents[0]
            self.circle.remove()

            self.circle = patches.Circle((agent[0], agent[1]), 0.15, edgecolor=color, facecolor=color)
            ax.add_patch(self.circle)

        # Construct the animation, using the update function as the animation director.
        animation = FuncAnimation(fig, update, frames=len(agents), interval=1)
        if '.mp4' in filename:
            writermp4 = FFMpegWriter(fps=10) 
            animation.save(filename, writer=writermp4)
        if '.gif' in filename:
            animation.save(filename, writer=PillowWriter(fps=10))# 'imagemagick', fps=10)
    