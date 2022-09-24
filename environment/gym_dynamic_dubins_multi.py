import numpy as np
import math
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter, PillowWriter
from matplotlib.collections import PatchCollection, EllipseCollection
from environment.gym_abstract import AbstractState, AbstractEnv
from scipy.spatial.distance import cdist
import torch

AGENT_TOP_K = 6
OBSTACLE_TOP_K = 1
AGENT_OBS_RADIUS = 4.0
OBSTACLE_OBS_RADIUS = 4.0
AGENT_DISTANCE_THRESHOLD = 0.3
OBSTACLE_DISTANCE_THRESHOLD = 0.3
GOAL_THRESHOLD = 0.45
STEER = 0.005
n_candidates = 2000     


class DoubleDubinsCarState(AbstractState):

    def scanForAgents(self):
        obstacles = []
        agents = [(-1,-1,0,0) for i in range(self.num_agents)]     
        agent_goals = [(-1,-1) for i in range(self.num_agents)]        
        for i in range(self.state.shape[0]):
            for j in range(self.state.shape[1]):
                if(self.goals[i,j]>0):
                    agent_goals[self.goals[i,j]-1] = (i+0.5,j+0.5)
        for i in range(self.state.shape[0]):
            for j in range(self.state.shape[1]):                    
                if(self.state[i,j]>0):
                    angle = np.arctan2(agent_goals[self.state[i,j]-1][1]-(j+0.5), agent_goals[self.state[i,j]-1][0]-(i+0.5))%(2*math.pi)
                    agents[self.state[i,j]-1] = (i+0.5,j+0.5,0.,angle)
        # add random obstacles    
        map_size = len(self.state)
        if self.keep_sample_obs:
            num_candid = 1
        else:
            num_candid = int(self.prob*(map_size**2))        
        while True:
            if (self.keep_sample_obs) and ((len(obstacles) >= int(self.prob)) or (num_candid > 1000000)):
                break
            new_obstacles = np.random.uniform(0, map_size, size=(num_candid, 2))
            distance_agent = cdist(new_obstacles, np.array(agents)[:,:2]).min(axis=-1)
            distance_goal = cdist(new_obstacles, np.array(agent_goals)[:,:2]).min(axis=-1)        
            valid = ((distance_goal > GOAL_THRESHOLD) & (distance_agent > (0.1+OBSTACLE_DISTANCE_THRESHOLD)))
            new_obstacles = new_obstacles[valid]
            for new_obstacle in new_obstacles:
                if len(obstacles):
                    distance_obs = cdist([new_obstacle], obstacles).min(axis=-1)
                else:
                    distance_obs = np.array([float('inf')])
                valid = (distance_obs > (0.2+2*OBSTACLE_DISTANCE_THRESHOLD))
                if valid.all():
                    obstacles.append(new_obstacle)
            if (not self.keep_sample_obs):
                break
            else:
                num_candid = num_candid * 2
        return obstacles, agents, agent_goals
    
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


class MultiDynamicDubinsEnv(AbstractEnv):

    def __init__(self, steer=None,
                 agent_top_k=None, obstacle_top_k=None,
                 agent_obs_radius=None, obstacle_obs_radius=None, **kwargs):
        if agent_top_k is None:
            agent_top_k = AGENT_TOP_K
            
        if obstacle_top_k is None:
            obstacle_top_k = OBSTACLE_TOP_K
            
        if agent_obs_radius is None:
            agent_obs_radius = AGENT_OBS_RADIUS
            
        if obstacle_obs_radius is None:
            obstacle_obs_radius = OBSTACLE_OBS_RADIUS
            
        if steer is None:
            self.steer = STEER
        else:
            self.steer = steer

        super().__init__(**kwargs, absState=DoubleDubinsCarState, 
                         action_dim=2, space_dim=2, state_dim=4, 
                         angle_dim=1,
                            agent_top_k=agent_top_k, obstacle_top_k=obstacle_top_k,
                            obstacle_threshold=OBSTACLE_DISTANCE_THRESHOLD, 
                            agent_threshold=AGENT_DISTANCE_THRESHOLD, 
                            goal_threshold=GOAL_THRESHOLD,
                            agent_obs_radius=agent_obs_radius,
                            obstacle_obs_radius=obstacle_obs_radius,) 
    
    def _render(self):
        env = self
        plt.clf()
        plt.close('all')
        plt.figure(figsize=(10, 10))
        environment_map = env.world.state
        map_x, map_y = env.world.state.shape

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
            arrow = patches.Arrow(agent[0]*d_x-np.cos(agent[3])*(0.15*d_x), agent[1]*d_y-np.sin(agent[3])*(0.15*d_y), np.cos(agent[3])*0.3*d_x, np.sin(agent[3])*0.3*d_y, width=0.3*d_x, edgecolor='black', facecolor='black')
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
            
            arrow = patches.Arrow(agent[0]*d_x-np.cos(agent[3])*(0.25*d_x), agent[1]*d_y-np.sin(agent[3])*(0.25*d_y), np.cos(agent[3])*0.5*d_x, np.sin(agent[3])*0.5*d_y, width=0.5*d_x, edgecolor='black', facecolor='black')
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
    
    
    def dynamic(self, pos, action):
        next_pos = pos.copy()
        dt = 0.05
        # pos: x, y, vel, theta
        # action: acc, angular velocity
        next_pos[:, 3] = next_pos[:, 3] + 0.2 * action.reshape(-1, self.action_dim)[:, 1].copy()
        next_pos[:, 2] = next_pos[:, 2] + dt * action.reshape(-1, self.action_dim)[:, 0].copy()
        next_pos[:, 2] = np.clip(next_pos[:, 2], 0, 1)
        next_pos[:, 0] = next_pos[:, 0] + dt * next_pos[:, 2] * np.cos(next_pos[:, 3])
        next_pos[:, 1] = next_pos[:, 1] + dt * next_pos[:, 2] * np.sin(next_pos[:, 3])
        next_pos[:, 3] = next_pos[:, 3]%(2*math.pi)
        return next_pos
    

    def dynamic_torch(self, pos, action):
        next_pos = pos.clone()
        next_theta = pos[:, 3] + self.steer * action.reshape(-1, self.action_dim)[:, 1]
        next_vel = pos[:, 2] + 0.05 * action.reshape(-1, self.action_dim)[:, 0]
        next_pos[:, 0] = pos[:, 0] + 0.05 * next_vel * torch.cos(next_theta)
        next_pos[:, 1] = pos[:, 1] + 0.05 * next_vel * torch.sin(next_theta)
        next_pos[:, 2] = next_vel
        next_pos[:, 3] = next_theta
        return next_pos
    
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
        
        diff = np.expand_dims(self.world.agent_goals, axis=1)[:,:,:2]-next_pos[:, :, :2]
        angle = np.arctan2(diff[:,:,1]+1e-5, diff[:,:,0]+1e-5)
        diff_angle = next_pos[:, :, 3] - angle
        diff_angle = (diff_angle + math.pi) % (2*math.pi) - math.pi
        direction_force = np.abs(diff_angle)
        
        score = -K2 * (goal_force.reshape(-1) + 0.1 * direction_force.reshape(-1))
        score = score.reshape(self.num_agents, n_candidates)
        return score

    
    def save_fig(self, agents, goals, obstacles, filename):
        plt.clf()
        plt.close('all')

        env = self
        # fig = plt.figure(figsize=(7, 7))
        # ax = fig.add_axes([0, 0, 1, 1], frameon=False)
        # ax.set_xlim(0, 1), ax.set_xticks([])
        # ax.set_ylim(0, 1), ax.set_yticks([])

        env = self
        plt.clf()
        plt.close('all')
        environment_map = env.world.state
        map_x, map_y = env.world.state.shape
        
        fig = plt.figure(figsize=(10*max(1, int(map_x/16)), 10*max(1, int(map_x/16))))
        ax = fig.add_subplot(111)        

        map_width = env.world.state.shape
        d_x = 2.0 / map_width[0]
        colors=env.initColors()
        colors_v = np.array(list(colors.values()))
        
        d_y = 2.0 / map_width[0]
        for obstacle in obstacles:
            circle = patches.Circle((obstacle[0] * d_x, obstacle[1] * d_y), 0.15*d_x, edgecolor='black', facecolor='#253494')
            plt.gca().add_patch(circle)

        curr_agents = agents[0]
        num_agents = self.num_agents

        agent_circles = EllipseCollection([0.3*d_x]*num_agents, [0.3*d_x]*num_agents,
                                                        np.zeros(num_agents),
                                                        offsets=curr_agents[:,:2]*np.array([[d_x, d_y]]), units='x',
                                                        color = colors_v,
                                                        transOffset=ax.transData,)
        ax.add_collection(agent_circles)
        self.agent_circles = agent_circles
        
        arrows = [patches.Arrow(agent[0]*d_x-np.cos(agent[3])*(0.15*d_x), agent[1]*d_y-np.sin(agent[3])*(0.15*d_y), np.cos(agent[3])*0.3*d_x, np.sin(agent[3])*0.3*d_y, width=0.3*d_x) for agent in curr_agents]
        agent_arrows = PatchCollection(arrows, zorder=10, color='black')
        ax.add_collection(agent_arrows)
        self.agent_arrows = agent_arrows

        goal_circles = EllipseCollection([0.6*d_x]*num_agents, [0.6*d_x]*num_agents,
                                                        np.zeros(num_agents),
                                                        offsets=goals[:,:2]*np.array([[d_x, d_y]]), units='x',
                                                        color = colors_v,
                                                        linewidth = 2,
                                                        transOffset=ax.transData, zorder=-1)
        goal_circles.set_edgecolor('black')
        ax.add_collection(goal_circles)        

        ax.set(xlim=(0, d_x * map_width[0]))
        ax.set(ylim=(0, d_y * map_width[1]))  
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        plt.tight_layout()

        def update(frame_number):
            curr_agents = agents[frame_number]
            num_agents = self.num_agents
            self.agent_circles.remove()
            self.agent_arrows.remove()

            agent_circles = EllipseCollection([0.3*d_x]*num_agents, [0.3*d_x]*num_agents,
                                                            np.zeros(num_agents),
                                                            offsets=curr_agents[:,:2]*np.array([[d_x, d_y]]), units='x',
                                                            color = colors_v,
                                                            transOffset=ax.transData,)
            ax.add_collection(agent_circles)
            self.agent_circles = agent_circles
            
            arrows = [patches.Arrow(agent[0]*d_x-np.cos(agent[3])*(0.15*d_x), agent[1]*d_y-np.sin(agent[3])*(0.15*d_y), np.cos(agent[3])*0.3*d_x, np.sin(agent[3])*0.3*d_y, width=0.3*d_x) for agent in curr_agents]
            agent_arrows = PatchCollection(arrows, zorder=10, color='black')
            ax.add_collection(agent_arrows)
            self.agent_arrows = agent_arrows

        # Construct the animation, using the update function as the animation director.
        animation = FuncAnimation(fig, update, frames=len(agents), interval=1)
        if '.mp4' in filename:
            writermp4 = FFMpegWriter(fps=10) 
            animation.save(filename, writer=writermp4)
        if '.gif' in filename:
            animation.save(filename, writer=PillowWriter(fps=10))# 'imagemagick', fps=10)
    