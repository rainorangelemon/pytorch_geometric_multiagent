# import sys
# sys.dont_write_bytecode = True

import torch
import numpy as np
import matplotlib.pyplot as plt
from environment.quad3d import Quad3D
import torch
import math
from environment.gym_abstract import AbstractState, AbstractEnv
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter, PillowWriter
from matplotlib.collections import PatchCollection, EllipseCollection
from matplotlib.patches import Circle
from scipy.spatial.distance import cdist
from scipy.linalg import block_diag

quad3d = Quad3D()

AGENT_TOP_K = 6
HEIGHT = 2
OBSTACLE_TOP_K = 2
AGENT_OBS_RADIUS = 3.0
OBSTACLE_OBS_RADIUS = 2.0
AGENT_DISTANCE_THRESHOLD = 0.3
OBSTACLE_DISTANCE_THRESHOLD = 0.3
GOAL_THRESHOLD = 0.45
STEER = 2 * np.pi / 3
n_candidates = 2000

class DroneState(AbstractState):

    def scanForAgents(self):
        obstacles = []
        agents = [(-1,-1,0) for i in range(self.num_agents)]     
        agent_goals = [(-1,-1) for i in range(self.num_agents)]        
        for i in range(self.state.shape[0]):
            for j in range(self.state.shape[1]):
                if(self.state[i,j]>0):
                    agents[self.state[i,j]-1] = [i+0.5,j+0.5,np.random.uniform(0,HEIGHT)]+[0.]*6
                if(self.goals[i,j]>0):
                    agent_goals[self.goals[i,j]-1] = [i+0.5,j+0.5, np.random.uniform(0,HEIGHT)]+[0.]*6
        # add random obstacles 
        map_size = len(self.state)
        new_obstacles = np.random.uniform(0, map_size, size=(int(self.prob*(map_size**2)), 2))
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
        
        # NEW CHANGE: the velocities and angles in the states of agents are randomized
        agents = np.array(agents)
        agents[:, 3:6] = np.random.uniform(-1, 1, size=agents[:, 3:6].shape)
        agents[:, 6:] = np.random.uniform(-math.pi/2, math.pi/2, size=agents[:, 6:].shape)
        return obstacles, agents, agent_goals        
    
    def sample_agents(self, n_agents, prob=0.1):
        if np.random.uniform() < prob:
            agents = np.random.uniform(-math.pi/2, math.pi/2, size=(n_agents, 9))
            agents[:, 3:6] = np.random.uniform(-1, 1, size=(n_agents, 3))
            agents = agents + self.agent_goals
            agents[:, :2] = agents[:, :2].clip(0, len(self.state))
            agents[:, 2] = agents[:, 2].clip(0, HEIGHT)
            return agents
        else:
            agents = np.random.uniform(0, len(self.state), size=(n_agents, 9))
            agents[:, 2] = np.random.uniform(0, HEIGHT, size=(n_agents))
            agents[:, 3:6] = np.random.uniform(-1, 1, size=(n_agents, 3))
            agents[:, 6:] = np.random.uniform(-math.pi/2, math.pi/2, size=(n_agents, 3))
            return agents


class DroneEnv(AbstractEnv):
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

        super().__init__(**kwargs, absState=DroneState, 
                         action_dim=4, space_dim=3, state_dim=9, angle_dim=3,
                            agent_top_k=agent_top_k, obstacle_top_k=obstacle_top_k,
                            obstacle_threshold=OBSTACLE_DISTANCE_THRESHOLD, 
                            agent_threshold=AGENT_DISTANCE_THRESHOLD, 
                            goal_threshold=GOAL_THRESHOLD,
                            agent_obs_radius=agent_obs_radius,
                            obstacle_obs_radius=obstacle_obs_radius,)
    
    def show_obstacles(self, obs, ax, zs=None, alpha=0.6, colors=None):
        
        if colors is None:
            colors = ['deepskyblue']*len(obs)
            
        if zs is None:
            zs = [[0,10]]*len(obs)
        
        def data_for_cylinder_along_z(center_x,center_y,radius,zmin,zmax):
            z = np.linspace(zmin,zmax, 50)
            theta = np.linspace(0, 2*np.pi, 50)
            theta_grid, z_grid=np.meshgrid(theta, z)
            x_grid = radius*np.cos(theta_grid) + center_x
            y_grid = radius*np.sin(theta_grid) + center_y
            return x_grid,y_grid,z_grid
        
        
        for pos, color, z in zip(obs, colors, zs):
            x, y = pos
            p = Circle((x, y), 0.15, color=color, alpha=alpha)
            ax.add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=z[0], zdir="z")
            p = Circle((x, y), 0.15, color=color, alpha=alpha)
            ax.add_patch(p)            
            art3d.pathpatch_2d_to_3d(p, z=z[1], zdir="z")

            xs, ys, zs = data_for_cylinder_along_z(x, y, 0.15, z[0],z[1])
            ax.plot_surface(xs, ys, zs, alpha=alpha, color=color)
    
    def _render(self, elev=85, azim=0, **kwargs):
        
        colors=list(self.initColors().values())
        
        plt.clf()
        plt.close('all')
        
        dpi = 50
        
        fig = plt.figure(figsize=(10, 10), dpi=dpi)
        ax_1 = fig.add_subplot(111, projection='3d')
        ax_1.clear()
        ax_1.view_init(elev=85, azim=0)
        ax_1.axis('off')
        self.show_obstacles(self.world.obstacles, ax_1)

        ax_1.set_xlim(0, len(self.world.state))
        ax_1.set_ylim(0, len(self.world.state))
        ax_1.set_zlim(0, 10)
        
        d_x = dpi*10/len(self.world.state)
        s = ((d_x*0.15) ** 2)
        ax_1.scatter(self.world.agents[:, 0], self.world.agents[:, 1], self.world.agents[:, 2], s=s, color=colors, label='Agent')
        
        
        N=50
        u = np.linspace(0, 2 * np.pi, N)
        v = np.linspace(0, np.pi, N)
        x_sphere = 0.3*np.outer(np.cos(u), np.sin(v))
        y_sphere = 0.3*np.outer(np.sin(u), np.sin(v))
        z_sphere = 0.3*np.outer(np.ones(np.size(u)), np.cos(v))        
        for color, goal in zip(colors, self.world.agent_goals):
            alpha = 0.2
            x, y, z = goal[:3]
            ax_1.plot_surface(x+x_sphere, y+y_sphere, z+z_sphere, linewidth=0.0, alpha=alpha, color=color)

        fig.tight_layout()
        fig.canvas.draw()     

        # convert canvas to image using numpy
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return ax_1
    
    def save_fig_chasing(self, agents, obstacles, filename, dpi=50, timestep=25, **kwargs):
        env = self
        colors=list(env.initColors().values())

        plt.clf()
        plt.close('all')
        
        environment_map = env.world.state
        map_x, map_y = env.world.state.shape
        num_agents = self.num_agents
        
        figsize = 10*max(1, int(map_x/16))
        d_x = dpi*figsize/map_x
        s = ((0.15*d_x) ** 2)

        fig = plt.figure(figsize=(figsize, figsize), dpi=dpi)
        ax_1 = fig.add_subplot(111)
        ax_1.clear()
        ax_1.axis('off')

        ax_1.set_xlim(0, len(env.world.state))
        ax_1.set_ylim(0, len(env.world.state))

        for obstacle in env.world.obstacles:
            circle = patches.Circle((obstacle[0], obstacle[1]), 0.15, facecolor='royalblue')
            plt.gca().add_patch(circle)
        
        frame_number = 0        
        subpath = np.array(agents[max(0, frame_number-timestep):(1+frame_number)])
        for agent_id in list(range(env.num_agents))[::-1]:
            lii,=ax_1.plot(subpath[:,agent_id,0],subpath[:,agent_id,1],color=colors[agent_id],linewidth=2)
            lii.set_solid_capstyle('round')
        agent_circles = EllipseCollection([0.15]*num_agents, [0.15]*num_agents,
                                                        np.zeros(num_agents),
                                                        offsets=subpath[-1,::-1,:2], units='x',
                                                        color = colors[::-1],
                                                        transOffset=ax_1.transData,zorder=100)
        ax_1.add_collection(agent_circles)
        self.agent_circles = agent_circles
        
        ax_1.xaxis.set_visible(False)
        ax_1.yaxis.set_visible(False)
        plt.tight_layout()

        def update(frame_number):
            ax_1.lines.clear()
            subpath = np.array(agents[max(0, frame_number-timestep):(1+frame_number),:,:])
            for agent_id in list(range(env.num_agents))[::-1]:
                lii,=ax_1.plot(subpath[:,agent_id,0],subpath[:,agent_id,1],color=colors[agent_id],linewidth=2)
                lii.set_solid_capstyle('round')
            self.agent_circles.set_offsets(subpath[-1,::-1,:2])

        # Construct the animation, using the update function as the animation director.
        animation = FuncAnimation(fig, update, frames=len(agents), interval=1)
        if '.mp4' in filename:
            writermp4 = FFMpegWriter(fps=10) 
            animation.save(filename, writer=writermp4)
        if '.gif' in filename:
            animation.save(filename, writer=PillowWriter(fps=10))# 'imagemagick', fps=10)    

        return ax_1        
    
    def save_fig(self, agents, goals, obstacles, filename):
        plt.clf()
        plt.close('all')
        env = self
        
        environment_map = env.world.state
        map_x, map_y = env.world.state.shape
        
        figsize = 10*max(1, int(map_x/16))
        dpi = 50
        fig = plt.figure(figsize=(figsize, figsize), dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
            
        ax.clear()
        ax.view_init(elev=85, azim=0)
        ax.axis('off')        

        map_width = env.world.state.shape
        colors=env.initColors()
        colors_v = np.array(list(colors.values()))
        
        self.show_obstacles(obstacles, ax)

        d_x = dpi*figsize/len(self.world.state)
        s = ((d_x*0.15) ** 2)
        curr_agents = agents[0]
        self.scat = ax.scatter(curr_agents[:, 0], curr_agents[:, 1], curr_agents[:, 2], s=s, color=colors_v, label='Agent') 
        
        N=50
        u = np.linspace(0, 2 * np.pi, N)
        v = np.linspace(0, np.pi, N)
        x_sphere = 0.3*np.outer(np.cos(u), np.sin(v))
        y_sphere = 0.3*np.outer(np.sin(u), np.sin(v))
        z_sphere = 0.3*np.outer(np.ones(np.size(u)), np.cos(v))        
        for color, goal in zip(colors_v, goals):
            alpha = 0.2
            x, y, z = goal[:3]
            ax.plot_surface(x+x_sphere, y+y_sphere, z+z_sphere, linewidth=0.0, alpha=alpha, color=color)  
        ax.set(xlim=(0, map_width[0]))
        ax.set(ylim=(0, map_width[1]))  
        ax.set(zlim=(0, 10))
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        plt.tight_layout()

        def update(frame_number):
            curr_agents = agents[frame_number]
            self.scat._offsets3d = (curr_agents[:,0], curr_agents[:,1], curr_agents[:,2])

        # Construct the animation, using the update function as the animation director.
        animation = FuncAnimation(fig, update, frames=len(agents), interval=1)
        if '.mp4' in filename:
            writermp4 = FFMpegWriter(fps=10) 
            animation.save(filename, writer=writermp4)
        if '.gif' in filename:
            animation.save(filename, writer=PillowWriter(fps=10))# 'imagemagick', fps=10)

    def _render_with_contour(self, xys, values, **kwargs):
        pass
    
    def dynamic(self, pos, action):
        action = action * 4
        pos = pos.copy()
        for _ in range(10):
            pos = pos + 0.01 * quadrotor_dynamics_np(pos, action)
        pos[:, 3:6] = np.clip(pos[:, 3:6], -1, 1)
        pos[:, 6:] = np.clip(pos[:, 6:], -math.pi/2, math.pi/2)
        return pos
    
    @staticmethod
    def dynamic_torch(pos, action):
        action = action * 4
        for _ in range(10):
            pos = (pos + 0.01 * quad3d.closed_loop_dynamics(pos, action)).clone()
        new_pos = pos.clone()
        new_pos[:, 3:6] = torch.clip(pos[:, 3:6], -1, 1)
        new_pos[:, 6:] = torch.clip(pos[:, 6:], -math.pi/2, math.pi/2)
        return new_pos
    
    def potential_field(self, actions, K1, K2, ignore_agent=False):
        
        # size of actions: (num_agents, n_candidates, action_dim)
        
        assert actions.shape[0]==self.num_agents
        assert actions.shape[-1]==self.action_dim
        n_candidates = actions.shape[1]
        origin_pos = np.copy(self.world.agents)

        if K1!=0:
        
            next_pos = np.expand_dims(origin_pos, 1)
            next_pos = np.tile(next_pos, (1, n_candidates, 1)) # num_agents x n_candidates x state_dim

            next_pos = self.dynamic(next_pos.reshape(-1, self.state_dim), actions.reshape(-1, self.action_dim))
            next_pos = next_pos.reshape((self.num_agents, n_candidates, self.state_dim))

            if len(self.world.obstacles)!=0:
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
            D = self.obstacle_threshold + 0.2
            obs_force = (1.0 / dist - 1.0 / D)**2
            obs_force = (obs_force)*(dist < D)
            
        else:
            obs_force = np.zeros((self.num_agents, n_candidates)).reshape(-1)
        
        nominal_control = get_simple_direction(origin_pos, self.world.agent_goals)
        goal_force = ((actions - nominal_control.reshape(self.num_agents, 1, self.action_dim))**2).sum(axis=-1)  # num_agents x n_candidates
        score = K1 * obs_force - K2 * goal_force.reshape(-1)
        score = score.reshape(self.num_agents, n_candidates)
        return score    


grav = 0.98
PX = 0
PY = 1
PZ = 2

VX = 3
VY = 4
VZ = 5

PHI = 6
THETA = 7
PSI = 8

F = 0
PHI_DOT = 1
THETA_DOT = 2
PSI_DOT = 3
def quadrotor_dynamics_np(x, u):
    
    batch_size = x.shape[0]
    f = np.zeros((batch_size, 9, 1))

    # Derivatives of positions are just velocities
    f[:, PX, 0] = x[:, VX]  # x
    f[:, PY, 0] = x[:, VY]  # y
    f[:, PZ, 0] = x[:, VZ]  # z

    # Constant acceleration in z due to gravity
    f[:, VZ, 0] = grav

    # Extract batch size and set up a tensor for holding the result
    batch_size = x.shape[0]
    g = np.zeros((batch_size, 9, 4))
    
    # Derivatives of linear velocities depend on thrust f
    s_theta = np.sin(x[:, THETA])
    c_theta = np.cos(x[:, THETA])
    s_phi = np.sin(x[:, PHI])
    c_phi = np.cos(x[:, PHI])
    g[:, VX, F] = -s_theta
    g[:, VY, F] = c_theta * s_phi
    g[:, VZ, F] = -c_theta * c_phi

    # Derivatives of all orientations are control variables
    g[:, PHI:, PHI_DOT:] = np.eye(3)
    
    dsdt = f + np.einsum('ijk,ikl->ijl', g, u[...,np.newaxis])
    return dsdt.reshape(x.shape)

upper_limit = np.array([100, 50, 50, 50])
lower_limit = -1.0 * upper_limit

def get_simple_direction(pos, goal):
    
    diff = pos - goal
    diff[:, :3] = diff[:, :3].clip(-4, 4)
    
    # K = quad3d.K.type_as(x)
    # goal = self.goal_point.squeeze().type_as(x)
    u_nominal = -(quad3d.K_np @ (diff).T).T

    
    u_eq = np.zeros((1, 4))
    u_eq[0, F] = grav    
    # Adjust for the equilibrium setpoint
    u = u_nominal + u_eq
    u = u.clip(lower_limit, upper_limit)  
    
    # u = quad3d.u_nominal(torch.FloatTensor(diff)).data.cpu().numpy()
    u = u / 4
    u = u.clip(-1, 1)
    return u
