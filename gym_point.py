__credits__ = ["Chenning Yu"]

import numpy as np

class PointEnv():
    
    state_dim = 2
    action_dim = 2
    goal_dim = 2
    max_episode_steps = 100
    state_range = np.array([(-3, 3),
         (-3, 3)]).T

    def __init__(
        self,
        goal=(2,0),
    ):  
        self.t = 0
        self.goal = np.array(goal)
        self.state = np.array([-2,0])
        
    def reset(self, **kwargs):
        self.t = 0
        self.state = np.array([-2,0])
        return self.state

    def step(self, action):
        xys_before = self.state.copy()
        is_collide_prev = self.collision_check()
        xy_position_before = self.state.copy()
        
        xy_position_after = xy_position_before + 0.3 * action
        self.t += 1
        xys_after = xy_position_after.copy()
        self.state = xy_position_after
        
        observation = self._get_obs()
        done = False
        
        dist2goal_before = np.linalg.norm(self.goal.reshape(1, -1)-xys_before, axis=-1).min()
        dist2goal_after = np.linalg.norm(self.goal.reshape(1, -1)-xys_after, axis=-1).min()
        displace = dist2goal_before-dist2goal_after
        is_collide = self.collision_check()
        
        if dist2goal_after < 0.1 and (not is_collide):
            reach_goal = True
            done = True
        else:
            reach_goal = False
            
        if self.t > self.max_episode_steps:
            done = True
            
        reward =  -2*is_collide + displace  # + 10*reach_goal + - ctrl_cost
        
        info = {
            "x": xys_before,
            "next_x": observation,
            "action": action,
            "goal": self.goal,
            "prev_free": not is_collide_prev,
            "prev_danger": is_collide_prev,
            "next_goal": reach_goal,
            "next_free": not is_collide,
            "next_danger": is_collide,
            # "reward_fwd": forward_reward,
            # "reward_ctrl": -ctrl_cost,
            # "x_position": xy_position_after[0],
            # "y_position": xy_position_after[1],
            # "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            # "x_velocity": x_velocity,
            # "y_velocity": y_velocity,
            # "forward_reward": forward_reward,
            # "collision": is_collide,
            # "dist2goal": dist2goal,
        }

        return observation, reward, done, info
    
    def collision_check(self):
        return np.linalg.norm(self.state)<=1

    def _get_obs(self):
        return self.state.copy()

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)