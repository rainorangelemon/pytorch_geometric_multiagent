import torch
import random
import numpy as np
from copy import deepcopy

def infer_p(env, n_action=2000, max_episode_length=256, ignore_agent=True, need_gif=None,
            verbose=False, seed=0, K1=1e-1, K2=-3e-2, **kwargs):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    paths = [deepcopy(env.world.agents)]
    total_trans=0; n_danger=0; no_feasible=0; collided=np.zeros(env.num_agents).astype(bool)

    while True:
        o = env._get_obs()
        a_all = np.random.uniform(-1, 1, size=(env.num_agents, n_action, env.action_dim))
        dists = env.potential_field(a_all, K1=K1, K2=K2, ignore_agent=ignore_agent)

        v = np.zeros(env.num_agents)
        a = np.zeros((env.num_agents, env.action_dim))
        for agent_id, a_refine, dist in zip(np.arange(env.num_agents), a_all, dists):
            a[agent_id] = a_refine[np.argmin(dist)]
            v[agent_id] = dist[np.argmin(dist)]
            
        next_o, rw, done, info = env.step(a)

        prev_danger = info['prev_danger'].data.cpu().numpy().astype(bool)
        next_danger = info['next_danger'].data.cpu().numpy().astype(bool)
        if np.any(next_danger):
            collided = collided | next_danger
        if verbose:
            print(env.world.agents, dist.min(axis=-1), dist.max(axis=-1), v, next_danger)

        total_trans += 1
        paths.append(deepcopy(env.world.agents))

        if done or (total_trans >= max_episode_length):
            break
    
    if need_gif is not None:
        env.save_fig(paths, env.world.agent_goals, env.world.obstacles, need_gif[:-4]+'_'+str(np.any(collided))+'_'+str(done)+need_gif[-4:])

    return collided, done, paths