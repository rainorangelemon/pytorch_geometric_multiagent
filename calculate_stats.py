import copy

collideds = []
dones = []
lengths = []
neighbor_o = []
neighbor_a = []
for v_idx, data in tqdm(enumerate(valid_dataset)):
    env = create_env()
    env.world.obstacles, env.world.agent_goals, env.world.agents = deepcopy(data)
    # env.world.agents[1:,:]=-100
    # env.world.obstacles = [[1.499, 1.499]]
    gif_file = 'gifs/0426/v122/'+str(v_idx)+'_3_2e-2_vanilla_lie.gif'
    collided, done, gifs = infer(env,bnn,verbose=False,spatial_prop=False,stop_at_collision=True,decompose=None,need_gif=gif_file)
    # lie_derive_safe=True,threshold=1e-1
    collideds.append(collided)
    dones.append(done)
    lengths.append(len(gifs))

    for gif in gifs:
        env.world.agents = copy.deepcopy(gif)
        neighbor_a.append(int((env._get_obs(**OBS_CONFIG)['a_near_a'].edge_index[1,:]==1).sum()))
        neighbor_o.append(int((env._get_obs(**OBS_CONFIG)['o_near_a'].edge_index[1,:]==1).sum()))

print(np.any(collideds, axis=-1).mean(), np.mean(collideds), np.mean(dones), np.mean(lengths))
print(np.min(neighbor_o), np.max(neighbor_o), np.min(neighbor_a), np.max(neighbor_a))

import matplotlib.pyplot as plt
plt.clf()
plt.close('all')
plt.hist(neighbor_a, bins='auto')
plt.show()
