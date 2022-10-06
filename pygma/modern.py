from tqdm import tqdm
import gc
from copy import deepcopy
from environment.gym_dubins_car import DubinsCarEnv, STEER

import faulthandler
faulthandler.enable()

import torch
import numpy as np
from torch import nn
import math
from models import *
from core import generate_default_model_name
import wandb

import scipy
import random
from PIL import Image
from random import shuffle
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from collections import defaultdict
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch.utils.data import DataLoader, Dataset

'''
compare to v5: 
1. make the valid dataset smaller 
2. make the warmup longer
3. make the cbuf size larger
4. change the data of the cbuf to be first add to cbuf, then relabel
5. no valid dataset
6. no repeat dataset
7. use Adam
8. no lr decay, default LR 3e-4, and add weight decay 1e-8
9. larger batch size 256->1024
10. add nominal eps
11. add track on success rate
12. use potential field (has repulsive force on obstacles)
13: density = 0, map size = 4, # agents = 8
14. explore eps faster converge to 0, larger n_epoch
15. cyclic explore eps
'''

from configs.v68 import *

Env = DubinsCarEnv
name_dict = generate_default_model_name(Env)
TXT_NAME = '1model_'+Env.__name__+'_'+version_name+'.txt'
BMODEL_PATH = name_dict['db'].replace('.pt', '_'+version_name+'.pt')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)

if POTENTIAL_OBS:
    K1 = 1e-1
    K2 = -3e-2
else:
    K1 = 0.
    K2 = -3e-2


def create_network():
    if OBSTACLE_DENSITY == 0:
        bnn = OriginGNNv5(HIDDEN_SIZE, 3, keys=['agent'])
    else:
        bnn = OriginGNNv5(HIDDEN_SIZE, 3, keys=['agent', 'obstacle'])
    bnn.to(device)
    return bnn


def train_barrier(bnn, optimizer, buf, n_iter=10):
    
    # Set up function for computing value loss
    def compute_loss(bnn, data, next_data):
        value = bnn(data)
        next_value = bnn(next_data)
        
        bloss1 = ((1e-2-value).relu())*data['next_free'] / (1e-9 + (data['next_free']).sum())
        bloss2 = ((1e-2+value).relu())*data['next_danger'] / (1e-9 + (data['next_danger'].sum()))
        bloss = bloss1.sum() + bloss2.sum()
        
        if not ALL_LIE:
            deriv = next_value-value+0.1*value
            dloss = ((-deriv+1e-2).relu())*data['next_free']*next_data['next_free']
            dloss = dloss.sum() / (1e-9 + (data['next_free']*next_data['next_free']).sum())
        else:
            deriv = next_value-value+0.1*value
            dloss = ((-deriv+1e-2).relu())*data['next_free']*next_data['next_free']
            dloss = dloss.mean()
        
        
        # # contrastive loss
        # x = next_data['x'].clone()
        # a = torch.rand(len(next_data['action']), 100, next_data['action'].shape[-1]).to(device).uniform_(-1, 1.)
        # next_value_neg = bnn(x=x.unsqueeze(1).repeat(1, 100, 1), action=a)
        # deriv_neg = next_value_neg-value.reshape(len(next_value_neg)).unsqueeze(1)+0.1*value.reshape(len(next_value_neg)).unsqueeze(1)
        # good_noise = ((deriv+1e-2).relu())
        # closs = good_noise.mean()

        return bloss, dloss, ((value-data['old_v'])**2).mean()
    
    buf.construct_dataset()
    loader = buf.get()
    for i in range(n_iter):
        for data, next_data in loader:
            optimizer.zero_grad()
            data = data.to(device)
            next_data = next_data.to(device)
            bloss, dloss, closs = compute_loss(bnn, data, next_data)
            if USE_CLOSS:
                loss = bloss + dloss + closs
            else:
                loss = bloss + dloss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bnn.parameters(), 5.0)
            optimizer.step()
            optimizer.zero_grad()
    
            wandb.log({"bloss": bloss,
                       "dloss": dloss,
                       "closs": closs,})
            
    return


# create replay buffer
import scipy
from random import shuffle


class DotDict(dict):
    """
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value


class GlobalReplayBuffer:
    """
    A buffer for storing trajectories experiences
    """

    def __init__(self):
        self.obs_buf = []       
        
    def store(self, obs):
        """
        Append one timestep of agent-environment interaction to the buffer.
        expected keys: 'prev_free', 'next_free', 'prev_danger', 'next_danger', 'bvalue', 'action'
        """
        self.obs_buf.append(obs)
    
    def relabel(self, prob):

        n_relabels = 0
        # back-prop the danger set
        for idx, obs in list(zip(range(len(self.obs_buf)), self.obs_buf))[::-1]:
            if (idx < (len(self.obs_buf)-1)):
                self.obs_buf[idx]['next_danger'] = self.obs_buf[idx+1]['prev_danger']
                self.obs_buf[idx]['next_free'] = self.obs_buf[idx+1]['prev_free']
                    
            if RELABEL_ONLY_AGENT:
                obs['feasible'] = ((obs['feasible'] + obs['meet_obstacle']) >= 1).float()
            inadmissible = ((torch.rand(*(obs['next_danger'].shape)) < prob) * obs['next_danger'] * (1-obs['feasible']))

            obs['prev_danger'] = ((obs['prev_danger'] + inadmissible) >= 1).float()
            obs['prev_free'] = obs['prev_free'] * (1-obs['prev_danger'])
            n_relabels += inadmissible.sum()
        return n_relabels


class MyDataset(Dataset):
    def __init__(self, data, next_data):
        self.data = data
        self.next_data = next_data
        
    def __getitem__(self, index):
        xA = self.data[index]
        xB = self.next_data[index]
        return xA, xB
    
    def __len__(self):
        return len(self.data)            

    
def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    return batchA, batchB      
    
            
class GatherReplayBuffer:

    def __init__(self, batch=64):
        self.buffers = []
        self.batch = batch
        
    def append(self, buffer):
        self.buffers.append(buffer)
        
    def construct_dataset(self):
        prev_o = []
        prev_o.extend([o for b in self.buffers for o in b.obs_buf[:-1]])
        next_o = []
        next_o.extend([o for b in self.buffers for o in b.obs_buf[1:]])        
        self.dataset = MyDataset(prev_o, next_o)
        
    def get(self):          

        # def collate_fn(data):
        #     """
        #        data: is a list of tuples with (example, label, length)
        #              where 'example' is a tensor of arbitrary shape
        #              and label/length are scalars
        #     """
        #     o, next_o = [d[0] for d in data], [d[1] for d in data]
        #     # datas = []
        #     # for data in [o, next_o]:
        #     #     data = default_collate(data)
        #     #     for k, v in data.items():
        #     #         data[k] = v.to(device)
        #     #     datas.append(data)
        #     assert False
        #     o_batch = Batch.from_data_list(o)
        #     next_o_batch = Batch.from_data_list(next_o)
        #     return o_batch, next_o_batch

        loader = DataLoader(self.dataset, shuffle=True, batch_size=self.batch, num_workers=8, drop_last=True, collate_fn=collate)

        return loader
    

@torch.no_grad()    
def eval_action(bnn, o, a, threshold, max_iter=30):
    # size of a: (num_agents, n_candidates, action_dim)
    n_candidate = a.shape[1]

    input_ = o.to(device)
    tensor_a = torch.FloatTensor(a).to(device)

    input_['action'] = tensor_a
    vec = bnn.get_vec(input_)
    vec = vec.unsqueeze(1).repeat((1, n_candidate, 1))
    bvalue = bnn.get_field(vec, tensor_a)
    return tensor_a.data.cpu().numpy(), bvalue.data.cpu().numpy()


def choose_action(a_refine, values, threshold, explore_eps=0.1, nominal_eps=0.1):
    a = np.zeros(a_refine.shape[-1])
    if values[0] > threshold:
        a = a_refine[0]
        a_value = values[0]
    elif np.random.rand() < nominal_eps:
        a = a_refine[0]
        a_value = values[0]  
    elif np.any(values > threshold):
        idx_candidates = np.arange(len(values))[values > threshold]
        idx_candidate = np.random.choice(idx_candidates, 1)[0]
        a = a_refine[idx_candidate, :]
        a_value = values[idx_candidate]        
    else: # np.random.rand() < explore_eps:       
        idx_candidate = np.random.choice(len(values))
        a = a_refine[idx_candidate, :]
        a_value = values[idx_candidate]
    # else:
    #     # if np.any(values > threshold):
    #     #     idx_candidates = np.arange(len(values))[values > threshold]
    #     #     idx_candidate = np.random.choice(idx_candidates, 1)[0]
    #     #     a = a_refine[idx_candidate, :]
    #     #     a_value = values[idx_candidate]
    #     # else:
    #     a = a_refine[np.argmax(values), :]
    #     a_value = np.amax(values)
    return a, a_value


def save_gif(gifs, name="play.gif"):
    a_frames = []
    for img in gifs:
        a_frames.append(np.asarray(img))
    a_frames = np.stack(a_frames)
    ims = [Image.fromarray(a_frame) for a_frame in a_frames]
    ims[0].save(name, save_all=True, append_images=ims[1:], loop=0, duration=10)


@torch.no_grad()
def infer(env, bnn, threshold=None, max_episode_length=256, verbose=False, seed=0, stop_at_collision=False):
    if threshold is None:
        threshold=THRESHOLD
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    gifs = [env._render()]
    total_trans=0; n_danger=0; no_feasible=0; collided=np.zeros(env.num_agents).astype(bool)

    while True:
        o = env._get_obs()
        a_all = np.random.uniform(-1, 1, size=(env.num_agents, n_candidates, env.action_dim))
        a_refines, bvalues = eval_action(bnn, o, a_all, max_iter=0, threshold=threshold)

        dists = env.potential_field(a_refines, K1=K1, K2=K2, ignore_agent=True)
        
        v = np.zeros(env.num_agents)
        a = np.zeros((env.num_agents, env.action_dim))
        for agent_id, a_refine, bvalue, dist in zip(np.arange(env.num_agents), a_refines, bvalues, dists):
            if np.any(bvalue>threshold):
                for a_idx in np.argsort(dist):
                    if bvalue[a_idx] > threshold:
                        a[agent_id] = a_refine[a_idx]
                        v[agent_id] = bvalue[a_idx]
                        break
            else:
                no_feasible += 1
                a[agent_id] = a_refine[np.argmax(bvalue)]
                v[agent_id] = bvalue[np.argmax(bvalue)]

        next_o, rw, done, info = env.step(a)

        prev_danger = info['prev_danger'].data.cpu().numpy().astype(bool)
        next_danger = info['next_danger'].data.cpu().numpy().astype(bool)
        if np.any(next_danger):
            collided = collided | next_danger
        if verbose:
            print(bvalues.min(axis=-1), bvalues.max(axis=-1), v, next_danger)

        total_trans += 1
        gifs.append(env._render())

        if np.any(next_danger) and stop_at_collision:
            break        
        
        if done or (total_trans >= max_episode_length):
            break

    return collided, done, gifs


if __name__ == '__main__':
    
    wandb.init(
        project="dubins_car",
        config={
            "num_agents": NUM_AGENTS,
            "map_size": MAP_SIZE,
            "obstacle_density": OBSTACLE_DENSITY, 
            "n_candidates": n_candidates,
            "batch": BATCH,
            "n_traj": N_TRAJ,
            "n_iter": N_ITER,
            "threshold": THRESHOLD,
            "n_step_per_epoch": N_STEP_PER_EPOCH,
            "n_buffer": N_BUFFER,
            "n_cbuf": N_CBUF,
            "n_evaluate": N_EVALUATE,
            "patience": PATIENCE,
            "lr": LR,
            "hidden_size": HIDDEN_SIZE,
            "steer": STEER,
            "relabel": RELABEL,
            "cyclic": CYCLIC,
            "decay_relabel": DECAY_RELABEL,
            "scheduler": USE_SCHEDULER,
            "optimizer": OPTIMIZER,
            "min_explore_eps": MIN_EXPLORE_EPS,
            "max_explore_eps": MAX_EXPLORE_EPS,
            "decay_explore_rate": DECAY_EXPLORE_RATE,
            "decay_nominal_rate": DECAY_NOMINAL_RATE,
            "potential_obs": POTENTIAL_OBS,
            "train_on_hard": TRAIN_ON_HARD,
            "refine": REFINE_EPS,
            "relabel_only_agent": RELABEL_ONLY_AGENT,
            "all_lie": ALL_LIE,
        },
        name=version_name,)
    
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # generate training data
    dataset = []
    for _ in tqdm(range(N_DATASET)):
        env = Env(num_agents=NUM_AGENTS, PROB=(0.,OBSTACLE_DENSITY), SIZE=(MAP_SIZE,MAP_SIZE))
        # while True:
        #     env = Env(num_agents=NUM_AGENTS, PROB=(0.,OBSTACLE_DENSITY), SIZE=(MAP_SIZE,MAP_SIZE))
        #     if (np.linalg.norm(env.world.agents - env.world.agent_goals, axis=-1).min() >= 2):
        #         break
        dataset.append([env.world.obstacles.copy(), env.world.agent_goals.copy(), env.world.agents.copy(), 0])


    Env = Env
    env = Env(num_agents=NUM_AGENTS, PROB=(0.,OBSTACLE_DENSITY), SIZE=(MAP_SIZE,MAP_SIZE))
    bnn = create_network()

    name_dict = generate_default_model_name(Env)
    # bnn.load_state_dict(torch.load(name_dict['b'].replace('.pt', '_1model.pt'), map_location=device))

    if OPTIMIZER=='SGD':
        boptimizer = torch.optim.SGD(bnn.parameters(), lr=LR, momentum=0.9)#, weight_decay=1e-8)
    elif OPTIMIZER=='Adam':
        boptimizer = torch.optim.Adam(bnn.parameters(), lr=LR, weight_decay=1e-8)
    else:
        assert False
    if USE_SCHEDULER:
        bscheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(boptimizer, factor=0.5, patience=PATIENCE)

    max_episode_length = 256

    running_unsafe_rate = 0
    best_unsafe_rate = float('inf')
    nominal_eps = 1.0
    explore_eps = 1.0

    trajs = defaultdict(list)
    # open(TXT_NAME, 'w+').close()
    bbuf_gather = GatherReplayBuffer(BATCH)
    cbuf = GlobalReplayBuffer()
    fbuf = GlobalReplayBuffer()
    unsafe_rates = []
    uncollide_rates = []
    success_rates = []     

    for epoch_i in range(N_TRAJ):
        
        bbuf_gather = GatherReplayBuffer(BATCH)

        if CYCLIC:
            explore_eps = (MAX_EXPLORE_EPS-(MAX_EXPLORE_EPS-MIN_EXPLORE_EPS)*(epoch_i % N_VALID) / (N_VALID-1))  # if ((epoch_i % 200) < 100) else 0.
        else:
            explore_eps = np.clip(MAX_EXPLORE_EPS * (DECAY_EXPLORE_RATE ** ((epoch_i // N_VALID))), MIN_EXPLORE_EPS, MAX_EXPLORE_EPS)

        nominal_eps = DECAY_NOMINAL_RATE ** ((epoch_i // N_VALID))
        
        if nominal_eps < 0.01:
            nominal_eps = 0

        if DECAY_RELABEL:
            relabel_eps = 1 - explore_eps # (epoch_i // N_VALID) / (N_TRAJ // N_VALID)
        else:
            relabel_eps = REFINE_EPS

        if (epoch_i % len(dataset) == 0):
            shuffle(dataset)

        bbuf = GlobalReplayBuffer(); total_step = 0
        
        while total_step < N_STEP_PER_EPOCH:

            env = Env(num_agents=NUM_AGENTS, PROB=(0.,OBSTACLE_DENSITY), SIZE=(MAP_SIZE,MAP_SIZE))
            dataset[epoch_i%len(dataset)][-1] += 1; visit_time = 0
            if TRAIN_ON_HARD:
                if epoch_i > N_WARMUP:
                    env.world.obstacles, env.world.agent_goals, env.world.agents, visit_time = deepcopy(dataset[epoch_i%len(dataset)])

            total_trans=0; n_danger=0; threshold=THRESHOLD; no_feasible=0; collided=np.zeros(env.num_agents).astype(bool)

            while True:
                o = env._get_obs()

                a_all = np.random.uniform(-1, 1, size=(env.num_agents, n_candidates, env.action_dim))

                a_refines, bvalues = eval_action(bnn, o, a_all, max_iter=0, threshold=threshold)

                dists = env.potential_field(a_refines, K1=K1, K2=K2, ignore_agent=True)

                feasibles = np.zeros(env.num_agents)
                a = np.zeros((env.num_agents, env.action_dim))
                b = np.zeros(env.num_agents)
                for agent_id, a_refine, bvalue, dist in zip(np.arange(env.num_agents), a_refines, bvalues, dists):
                    if np.random.rand() < nominal_eps:
                        a[agent_id] = a_refine[np.argsort(dist)[0]]
                        b[agent_id] = bvalue[np.argsort(dist)[0]]
                        feasibles[agent_id] = 1
                    elif np.any(bvalue>threshold):
                        feasibles[agent_id] = 1
                        for a_idx in np.argsort(dist):
                            if bvalue[a_idx] > threshold:
                                a[agent_id] = a_refine[a_idx]
                                b[agent_id] = bvalue[a_idx]
                                break
                    else:
                        no_feasible += 1
                        if np.random.rand()<explore_eps:
                            a_idx = np.random.randint(n_candidates)
                            a[agent_id] = a_refine[a_idx]
                            b[agent_id] = bvalue[a_idx]
                            if a_idx != np.argmax(bvalue):
                                feasibles[agent_id] = 1  # mask the random action
                        else:
                            a[agent_id] = a_refine[np.argmax(bvalue)]
                            b[agent_id] = bvalue[np.argmax(bvalue)]

                next_o, rw, done, info = env.step(a)

                info['feasible'] = torch.FloatTensor(feasibles)
                info['old_v'] = torch.FloatTensor(b)
                bbuf.store(info.clone())
                prev_danger = info['prev_danger'].data.cpu().numpy().astype(bool)
                next_danger = info['next_danger'].data.cpu().numpy().astype(bool)
                if np.any(next_danger):
                    collided = collided | next_danger

                total_trans += 1
                total_step += 1
                n_danger += np.array(next_danger).sum()

                if done or (total_trans >= max_episode_length):
                    unsafe_rates.append(collided.mean())
                    uncollide_rates.append(np.any(collided))
                    success_rates.append(done and (not np.any(collided)))
                    running_unsafe_rate = np.mean(unsafe_rates)
                    
                    if (n_danger==0) or (visit_time > 3):
                        # only preserve the hard envs
                        dataset.pop(epoch_i%len(dataset))
                        env = Env(num_agents=NUM_AGENTS, PROB=(0.,OBSTACLE_DENSITY), SIZE=(MAP_SIZE,MAP_SIZE))
                        # while True:
                        #     env = Env(num_agents=NUM_AGENTS, PROB=(0.,OBSTACLE_DENSITY), SIZE=(MAP_SIZE,MAP_SIZE))
                        #     if (np.linalg.norm(env.world.agents[:,:2] - env.world.agent_goals[:,:2], axis=-1).min() >= 2):
                        #         break
                        dataset.append([env.world.obstacles.copy(), env.world.agent_goals.copy(), env.world.agents.copy(), 0])

                    if (n_danger!=0):
                        for info in bbuf.obs_buf:
                            prev_danger = info['prev_danger'].data.cpu().numpy().astype(bool)
                            next_danger = info['next_danger'].data.cpu().numpy().astype(bool)                        
                            if (next_danger).any():
                                data = info.clone()
                                data['prev_free'] = torch.FloatTensor([False]*env.num_agents)
                                data['next_free'] = torch.FloatTensor([False]*env.num_agents)
                                cbuf.store(data)
                                if len(cbuf.obs_buf) > N_CBUF:
                                    cbuf.obs_buf.pop(0)  

                        if RELABEL:
                            n_relabels = bbuf.relabel(relabel_eps)  
                            wandb.log({"n_relabels": n_relabels,
                                       "no_feasible": no_feasible,})

                    bbuf_gather.append(bbuf)
                    break  
                                         
            
        if (epoch_i % (N_VALID) == (N_VALID-1)):

            torch.save(bnn.state_dict(), BMODEL_PATH.replace('.pt', '_{0:d}.pt'.format(epoch_i)))
            
            wandb.log({"uncollide_rates": np.mean(uncollide_rates),
                       "success_rates": np.mean(success_rates),
                       "running_unsafe_rate": running_unsafe_rate,})             
            
            if USE_SCHEDULER:
                bscheduler.step(running_unsafe_rate)     
            
            unsafe_rates = []
            uncollide_rates = []
            success_rates = []               

        if running_unsafe_rate < best_unsafe_rate:
            best_unsafe_rate = running_unsafe_rate
            torch.save(bnn.state_dict(), BMODEL_PATH)        

        subcbuf = GlobalReplayBuffer()
        shuffle(cbuf.obs_buf)
        subcbuf.obs_buf = cbuf.obs_buf[:100]
        bbuf_gather.buffers.append(subcbuf)
        bnn.train()
        train_barrier(bnn, boptimizer, bbuf_gather, n_iter=N_ITER)
        torch.save(bnn.state_dict(), BMODEL_PATH.replace('.pt', '_current.pt'))
        bbuf_gather.buffers.pop(-1)
        
        
        wandb.log({"lr": boptimizer.param_groups[0]['lr'],
                   "explore_eps": explore_eps,
                   "nominal_eps": nominal_eps,
                   "relabel_prob": relabel_eps,})
        