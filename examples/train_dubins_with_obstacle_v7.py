from tqdm import tqdm
import gc
from copy import deepcopy
from gym_dubins_car import DubinsCarEnv, STEER

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
compare to v7: 
1. make the cbuf size larger
2. make the warmup longer
3. make the valid dataset smaller 
4. change the preference to the potential field controller
'''

Env = DubinsCarEnv
name_dict = generate_default_model_name(Env)
TXT_NAME = '1model_'+Env.__name__+'_v7.txt'
BMODEL_PATH = name_dict['db'].replace('.pt', '_v7.pt')


N_TRAJ = N_EPOCH = 4000
N_CBUF = 1000000
PATIENCE = 2

NUM_AGENTS = 3
MAP_SIZE = 3
OBSTACLE_DENSITY = 1.
n_candidates = 2000
BATCH = 256
N_ITER = 100
N_TRAJ_PER_EPOCH = 10
N_BUFFER = 20
N_EVALUATE = 100 # len(dataset)
N_VALID = 100
N_WARMUP = 100 # len(dataset)
N_DATASET = 10
N_VALID_DATASET = 20
THRESHOLD = 5e-2
HIDDEN_SIZE = 128
LR = 1e-2
RELABEL = True
CYCLIC = False
DECAY_RELABEL = False
SAVE_GIF = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)


def train_barrier(bnn, optimizer, buf, pbar, n_iter=10):
    
    # Set up function for computing value loss
    def compute_loss(bnn, data, next_data):
        value = bnn(data)
        next_value = bnn(next_data)
        
        bloss1 = ((1e-2-value).relu())*data['next_free'] / (1e-9 + (data['next_free']).sum())
        bloss2 = ((1e-2+value).relu())*data['next_danger'] / (1e-9 + (data['next_danger'].sum()))
        bloss = bloss1.sum() + bloss2.sum()
        
        deriv = next_value-value+0.1*value
        near_boundary = 1  # torch.minimum(next_value, value) < 5e-2
        dloss = ((-deriv+1e-2).relu())*data['next_free']*next_data['next_free']*near_boundary
        dloss = dloss.sum() / (1e-9 + (data['next_free']*next_data['next_free']*near_boundary).sum())
        
        
        # # contrastive loss
        # x = next_data['x'].clone()
        # a = torch.rand(len(next_data['action']), 100, next_data['action'].shape[-1]).to(device).uniform_(-1, 1.)
        # next_value_neg = bnn(x=x.unsqueeze(1).repeat(1, 100, 1), action=a)
        # deriv_neg = next_value_neg-value.reshape(len(next_value_neg)).unsqueeze(1)+0.1*value.reshape(len(next_value_neg)).unsqueeze(1)
        # good_noise = ((deriv+1e-2).relu())
        # closs = good_noise.mean()

        return bloss, dloss, 0
    
    # imitation learning
    loader = buf.get()
    for i in range(n_iter):
        for data, next_data in loader:
            optimizer.zero_grad()
            bloss, dloss, closs = compute_loss(bnn, data.to(device), next_data.to(device))
            loss = bloss + dloss + closs
            loss.backward()            
            optimizer.step()
            optimizer.zero_grad()
            pbar.set_description("iter #%d, bloss %.6f, dloss %.6f, closs %.6f" % (i, bloss, dloss, closs))
            wandb.log({"bloss": bloss,
                       "dloss": dloss})
    
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
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
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

        # back-prop the danger set
        for idx, obs in list(zip(range(len(self.obs_buf)), self.obs_buf))[::-1]:
            if idx < (len(self.obs_buf)-1):
                self.obs_buf[idx]['next_danger'] = self.obs_buf[idx+1]['prev_danger']
                self.obs_buf[idx]['next_free'] = self.obs_buf[idx+1]['prev_free']
            obs['prev_danger'] = ((obs['prev_danger'] + ((torch.rand(*(obs['next_danger'].shape)) < prob) * obs['next_danger'] * (1-obs['feasible']))) >= 1).float()
            obs['prev_free'] = obs['prev_free'] * (1-obs['prev_danger'])


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

            
class GatherReplayBuffer:

    def __init__(self, batch=64):
        self.buffers = []
        self.batch = batch
        self.construct = False
        
    def append(self, buffer):
        self.buffers.append(buffer)
        self.construct = False
        
    def construct_dataset(self):
        prev_o = []
        prev_o.extend([o for b in self.buffers for o in b.obs_buf[:-1]])
        next_o = []
        next_o.extend([o for b in self.buffers for o in b.obs_buf[1:]])        
        self.dataset = MyDataset(prev_o, next_o)
        
    def get(self):
        if not self.construct:
            self.construct_dataset()  
            self.construct = True

        def collate(data_list):
            batchA = Batch.from_data_list([data[0] for data in data_list])
            batchB = Batch.from_data_list([data[1] for data in data_list])
            return batchA, batchB            
            
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

        loader = DataLoader(self.dataset, shuffle=True, batch_size=self.batch, collate_fn=collate)

        return loader
    
    
def iter_action(bnn, o, a, threshold, max_iter=30):
    # size of a: (num_agents, n_candidates, action_dim)
    n_candidate = a.shape[1]
    
    bnn.eval()
    
    input_ = o.to(device)
    tensor_a = torch.FloatTensor(a).to(device)
    origin_a = torch.FloatTensor(a).to(device)

    tensor_a.requires_grad = True
    input_['action'] = tensor_a
    vec = bnn.get_vec(input_).detach()
    vec = vec.unsqueeze(1).repeat((1, n_candidate, 1))
    
    aoptimizer = torch.optim.SGD([tensor_a], lr=1)
    
    iter_ = 0
    while iter_ < max_iter:
        bvalue = bnn.get_field(vec, tensor_a)
        if bvalue[0]>threshold:
            break
        aoptimizer.zero_grad()
        ((-bvalue+threshold).relu().sum()+((tensor_a-origin_a)**2).sum()).backward()
        torch.nn.utils.clip_grad_value_([tensor_a], 1e-2)
        aoptimizer.step()        
        iter_ += 1

    with torch.no_grad():
        tensor_a[:] = tensor_a.clamp(-1., 1.)
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
def infer(env, bnn, verbose=False, seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    gifs = [env._render()]
    max_episode_length = 256
    total_trans=0; n_danger=0; threshold=THRESHOLD; no_feasible=0; collided=np.zeros(env.num_agents).astype(bool)

    while True:
        o = env._get_obs()
        a_all = np.random.uniform(-1, 1, size=(env.num_agents, n_candidates, env.action_dim))
        a_refines, bvalues = iter_action(bnn, o, a_all, max_iter=0, threshold=threshold)

        next_states = np.tile(env.world.agents.reshape(env.num_agents, 1, -1), (1, n_candidates, 1))
        ax, ay, theta = next_states[:, :, 0], next_states[:, :, 1], next_states[:, :, 2]
        theta = theta + STEER*a_refines.squeeze(-1)
        dx, dy = 0.05*np.cos(theta), 0.05*np.sin(theta)
        next_states[:, :, 0] = ax + dx
        next_states[:, :, 1] = ay + dy
        theta = theta + (theta - 2* np.pi)*(theta>np.pi) + (theta + 2* np.pi)*(theta<-np.pi)
        next_states[:, :, 2] = theta

        dists = np.linalg.norm((next_states-env.world.agent_goals.reshape(env.num_agents, 1, -1))[:,:,:2], axis=-1)
        
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
            "n_traj_per_epoch": N_TRAJ_PER_EPOCH,
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
    })
    
    env = Env(num_agents=3, mode='barrier', PROB=(0,0.1), SIZE=(8,8))

    # generate training data
    dataset = []
    for _ in tqdm(range(N_DATASET)):
        while True:
            env = Env(num_agents=NUM_AGENTS, mode='barrier', PROB=(0.,OBSTACLE_DENSITY), SIZE=(MAP_SIZE,MAP_SIZE))
            if (np.linalg.norm(env.world.agents - env.world.agent_goals, axis=-1).min() >= 2):
                break
        dataset.append([env.world.obstacles.copy(), env.world.agent_goals.copy(), env.world.agents.copy(), 0])

    # generate valid data
    valid_dataset = []
    for _ in tqdm(range(N_VALID_DATASET)):
        while True:
            env = Env(num_agents=NUM_AGENTS, mode='barrier', PROB=(0.,OBSTACLE_DENSITY), SIZE=(MAP_SIZE,MAP_SIZE))
            if (np.linalg.norm(env.world.agents - env.world.agent_goals, axis=-1).min() >= 2):
                break
        valid_dataset.append((env.world.obstacles.copy(), env.world.agent_goals.copy(), env.world.agents.copy()))

    
    Env = Env
    env = Env(num_agents=NUM_AGENTS, mode='barrier', PROB=(0.,OBSTACLE_DENSITY), SIZE=(MAP_SIZE,MAP_SIZE))

    # bnn = HeteroGNN(HIDDEN_SIZE, 3, keys=['agent', 'obstacle'])
    if OBSTACLE_DENSITY == 0:
        bnn = OriginGNNv3(HIDDEN_SIZE, 3, keys=['agent'])
    else:
        bnn = OriginGNNv3(HIDDEN_SIZE, 3, keys=['agent', 'obstacle'])
    bnn.to(device)

    name_dict = generate_default_model_name(Env)
    # bnn.load_state_dict(torch.load(name_dict['b'].replace('.pt', '_1model.pt'), map_location=device))

    boptimizer = torch.optim.SGD(bnn.parameters(), lr=LR, momentum=0.9)#, weight_decay=1e-8)
    # boptimizer = torch.optim.Adam(bnn.parameters(), lr=1e-4)#, weight_decay=1e-8)
    bscheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(boptimizer, factor=0.5, patience=PATIENCE, eps=1e-6)  

    max_episode_length = 256

    running_unsafe_rate = 0
    best_unsafe_rate = float('inf')
    unsafe_rates = [1.]*N_EVALUATE
    uncollide_rates = [1.]*N_EVALUATE
    nominal_eps = 1.0
    explore_eps = 1.0

    trajs = defaultdict(list)
    open(TXT_NAME, 'w+').close()
    bbuf_gather = GatherReplayBuffer(BATCH)
    cbuf = GlobalReplayBuffer()
    fbuf = GlobalReplayBuffer()

    pbar = tqdm(range(N_TRAJ))
    for epoch_i in pbar:
        
        if CYCLIC:
            explore_eps = (1.-(epoch_i % 100)/50.) if ((epoch_i % 100) < 50) else 0.
        else:
            explore_eps = np.clip(1. - 1.5 * (epoch_i // N_VALID) / (N_TRAJ // N_VALID), 0, 0.5)
        
        if DECAY_RELABEL:
            relabel_eps = 1 - explore_eps # (epoch_i // N_VALID) / (N_TRAJ // N_VALID)
        else:
            relabel_eps = 1.

        if (epoch_i % len(dataset) == 0):
            shuffle(dataset)

        bbuf = GlobalReplayBuffer()

        while True:
            env = Env(num_agents=NUM_AGENTS, mode='barrier', PROB=(0.,OBSTACLE_DENSITY), SIZE=(MAP_SIZE,MAP_SIZE))
            if (np.linalg.norm(env.world.agents[:,:2] - env.world.agent_goals[:,:2], axis=-1).min() >= 2):
                break
        dataset[epoch_i%len(dataset)][-1] += 1
        env.world.obstacles, env.world.agent_goals, env.world.agents, visit_time = deepcopy(dataset[epoch_i%len(dataset)])
    #     half_size = MAP_SIZE / 2
    #     env.world.agents = np.array([[half_size+(half_size-0.5)*np.cos(a), half_size+(half_size-0.5)*np.sin(a)]+[0.]*(Env.state_dim-2) for a in np.linspace(0, 2*np.pi, NUM_AGENTS, endpoint=False)])
    #     env.world.agent_goals = np.array([[half_size-(half_size-0.5)*np.cos(a), half_size-(half_size-0.5)*np.sin(a)]+[0.]*(Env.state_dim-2) for a in np.linspace(0, 2*np.pi, NUM_AGENTS, endpoint=False)])

        total_trans=0; n_danger=0; threshold=THRESHOLD; no_feasible=0; collided=np.zeros(env.num_agents).astype(bool)

        while True:
            o = env._get_obs()

            a_all = np.random.uniform(-1, 1, size=(env.num_agents, n_candidates, env.action_dim))  # np.hstack([i.reshape(-1, 1) for i in np.meshgrid(*([np.linspace(-1, 1, 20)]*2))])

            # o = {'x': torch.FloatTensor(o), 'goal': torch.FloatTensor(env.goal)}
            a_refines, bvalues = iter_action(bnn, o, a_all, max_iter=0, threshold=threshold)  # min(epoch_i//10, 30)

            next_states = np.tile(env.world.agents.reshape(env.num_agents, 1, -1), (1, n_candidates, 1))
            ax, ay, theta = next_states[:, :, 0], next_states[:, :, 1], next_states[:, :, 2]
            theta = theta + STEER*a_refines.squeeze(-1)
            dx, dy = 0.05*np.cos(theta), 0.05*np.sin(theta)
            next_states[:, :, 0] = ax + dx
            next_states[:, :, 1] = ay + dy
            theta = theta + (theta - 2* np.pi)*(theta>np.pi) + (theta + 2* np.pi)*(theta<-np.pi)
            next_states[:, :, 2] = theta

            dists = np.linalg.norm((next_states-env.world.agent_goals.reshape(env.num_agents, 1, -1))[:,:,:2], axis=-1)
            
            feasibles = np.zeros(env.num_agents)
            a = np.zeros((env.num_agents, env.action_dim))
            for agent_id, a_refine, bvalue, dist in zip(np.arange(env.num_agents), a_refines, bvalues, dists):
                if epoch_i < N_WARMUP:  # warmup
                    a[agent_id] = a_refine[np.argsort(dist)[0]]
                    feasibles[agent_id] = 1
                elif np.any(bvalue>threshold):
                    feasibles[agent_id] = 1
                    for a_idx in np.argsort(dist):
                        if bvalue[a_idx] > threshold:
                            a[agent_id] = a_refine[a_idx]
                            break
                else:
                    no_feasible += 1
                    if np.random.rand()<explore_eps:
                        a_idx = np.random.randint(n_candidates)
                        a[agent_id] = a_refine[a_idx]
                        if a_idx != np.argmax(bvalue):
                            feasibles[agent_id] = 1  # mask the random action
                    else:
                        a[agent_id] = a_refine[np.argmax(bvalue)]

            # for zip(a, dists)

            # a, a_value = choose_action(a_refine, bvalue, threshold=threshold, explore_eps=0, nominal_eps=nominal_eps)  # 

            next_o, rw, done, info = env.step(a)

            info['feasible'] = torch.FloatTensor(feasibles)
            bbuf.store(info.clone())
            prev_danger = info['prev_danger'].data.cpu().numpy().astype(bool)
            next_danger = info['next_danger'].data.cpu().numpy().astype(bool)
            if np.any(next_danger):
                collided = collided | next_danger

            total_trans += 1
            n_danger += np.array(next_danger).sum()

            if done or (total_trans >= max_episode_length):
                if (n_danger==0) or (visit_time > 3):
                    # only preserve the hard envs
                    dataset.pop(epoch_i%len(dataset))
                    while True:
                        env = Env(num_agents=NUM_AGENTS, mode='barrier', PROB=(0.,OBSTACLE_DENSITY), SIZE=(MAP_SIZE,MAP_SIZE))
                        if (np.linalg.norm(env.world.agents[:,:2] - env.world.agent_goals[:,:2], axis=-1).min() >= 2):
                            break
                    dataset.append([env.world.obstacles.copy(), env.world.agent_goals.copy(), env.world.agents.copy(), 0])

                if (n_danger!=0) and (epoch_i > N_WARMUP):
                    if RELABEL:
                        bbuf.relabel(relabel_eps)
                    for info in bbuf.obs_buf:
                        prev_danger = info['prev_danger'].data.cpu().numpy().astype(bool)
                        next_danger = info['next_danger'].data.cpu().numpy().astype(bool)                        
                        if (next_danger & (~prev_danger)).any():
                            data = info.clone()
                            data['prev_free'] = torch.FloatTensor([False]*env.num_agents)
                            data['next_free'] = torch.FloatTensor([False]*env.num_agents)
                            cbuf.store(data)
                            if len(cbuf.obs_buf) > N_CBUF:
                                cbuf.obs_buf.pop(0)                 
                    
                    
                #     meet_danger = False
                #     for o in bbuf.obs_buf:
                #         if meet_danger:
                #             o['prev_free'] = 0 * o['prev_free']
                #             o['next_free'] = 0 * o['next_free']
                #         if o['next_danger']:
                #             meet_danger = True
                    # TODO: not feasible if next state is not feasible
                    # for o in bbuf.obs_buf:
                bbuf_gather.append(bbuf)
                # else:
                #     fbuf.obs_buf.append(bbuf)
                break

        unsafe_rates.append(collided.mean())
        unsafe_rates.pop(0)
        uncollide_rates.append(np.any(collided))
        uncollide_rates.pop(0)    
        running_unsafe_rate = np.mean(unsafe_rates)

        if len(bbuf_gather.buffers) > N_BUFFER:
            bbuf_gather.buffers.pop(0)

        if (epoch_i % N_TRAJ_PER_EPOCH) == (N_TRAJ_PER_EPOCH-1) and (running_unsafe_rate!=0):
            if running_unsafe_rate < best_unsafe_rate:
                best_unsafe_rate = running_unsafe_rate
                torch.save(bnn.state_dict(), BMODEL_PATH)        

            subcbuf = GlobalReplayBuffer()
            shuffle(cbuf.obs_buf)
            subcbuf.obs_buf = cbuf.obs_buf[:100]
            bbuf_gather.buffers.append(subcbuf)
            bnn.train()
            train_barrier(bnn, boptimizer, bbuf_gather, pbar=pbar, n_iter=N_ITER)
            torch.save(bnn.state_dict(), BMODEL_PATH.replace('.pt', '_current.pt'))
            bbuf_gather.buffers.pop(-1)
            
            if epoch_i == 9:
                print(bnn)

            
        with open(TXT_NAME, 'a+') as f:
            f.write(pbar.desc+'\t'+str(epoch_i)+'\t'+'collided: {0:.2f}'.format(collided.mean())+' running rate: {0:.6f}, {1:6f}'.format(running_unsafe_rate, np.mean(uncollide_rates))+' no feasible: {0:d}'.format(no_feasible)+', buf size {0:d} {1:d} {2:d}'.format(len(bbuf.obs_buf), len(cbuf.obs_buf), visit_time)+' lr: {0:.2e}'.format(boptimizer.param_groups[0]['lr'])+'\n')
        
        wandb.log({"lr": boptimizer.param_groups[0]['lr'],
                   "uncollide_rates": np.mean(uncollide_rates),
                   "running_unsafe_rate": running_unsafe_rate,
                   "explore_eps": explore_eps,
                   "relabel_prob": relabel_eps,
                   "no_feasible": no_feasible,})

        if (epoch_i % (N_VALID) == (N_VALID-1)):

            valid_loss = 0
            valid_length = 0
            for v_idx, data in enumerate(valid_dataset):
                env = Env(num_agents=NUM_AGENTS, mode='barrier', PROB=(0.,OBSTACLE_DENSITY), SIZE=(MAP_SIZE,MAP_SIZE))
                env.world.obstacles, env.world.agent_goals, env.world.agents = deepcopy(data)
                collided, done, gifs = infer(env, bnn)
                valid_loss += np.any(collided)
                valid_length += len(gifs)
                if SAVE_GIF and np.any(collided):
                    save_gif(gifs, 'gifs/'+Env.__name__+str(v_idx)+'.gif')
            torch.save(bnn.state_dict(), BMODEL_PATH.replace('.pt', '{0:d}_{1:.2f}.pt'.format(epoch_i, valid_loss/len(valid_dataset))))
            
            bscheduler.step(valid_loss/len(valid_dataset))
            wandb.log({"valid loss": valid_loss/len(valid_dataset),
                       "valid length": valid_length/len(valid_dataset)})
        