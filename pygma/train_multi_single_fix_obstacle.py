from tensorboardX import SummaryWriter
from tqdm import tqdm
import gc
from copy import deepcopy
from gym_multi_point import MultiPointEnv

import faulthandler
faulthandler.enable()

import torch
import numpy as np
from torch import nn
import math
from models import *
from core import generate_default_model_name

import scipy
from random import shuffle
import random
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from collections import defaultdict
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch.utils.data import DataLoader, Dataset


NUM_AGENTS = 3
MAP_SIZE = 3
OBSTACLE_DENSITY = 0.1
n_candidates = 2000
BATCH = 256
N_ITER = 100
N_TRAJ = N_EPOCH = 100
N_TRAJ_PER_EPOCH = 1
N_BUFFER = 10
N_CBUF = 10000
N_EVALUATE = 20 # len(dataset)
N_LOG = 25000
N_WARMUP = 4 # len(dataset)
THRESHOLD=2e-2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_barrier(bnn, optimizer, buf, writer, pbar, n_iter=10):
    
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
    
    writer.add_scalar('Losses/Barrier Loss', loss, writer.step)    
    writer.step += 1 
    
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
    
    def relabel(self, theta=5e-2):

        # back-prop the danger set
        for idx, obs in list(zip(range(len(self.obs_buf)), self.obs_buf))[::-1]:
            if idx < (len(self.obs_buf)-1):
                self.obs_buf[idx]['next_danger'] = self.obs_buf[idx+1]['prev_danger']
                self.obs_buf[idx]['next_free'] = self.obs_buf[idx+1]['prev_free']
            if idx <  (len(self.obs_buf)-1):
                obs['prev_danger'] = ((obs['prev_danger'] + (obs['next_danger'] * (obs['bvalue']<theta)) + ((1-self.obs_buf[idx+1]['feasible'])*(obs['bvalue']<theta))) >= 1).float()
            else:
                obs['prev_danger'] = ((obs['prev_danger'] + (obs['next_danger'] * (obs['bvalue']<theta))) >= 1).float()
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


@torch.no_grad()
def infer(env, verbose=False, seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    max_episode_length = 256
    total_trans=0; n_danger=0; threshold=THRESHOLD; no_feasible=0; collided=np.zeros(env.num_agents).astype(bool)

    while True:
        o = env._get_obs()
        a_all = np.random.uniform(-1, 1, size=(env.num_agents, n_candidates, env.action_dim))
        a_refines, bvalues = iter_action(bnn, o, a_all, max_iter=0, threshold=threshold)

        dists = []
        next_states = env.world.agents.reshape(env.num_agents, 1, -1)+0.3*a_refines
        dists = np.linalg.norm(next_states-env.world.agent_goals.reshape(env.num_agents, 1, -1), axis=-1)
        a = np.zeros((env.num_agents, env.action_dim))
        v = np.zeros(env.num_agents)
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

        if done or (total_trans >= max_episode_length):
            break

    return collided, done


if __name__ == '__main__':
    Env = MultiPointEnv
    env = MultiPointEnv(num_agents=3, mode='barrier', PROB=(0,0.1), SIZE=(8,8))
    name_dict = generate_default_model_name(Env)

    pos = np.array(np.meshgrid(np.linspace(0.5, 2.5, 3), np.linspace(0.5, 2.5, 3))).reshape(2, -1).T.tolist()
    pos.pop(4)
    pos = np.array(pos)

    # generate training data
    dataset = []
    for _ in tqdm(range(20)):
        while True:
            env = MultiPointEnv(num_agents=NUM_AGENTS, mode='barrier', PROB=(0,OBSTACLE_DENSITY), SIZE=(MAP_SIZE,MAP_SIZE))
            env.world.obstacles = [[1.5, 1.5]]
            env.world.agents = pos[np.random.choice(len(pos), size=(3,), replace=False)]
            env.world.agent_goals = pos[np.random.choice(len(pos), size=(3,), replace=False)]
            if np.linalg.norm(env.world.agents - env.world.agent_goals, axis=-1).min() > 2:
                break
        dataset.append((env.world.obstacles.copy(), env.world.agent_goals.copy(), env.world.agents.copy()))

    # generate valid data
    valid_dataset = []
    for _ in tqdm(range(20)):
        while True:
            env = MultiPointEnv(num_agents=NUM_AGENTS, mode='barrier', PROB=(0,OBSTACLE_DENSITY), SIZE=(MAP_SIZE,MAP_SIZE))
            env.world.obstacles = [[1.5, 1.5]]
            env.world.agents = pos[np.random.choice(len(pos), size=(3,), replace=False)]
            env.world.agent_goals = pos[np.random.choice(len(pos), size=(3,), replace=False)]
            if np.linalg.norm(env.world.agents - env.world.agent_goals, axis=-1).min() > 2:
                break
        valid_dataset.append((env.world.obstacles.copy(), env.world.agent_goals.copy(), env.world.agents.copy()))

    
    Env = MultiPointEnv
    env = MultiPointEnv(num_agents=NUM_AGENTS, mode='barrier', PROB=(0,OBSTACLE_DENSITY), SIZE=(MAP_SIZE,MAP_SIZE))

    bnn = HeteroGNN(64, 3, keys=['agent', 'obstacle'])
    bnn.to(device)
    print(bnn.train())

    name_dict = generate_default_model_name(Env)
    # bnn.load_state_dict(torch.load(name_dict['b'].replace('.pt', '_1model.pt'), map_location=device))

    boptimizer = torch.optim.SGD(bnn.parameters(), lr=1e-3, momentum=0.9)#, weight_decay=1e-8)
    bscheduler = torch.optim.lr_scheduler.ExponentialLR(boptimizer, gamma=0.89)  

    writer = SummaryWriter()
    writer.step = 0
    max_episode_length = 256

    running_unsafe_rate = 0
    best_unsafe_rate = float('inf')
    unsafe_rates = [1.]*N_EVALUATE
    uncollide_rates = [1.]*N_EVALUATE
    nominal_eps = 1.0
    explore_eps = 1.0

    trajs = defaultdict(list)
    BMODEL_PATH = name_dict['db'].replace('.pt', '_fix.pt')
    open('1model_fix_'+Env.__name__+'.txt', 'w+').close()
    bbuf_gather = GatherReplayBuffer(BATCH)
    cbuf = GlobalReplayBuffer()
    fbuf = GlobalReplayBuffer()

    pbar = tqdm(range(N_TRAJ))
    for epoch_i in pbar:

        if (epoch_i % len(dataset) == 0):
            shuffle(dataset)

        bbuf = GlobalReplayBuffer()

        while True:
            env = MultiPointEnv(num_agents=NUM_AGENTS, mode='barrier', PROB=(0,OBSTACLE_DENSITY), SIZE=(MAP_SIZE,MAP_SIZE))
            env.world.obstacles = [[1.5, 1.5]]
            env.world.agents = pos[np.random.choice(len(pos), size=(3,), replace=False)]
            env.world.agent_goals = pos[np.random.choice(len(pos), size=(3,), replace=False)]
            if np.linalg.norm(env.world.agents - env.world.agent_goals, axis=-1).min() > 2:
                break
        # env.world.obstacles, env.world.agent_goals, env.world.agents = deepcopy(dataset[epoch_i%len(dataset)])
    #     half_size = MAP_SIZE / 2
    #     env.world.agents = np.array([[half_size+(half_size-0.5)*np.cos(a), half_size+(half_size-0.5)*np.sin(a)]+[0.]*(Env.state_dim-2) for a in np.linspace(0, 2*np.pi, NUM_AGENTS, endpoint=False)])
    #     env.world.agent_goals = np.array([[half_size-(half_size-0.5)*np.cos(a), half_size-(half_size-0.5)*np.sin(a)]+[0.]*(Env.state_dim-2) for a in np.linspace(0, 2*np.pi, NUM_AGENTS, endpoint=False)])

        # env.state = env.state + np.random.uniform(-2e-2, 2e-2, size=(2,))
    #     env.world.agents = env.world.sample_agents(env.num_agents, prob=0.0)
        total_trans=0; n_danger=0; threshold=THRESHOLD; no_feasible=0; collided=np.zeros(env.num_agents).astype(bool)

        while True:
            o = env._get_obs()

            a_all = np.random.uniform(-1, 1, size=(env.num_agents, n_candidates, env.action_dim))  # np.hstack([i.reshape(-1, 1) for i in np.meshgrid(*([np.linspace(-1, 1, 20)]*2))])

            # o = {'x': torch.FloatTensor(o), 'goal': torch.FloatTensor(env.goal)}
            a_refines, bvalues = iter_action(bnn, o, a_all, max_iter=0, threshold=threshold)  # min(epoch_i//10, 30)

            dists = []
            next_states = env.world.agents.reshape(env.num_agents, 1, -1)+0.3*a_refines
            dists = np.linalg.norm(next_states-env.world.agent_goals.reshape(env.num_agents, 1, -1), axis=-1)
            a = np.zeros((env.num_agents, env.action_dim))
            for agent_id, a_refine, bvalue, dist in zip(np.arange(env.num_agents), a_refines, bvalues, dists):
                if epoch_i < N_WARMUP:  # warmup
                    a[agent_id] = a_refine[np.argsort(dist)[0]]
                elif np.any(bvalue>threshold):
                    for a_idx in np.argsort(dist):
                        if bvalue[a_idx] > threshold:
                            a[agent_id] = a_refine[a_idx]
                            break
                else:
                    no_feasible += 1
                    if np.random.rand()<explore_eps:
                        a[agent_id] = a_refine[np.random.randint(n_candidates)]
                    else:
                        a[agent_id] = a_refine[np.argmax(bvalue)]

            # for zip(a, dists)

            # a, a_value = choose_action(a_refine, bvalue, threshold=threshold, explore_eps=0, nominal_eps=nominal_eps)  # 

            next_o, rw, done, info = env.step(a)

            bbuf.store(info)
            prev_danger = info['prev_danger'].data.cpu().numpy().astype(bool)
            next_danger = info['next_danger'].data.cpu().numpy().astype(bool)
            if next_danger.any():
                info['prev_free'] = torch.FloatTensor([False]*env.num_agents)
                info['next_free'] = torch.FloatTensor([False]*env.num_agents)
                cbuf.store(info)
                if len(cbuf.obs_buf) > N_CBUF:
                    cbuf.obs_buf.pop(0)
                collided = collided | next_danger

            total_trans += 1
            n_danger += np.array(next_danger).sum()

            if done or (total_trans >= max_episode_length):
                if n_danger==0:
                    # only preserve the hard envs
                    dataset.pop(epoch_i%len(dataset))
                    while True:
                        env = MultiPointEnv(num_agents=NUM_AGENTS, mode='barrier', PROB=(0,OBSTACLE_DENSITY), SIZE=(MAP_SIZE,MAP_SIZE))
                        env.world.obstacles = [[1.5, 1.5]]
                        env.world.agents = pos[np.random.choice(len(pos), size=(3,), replace=False)]
                        env.world.agent_goals = pos[np.random.choice(len(pos), size=(3,), replace=False)]
                        if np.linalg.norm(env.world.agents - env.world.agent_goals, axis=-1).min() > 2:
                            break
                    dataset.append((env.world.obstacles.copy(), env.world.agent_goals.copy(), env.world.agents.copy()))            

                # if n_danger!=0:
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

            bbuf_gather.buffers.append(cbuf)
            bnn.train()
            train_barrier(bnn, boptimizer, bbuf_gather, writer, pbar=pbar, n_iter=N_ITER)
            torch.save(bnn.state_dict(), BMODEL_PATH.replace('.pt', '_current.pt'))
            bbuf_gather.buffers.pop(-1)

            # if (epoch_i % 10 == 9):
            #     bscheduler.step()
            
        with open('1model_fix_'+Env.__name__+'.txt', 'a+') as f:
            f.write(pbar.desc+'\t'+str(epoch_i)+'\t'+'collided: {0:.2f}'.format(collided.mean())+' running rate: {0:.6f}, {1:6f}'.format(running_unsafe_rate, np.mean(uncollide_rates))+' no feasible: {0:d}'.format(no_feasible)+', buf size {0:d} {1:d} {2:d}'.format(len(bbuf.obs_buf), len(cbuf.obs_buf), len(fbuf.obs_buf))+'\n')

        if (epoch_i % (N_TRAJ//20) == ((N_TRAJ//20)-1)):
            explore_eps = max(explore_eps * 0.8, 0.01)

            valid_loss = 0
            for data in valid_dataset:
                env = MultiPointEnv(num_agents=NUM_AGENTS, mode='barrier', PROB=(0,OBSTACLE_DENSITY), SIZE=(MAP_SIZE,MAP_SIZE))
                env.world.obstacles, env.world.agent_goals, env.world.agents = deepcopy(data)
                collided, done = infer(env)
                valid_loss += np.any(collided)
            torch.save(bnn.state_dict(), BMODEL_PATH.replace('.pt', '{0:d}_{1:.2f}.pt'.format(epoch_i, valid_loss/len(valid_dataset))))
        