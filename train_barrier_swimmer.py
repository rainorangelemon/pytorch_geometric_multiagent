from tensorboardX import SummaryWriter
from tqdm import tqdm
import gc
from copy import deepcopy
from gym_swimmer import SwimmerEnv

import torch
import numpy as np
from torch import nn
import math
from models import *   
from core import generate_default_model_name

import torch
import numpy as np
from torch import nn
import math
import scipy
from random import shuffle
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from collections import defaultdict

env_name = 'SwimmerEnv'
n_candidates = 2000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

N_TRAJ = 100
NUM_AGENTS = 16
MAP_SIZE = 8

def train_barrier(bnn, optimizer, buf, writer, pbar, n_iter=10):
    
    # Set up function for computing value loss
    def compute_loss(bnn, data, next_data):
        value = bnn(**data)
        next_value = bnn(**next_data)
        
        bloss1 = ((1e-2-value).relu())*data['next_free'] / (1e-9 + (data['next_free']).sum())
        bloss2 = ((1e-2+value).relu())*data['next_danger'] / (1e-9 + (data['next_danger'].sum()))
        bloss = bloss1.sum() + bloss2.sum()
        
        deriv = next_value-value+0.1*value
        near_boundary = torch.minimum(next_value, value) < 5e-2
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
            bloss, dloss, closs = compute_loss(bnn, data, next_data)
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
        
    def store(self, **kwargs):
        """
        Append one timestep of agent-environment interaction to the buffer.
        expected keys: 'prev_free', 'next_free', 'prev_danger', 'next_danger', 'bvalue', 'action'
        """
        obs = DotDict({})
        for key, value in kwargs.items():
            obs[key] = torch.as_tensor(value, dtype=torch.float)
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
        self.concat_goal = False
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

        def collate_fn(data):
            """
               data: is a list of tuples with (example, label, length)
                     where 'example' is a tensor of arbitrary shape
                     and label/length are scalars
            """
            o, next_o = [d[0] for d in data], [d[1] for d in data]
            datas = []
            for data in [o, next_o]:
                data = default_collate(data)
                for k, v in data.items():
                    data[k] = v.to(device)
                if self.concat_goal:
                    data['x'] = torch.cat((data['x'], data['goal']), dim=-1)
                    data['next_x'] = torch.cat((data['next_x'], data['goal']), dim=-1)
                datas.append(data)
            return datas             

        loader = DataLoader(self.dataset, shuffle=True, batch_size=self.batch, collate_fn=collate_fn)

        return loader
    
    
def iter_action(bnn, o, a, threshold, max_iter=30):
    # size of a: (num_agents, n_candidates, action_dim)
    n_candidate = a.shape[0]
    
    bnn.eval()
    
    input_ = {k: v.to(device) for k, v in o.items()}
    tensor_a = torch.FloatTensor(a).to(device)
    origin_a = torch.FloatTensor(a).to(device)

    tensor_a.requires_grad = True
    vec = bnn.get_vec(**(input_)).detach()
    vec = vec.unsqueeze(0).repeat((n_candidate, 1))
    
    aoptimizer = torch.optim.Adam([tensor_a], lr=2)
    
    iter_ = 0
    while iter_ < max_iter:
        bvalue = bnn.get_field(vec, tensor_a)
        if bvalue[0]>threshold:
            break
        aoptimizer.zero_grad()
        ((-bvalue+threshold).relu().sum()+((tensor_a-origin_a)**2).sum()).backward()
        torch.nn.utils.clip_grad_value_([tensor_a], 1e-1)
        aoptimizer.step()
        with torch.no_grad():
            tensor_a[:] = tensor_a.clamp(-1., 1.)        
        iter_ += 1

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


if __name__=='__main__':
    from gym_swimmer import SwimmerEnv
    from stable_baselines3 import PPO
    from tqdm import tqdm
    
    Env = PointEnv
    env = PointEnv()
    nominal_control = lambda x: (np.array([2, 0])-x).clip(-1, 1)

    bnn = DMLP(state_dim=Env.state_dim, action_dim=Env.action_dim, mode='straight')
    bnn.to(device)
    bnn.train()

    name_dict = generate_default_model_name(Env)
    # bnn.load_state_dict(torch.load(name_dict['b'].replace('.pt', '_1model.pt'), map_location=device))

    boptimizer = torch.optim.Adam(bnn.parameters(), lr=1e-4, weight_decay=1e-8)
    bscheduler = torch.optim.lr_scheduler.ExponentialLR(boptimizer, gamma=0.996)  

    writer = SummaryWriter()
    writer.step = 0
    max_episode_length = 256
    BATCH = 64
    N_ITER = 1
    N_EPOCH = 12000

    # threshold=1e-2
    running_unsafe_rate = 0
    best_unsafe_rate = float('inf')
    unsafe_rates = [1.]*90

    BMODEL_PATH = name_dict['db'].replace('dbgnn', 'dbnn')
    open('1model_'+Env.__name__+'.txt', 'w+').close()
    bbuf_gather = GatherReplayBuffer(BATCH)
    cbuf = GlobalReplayBuffer()
    fbuf = GlobalReplayBuffer()

    pbar = tqdm(range(N_EPOCH))
    for epoch_i in pbar:

        bbuf = GlobalReplayBuffer()

        env.reset()
    #     env.world.agents = env.world.sample_agents(env.num_agents, prob=0.0)
        total_trans=0; n_danger=0; threshold=1e-2

        while True:
            o = env._get_obs()
            a_best = nominal_control(o)
            noise = 2. * min(1., epoch_i/500.)
            if noise != 1.:
                a_other = np.expand_dims(a_best, axis=0) + np.random.uniform(-noise, noise, size=(n_candidates-1, env.action_dim))
                a_other = a_other.clip(-1., 1.)
            else:
                a_other = np.random.uniform(-1., 1., size=(n_candidates-1, env.action_dim))
            a_all = np.zeros((n_candidates, env.action_dim))
            a_all[0, :] = a_best
            a_all[1:, :] = a_other

            o = {'x': torch.FloatTensor(o), 'goal': torch.FloatTensor(env.goal)}
            a_refine, bvalue = iter_action(bnn, o, a_all, max_iter=0, threshold=threshold)  # min(epoch_i//10, 30)
            
            a, a_value = choose_action(a_refine, bvalue, threshold=threshold)

            next_o, rw, done, info = env.step(a)
            
            bbuf.store(**info,  bvalue=a_value, feasible=a_value>=threshold)
            prev_danger = info['prev_danger']
            next_danger = info['next_danger']
            if np.any(prev_danger) or np.any(next_danger):
                cbuf.store(**info, bvalue=1, feasible=a_value>=threshold)

            threshold = np.maximum(a_value*0.9+1e-2, 1e-2)
            total_trans += 1
            n_danger += np.array(next_danger).sum()

            if done or (total_trans >= max_episode_length):
                if n_danger!=0:
                    bbuf.relabel(-1e4 if epoch_i<100 else 5e-2)
                else:
                    fbuf.obs_buf.append(bbuf)
                bbuf_gather.append(bbuf)
                break

        unsafe_rate = n_danger / total_trans
        unsafe_rates.append(unsafe_rate)
        running_unsafe_rate = np.mean(unsafe_rates)
        if running_unsafe_rate < best_unsafe_rate:
            best_unsafe_rate = running_unsafe_rate
            torch.save(bnn.state_dict(), BMODEL_PATH)

        if len(bbuf_gather.buffers) > N_TRAJ:
            bbuf_gather.buffers.pop(0)
        if len(unsafe_rates) > N_TRAJ:
            unsafe_rates.pop(0)

        bbuf_gather.append(cbuf)
        for b in fbuf.obs_buf:
            bbuf_gather.append(b)
        bnn.train()
        train_barrier(bnn, boptimizer, bbuf_gather, writer, pbar=pbar, n_iter=N_ITER)
        bbuf_gather.buffers.pop(-1)
        for _ in range(len(fbuf.obs_buf)):
            bbuf_gather.buffers.pop(-1)

        with open('1model_'+Env.__name__+'.txt', 'a+') as f:
            f.write(pbar.desc+'\t'+str(pbar.last_print_n)+'\t'+'unsafe rate: {0:.6f}'.format(unsafe_rate)+' running rate: {0:.6f}'.format(running_unsafe_rate)+', buf size {0:d} {1:d} {2:d}'.format(len(bbuf.obs_buf), len(cbuf.obs_buf), len(fbuf.obs_buf))+'\n')

        torch.save(bnn.state_dict(), BMODEL_PATH.replace('.pt', '_current.pt'))