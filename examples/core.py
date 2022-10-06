import torch
import numpy as np
from models import device
from PIL import Image
import sys
import random

def iter_action(dbarriergnn, o_b, a, threshold=-1.1e-2, max_iter=30):
    # size of a: (num_agents, n_candidates, action_dim)
    
    a = a.reshape((a.shape[0], -1, a.shape[-1]))
    n_candidate = a.shape[1]
    
    dbarriergnn.eval()
    
    tensor_a = torch.FloatTensor(a).to(device)
    tensor_a.requires_grad = True
    
    input_ = {k: v.to(device) for k, v in o_b.items()}
    vec = dbarriergnn.get_vec(**(input_)).detach()
    vec = vec.unsqueeze(1).repeat((1, n_candidate, 1))
    
    aoptimizer = torch.optim.Adam([tensor_a], lr=2)

    iter_ = 0
    while iter_ < max_iter:
        bvalue = dbarriergnn.get_field(vec, tensor_a, threshold=threshold)
        admissible = (((bvalue+threshold).relu()).min(dim=-1)[0])==0
        if (bvalue>threshold).float().sum()==0:
            break
        aoptimizer.zero_grad()
        norm_a = tensor_a.norm(dim=-1)
#         (bvalue+norm_a)[bvalue>threshold].sum().backward()
        filter_value = bvalue[~admissible, :]
        filter_value[filter_value>threshold].sum().backward()
        torch.nn.utils.clip_grad_value_([tensor_a], 1e-1)
        aoptimizer.step()
        with torch.no_grad():
            tensor_a[:] = tensor_a.clamp(-0.3, 0.3)        
        iter_ += 1

    dbvalue = dbarriergnn.get_field(vec, tensor_a)
    return tensor_a.data.cpu().numpy(), dbvalue.data.cpu().numpy()


def iter_action_combine(dlgnn, dbgnn, o_l, o_b, a, threshold=-1.1e-2, max_iter=30):
    # size of a: (num_agents, n_candidates, action_dim)
    
    a = a.reshape((a.shape[0], -1, a.shape[-1]))
    n_candidate = a.shape[1]
    
    dlgnn.eval()
    dbgnn.eval()
    
    tensor_a = torch.FloatTensor(a).to(device)
    tensor_a.requires_grad = True
    
    input_ = {k: v.to(device) for k, v in o_l.items()}
    vec_l = dlgnn.get_vec(**(input_)).detach()
    vec_l = vec_l.unsqueeze(1).repeat((1, n_candidate, 1))    
    
    input_ = {k: v.to(device) for k, v in o_b.items()}
    vec_b = dbgnn.get_vec(**(input_)).detach()
    vec_b = vec_b.unsqueeze(1).repeat((1, n_candidate, 1))
    
    aoptimizer = torch.optim.Adam([tensor_a], lr=2)

    iter_ = 0
    while iter_ < max_iter:
        lvalue = dlgnn.get_field(vec_l, tensor_a, threshold=threshold)
        bvalue = dbgnn.get_field(vec_b, tensor_a, threshold=threshold)
        if (((bvalue+threshold).relu()+(lvalue+threshold).relu()).min(dim=-1)[0]).sum()==0:
            break
        aoptimizer.zero_grad()
        norm_a = tensor_a.norm(dim=-1)
#         (bvalue+norm_a)[bvalue>threshold].sum().backward()
        admissble = (((bvalue+threshold).relu()+(lvalue+threshold).relu()).min(dim=-1)[0]==0)
        filter_value = ((bvalue+threshold).relu()+(lvalue+threshold).relu())[~admissble, :]
        filter_value[filter_value>0].sum().backward()
        torch.nn.utils.clip_grad_value_([tensor_a], 1e-1)
        aoptimizer.step()
        with torch.no_grad():
            tensor_a[:] = tensor_a.clamp(-0.3, 0.3)        
        iter_ += 1

    lvalue = dlgnn.get_field(vec_l, tensor_a, threshold=threshold)
    bvalue = dbgnn.get_field(vec_b, tensor_a, threshold=threshold)
    return tensor_a.data.cpu().numpy(), lvalue.data.cpu().numpy(), bvalue.data.cpu().numpy()


def calculate_dl_value(dlyapunovgnn, o_l, a):
    n_candidate = a.shape[1]
    
    input_ = {k: v.to(device) for k, v in o_l.items()}
    vec_l = dlyapunovgnn.get_vec(**(input_)).detach()
    vec_l = vec_l.unsqueeze(1).repeat((1, n_candidate, 1))
    
    tensor_a = torch.FloatTensor(a).to(device)
    
    return dlyapunovgnn.get_field(vec_l, tensor_a).data.cpu().numpy()
    

def choose_action(a_refine, dbvalue, dlvalue, threshold=-1.1e-2):
    a = np.zeros((a_refine.shape[0], a_refine.shape[-1]))
    a_value = np.zeros(a_refine.shape[0])
    for idx, candidates, dbvalues, dlvalues in zip(np.arange(a_refine.shape[0]), a_refine, dbvalue, dlvalue):
#         if dbvalues[0] < threshold:
#             a[idx, :] = candidates[0]
#             a_value[idx] = dbvalues[0]
#         else:
        if np.any(dbvalues < threshold):
            idx_candidates = np.arange(len(dbvalues))[dbvalues < threshold]
            filter_lvalue = dlvalues[idx_candidates]
            # choose a random control that minimizes the d-CLF value
            # randomly choose an argmin if multiple control exists
            idx_candidate = idx_candidates[np.random.choice(np.flatnonzero(np.isclose(filter_lvalue, filter_lvalue.min())))]
#                 idx_candidate = np.random.choice(idx_candidates, 1)[0]
            a[idx, :] = candidates[idx_candidate, :]
            a_value[idx] = dbvalues[idx_candidate]
        else:
            a[idx, :] = candidates[np.argmin(dbvalues), :]
            a_value[idx] = np.amin(dbvalues)
    return a, a_value


def choose_action_combine(a_refine, dlvalue, dbvalue, threshold=-1.1e-2):
    a = np.zeros((a_refine.shape[0], a_refine.shape[-1]))
    a_value = np.zeros(a_refine.shape[0])
    for idx, candidates, dlvalues, dbvalues in zip(np.arange(a_refine.shape[0]), a_refine, dlvalue, dbvalue):
        dcvalues = np.maximum(dlvalues+threshold, 0) + np.maximum(dbvalues+threshold, 0)
        if np.any(dcvalues==0):
            idx_candidates = np.arange(len(dcvalues))[dcvalues==0]
            idx_candidate = np.random.choice(idx_candidates)
            a[idx, :] = candidates[idx_candidate, :]
            a_value[idx] = dcvalues[idx_candidate]
        else:
            a[idx, :] = candidates[np.argmin(dcvalues), :]
            a_value[idx] = np.amin(dcvalues)
    return a, a_value


def generate_default_model_name(Env):
    return {
        'b': 'model_gnn/bgnn_{0}.pt'.format(Env.__name__),
        'db': 'model_gnn/dbgnn_{0}.pt'.format(Env.__name__),
        'l': 'model_gnn/lgnn_{0}.pt'.format(Env.__name__),
        'dl': 'model_gnn/dlgnn_{0}.pt'.format(Env.__name__),
    }


def load_models(state_dim, action_dim, b=None, db=None, l=None, dl=None):
    '''
        return all the models
    '''
    from models import BarrierGNN, DBarrierGNN, LyapunovGNN, DLyapunovGNN, device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    barriergnn = BarrierGNN(state_dim=state_dim)
    # barriergnn.load_state_dict(torch.load('model_gnn/barriergnn_no_obstacle3.pt', map_location=device))
    if b is not None:
        barriergnn.load_state_dict(torch.load(b, map_location=device))
    barriergnn.to(device)
    barriergnn.eval()

    dbarriergnn = DBarrierGNN(state_dim=state_dim, action_dim=action_dim)
    # dbarriergnn.load_state_dict(torch.load('model_gnn/dbarriergnn_no_obstacle3.pt', map_location=device))
    if db is not None:
        dbarriergnn.load_state_dict(torch.load(db, map_location=device))
    dbarriergnn.to(device)
    dbarriergnn.eval()
    
    lyapunovgnn = LyapunovGNN(state_dim=state_dim)
    if l is not None:
        lyapunovgnn.load_state_dict(torch.load(l, map_location=device))
    lyapunovgnn.to(device)
    lyapunovgnn.eval()

    dlyapunovgnn = DLyapunovGNN(state_dim=state_dim, action_dim=action_dim)
    if dl is not None:
        dlyapunovgnn.load_state_dict(torch.load(dl, map_location=device))
    dlyapunovgnn.to(device)
    dlyapunovgnn.eval()    
    
    return barriergnn, dbarriergnn, lyapunovgnn, dlyapunovgnn


def generate_maze(size, prob, num_agents):
    def getConnectedRegion(world, regions_dict, x, y):
        sys.setrecursionlimit(1000000)
        '''returns a list of tuples of connected squares to the given tile
        this is memoized with a dict'''
        if (x,y) in regions_dict:
            return regions_dict[(x,y)]
        visited=set()
        sx,sy=world.shape[0],world.shape[1]
        work_list=[(x,y)]
        while len(work_list)>0:
            (i,j)=work_list.pop()
            if(i<0 or i>=sx or j<0 or j>=sy):#out of bounds, return
                continue
            if(world[i,j]==-1):
                continue#crashes
            if world[i,j]>0:
                regions_dict[(i,j)]=visited
            if (i,j) in visited:continue
            visited.add((i,j))
            work_list.append((i+1,j))
            work_list.append((i,j+1))
            work_list.append((i-1,j))
            work_list.append((i,j-1))
        regions_dict[(x,y)]=visited
        return visited
    
    world = -(np.random.rand(int(size),int(size))<prob).astype(int)

    #RANDOMIZE THE POSITIONS OF AGENTS
    agent_counter = 1
    agent_locations=[]
    while agent_counter<=num_agents:
        x,y = np.random.randint(0, world.shape[0]),np.random.randint(0,world.shape[1])
        if(world[x,y] == 0):
            world[x,y]=agent_counter
            agent_locations.append((x,y))
            agent_counter += 1        

    #RANDOMIZE THE GOALS OF AGENTS
    goals = np.zeros(world.shape).astype(int)
    goal_counter = 1
    agent_regions=dict()     
    while goal_counter<=num_agents:
        agent_pos=agent_locations[goal_counter-1]
        valid_tiles=getConnectedRegion(world,agent_regions,agent_pos[0],agent_pos[1])
        x,y  = random.choice(list(valid_tiles))
        if(goals[x,y]==0 and world[x,y]!=-1):
            goals[x,y]    = goal_counter
            goal_counter += 1

    return world, agent_locations, goals


def save_gif(gifs, name="play.gif"):
    a_frames = []
    for img in gifs:
        a_frames.append(np.asarray(img))
    a_frames = np.stack(a_frames)
    ims = [Image.fromarray(a_frame) for a_frame in a_frames]
    ims[0].save(name, save_all=True, append_images=ims[1:], loop=0, duration=10)
