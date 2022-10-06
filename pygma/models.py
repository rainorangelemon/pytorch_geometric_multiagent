import torch
import math
import numpy as np
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch.nn import LazyLinear
from torch_geometric.nn.pool import knn
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch_geometric.nn.conv import MessagePassing
from torch import nn
from torch_sparse import coalesce
from torch_geometric.nn import HeteroConv
from typing import Tuple, Union
from torch import Tensor
from torch_geometric.typing import PairTensor
from torch_geometric.nn import GATv2Conv, HGTConv
from torch_scatter import scatter_max

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def numpy2tensor(arr):
    tensor = torch.from_numpy(arr)
    if isinstance(tensor, torch.DoubleTensor):
        tensor = tensor.float()
    return tensor


class MPNN(MessagePassing):
    def __init__(self, embed_size, aggr: str = 'max', **kwargs):
        super(MPNN, self).__init__(aggr=aggr, **kwargs)
        self.fx = Seq(Lin(embed_size*3, embed_size), ReLU(), Lin(embed_size, embed_size))

    def forward(self, x, coord, edge_index):
        """"""
        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=(x, x), coord=(coord, coord))
        return x + out

    def message(self, x_i, x_j, coord_i, coord_j):
        values = self.fx(torch.cat((x_j, x_i, coord_j - coord_i), dim=-1))
        return values


class GNNBlock(torch.nn.Module):
    def __init__(self, n_node, n_coord, n_block=3, embed_size=256):   
        super(GNNBlock, self).__init__()
        self.h_nodes = Seq(Lin(n_node, embed_size), ReLU(), Lin(embed_size, embed_size))
        self.h_coords = Seq(Lin(n_coord, embed_size), ReLU(), Lin(embed_size, embed_size))
        self.mpnns = torch.nn.ModuleList([MPNN(embed_size) for i in range(n_block)])
    
    def forward(self, nodes, coords, edge_index, **kwargs):
        h_nodes = self.h_nodes(nodes)
        h_coords = self.h_coords(coords)        
        for mpnn in self.mpnns:
            h_nodes = mpnn(h_nodes, h_coords, edge_index)
        return h_nodes
    
    
class BarrierGNN(torch.nn.Module):
    def __init__(self, state_dim=2, piembed=1024):
        super(BarrierGNN, self).__init__()

        self.blockpi = GNNBlock(3, state_dim, 3, piembed)
        self.headpi = Seq(Lin(piembed, piembed), ReLU(), Lin(piembed, 1))   

    def forward(self, x, edge_index, label, goal, **kwargs):
        edge_index = edge_index.long()
        is_agents = (label[:,1]==1)
        
        nodes = label
        coords = x
        h_pi = self.blockpi(nodes, coords, edge_index)

        vec = h_pi[is_agents, :]
#         field = (vec**2).sum(dim=-1) + dist[is_agents]
        field = self.headpi(vec).squeeze(dim=-1)

        return field 
    
    
class DBarrierGNN(torch.nn.Module):
    def __init__(self, state_dim=2, action_dim=2, piembed=1024):
        super(DBarrierGNN, self).__init__()

        self.blockpi = GNNBlock(3, state_dim, 3, piembed)
        self.headpi = Seq(Lin(piembed+action_dim, piembed), ReLU(), Lin(piembed, 1))   

    def forward(self, x, action, edge_index, label, goal, **kwargs):
        edge_index = edge_index.long()
        is_agents = (label[:,1]==1)
        
        nodes = label
        coords = x
        h_pi = self.blockpi(nodes, coords, edge_index)

        vec = h_pi[is_agents, :]
        field = self.headpi(torch.cat((vec, action), dim=-1)).squeeze(dim=-1)

        return field
    
    def get_vec(self, x, edge_index, label, goal, **kwargs):
        edge_index = edge_index.long()
        is_agents = (label[:,1]==1)
        
        nodes = label
        coords = x
        h_pi = self.blockpi(nodes, coords, edge_index)

        vec = h_pi[is_agents, :]

        return vec
    
    def get_field(self, vec, action, **kwargs):
        return self.headpi(torch.cat((vec, action), dim=-1)).squeeze(dim=-1)  

    
class LyapunovGNN(torch.nn.Module):
    def __init__(self, state_dim=2, piembed=1024):
        super(LyapunovGNN, self).__init__()

        self.blockpi = GNNBlock(3, state_dim, 3, piembed)
        self.headpi = Seq(Lin(piembed, piembed), ReLU(), Lin(piembed, 1))   

    def forward(self, x, edge_index, label, **kwargs):
        edge_index = edge_index.long()
        is_agents = (label[:,1]==1)
        
        nodes = label
        h_pi = self.blockpi(nodes, x, edge_index)

        vec = h_pi[is_agents, :]
        field = (vec**2).sum(dim=-1)

        return field


class DLyapunovGNN(torch.nn.Module):
    def __init__(self, state_dim=2, action_dim=2, piembed=1024):
        super(DLyapunovGNN, self).__init__()

        self.blockpi = GNNBlock(3, state_dim, 3, piembed)
        self.headpi = Seq(Lin(piembed+action_dim, piembed), ReLU(), Lin(piembed, 1))   

    def forward(self, **kwargs):
        
        vec = self.get_vec(**kwargs)
        field = self.get_field(vec=vec, **kwargs).squeeze(-1)
                
        return field
    
    def get_vec(self, x, edge_index, label, **kwargs):
        edge_index = edge_index.long()
        is_agents = (label[:,1]==1)    
        
        nodes = label
        vec = self.blockpi(nodes, x, edge_index)[is_agents, :]

        return vec
    
    def get_field(self, vec, action, **kwargs):
        return self.headpi(torch.cat((vec, action), dim=-1)).squeeze(dim=-1)


class MLP(torch.nn.Module):
    def __init__(self, state_dim=2, piembed=1024, mode=None):
        super(MLP, self).__init__()

        assert mode=='sum' or mode=='straight'
        self.mode = mode
        
        if mode=='sum':
            self.field = Seq(Lin(state_dim, piembed), ReLU(), Lin(piembed, piembed), ReLU(), Lin(piembed, piembed))
        else:
            self.field = Seq(Lin(state_dim, piembed), ReLU(), Lin(piembed, piembed), ReLU(), Lin(piembed, 1))

    def forward(self, x, **kwargs):
        
        field = self.field(x)
        if self.mode=='sum':
            field = (field**2).sum(dim=-1)
        else:
            field = field.squeeze(dim=-1)
                
        return field


class DMLP(torch.nn.Module):
    def __init__(self, state_dim=2, action_dim=2, piembed=1024, mode=None):
        super(DMLP, self).__init__()

        assert mode=='sum' or mode=='straight'
        self.mode = mode
        self.alpha_ = torch.randn(1, requires_grad=True).reshape([])
        
        self.vec = Seq(LazyLinear(piembed), ReLU(), Lin(piembed, piembed), ReLU(), Lin(piembed, piembed))
        if mode=='sum':
            self.field = Seq(LazyLinear(piembed), ReLU(), Lin(piembed, piembed), ReLU(), Lin(piembed, piembed))
        else:
            self.field = Seq(LazyLinear(piembed), ReLU(), Lin(piembed, piembed), ReLU(), Lin(piembed, 1))

    @property
    def alpha(self):
        return torch.nn.functional.softplus(self.alpha_)
            
    def forward(self, **kwargs):

        vec = self.get_vec(**kwargs)
        field = self.get_field(vec=vec, **kwargs).squeeze(-1)

        return field
    
    def get_vec(self, x, **kwargs): 

        vec = self.vec(x)

        return vec
    
    def get_field(self, vec, action, **kwargs):
        field = self.field(torch.cat((vec, action), dim=-1))
        if self.mode=='sum':
            field = (field**2).sum(dim=-1)
        else:
            field = field.squeeze(dim=-1)
        return field
    
    
class OldHeteroMPNN(MessagePassing):
    def __init__(self, embed_size, aggr: str = 'max', **kwargs):
        super(OldHeteroMPNN, self).__init__(aggr=aggr, **kwargs)
        self.fx = Seq(LazyLinear(embed_size), ReLU(), Lin(embed_size, embed_size))

    def forward(self, x: Union[Tensor, PairTensor], edge_index, edge_attr):
        """"""
        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        # if len(edge_attr) != 0:
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        # out += x[1]
        return out
        # else:
        #     return x[1]

    def message(self, x_i, x_j, edge_attr):
        values = self.fx(torch.cat((x_i, x_j, edge_attr), dim=-1))
        return values
    
    
class OldHeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels=1024, num_layers=3, keys=None, mode='straight'):
        super().__init__()
        assert mode=='sum' or mode=='straight'
        
        self.mode = mode

        if keys is None:
            keys = {'obstacle', 'agent', 'goal'}
        
        edge_keys = {('obstacle', 'o_near_a', 'agent'), 
                        ('agent', 'a_near_a', 'agent'), 
                        ('goal', 'toward', 'agent')}
        
        self.embed = torch.nn.ModuleDict()
        for key in keys:
            self.embed[key] = Seq(LazyLinear(hidden_channels),)
        
        self.edge_embed = torch.nn.ModuleDict()
        for edge_key in edge_keys:
            if edge_key[0] in keys:
                self.edge_embed[str(edge_key)] = Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, hidden_channels))
        
        self.convs = torch.nn.ModuleList()
        # mpnn = HeteroMPNN(hidden_channels) 
        for _ in range(num_layers):
            mpnns = {}
            for edge_key in edge_keys:
                if edge_key[0] in keys:
                    mpnns[edge_key] = HeteroMPNN(hidden_channels)         

            conv = HeteroConv(mpnns, aggr='max')
            self.convs.append(conv)

        self.field = Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, 1))

    def get_vec(self, data):
        x_dict, edge_index_dict, edge_attr_dict = data.x_dict, data.edge_index_dict, data.edge_attr_dict
        for key in self.embed.keys():
            x_dict[key] = self.embed[key](x_dict[key])
        for key in edge_attr_dict.keys():
            if key[0] in self.embed.keys():
                edge_attr_dict[key] = self.edge_embed[str(key)](edge_attr_dict[key])
        for conv in self.convs:
            new_dict = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            for key, value in new_dict.items():
                x_dict[key] = x_dict[key] + value
        
        vec = x_dict['agent']
        return vec
    
    def get_field(self, vec, action):
        field = self.field(torch.cat((vec, action), dim=-1))
        if self.mode=='sum':
            field = (field**2).sum(dim=-1)
        else:
            field = field.squeeze(dim=-1)
        return field
    
    def forward(self, data):
        
        vec = self.get_vec(data)
        field = self.get_field(vec=vec, action=data['action']).squeeze(-1)
                
        return field


class HeteroMPNN(MessagePassing):
    def __init__(self, embed_size, aggr: str = 'max', **kwargs):
        super(HeteroMPNN, self).__init__(aggr=aggr, **kwargs)
        self.fx = Seq(LazyLinear(embed_size), ReLU(), Lin(embed_size, embed_size))

    def forward(self, x: Union[Tensor, PairTensor], edge_index, edge_attr):
        """"""
        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        # if len(edge_attr) != 0:
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        # out += x[1]
        return out
        # else:
        #     return x[1]

    def message(self, x_i, x_j, edge_attr):
        values = self.fx(edge_attr)
        return values    
    

class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels=1024, num_layers=3, keys=None, mode='straight'):
        super().__init__()
        assert mode=='sum' or mode=='straight'
        
        self.mode = mode

        if keys is None:
            self.keys = {'obstacle', 'agent', 'goal'}
        else:
            self.keys = keys
        
        edge_keys = {('obstacle', 'o_near_a', 'agent'), 
                        ('agent', 'a_near_a', 'agent'), 
                        ('goal', 'toward', 'agent')}
        
        self.embed = LazyLinear(hidden_channels, bias=False)
        self.edge_embed = torch.nn.ModuleDict()
        for edge_key in edge_keys:
            if edge_key[0] in self.keys:
                self.edge_embed[str(edge_key)] = Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, hidden_channels))
        
        self.convs = torch.nn.ModuleList()
        # mpnn = HeteroMPNN(hidden_channels) 
        for _ in range(num_layers):
            mpnns = {}
            for edge_key in edge_keys:
                if edge_key[0] in self.keys:
                    mpnns[edge_key] = HeteroMPNN(hidden_channels)         

            conv = HeteroConv(mpnns, aggr='max')
            self.convs.append(conv)
            
        self.edge_mlps = torch.nn.ModuleDict()
        for layer_i in range(num_layers):
            for edge_key in edge_keys:
                if edge_key[0] in self.keys:
                    self.edge_mlps[str(layer_i)+str(edge_key)] = Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, hidden_channels))

        self.field = Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, 1))

    def get_vec(self, data):
        x_dict, edge_index_dict, edge_attr_dict = data.x_dict, data.edge_index_dict, data.edge_attr_dict
        for key in self.keys:
            x_dict[key] = self.embed(x_dict[key])
        for key in edge_attr_dict.keys():
            if key[0] in self.keys:
                edge_attr_dict[key] = self.edge_embed[str(key)](edge_attr_dict[key])
        for layer_i, conv in enumerate(self.convs):
            for key in edge_attr_dict.keys():
                if key[0] in self.keys:
                    edge = edge_index_dict[key]
                    x0, x1 = x_dict[key[0]], x_dict[key[2]]
                    edge_attr_dict[key] = edge_attr_dict[key] + self.edge_mlps[str(layer_i)+str(key)](torch.cat((edge_attr_dict[key], x0[edge[0,:]], x1[edge[1,:]]), dim=-1))
            
            new_dict = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            for key, value in new_dict.items():
                x_dict[key] = x_dict[key] + value
        
        vec = x_dict['agent']
        return vec
    
    def get_field(self, vec, action):
        field = self.field(torch.cat((vec, action), dim=-1))
        if self.mode=='sum':
            field = (field**2).sum(dim=-1)
        else:
            field = field.squeeze(dim=-1)
        return field
    
    def forward(self, data):
        
        vec = self.get_vec(data)
        field = self.get_field(vec=vec, action=data['action'])
                
        return field    
    
    
class OriginMPNN(MessagePassing):
    def __init__(self, embed_size, aggr: str = 'max', **kwargs):
        super(OriginMPNN, self).__init__(aggr=aggr, **kwargs)
        self.fx = Seq(LazyLinear(embed_size), ReLU(), Lin(embed_size, embed_size))

    def forward(self, x: Union[Tensor, PairTensor], edge_index, edge_attr):
        """"""
        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        # if len(edge_attr) != 0:
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        # out += x[1]
        return out
        # else:
        #     return x[1]

    def message(self, x_i, x_j, edge_attr):
        values = self.fx(torch.cat((x_i, x_j, edge_attr), dim=-1))
        return values
    

class OriginGNN(torch.nn.Module):
    def __init__(self, hidden_channels=1024, num_layers=3, keys=None, mode='straight'):
        super().__init__()
        assert mode=='sum' or mode=='straight'
        
        self.mode = mode

        if keys is None:
            self.keys = {'obstacle', 'agent', 'goal'}
        else:
            self.keys = keys
        
        edge_keys = {('obstacle', 'o_near_a', 'agent'), 
                        ('agent', 'a_near_a', 'agent'), 
                        ('goal', 'toward', 'agent')}
        
        self.embed = LazyLinear(hidden_channels, bias=False)
        self.edge_embed = Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, hidden_channels))
        
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mpnns = {}
            mpnn = OriginMPNN(hidden_channels)
            for edge_key in edge_keys:
                if edge_key[0] in self.keys:
                    mpnns[edge_key] = mpnn

            conv = HeteroConv(mpnns, aggr='max')
            self.convs.append(conv)

        # self.edge_mlps = torch.nn.ModuleDict()
        # for layer_i in range(num_layers):
        #     for edge_key in edge_keys:
        #         if edge_key[0] in self.keys:
        #             self.edge_mlps[str(layer_i)+str(edge_key)] = Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, hidden_channels))

        self.field = Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, 1))

    def get_vec(self, data):
        x_dict, edge_index_dict, edge_attr_dict = data.x_dict, data.edge_index_dict, data.edge_attr_dict
        for key in self.keys:
            x_dict[key] = self.embed(x_dict[key])
        for key in edge_attr_dict.keys():
            if key[0] in self.keys:
                edge_attr_dict[key] = self.edge_embed(edge_attr_dict[key])
        for layer_i, conv in enumerate(self.convs):            
            new_dict = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            for key, value in new_dict.items():
                x_dict[key] = x_dict[key] + value
        
        vec = x_dict['agent']
        return vec
    
    def get_field(self, vec, action):
        field = self.field(torch.cat((vec, action), dim=-1))
        if self.mode=='sum':
            field = (field**2).sum(dim=-1)
        else:
            field = field.squeeze(dim=-1)
        return field
    
    def forward(self, data):
        
        vec = self.get_vec(data)
        field = self.get_field(vec=vec, action=data['action'])
                
        return field   

    
class OriginGNNv2(torch.nn.Module):
    def __init__(self, hidden_channels=1024, num_layers=3, keys=None, mode='straight'):
        super().__init__()
        assert mode=='sum' or mode=='straight'
        
        self.mode = mode

        if keys is None:
            self.keys = {'obstacle', 'agent', 'goal'}
        else:
            self.keys = keys
        
        edge_keys = {('obstacle', 'o_near_a', 'agent'), 
                        ('agent', 'a_near_a', 'agent'), 
                        ('goal', 'toward', 'agent')}
        
        self.embed = torch.nn.ModuleDict()
        for key in self.keys:
            self.embed[key] = LazyLinear(hidden_channels)
        self.edge_embed = Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, hidden_channels))
        
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mpnns = {}
            mpnn = OriginMPNN(hidden_channels)
            for edge_key in edge_keys:
                if edge_key[0] in self.keys:
                    mpnns[edge_key] = mpnn

            conv = HeteroConv(mpnns, aggr='max')
            self.convs.append(conv)

        # self.edge_mlps = torch.nn.ModuleDict()
        # for layer_i in range(num_layers):
        #     for edge_key in edge_keys:
        #         if edge_key[0] in self.keys:
        #             self.edge_mlps[str(layer_i)+str(edge_key)] = Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, hidden_channels))

        self.field = Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, 1))

    def get_vec(self, data):
        x_dict, edge_index_dict, edge_attr_dict = data.x_dict, data.edge_index_dict, data.edge_attr_dict
        for key in self.keys:
            x_dict[key] = self.embed[key](x_dict[key])
        for key in edge_attr_dict.keys():
            if key[0] in self.keys:
                edge_attr_dict[key] = self.edge_embed(edge_attr_dict[key])
        for layer_i, conv in enumerate(self.convs):            
            new_dict = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            for key, value in new_dict.items():
                x_dict[key] = x_dict[key] + value
        
        vec = x_dict['agent']
        return vec
    
    def get_field(self, vec, action):
        field = self.field(torch.cat((vec, action), dim=-1))
        if self.mode=='sum':
            field = (field**2).sum(dim=-1)
        else:
            field = field.squeeze(dim=-1)
        return field
    
    def forward(self, data):
        
        vec = self.get_vec(data)
        field = self.get_field(vec=vec, action=data['action'])
                
        return field       

    
    
class OriginGNNv3(torch.nn.Module):
    def __init__(self, hidden_channels=1024, num_layers=3, keys=None, mode='straight'):
        super().__init__()
        assert mode=='sum' or mode=='straight'
        
        self.mode = mode

        if keys is None:
            self.keys = {'obstacle', 'agent', 'goal'}
        else:
            self.keys = keys
        
        edge_keys = {('obstacle', 'o_near_a', 'agent'), 
                        ('agent', 'a_near_a', 'agent'), 
                        ('goal', 'toward', 'agent')}
        
        self.embed = torch.nn.ModuleDict()
        for key in self.keys:
            self.embed[key] = LazyLinear(hidden_channels)
        self.edge_embed = torch.nn.ModuleDict()
        for edge_key in edge_keys:
            if edge_key[0] in self.keys:
                self.edge_embed[str(edge_key)] = Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, hidden_channels))
        
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mpnns = {}
            mpnn = OriginMPNN(hidden_channels)
            for edge_key in edge_keys:
                if edge_key[0] in self.keys:
                    mpnns[edge_key] = mpnn

            conv = HeteroConv(mpnns, aggr='max')
            self.convs.append(conv)

        # self.edge_mlps = torch.nn.ModuleDict()
        # for layer_i in range(num_layers):
        #     for edge_key in edge_keys:
        #         if edge_key[0] in self.keys:
        #             self.edge_mlps[str(layer_i)+str(edge_key)] = Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, hidden_channels))

        self.field = Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, 1))

    def get_vec(self, data):
        x_dict, edge_index_dict, edge_attr_dict = data.x_dict, data.edge_index_dict, data.edge_attr_dict
        for key in self.keys:
            x_dict[key] = self.embed[key](x_dict[key])
        for key in edge_attr_dict.keys():
            if key[0] in self.keys:
                edge_attr_dict[key] = self.edge_embed[str(key)](edge_attr_dict[key])
        for layer_i, conv in enumerate(self.convs):            
            new_dict = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            for key, value in new_dict.items():
                x_dict[key] = x_dict[key] + value
        
        vec = x_dict['agent']
        return vec
    
    def get_field(self, vec, action):
        field = self.field(torch.cat((vec, action), dim=-1))
        if self.mode=='sum':
            field = (field**2).sum(dim=-1)
        else:
            field = field.squeeze(dim=-1)
        return field
    
    def forward(self, data):
        
        vec = self.get_vec(data)
        field = self.get_field(vec=vec, action=data['action'])
                
        return field    
    
    
    
class OriginGNNv4(torch.nn.Module):
    def __init__(self, hidden_channels=1024, num_layers=3, keys=None, mode='straight'):
        super().__init__()
        assert mode=='sum' or mode=='straight'
        
        self.mode = mode

        if keys is None:
            self.keys = {'obstacle', 'agent', 'goal'}
        else:
            self.keys = keys
        
        edge_keys = {('obstacle', 'o_near_a', 'agent'), 
                        ('agent', 'a_near_a', 'agent'), 
                        ('goal', 'toward', 'agent')}
        
        self.embed = torch.nn.ModuleDict()
        for key in self.keys:
            self.embed[key] = LazyLinear(hidden_channels)
        self.edge_embed = torch.nn.ModuleDict()
        for edge_key in edge_keys:
            if edge_key[0] in self.keys:
                self.edge_embed[str(edge_key)] = Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, hidden_channels))
        
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mpnns = {}
            for edge_key in edge_keys:
                if edge_key[0] in self.keys:
                    mpnns[edge_key] = OriginMPNN(hidden_channels)

            conv = HeteroConv(mpnns, aggr='max')
            self.convs.append(conv)

        # self.edge_mlps = torch.nn.ModuleDict()
        # for layer_i in range(num_layers):
        #     for edge_key in edge_keys:
        #         if edge_key[0] in self.keys:
        #             self.edge_mlps[str(layer_i)+str(edge_key)] = Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, hidden_channels))

        self.field = Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, 1))

    def get_vec(self, data):
        x_dict, edge_index_dict, edge_attr_dict = data.x_dict, data.edge_index_dict, data.edge_attr_dict
        for key in self.keys:
            x_dict[key] = self.embed[key](x_dict[key])
        for key in edge_attr_dict.keys():
            if key[0] in self.keys:
                edge_attr_dict[key] = self.edge_embed[str(key)](edge_attr_dict[key])
        for layer_i, conv in enumerate(self.convs):            
            new_dict = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            for key, value in new_dict.items():
                x_dict[key] = x_dict[key] + value
        
        vec = x_dict['agent']
        return vec
    
    def get_field(self, vec, action):
        field = self.field(torch.cat((vec, action), dim=-1))
        if self.mode=='sum':
            field = (field**2).sum(dim=-1)
        else:
            field = field.squeeze(dim=-1)
        return field
    
    def forward(self, data):
        
        vec = self.get_vec(data)
        field = self.get_field(vec=vec, action=data['action'])
                
        return field    

    
class OriginGNNv5(torch.nn.Module):
    def __init__(self, hidden_channels=1024, num_layers=3, keys=None, mode='straight', pos_encode=None):
        super().__init__()
        assert mode=='sum' or mode=='straight'
        
        self.mode = mode
        self.pos_encode = pos_encode
        
        if self.pos_encode is not None:
            div_term = torch.exp(torch.arange(0, pos_encode, 2))
            self.register_buffer('div_term', div_term)
        
        if keys is None:
            self.keys = {'obstacle', 'agent', 'goal'}
        else:
            self.keys = keys
        
        edge_keys = {('obstacle', 'o_near_a', 'agent'), 
                        ('agent', 'a_near_a', 'agent'), 
                        ('goal', 'toward', 'agent')}
        
        self.embed = torch.nn.ModuleDict()
        for key in self.keys:
            self.embed[key] = LazyLinear(hidden_channels)
        self.edge_embed = torch.nn.ModuleDict()
        for edge_key in edge_keys:
            if edge_key[0] in self.keys:
                self.edge_embed[str(edge_key)] = Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, hidden_channels))
        
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mpnns = {}
            for edge_key in edge_keys:
                if edge_key[0] in self.keys:
                    mpnns[edge_key] = OriginMPNN(hidden_channels)

            conv = HeteroConv(mpnns, aggr='max')
            self.convs.append(conv)

        # self.edge_mlps = torch.nn.ModuleDict()
        # for layer_i in range(num_layers):
        #     for edge_key in edge_keys:
        #         if edge_key[0] in self.keys:
        #             self.edge_mlps[str(layer_i)+str(edge_key)] = Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, hidden_channels))

        self.field = Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, 1))

    def get_vec(self, data):
        x_dict, edge_index_dict, edge_attr_dict = data.x_dict, data.edge_index_dict, data.edge_attr_dict
        for key in self.keys:
            x_dict[key] = self.embed[key](x_dict[key])
        for key in edge_attr_dict.keys():
            if key[0] in self.keys:
                edge_attr_dict[key] = self.edge_embed[str(key)](edge_attr_dict[key])
        for layer_i, conv in enumerate(self.convs):            
            new_dict = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            for key, value in new_dict.items():
                x_dict[key] = x_dict[key] + value
        
        vec = x_dict['agent']
        return vec
    
    def get_field(self, vec, action):
        if self.pos_encode is not None:
            action = torch.flatten(torch.cat((torch.sin(action.unsqueeze(-1)*self.div_term), 
                                torch.cos(action.unsqueeze(-1)*self.div_term)), dim=-1), start_dim=-2)
        field = self.field(torch.cat((vec, action), dim=-1))
        if self.mode=='sum':
            field = (field**2).sum(dim=-1)
        else:
            field = field.squeeze(dim=-1)
        return field
    
    def forward(self, data):
        
        vec = self.get_vec(data)
        field = self.get_field(vec=vec, action=data['action'])
                
        return field    
    
    
    
class OriginGNNv6(torch.nn.Module):
    def __init__(self, hidden_channels=1024, num_layers=3, keys=None, mode='straight', pos_encode=None):
        super().__init__()
        assert mode=='sum' or mode=='straight'
        
        self.mode = mode
        self.pos_encode = pos_encode
        
        if self.pos_encode is not None:
            div_term = torch.exp(torch.arange(0, pos_encode, 2))
            self.register_buffer('div_term', div_term)
        
        if keys is None:
            self.keys = {'obstacle', 'agent', 'goal'}
        else:
            self.keys = keys
        
        edge_keys = {('obstacle', 'o_near_a', 'agent'), 
                        ('agent', 'a_near_a', 'agent'), 
                        ('goal', 'toward', 'agent')}
        
        self.embed = torch.nn.ModuleDict()
        for key in self.keys:
            self.embed[key] = LazyLinear(hidden_channels)
        self.edge_embed = torch.nn.ModuleDict()
        self.edge_embed_latent = torch.nn.ParameterDict()
        for edge_key in edge_keys:
            if edge_key[0] in self.keys:
                self.edge_embed[str(edge_key)] = Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, hidden_channels))
                self.edge_embed_latent[str(edge_key)] = torch.nn.Parameter(torch.rand(hidden_channels))
        
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mpnns = {}
            mpnn = OriginMPNN(hidden_channels)
            for edge_key in edge_keys:
                if edge_key[0] in self.keys:
                    mpnns[edge_key] = mpnn

            conv = HeteroConv(mpnns, aggr='max')
            self.convs.append(conv)

        # self.edge_mlps = torch.nn.ModuleDict()
        # for layer_i in range(num_layers):
        #     for edge_key in edge_keys:
        #         if edge_key[0] in self.keys:
        #             self.edge_mlps[str(layer_i)+str(edge_key)] = Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, hidden_channels))

        self.field = Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, 1))

    def get_vec(self, data):
        x_dict, edge_index_dict, edge_attr_dict = data.x_dict, data.edge_index_dict, data.edge_attr_dict
        for key in self.keys:
            x_dict[key] = self.embed[key](x_dict[key])
        for key in edge_attr_dict.keys():
            if key[0] in self.keys:
                edge_attr_dict[key] = self.edge_embed[str(key)](edge_attr_dict[key])
                edge_attr_dict[key] = edge_attr_dict[key] + self.edge_embed_latent[str(key)]
        for layer_i, conv in enumerate(self.convs):            
            new_dict = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            for key, value in new_dict.items():
                x_dict[key] = x_dict[key] + value
        
        vec = x_dict['agent']
        return vec
    
    def get_field(self, vec, action):
        if self.pos_encode is not None:
            action = torch.flatten(torch.cat((torch.sin(action.unsqueeze(-1)*self.div_term), 
                                torch.cos(action.unsqueeze(-1)*self.div_term)), dim=-1), start_dim=-2)
        field = self.field(torch.cat((vec, action), dim=-1))
        if self.mode=='sum':
            field = (field**2).sum(dim=-1)
        else:
            field = field.squeeze(dim=-1)
        return field
    
    def forward(self, data):
        
        vec = self.get_vec(data)
        field = self.get_field(vec=vec, action=data['action'])
                
        return field  
    
    
    
class OriginGNNv7(torch.nn.Module):
    def __init__(self, hidden_channels=1024, 
                 num_layers: dict = None, 
                 keys=None, mode='straight', pos_encode=None):
        super().__init__()
        assert mode=='sum' or mode=='straight'

        self.mode = mode
        self.pos_encode = pos_encode
        self.hidden_channels = hidden_channels

        if self.pos_encode is not None:
            div_term = torch.exp(torch.arange(0, pos_encode, 2))
            self.register_buffer('div_term', div_term)

        if keys is None:
            self.keys = {'obstacle', 'agent', 'goal'}
        else:
            self.keys = keys

        self.edge_keys = {('obstacle', 'o_near_a', 'agent'), 
                        ('agent', 'a_near_a', 'agent'), 
                        ('goal', 'toward', 'agent')}
        
        for edge in list(self.edge_keys):
            if edge[0] not in self.keys:
                self.edge_keys.remove(edge)

        if num_layers is None:
            self.num_layers = dict()
            for edge in self.edge_keys:
                if 'a_near_a' in edge:
                    self.num_layers[edge] = 3
                else:
                    self.num_layers[edge] = 1

        # self.x_embed = torch.nn.ModuleDict()
        # for edge in self.edge_keys:
        #     x_embed = torch.nn.ModuleDict()
        #     x_embed[edge[0]] = LazyLinear(hidden_channels)
        #     x_embed[edge[-1]] = LazyLinear(hidden_channels)
        # self.x_embed[str(edge)] = x_embed
        
        self.edge_embed = torch.nn.ModuleDict()
        for edge_key in self.edge_keys:
            self.edge_embed[str(edge_key)] = Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, hidden_channels))
            
        self.convs = torch.nn.ModuleDict()
        for edge in self.edge_keys:
            list_ = torch.nn.ModuleList()
            num_layers = self.num_layers[edge]
            for _ in range(num_layers):
                conv = OriginMPNNv2(hidden_channels)
                list_.append(conv)
            self.convs[str(edge)] = list_

        self.edge_mlps = torch.nn.ModuleDict()
        for edge_key in self.edge_keys:
            list_ = torch.nn.ModuleList()
            num_layers = self.num_layers[edge_key]
            for _ in range(num_layers):
                list_.append(Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, hidden_channels)))
            self.edge_mlps[str(edge_key)] = list_

        self.field = Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, 1))

    def get_vec(self, data):
        x_dict, edge_index_dict, edge_attr_dict = data.x_dict, data.edge_index_dict, data.edge_attr_dict
        h_final_agent = torch.zeros((len(x_dict['agent']), self.hidden_channels), device=x_dict['agent'].device)

        for key in edge_attr_dict.keys():
            if key not in self.edge_keys:
                continue

            h_agent = torch.zeros((len(x_dict['agent']), self.hidden_channels), device=x_dict['agent'].device)
            edge_attr_dict[key] = self.edge_embed[str(key)](edge_attr_dict[key])
            for conv, edge_mlp in zip(self.convs[str(key)], self.edge_mlps[str(key)]):            
                h_agent = conv(x=(x_dict[key[0]], h_agent),
                                edge_index=edge_index_dict[key], 
                                edge_attr=edge_attr_dict[key])
                edge_attr_dict[key] = edge_attr_dict[key] + edge_mlp(torch.cat((edge_attr_dict[key], h_agent[edge_index_dict[key][1,:]]), dim=-1))

            h_final_agent = torch.maximum(h_final_agent, h_agent)

        vec = h_final_agent
        return vec

    def get_field(self, vec, action):
        if self.pos_encode is not None:
            action = torch.flatten(torch.cat((torch.sin(action.unsqueeze(-1)*self.div_term), 
                                torch.cos(action.unsqueeze(-1)*self.div_term)), dim=-1), start_dim=-2)
        field = self.field(torch.cat((vec, action), dim=-1))
        if self.mode=='sum':
            field = (field**2).sum(dim=-1)
        else:
            field = field.squeeze(dim=-1)
        return field

    def forward(self, data):

        vec = self.get_vec(data)
        field = self.get_field(vec=vec, action=data['action'])

        return field    


class OriginMPNNv2(MessagePassing):
    def __init__(self, embed_size, aggr: str = 'max', **kwargs):
        super(OriginMPNNv2, self).__init__(aggr=aggr, **kwargs)
        self.fx = Seq(LazyLinear(embed_size), ReLU(), Lin(embed_size, embed_size))

    def forward(self, x, edge_index, edge_attr):
        """"""
        if not isinstance(x, tuple):
            x=(x,x)
            
        out = self.propagate(edge_index, x=x, size=(len(x[0]), len(x[1])), edge_attr=edge_attr)
        return x[1] + out

    def message(self, x_i, x_j, edge_attr):
        values = self.fx(edge_attr)
        return values

    
    
class OriginGNNv8(torch.nn.Module):
    def __init__(self, hidden_channels=1024, 
                 num_layers: dict = None, 
                 keys=None, mode='straight', pos_encode=None):
        super().__init__()
        assert mode=='sum' or mode=='straight'

        self.mode = mode
        self.pos_encode = pos_encode
        self.hidden_channels = hidden_channels

        if self.pos_encode is not None:
            div_term = torch.exp(torch.arange(0, pos_encode, 2))
            self.register_buffer('div_term', div_term)

        if keys is None:
            self.keys = {'obstacle', 'agent', 'goal'}
        else:
            self.keys = keys

        self.edge_keys = {('obstacle', 'o_near_a', 'agent'), 
                        ('agent', 'a_near_a', 'agent'), 
                        ('goal', 'toward', 'agent')}
        
        for edge in list(self.edge_keys):
            if edge[0] not in self.keys:
                self.edge_keys.remove(edge)

        if num_layers is None:
            self.num_layers = dict()
            for edge in self.edge_keys:
                if 'a_near_a' in edge:
                    self.num_layers[edge] = 3
                else:
                    self.num_layers[edge] = 1

        # self.x_embed = torch.nn.ModuleDict()
        # for edge in self.edge_keys:
        #     x_embed = torch.nn.ModuleDict()
        #     x_embed[edge[0]] = LazyLinear(hidden_channels)
        #     x_embed[edge[-1]] = LazyLinear(hidden_channels)
        # self.x_embed[str(edge)] = x_embed
        
        self.edge_embed = torch.nn.ModuleDict()
        for edge_key in self.edge_keys:
            self.edge_embed[str(edge_key)] = Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, hidden_channels))
            
        self.convs = torch.nn.ModuleDict()
        for edge in self.edge_keys:
            list_ = torch.nn.ModuleList()
            num_layers = self.num_layers[edge]
            for _ in range(num_layers):
                conv = OriginMPNNv2(hidden_channels)
                list_.append(conv)
            self.convs[str(edge)] = list_

        self.edge_mlps = torch.nn.ModuleDict()
        for edge_key in self.edge_keys:
            list_ = torch.nn.ModuleList()
            num_layers = self.num_layers[edge_key]
            for _ in range(num_layers):
                list_.append(Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, hidden_channels)))
            self.edge_mlps[str(edge_key)] = list_

        self.field = Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, 1))

    def get_vec(self, data):
        x_dict, edge_index_dict, edge_attr_dict = data.x_dict, data.edge_index_dict, data.edge_attr_dict
        h_final_agent = []

        for key in self.edge_keys:
            h_agent = torch.zeros((len(x_dict['agent']), self.hidden_channels), device=x_dict['agent'].device)
            
            if key not in edge_attr_dict.keys():
                h_final_agent.append(h_agent)
                continue

            edge_attr_dict[key] = self.edge_embed[str(key)](edge_attr_dict[key])
            for conv, edge_mlp in zip(self.convs[str(key)], self.edge_mlps[str(key)]):            
                h_agent = conv(x=(x_dict[key[0]], h_agent),
                                edge_index=edge_index_dict[key], 
                                edge_attr=edge_attr_dict[key])
                edge_attr_dict[key] = edge_attr_dict[key] + edge_mlp(torch.cat((edge_attr_dict[key], h_agent[edge_index_dict[key][1,:]]), dim=-1))

            h_final_agent.append(h_agent)

        vec = torch.cat(h_final_agent, dim=-1)
        return vec

    def get_field(self, vec, action):
        if self.pos_encode is not None:
            action = torch.flatten(torch.cat((torch.sin(action.unsqueeze(-1)*self.div_term), 
                                torch.cos(action.unsqueeze(-1)*self.div_term)), dim=-1), start_dim=-2)
        field = self.field(torch.cat((vec, action), dim=-1))
        if self.mode=='sum':
            field = (field**2).sum(dim=-1)
        else:
            field = field.squeeze(dim=-1)
        return field

    def forward(self, data):

        vec = self.get_vec(data)
        field = self.get_field(vec=vec, action=data['action'])

        return field         
    
    
    
class OriginGNNv9(torch.nn.Module):
    def __init__(self, hidden_channels=1024, num_layers=3, keys=None, mode='straight', pos_encode=None):
        super().__init__()
        assert mode=='sum' or mode=='straight'
        
        self.mode = mode
        self.pos_encode = pos_encode
        
        if self.pos_encode is not None:
            div_term = torch.exp(torch.arange(0, pos_encode, 2))
            self.register_buffer('div_term', div_term)
        
        if keys is None:
            self.keys = {'obstacle', 'agent', 'goal'}
        else:
            self.keys = keys
        
        edge_keys = {('obstacle', 'o_near_a', 'agent'), 
                        ('agent', 'a_near_a', 'agent'), 
                        ('goal', 'toward', 'agent')}
        
        for edge in list(edge_keys):
            if edge[0] not in self.keys:
                edge_keys.remove(edge)
        self.edge_keys = edge_keys
        
        self.embed = torch.nn.ModuleDict()
        for key in self.keys:
            self.embed[key] = LazyLinear(hidden_channels)
        self.edge_embed = torch.nn.ModuleDict()
        for edge_key in edge_keys:
            self.edge_embed[str(edge_key)] = Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, hidden_channels))
        
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mpnns = {}
            mpnn = OriginMPNNv2(hidden_channels)
            for edge_key in edge_keys:
                mpnns[edge_key] = mpnn

            conv = HeteroConv(mpnns, aggr='max')
            self.convs.append(conv)
            
        self.field = Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, 1))

    def get_vec(self, data):
        x_dict, edge_index_dict, edge_attr_dict = data.x_dict, data.edge_index_dict, data.edge_attr_dict
        for key in self.keys:
            x_dict[key] = self.embed[key](x_dict[key])
        for key in edge_attr_dict.keys():
            if key[0] in self.keys:
                edge_attr_dict[key] = self.edge_embed[str(key)](edge_attr_dict[key])
        for layer_i, conv in enumerate(self.convs):            
            new_dict = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            for key, value in new_dict.items():
                x_dict[key] = new_dict[key]
        
        vec = x_dict['agent']
        return vec
    
    def get_field(self, vec, action):
        if self.pos_encode is not None:
            action = torch.flatten(torch.cat((torch.sin(action.unsqueeze(-1)*self.div_term), 
                                torch.cos(action.unsqueeze(-1)*self.div_term)), dim=-1), start_dim=-2)
        field = self.field(torch.cat((vec, action), dim=-1))
        if self.mode=='sum':
            field = (field**2).sum(dim=-1)
        else:
            field = field.squeeze(dim=-1)
        return field
    
    def forward(self, data):
        
        vec = self.get_vec(data)
        field = self.get_field(vec=vec, action=data['action'])
                
        return field        


class OriginGNNv10(torch.nn.Module):
    def __init__(self, hidden_channels=1024, num_layers=3, keys=None, mode='straight', pos_encode=None):
        super().__init__()
        assert mode=='sum' or mode=='straight'
        
        self.mode = mode
        self.pos_encode = pos_encode
        
        if self.pos_encode is not None:
            div_term = torch.exp(torch.arange(0, pos_encode, 2))
            self.register_buffer('div_term', div_term)
        
        if keys is None:
            self.keys = {'obstacle', 'agent', 'goal'}
        else:
            self.keys = keys
        
        edge_keys = {('obstacle', 'o_near_a', 'agent'), 
                        ('agent', 'a_near_a', 'agent'), 
                        ('goal', 'toward', 'agent')}
        
        for edge in list(edge_keys):
            if edge[0] not in self.keys:
                edge_keys.remove(edge)
        self.edge_keys = edge_keys        
        
        self.embed = torch.nn.ModuleDict()
        for key in self.keys:
            self.embed[key] = LazyLinear(hidden_channels)
        self.edge_embed = torch.nn.ModuleDict()
        for edge_key in self.edge_keys:
            self.edge_embed[str(edge_key)] = Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, hidden_channels))
        
        self.convs = torch.nn.ModuleList()
        self.edge_mlps = torch.nn.ModuleList()
        for _ in range(num_layers):
            mpnns = {}
            mpnn = OriginMPNNv2(hidden_channels)
            for edge_key in edge_keys:
                mpnns[edge_key] = mpnn

            conv = HeteroConv(mpnns, aggr='max')
            self.convs.append(conv)
            self.edge_mlps.append(Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, hidden_channels)))
            
        self.field = Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, 1))

    def get_vec(self, data):
        x_dict, edge_index_dict, edge_attr_dict = data.x_dict, data.edge_index_dict, data.edge_attr_dict
        for key in self.keys:
            x_dict[key] = self.embed[key](x_dict[key])
        for key in edge_attr_dict.keys():
            if key[0] in self.keys:
                edge_attr_dict[key] = self.edge_embed[str(key)](edge_attr_dict[key])
        for conv, edge_mlp in zip(self.convs, self.edge_mlps):            
            new_dict = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            for key, value in new_dict.items():
                x_dict[key] = new_dict[key]
            
            for key in self.edge_keys:
                if key in edge_attr_dict:
                    edge_attr_dict[key] = edge_attr_dict[key] + edge_mlp(torch.cat((edge_attr_dict[key], x_dict[key[-1]][edge_index_dict[key][1,:]]), dim=-1))
        
        vec = x_dict['agent']
        return vec
    
    def get_field(self, vec, action):
        if self.pos_encode is not None:
            action = torch.flatten(torch.cat((torch.sin(action.unsqueeze(-1)*self.div_term), 
                                torch.cos(action.unsqueeze(-1)*self.div_term)), dim=-1), start_dim=-2)
        field = self.field(torch.cat((vec, action), dim=-1))
        if self.mode=='sum':
            field = (field**2).sum(dim=-1)
        else:
            field = field.squeeze(dim=-1)
        return field
    
    def forward(self, data):
        
        vec = self.get_vec(data)
        field = self.get_field(vec=vec, action=data['action'])
                
        return field
    
    
class OriginGNNv11(torch.nn.Module):
    def __init__(self, hidden_channels=1024, num_layers=3, keys=None, mode='straight', pos_encode=None):
        super().__init__()
        assert mode=='sum' or mode=='straight'
        
        self.mode = mode
        self.pos_encode = pos_encode
        
        if self.pos_encode is not None:
            div_term = torch.exp(torch.arange(0, pos_encode, 2))
            self.register_buffer('div_term', div_term)
        
        if keys is None:
            self.keys = {'obstacle', 'agent', 'goal'}
        else:
            self.keys = keys
        
        edge_keys = {('obstacle', 'o_near_a', 'agent'), 
                        ('agent', 'a_near_a', 'agent'), 
                        ('goal', 'toward', 'agent')}
        
        for edge in list(edge_keys):
            if edge[0] not in self.keys:
                edge_keys.remove(edge)
        self.edge_keys = edge_keys        
        
        self.embed = Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, hidden_channels))
        self.edge_embed = Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, hidden_channels))
        
        self.convs = torch.nn.ModuleList()
        self.edge_mlps = torch.nn.ModuleList()
        for _ in range(num_layers):
            mpnns = {}
            mpnn = OriginMPNNv2(hidden_channels)
            for edge_key in edge_keys:
                mpnns[edge_key] = mpnn

            conv = HeteroConv(mpnns, aggr='max')
            self.convs.append(conv)
            self.edge_mlps.append(Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, hidden_channels)))
            
        self.field = Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, 1))

    def get_vec(self, data):
        x_dict, edge_index_dict, edge_attr_dict = data.x_dict, data.edge_index_dict, data.edge_attr_dict
        for key in self.keys:
            if key in x_dict:
                x_dict[key] = self.embed(x_dict[key])
        for key in edge_attr_dict.keys():
            if key[0] in self.keys:
                edge_attr_dict[key] = self.edge_embed(edge_attr_dict[key])
        for conv, edge_mlp in zip(self.convs, self.edge_mlps):            
            new_dict = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            for key, value in new_dict.items():
                x_dict[key] = new_dict[key]
            
            for key in self.edge_keys:
                if key in edge_attr_dict:
                    edge_attr_dict[key] = edge_attr_dict[key] + edge_mlp(torch.cat((edge_attr_dict[key], x_dict[key[-1]][edge_index_dict[key][1,:]]), dim=-1))

        vec = x_dict['agent']
        return vec
    
    def get_field(self, vec, action):
        if self.pos_encode is not None:
            action = torch.flatten(torch.cat((torch.sin(action.unsqueeze(-1)*self.div_term), 
                                torch.cos(action.unsqueeze(-1)*self.div_term)), dim=-1), start_dim=-2)
        field = self.field(torch.cat((vec, action), dim=-1))
        if self.mode=='sum':
            field = (field**2).sum(dim=-1)
        else:
            field = field.squeeze(dim=-1)
        return field
    
    def forward(self, data):
        
        vec = self.get_vec(data)
        field = self.get_field(vec=vec, action=data['action'])
                
        return field    
    
    
class OriginGNNv12(torch.nn.Module):
    def __init__(self, hidden_channels=1024, num_layers=3, keys=None, mode='straight', pos_encode=None):
        super().__init__()
        assert mode=='sum' or mode=='straight'
        
        self.mode = mode
        self.pos_encode = pos_encode
        
        if self.pos_encode is not None:
            div_term = torch.exp(torch.arange(0, pos_encode, 2))
            self.register_buffer('div_term', div_term)    
        
        self.embed = Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, hidden_channels))
        self.edge_embed = Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, hidden_channels))
        
        self.convs = torch.nn.ModuleList()
        self.edge_mlps = torch.nn.ModuleList()
        for _ in range(num_layers):
            mpnn = OriginMPNNv2(hidden_channels)
            self.convs.append(mpnn)
            self.edge_mlps.append(Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, hidden_channels)))
            
        self.field = Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, 1))

    def get_vec(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        origin_x = x.clone()
        origin_attr = edge_attr.clone()
        x = self.embed(x)
        edge_attr = self.edge_embed(edge_attr)
        
        for conv, edge_mlp in zip(self.convs, self.edge_mlps):     
            try:
                x = conv(x, edge_index, edge_attr)
            except:
                print(edge_index, origin_x, origin_attr)
            edge_attr = edge_attr + edge_mlp(torch.cat((edge_attr, x[edge_index[1,:]]), dim=-1))

        vec = x[origin_x[:,0]==1, :]
        return vec
    
    def get_field(self, vec, action):
        if self.pos_encode is not None:
            action = torch.flatten(torch.cat((torch.sin(action.unsqueeze(-1)*self.div_term), 
                                torch.cos(action.unsqueeze(-1)*self.div_term)), dim=-1), start_dim=-2)
        field = self.field(torch.cat((vec, action), dim=-1))
        if self.mode=='sum':
            field = (field**2).sum(dim=-1)
        else:
            field = field.squeeze(dim=-1)
        return field
    
    def forward(self, data):
        
        vec = self.get_vec(data)
        field = self.get_field(vec=vec, action=data['action'])
                
        return field    
        
        
        
        
class RLNet(torch.nn.Module):
    def __init__(self, hidden_channels=1024, 
                 output_channels=None,
                 use_tanh=True,
                 use_global=False,
                 num_layers=3, keys=None, pos_encode=None):
        super().__init__()
        self.pos_encode = pos_encode
        self.use_tanh = use_tanh
        self.use_global = use_global
        
        if self.pos_encode is not None:
            div_term = torch.exp(torch.arange(0, pos_encode, 2))
            self.register_buffer('div_term', div_term)
        
        if keys is None:
            self.keys = {'obstacle', 'agent', 'goal'}
        else:
            self.keys = keys
        
        edge_keys = {('obstacle', 'o_near_a', 'agent'), 
                        ('agent', 'a_near_a', 'agent'), 
                        ('goal', 'toward', 'agent')}
        
        for edge in list(edge_keys):
            if edge[0] not in self.keys:
                edge_keys.remove(edge)
        self.edge_keys = edge_keys        
        
        self.embed = Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, hidden_channels))
        self.edge_embed = Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, hidden_channels))
        
        self.convs = torch.nn.ModuleList()
        self.edge_mlps = torch.nn.ModuleList()
        for _ in range(num_layers):
            mpnns = {}
            mpnn = OriginMPNNv2(hidden_channels)
            for edge_key in edge_keys:
                mpnns[edge_key] = mpnn

            conv = HeteroConv(mpnns, aggr='max')
            self.convs.append(conv)
            self.edge_mlps.append(Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, hidden_channels)))
          
        self.mode = 'critic' if output_channels is None else 'actor'
        if output_channels is not None:
            self.transform = Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, output_channels))
        else:
            self.field = Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, 1))
        
        if use_global and self.mode=='critic':
            self.action_net = Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, hidden_channels))            
            self.global_net = Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, hidden_channels))            
            
    def get_vec(self, data):
        x_dict, edge_index_dict, edge_attr_dict = data.x_dict, data.edge_index_dict, data.edge_attr_dict
        for key in self.keys:
            if key in x_dict:
                x_dict[key] = self.embed(x_dict[key])
        for key in edge_attr_dict.keys():
            if key[0] in self.keys:
                edge_attr_dict[key] = self.edge_embed(edge_attr_dict[key])
        for conv, edge_mlp in zip(self.convs, self.edge_mlps):            
            new_dict = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            for key, value in new_dict.items():
                x_dict[key] = new_dict[key]
            
            for key in self.edge_keys:
                if key in edge_attr_dict:
                    edge_attr_dict[key] = edge_attr_dict[key] + edge_mlp(torch.cat((edge_attr_dict[key], x_dict[key[-1]][edge_index_dict[key][1,:]]), dim=-1))

        vec = x_dict['agent']
        return vec
    
    def get_field(self, vec, action):
        if self.pos_encode is not None:
            action = torch.flatten(torch.cat((torch.sin(action.unsqueeze(-1)*self.div_term), 
                                torch.cos(action.unsqueeze(-1)*self.div_term)), dim=-1), start_dim=-2)
        if self.use_global:
            feature = self.action_net(torch.cat((vec, action), dim=-1))
            max_pool, _ = torch.max(feature, dim=0, keepdim=True)
            feature = self.global_net(torch.cat((feature, max_pool.repeat(len(feature), 1)), dim=-1))
        else:
            feature = torch.cat((vec, action), dim=-1)
        field = self.field(feature)
        field = field.squeeze(dim=-1)
        return field
    
    def forward(self, data):
        
        if self.mode=='critic':
            vec = self.get_vec(data)
            field = self.get_field(vec=vec, action=data['action'])
            return field 
        else:
            vec = self.get_vec(data)
            if self.use_tanh:
                return self.transform(vec).tanh()
            else:
                return self.transform(vec)
                

class UR5Net(torch.nn.Module):
    def __init__(self, hidden_channels=1024, num_layers=3, **kwargs):
        super().__init__()
        
        self.action_embed = Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, hidden_channels))
        self.field = Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, 1))
        self.vec = Seq(LazyLinear(hidden_channels), ReLU(), Lin(hidden_channels, hidden_channels))

    def get_vec(self, data):
        x_dict, edge_index_dict, edge_attr_dict = data.x_dict, data.edge_index_dict, data.edge_attr_dict

        vec = self.vec(edge_attr_dict['agent', 'a_near_a', 'agent'])
        
        # new_vec = torch.zeros_like(vec)
        # new_vec[edge_index_dict['agent', 'a_near_a', 'agent'][1,:], :] = vec
        
        new_vec = scatter_max(vec, edge_index_dict['agent', 'a_near_a', 'agent'][1,:], dim_size=edge_index_dict['agent', 'a_near_a', 'agent'].max()+1, dim=0)[0]
        
        return new_vec
    
    def get_field(self, vec, action):
        feature = vec + self.action_embed(action)
        field = self.field(feature).squeeze(dim=-1)
        return field
    
    def forward(self, data):
        
        vec = self.get_vec(data)
        field = self.get_field(vec=vec, action=data['action'])
                
        return field