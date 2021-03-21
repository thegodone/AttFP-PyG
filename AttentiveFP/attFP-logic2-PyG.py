import torch
from torch import Tensor

from torch.nn import Linear, BatchNorm1d, Dropout
from torch.nn import Parameter as Param
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, EdgePooling
from torch_sparse import matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.typing import PairTensor, Adj, OptTensor, Size
from torch_scatter import scatter_add
import pickle

from typing import Union

class GatConvAtom(MessagePassing):
    """
    This function does only the atom embedding, not the molecule embedding
    """
    def __init__(self, atom_in_channels: int, bond_in_channels: int, fingerprint_dim: int, dropout: float, bias: bool = True, debug: bool = False, step = 0, compress = False, **kwargs):
        super(GatConvAtom, self).__init__()

        self.atom_in_channels = atom_in_channels
        self.bond_in_channels = bond_in_channels
        self.fingerprint_dim = fingerprint_dim
        self.step = step
        self.compress = compress
        if self.compress:
            self.DS = Linear(fingerprint_dim, 32, bias=bias)
            self.US = Linear(32, fingerprint_dim, bias=bias)
        
        if  self.step == 0 : 
            self.atom_fc = Linear(atom_in_channels, fingerprint_dim, bias=bias)
            self.neighbor_fc = Linear(atom_in_channels + bond_in_channels, fingerprint_dim, bias=bias)
        self.align = Linear(2*fingerprint_dim, 1, bias=bias)
        self.attend = Linear(fingerprint_dim, fingerprint_dim, bias=bias)
        self.debug = debug
        self.dropout = Dropout(p=dropout)
        self.rnn = torch.nn.GRUCell(fingerprint_dim, fingerprint_dim)

        
    def forward(self, x: Union[Tensor,PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        
        out = self.propagate(edge_index, x = x, edge_attr=edge_attr, size=size)
        return out

    def message(self, x_i, x_j, edge_index: Adj, edge_attr: OptTensor, size) -> Tensor:

        if self.debug:
            print('a x_j:',x_j.shape,'x_i:',x_i.shape,'edge_attr:',edge_attr.shape)
        if  self.step == 0 :

            x_i = F.leaky_relu(self.atom_fc(x_i)) # code 3 

            # neighbor_feature => neighbor_fc
            x_j = torch.cat([x_j, edge_attr], dim=-1) # code 8
            if self.debug:
                print('b neighbor_feature i = 0', x_j.shape)
            
            x_j = F.leaky_relu(self.neighbor_fc(x_j)) # code 9
            if self.debug:
                print('c neighbor_feature i = 0', x_j.shape)
            
        # align score
        evu = F.leaky_relu(self.align(torch.cat([x_i, x_j], dim=-1))) # code 10
        if self.debug:
            print('d align_score:', evu.shape)
        
        avu = EdgePooling.compute_edge_score_softmax(evu, edge_index, edge_index.max().item() + 1) # code 11
        if self.debug:
            print('e attention_weight:', avu.shape)
        # to do downscaling 200 fp => 32
        c_i = F.elu(torch.mul(avu, self.attend(self.dropout(x_i)))) # code 12
        # to do upscaling 32 => 200 fp 
        if self.debug:
            print('f context',c_i.shape)
            
        x_i = self.rnn(c_i, x_i)
        if self.debug:
            print('g gru',c_i.shape)            

        return x_i   

class GatConvMol(MessagePassing):
    """
    This function does the molecule embedding
    """
    def __init__(self, fingerprint_dim: int, dropout: int, debug: bool = False, step = 0):
        super(GatConvMol, self).__init__()
        # need to find the correct dimensions 
        self.step = step
        self.mol_align = Linear(2*fingerprint_dim,1)
        self.mol_attend = Linear(fingerprint_dim,fingerprint_dim)
        self.dropout = Dropout(p=dropout)
        self.debug = debug
        self.rnn = torch.nn.GRUCell(fingerprint_dim, fingerprint_dim)

    def forward(self, x: Union[Tensor,PairTensor], edge_index: Adj, size: Size = None) -> Tensor:
        
        out = self.propagate(edge_index, x = x, size=size)
        return out

    def message(self, x_i, x_j, edge_index: Adj, size) -> Tensor:
        if self.step == 0:
            h_s =  torch.sum(x_i, dim=-1)
            if self.debug:
                print('pre-h_s:',h_s.shape,',x_i:', x_i.shape)            
                
            h_s =  h_s.unsqueeze(1).repeat(1, x_i.size(1)) # code 2
            if self.debug:
                print('1 mol_feature expanded',h_s.shape)

        else:
            h_s = x_i
        
        if self.debug:
            print('2 activated_features', x_i.shape)
             
        esv = F.leaky_relu(self.mol_align(torch.cat([h_s, x_i], dim=-1))) # code 5
        if self.debug:
            print('3 mol_align_score:',esv.shape)
        asv = F.softmax(esv, dim=-1) # code 6
    
        if self.debug:
            print('4 mol_align_score:',asv.shape)
        
        # this is not correct it should be more hs and not x_i there based on the paper supplementary table 3!
        cs_i = F.elu(torch.mul(asv, self.mol_attend(self.dropout(h_s)))) # code 7 
        if self.debug:
            print('5 mol_context' ,cs_i.shape)
            
        x_i = self.rnn(cs_i, h_s) # code 8
        
        return x_i


class AtomEmbedding(torch.nn.Module):
    def __init__(self, atom_dim,  edge_dim, fp_dim, R=2, dropout = 0.2, debug=False):
        super(AtomEmbedding, self).__init__()
        self.R = R
        self.debug = debug
        self.conv = torch.nn.ModuleList([GatConvAtom(atom_in_channels=atom_dim, bond_in_channels= edge_dim, fingerprint_dim=fp_dim, dropout = dropout, debug=debug, step = i) for i in range(self.R)])  # GraphMultiHeadAttention

    def forward(self, x, edge_index, edge_attr):
        for i in range(self.R):
            if self.debug:
                print(x.shape)
            # skip connection
            x = self.conv[i](x, edge_index, edge_attr) # code 1-12
            if self.debug:
                print(x.shape)    
        return x

class MoleculeEmbedding(torch.nn.Module):
    def __init__(self, fp_dim, dropout, debug, T=2):
        super(MoleculeEmbedding, self).__init__()
        self.T = T
        self.debug = debug
        self.conv =torch.nn.ModuleList([GatConvMol(fp_dim, dropout, debug, step = i) for i in range(self.T)])

    def forward(self, x, edge_index):
        for i in range(self.T):
            x = self.conv[i](x, edge_index) # code 1-7
        return x

class AttentiveFPdebug(torch.nn.Module):
    def __init__(self, atom_in_dim, edge_in_dim, fingerprint_dim=32, R=2, T=2, dropout=0.2,  debug = False, outdim=1):
        super(AttentiveFPdebug, self).__init__()
        self.R = R
        self.T = T
        self.debug = debug
        self.dropout = dropout
        # call the atom embedding Phase
        self.convsAtom = AtomEmbedding(atom_in_dim, edge_in_dim, fingerprint_dim, R, debug) 
        self.convsMol = MoleculeEmbedding(fingerprint_dim, dropout, debug, T )

        # fast down project could be much more sofisticated! (ie  Feed Forward Network with multiple layers )
        self.out = Linear(fingerprint_dim, outdim) 
        
    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_feat
        x = F.dropout(self.convsAtom(x, edge_index, edge_attr), p=self.dropout, training=self.training) # atom Embedding       
        x = F.dropout(self.convsMol(x, edge_index), p=self.dropout, training=self.training) # molecule Embedding
        
        x = self.out(global_add_pool(x, batch))
        return x