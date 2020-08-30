from torch.nn import Module, Parameter
from torch.nn.init import xavier_normal
import torch
from torch_scatter import segment_coo

class DomainBlock(Module):
    def __init__(self, x_dim, ew_dim, out_dim, direct=False):
        super(DomainBlock, self).__init__()
        self.w_x    = Parameter(torch.Tensor(x_dim, out_dim))
        self.w_ew_i = Parameter(torch.Tensor(ew_dim, out_dim))
        self.w_ew_j = Parameter(torch.Tensor(ew_dim, out_dim))

        xavier_normal(self.w_x)
        xavier_normal(self.w_ew_i)
        xavier_normal(self.w_ew_j)

    def forward(self, x, edge_index, edge_weight):
        # incident nodes information
        mx_1 = x[edge_index[0,:],:].view(-1, x.shape[1])
        mx_2 = x[edge_index[1,:],:].view(-1, x.shape[1])
        mx = torch.matmul(mx_1 + mx_2, self.w_x)

        # edge information
        mew_i = torch.matmul(edge_weight, self.w_ew_i)

        # incident node's edges information
        sum_ew = segment_coo(edge_weight, edge_index[0,:], reduce="sum")
        mew_j1 = sum_ew[edge_index[0,:], :].view(-1, edge_weight.shape[1])
        mew_j2 = sum_ew[edge_index[1,:], :].view(-1, edge_weight.shape[1])
        mew_j  = torch.matmul(mew_j1 + mew_j2, self.w_ew_j)

        return mx + mew_i + mew_j