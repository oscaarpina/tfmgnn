import torch
from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_mean_pool
from models.domain_block import DomainBlock

class AALModel(torch.nn.Module):
  def __init__(self):
    super(AALModel, self).__init__()

    self.conv1 = GraphConv(1,3)
    self.dom1  = DomainBlock(1,1,2, direct=True)
    self.nn1   = torch.nn.Sequential(torch.nn.Linear(2,1), torch.nn.ReLU())

    self.conv2 = GraphConv(3,3)
    self.dom2  = DomainBlock(3,2,3, direct=True)
    self.nn2   = torch.nn.Sequential(torch.nn.Linear(3,1), torch.nn.ReLU())

    self.conv3 = GraphConv(3,5)
    self.dom3  = DomainBlock(3,3,4)
    self.nn3   = torch.nn.Sequential(torch.nn.Linear(4,1), torch.nn.ReLU())

    self.mlp = torch.nn.Sequential(
      torch.nn.Dropout(0.2),
      torch.nn.Linear(23,2),
      torch.nn.LogSoftmax(dim=-1)
    )

  def forward(self, data, cam=False):
    x, edge_index, edge_weight, g, batch = data.x, data.edge_index, data.edge_attr, data.g, data.batch

    x0, ew0 = x, edge_weight
    ew1 = torch.nn.ReLU()(self.dom1(x0, edge_index, ew0))
    x1 = torch.nn.ReLU()( self.conv1(x0, edge_index=edge_index, edge_weight=self.nn1(ew1)) )

    ew2 = torch.nn.ReLU()(self.dom2(x1, edge_index, ew1))
    x2 = torch.nn.ReLU()( self.conv2(x1, edge_index=edge_index, edge_weight=self.nn2(ew2)) )

    ew3 = self.dom3(x2, edge_index, ew2)
    x3 = torch.nn.ReLU()( self.conv3(x2, edge_index=edge_index, edge_weight=self.nn3(ew3)) )

    # ew4 = self.dom4(x3, edge_index, ew3)
    # x4 = torch.nn.ReLU()( self.conv4(x3, edge_index=edge_index, edge_weight=self.nn4(ew4)) )

    batch_e = batch[edge_index[0,:]].squeeze()

    x =  torch.cat([x0,  x1,  x2, x3],  dim=1)
    ew = torch.cat([ew0, ew1, ew2, ew3], dim=1)
    #x, ew = x3, ew3

    if cam:
      return x, ew
    #x = torch.cat([global_max_pool(ew, batch_e), global_mean_pool(ew, batch_e), global_max_pool(x, batch), global_mean_pool(x,batch)], dim=1)

    x = torch.cat([global_mean_pool(x,batch), global_mean_pool(ew,batch_e), g], dim=1)
    return self.mlp(x)

