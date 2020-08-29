import torch
from torch_geometric.data import Data
import numpy as np
from utils.data.storage import GDatabase
from abc import ABC, abstractmethod

class GConversor(ABC):
    def __init__(self, device):
        self.device = device

    @abstractmethod
    def convert_gdatabase(self, gdb: GDatabase, label):
        raise NotImplementedError()

class GSparseConversor(GConversor):
    def __init__(self, device):
        super(GSparseConversor, self).__init__(device)

    def convert_gdatabase(self, gdb: GDatabase, label="group_aet") -> list:
        df_graph, df_nodes, df_edges = gdb.graph, gdb.nodes, gdb.edges

        pyg_ls = list()

        default_nodes, default_edges = False, False
        # we can have either different node signal for each graph or the same for all graphs
        (nodes, default_nodes) = (df_nodes.groupby(gdb.id_feature),False) if gdb.id_feature in df_nodes.columns else (df_nodes, True)
        # we can have either different edges for each graph or the same for all graphs
        (edges, default_edges) = (df_edges.groupby(gdb.id_feature),False) if gdb.id_feature in df_edges.columns else (df_edges, True)

        for idx, row in df_graph.iterrows():
            graph_id = row["graph_id"]
            if (default_nodes or graph_id in nodes.indices) and (default_edges or graph_id in edges.indices):
                pyg_ls.append( \
                    self.create_data_object(row, nodes if default_nodes else nodes.get_group(row["graph_id"]),
                                            edges if default_edges else edges.get_group(row["graph_id"]), \
                                            label, gdb.id_feature))
        return pyg_ls

    def create_data_object(self, graph, nodes, edges, label, id_feature) -> Data:
        g_feats = [ c for c in graph.index if c != label and c != id_feature ]
        n_feats = [ c for c in nodes.columns if c != "node_num" and c != id_feature]
        e_index = ["node_src", "node_dst"]
        e_feats = [ c for c in edges.columns if c not in e_index and c != id_feature]

        y = int(graph[label])
        y = torch.tensor(y, dtype=torch.long, device=self.device).view(1)

        g = None
        if len(g_feats)>0:
            g = graph[g_feats].values
            g = torch.tensor(g.astype(np.float), dtype=torch.float, device=self.device).view(1,-1)

        num_nodes = nodes.shape[0]
        x = nodes[n_feats].values
        x = torch.tensor(x, dtype=torch.float, device=self.device).view(num_nodes, -1)

        edge_index = edges[e_index].T.values
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device)

        edge_attr = edges[e_feats].values
        edge_attr = torch.tensor(edge_attr, dtype=torch.float, device=self.device).view(edge_index.size(1), -1)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, g=g)
        return data