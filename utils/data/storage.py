import pandas as pd
from sklearn.utils import Bunch
class GDatabase(Bunch):

    r"""
    Graph-Database based on Pandas.
    The class does not contain a list or set of graphs but three pandas.DataFrame instead:
    - Graph
    - Nodes
    - Edges
    Graph Dataframe contains a row per each graph of the database.
    Nodes and edges contains a row per each node/edge of each graph, so they must include a reference to the graph, specified by \"id_feature
    """

    def __init__(self, graph, nodes, edges, id_feature="graph_id"):
        super(GDatabase, self).__init__(graph=graph, nodes=nodes, edges=edges, id_feature=id_feature)

    @classmethod
    def from_files(cls, path, graph_file="graph.csv", nodes_file="nodes.csv", edges_file="edges.csv"):
        df_graph = pd.read_csv(path + graph_file)
        df_nodes = pd.read_csv(path + nodes_file)
        df_edges = pd.read_csv(path + edges_file)

        return cls(df_graph, df_nodes, df_edges)

    def get_graph_by_id(self, id_feat):
        return self.graph[self.graph[self.id_feature] == id_feat], \
               self.nodes[self.nodes[self.id_feature] == id_feat], \
               self.edges[self.edges[self.id_feature] == id_feat]

    def filter_graph(self, filter):
        self.graph = self.graph[filter]
        df_filter = pd.DataFrame(self.graph[self.id_feature].values, columns=["filter"])

        node_cols, edge_cols = self.nodes.columns, self.edges.columns
        if self.id_feature in node_cols:
            self.nodes = self.nodes.merge(df_filter, left_on=self.id_feature, right_on="filter", how="inner")[node_cols]
        if self.id_feature in edge_cols:
            self.edges = self.edges.merge(df_filter, left_on=self.id_feature, right_on="filter", how="inner")[edge_cols]
