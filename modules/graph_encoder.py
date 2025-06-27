import torch
from graph.data_utils import *

class GraphEncoder(torch.nn.Module):
    def __init__(self, gmodel, device):
        super().__init__()
        self.model = gmodel
        self.device = device


    def forward(self, graph_data):
        graph_batch = molgraph_to_graph_data(graph_data)
        graph_batch = graph_batch.to(self.device)
        self.model = self.model.to(self.device)

        super_node_rep = self.model(graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr, graph_batch.num_part).to(self.device)

        return super_node_rep
