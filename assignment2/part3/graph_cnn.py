import torch.nn as nn
import torch

import torch.nn.functional as F
from torch_geometric.utils import add_self_loops


class MatrixGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(MatrixGraphConvolution, self).__init__()
        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
        self.B = nn.Parameter(torch.Tensor(out_features, in_features))

        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.B)

    def make_adjacency_matrix(self, edge_index, num_nodes):
        """
        Creates adjacency matrix from edge index.

        :param edge_index: [source, destination] pairs defining directed edges nodes. dims: [2, num_edges]
        :param num_nodes: number of nodes in the graph.
        :return: adjacency matrix with shape [num_nodes, num_nodes]

        Hint: A[i,j] -> there is an edge from node j to node i
        """
        adjacency_matrix = torch.zeros(num_nodes, num_nodes, device=edge_index.device, dtype=edge_index.dtype)
        src, dst = edge_index
        adjacency_matrix[src, dst] = 1.0
        return adjacency_matrix.to(edge_index.device)

    def make_inverted_degree_matrix(self, edge_index, num_nodes):
        """
        Creates inverted degree matrix from edge index.

        :param edge_index: [source, destination] pairs defining directed edges nodes. shape: [2, num_edges]
        :param num_nodes: number of nodes in the graph.
        :return: inverted degree matrix with shape [num_nodes, num_nodes]. Set degree of nodes without an edge to 1.
        """
        dst = edge_index[1]
        # degree is defined as the number of incoming edges to a node
        degree_vector = torch.bincount(dst, minlength=num_nodes).float()
        
        # for nodes without an edge, set the degree to 1 according to the instructions
        degree = degree_vector.masked_fill(degree_vector == 0, 1.0)
        
        inverted_degree_vector = 1 / degree
        inverted_degree_matrix = torch.diag(inverted_degree_vector).to(edge_index.device)
        return inverted_degree_matrix

    def forward(self, x, edge_index):
        """
        Forward propagation for GCNs using efficient matrix multiplication.

        :param x: values of nodes. shape: [num_nodes, num_features]
        :param edge_index: [source, destination] pairs defining directed edges nodes. shape: [2, num_edges]
        :return: activations for the GCN
        """
        A = self.make_adjacency_matrix(edge_index, x.size(0))
        D_inv = self.make_inverted_degree_matrix(edge_index, x.size(0))
        # following the derivation of the update rule in Q3.3.a
        out = D_inv @ A @ x @ self.W.T + x @ self.B.T
        return out

class MessageGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(MessageGraphConvolution, self).__init__()
        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
        self.B = nn.Parameter(torch.Tensor(out_features, in_features))

        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.B)

    @staticmethod
    def message(x, edge_index):
        """
        message step of the message passing algorithm for GCNs.

        :param x: values of nodes. shape: [num_nodes, num_features]
        :param edge_index: [source, destination] pairs defining directed edges nodes. shape: [2, num_edges]
        :return: message vector with shape [num_nodes, num_in_features]. Messages correspond to the old node values.

        Hint: check out torch.Tensor.index_add function
        """
        src, dst = edge_index
        num_nodes = x.size(0)
        
        # messages are the old node features
        messages = x[src]
        
        # sum messages per dest node
        aggregated_messages = torch.zeros_like(x).to(x.device)
        
        # index_add_ instead of index_add to avoid do the calculation in-place for efficiency
        aggregated_messages.index_add_(0, dst, messages)
        
        # degree == number of incoming edges to a node
        sum_weight = torch.bincount(dst, minlength=num_nodes).float()
        sum_weight = sum_weight.masked_fill(sum_weight == 0, 1.0)
        
        # mean aggregation as defined in equation 6
        aggregated_messages = aggregated_messages / sum_weight.unsqueeze(1)
        
        return aggregated_messages

    def update(self, x, messages):
        """
        update step of the message passing algorithm for GCNs.

        :param x: values of nodes. shape: [num_nodes, num_features]
        :param messages: messages vector with shape [num_nodes, num_in_features]
        :return: updated values of nodes. shape: [num_nodes, num_out_features]
        """
        # following equation 6. activation functions are applied in the model.py file according to the instructions
        # in the README.md file.
        # Shape: [num_nodes, num_out_features] = [num_nodes, num_in_features] @ [num_in_features, num_out_features] + [num_nodes, num_features] @ [num_features, num_out_features]
        x = messages @ self.W.T + x @ self.B.T
        return x

    def forward(self, x, edge_index):
        message = self.message(x, edge_index)
        x = self.update(x, message)
        return x


class GraphAttention(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphAttention, self).__init__()
        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
        self.a = nn.Parameter(torch.Tensor(out_features * 2))

        nn.init.xavier_uniform_(self.W)
        nn.init.uniform_(self.a, 0, 1)

    def forward(self, x, edge_index, debug=False):
        """
        Forward propagation for GATs.
        Follow the implementation of Graph attention networks (Veličković et al. 2018).

        :param x: values of nodes. shape: [num_nodes, num_features]
        :param edge_index: [source, destination] pairs defining directed edges nodes. shape: [2, num_edges]
        :param debug: used for tests
        :return: updated values of nodes. shape: [num_nodes, num_out_features]
        :return: debug data for tests:
                 messages -> messages vector with shape [num_nodes, num_out_features], i.e. Wh from Veličković et al.
                 edge_weights_numerator -> unnormalized edge weightsm i.e. exp(e_ij) from Veličković et al.
                 softmax_denominator -> per destination softmax normalizer

        Hint: the GAT implementation uses only 1 parameter vector and edge index with self loops
        Hint: It is easier to use/calculate only the numerator of the softmax
              and weight with the denominator at the end.

        Hint: check out torch.Tensor.index_add function
        """
        edge_index, _ = add_self_loops(edge_index)

        sources, destinations = edge_index
        num_nodes = x.size(0)
        Wh = x @ self.W.T # Shape: [num_nodes, num_out_features]
        
        
        # activations = ...
        messages = Wh # Shape: [num_edges, num_out_features]
        m_src = messages[sources] # Shape: [num_edges, num_out_features]
        m_dst = messages[destinations] # Shape: [num_edges, num_out_features]

        attention_inputs = torch.cat([m_src, m_dst], dim=1) # Shape: [num_edges, num_out_features * 2]

        e_ij = F.leaky_relu(attention_inputs @ self.a) # Shape: [num_edges, 1]
        edge_weights_numerator = torch.exp(e_ij) # Shape: [num_edges, 1]
        
        weighted_messages = edge_weights_numerator[:, None] * m_src # Shape: [num_edges, num_out_features]

        softmax_denominator = torch.zeros(num_nodes, device=edge_weights_numerator.device, dtype=edge_weights_numerator.dtype)
        softmax_denominator.index_add_(0, destinations, edge_weights_numerator) # Shape: [num_nodes, 1]§
        # clamping min to avoid division by zero
        softmax_denominator = torch.clamp(softmax_denominator, min=1e-12) # Shape: [num_nodes, 1]

        aggregated_messages = torch.zeros_like(Wh).to(Wh.device)
        aggregated_messages.index_add_(0, destinations, weighted_messages)
        # devide by softmax denominator
        aggregated_messages = aggregated_messages / softmax_denominator[:, None] # Shape: [num_nodes, num_out_features]
        return aggregated_messages, {'edge_weights': edge_weights_numerator, 'softmax_weights': softmax_denominator,
                                     'messages': messages}

