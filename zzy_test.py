import torch
from torch_geometric.utils import to_dense_batch, to_dense_adj


# a = torch.Tensor([1,1,2,2,3,3,4,4])
# print(a.shape)
# b = a.unsqueeze(1)
# print(b.shape)
# print(b)
# w = torch.tensor([[1,3],[2,4],[1,5],[8,8],[4,2]])
# print(w)
#
# w_dense , w_mask = to_dense_batch(w , torch.tensor([0,0,0,1,1]))
# print(w_dense)
# print("----------------------")
# print(w_dense.sum(dim = 1))
#
# adj = to_dense_adj(torch.tensor([[0,1,2,0,4],[1,2,0,2,3]]), torch.tensor([0,0,0,1,1]))
# print(adj)
# print(adj.size(0))
#
# abc = torch.tensor([[1,2,3],[3,2,1],[5,7,9]])
# print(abc.sum())

# adj = torch.tensor([
#     [
#         [0, 1, 0],
#         [1, 0, 1],
#         [0, 1, 0]
#     ],
#     [
#         [0, 1, 1],
#         [1, 0, 1],
#         [1, 1, 0]
#     ]
# ])
#
# # 假设我们处理第1个图（batch中的第2个图）
# b = 0
# node_index = 0
#
# # 找到与 node_index 相连的所有邻居节点的索引
# neighbors = torch.where(adj[b][node_index] == 1)[0]
# print(neighbors)  # 输出 tensor([1, 2])


# model = torch.load("./checkpoints/zzy_handsome.pkl")
# print(model)


# import torch
# from models import *
# from torch_geometric.data import Data
# from models import MISSolver
#
#
# num_nodes = 6
# w = torch.ones(num_nodes, dtype=torch.float32)
# edge_index = torch.tensor([[0 , 0 , 1 , 1 , 1 , 2 , 2 , 3 , 3 , 3 , 4 , 4 , 4 , 5],
#                            [1 , 4 , 0 , 2 , 4 , 1 , 3 , 2 , 4 , 5 , 0 , 1 , 3 , 3]])
# batch = torch.zeros(num_nodes, dtype=torch.long)
# data = Data(x = w , edge_index = edge_index)
#
#
# zzy = MISSolver()
# zzy.load_state_dict(torch.load("./checkpoints/zzy_handsome.pkl"))
#
# zzy.eval()
# with torch.no_grad():
#     loss , MIS = zzy(w , edge_index , batch)
# print(MIS)

import torch

# a = torch.tensor([1 , 2 , 3.] , requires_grad = True)
# print(a.grad)
# out = torch.sigmoid(a)
# out.sum().backward()
# print(a.grad)

a = torch.tensor([1 , 2 , 3 , 4 , 5])
b = a.indices[1]
print(b)