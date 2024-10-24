from models import *
from torch_geometric.data import Data
from models import MISSolver
from Read_AFIT import *

afit = AFIT("reqlf13")

num_nodes = afit.total_length()
print(f"--------afscn转化为图问题的总节点个数为{num_nodes}")

w = torch.ones(num_nodes, dtype=torch.float32)

edge_index = [[],[]]
with open("SRS_edge.txt" , 'r') as rf:
    for line in rf:
        start, end = map(int, line.strip().split())
        edge_index[0].append(start)
        edge_index[1].append(end)
edge_index = torch.tensor(edge_index , dtype = int)
print(f"--------邻接矩阵(2*ndim)为{edge_index}")

batch = torch.zeros(num_nodes, dtype=torch.long)

data = Data(x = w , edge_index = edge_index)

zzy = MISSolver()
zzy.load_state_dict(torch.load("./checkpoints/zzy_handsome.pkl"))

zzy.eval()
with torch.no_grad():
    loss , MIS = zzy(w , edge_index , batch)
MIS_tensor = torch.tensor(MIS , dtype = int)
print(f"-------GNN在afscn数据集上所求出的最大独立集节点个数为{len(MIS_tensor[0])}")
print(f'-------GNN在afscn数据集上所求出的最大独立集为{MIS_tensor}')
