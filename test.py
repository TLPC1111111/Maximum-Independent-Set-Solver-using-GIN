from models import *
import os.path as osp
import pprint
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from models import MISSolver

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')
dataset = Planetoid(path, name = 'Citeseer')

num_nodes = dataset.data.train_mask.size(0)
edge_index = dataset.data.edge_index
w = torch.ones(num_nodes, dtype=torch.float32)
batch = torch.zeros(num_nodes, dtype=torch.long)

zzy = MISSolver()
zzy.load_state_dict(torch.load("./checkpoints/zzy_handsome.pkl"))

zzy.eval()
with torch.no_grad():
    loss , MIS = zzy(w , edge_index , batch)
MIS_tensor = torch.tensor(MIS)
print(f"-------GNN在Cora-ML数据集上所求出的最大独立集节点个数为{len(MIS_tensor[0])}")
pprint.pprint(f'-------GNN在Cora-ML数据集上所求出的最大独立集为{MIS}')