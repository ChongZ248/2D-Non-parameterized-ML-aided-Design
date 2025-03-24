import torch

def cnntogra(pos, batch, datatomap):
    # this is a function to convert the batch of CNN data to the corresponding graph data based on the node position
    # the graph data is a batched graph therefore the "batch" is needed to distinguish the different sub graphs
    # datatomap is the CNN data with the shape of [batch_size, 1, height, width]
    # pos have the shape of [node_number, 2]
    # batch have the shape of [node_number]
    unique_indices, inverse_indices = torch.unique(batch, return_inverse=True)
    posindexT = torch.round(pos / 20 * (datatomap.shape[-1] - 1)).long()
    posindex = torch.stack((posindexT[:, 1], posindexT[:, 0]), dim=-1)
    selected_images = datatomap[unique_indices]
    result = selected_images[inverse_indices, :, posindex[:, 0], posindex[:, 1]]
    return result


def gratocnn(resultfromgnn,pos,batch,oldcnn):
    imageindex=batch
    # Example tensors for demonstration
    batchedimage =  torch.ones(oldcnn.shape[0], resultfromgnn.shape[-1], oldcnn.shape[2], oldcnn.shape[3]).to(oldcnn.device)*-10 # This can be of different sizes now
    pixelvalue = resultfromgnn  # Random values
    imageindex = imageindex.unsqueeze(dim=-1)  # Adjusted to batch size
    posindexT=torch.round(pos/20*(oldcnn.shape[-1]-1)).long()
    posindex=torch.stack((posindexT[:,1],posindexT[:,0]),dim=-1)
    # Flatten the batchedimage tensor
    batchedimage_flat = batchedimage.view(batchedimage.size(0), batchedimage.size(1), -1)
    # Get image dimensions
    img_height, img_width = batchedimage.size(2), batchedimage.size(3)
    # Calculate the linear indices
    linear_indices = posindex[:, 0] * img_width + posindex[:, 1]
    # Use advanced indexin to assign values
    batchedimage_flat[imageindex[:, 0], :, linear_indices] = pixelvalue
    # Reshape back to original shape
    batchedimage = batchedimage_flat.view(batchedimage.size(0), batchedimage.size(1), img_height, img_width)
    return batchedimage





import torch
import torch.nn.functional as F
from torch_scatter import scatter_add

class SAGraphLayerV3(torch.nn.Module):
    def __init__(self, in_channels, out_channels, aggregation='sum', bias=True):
        super(SAGraphLayerV3, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregation = aggregation  # 'sum' 或 'mean'

        self.weight_self = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.weight_neighbor = torch.nn.Linear(in_channels, out_channels, bias=False)

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_self.weight)
        torch.nn.init.xavier_uniform_(self.weight_neighbor.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x, edge_index):
        """
        x: (N, in_channels)  # 节点特征矩阵
        edge_index: (2, E)  # 边索引，表示图的连接关系
        """
        src, dst = edge_index  # 提取边的起点和终点

        out_self = self.weight_self(x)  # 计算每个节点的自特征变换 (N, out_channels)

        # 计算邻居的特征差值
        diff = x[src] - x[dst]  # (E, in_channels)

        # 对差值应用变换
        diff_transformed = self.weight_neighbor(diff)  # (E, out_channels)

        # 聚合邻居消息
        out_neighbors = scatter_add(diff_transformed, dst, dim=0, dim_size=x.size(0))  # (N, out_channels)

        if self.aggregation == 'mean':
            deg = scatter_add(torch.ones_like(dst, dtype=torch.float), dst, dim=0, dim_size=x.size(0)).unsqueeze(-1)
            deg[deg == 0] = 1  # 避免除零
            out_neighbors /= deg

        out = out_self + out_neighbors  # (N, out_channels)

        if self.bias is not None:
            out = out + self.bias

        return out



class SAGraphV3(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, dropout=0.0, aggregation='sum'):
        super(SAGraphV3, self).__init__()

        self.layers = torch.nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = torch.nn.Dropout(p=dropout)

        # Input layer
        self.layers.append(SAGraphLayerV3(in_channels, hidden_channels, aggregation))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(SAGraphLayerV3(hidden_channels, hidden_channels, aggregation))

        # Output layer
        self.layers.append(SAGraphLayerV3(hidden_channels, out_channels, aggregation))

    def forward(self, x, adj, mask=None):
        for i in range(self.num_layers):
            x = self.layers[i](x, adj)
            if i < self.num_layers - 1:
                x = self.dropout(torch.nn.functional.relu(x))

        return x

    def __repr__(self):
        return '{}(in_channels={}, hidden_channels={}, num_layers={}, out_channels={}, dropout={})'.format(
            self.__class__.__name__, self.layers[0].in_channels, self.layers[1].in_channels, self.num_layers, self.layers[-1].out_channels, self.dropout.p)

