import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, BatchNorm1d, Sequential, LeakyReLU, Dropout
from torch_geometric.nn import GCNConv, GATConv, TransformerConv, NNConv, SGConv, ARMAConv, TAGConv, ChebConv, DNAConv, LabelPropagation, \
EdgeConv, FiLMConv, FastRGCNConv, SSGConv, SAGEConv, GATv2Conv, BatchNorm, GraphNorm, MemPooling, SAGPooling, GINConv, CorrectAndSmooth

class GL_TAGConv_3l_128h_w_k3(torch.nn.Module):
    def __init__(self, data):
        super(GL_TAGConv_3l_128h_w_k3, self).__init__()
        self.conv1 = TAGConv(data.num_features, 128)
        self.conv2 = TAGConv(128, 128)
        self.conv3 = TAGConv(128, int(data.num_classes))

    def forward(self, data):
        x, edge_index, edge_attr = data.x.float(), data.edge_index, data.weight.float()
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = self.conv3(x, edge_index, edge_attr)
        return x


class GL_TAGConv_3l_512h_w_k3(torch.nn.Module):
    def __init__(self, data):
        super(GL_TAGConv_3l_512h_w_k3, self).__init__()
        self.conv1 = TAGConv(data.num_features, 512)
        self.conv2 = TAGConv(512, 512)
        self.conv3 = TAGConv(512, int(data.num_classes))

    def forward(self, data):
        x, edge_index, edge_attr = data.x.float(), data.edge_index, data.weight.float()
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = self.conv3(x, edge_index, edge_attr)
        return x


class GL_MLP_3l_128h(torch.nn.Module):
    def __init__(self, data):
        super().__init__()
        # self.norm = BatchNorm1d(3 * int(data.num_classes))
        self.fc1 = Linear(3 * int(data.num_classes), 128)
        self.fc2 = Linear(128, 128)
        self.fc3 = Linear(128, int(data.num_classes))

    def forward(self, data):
        h, edge_index, edge_weight = data.x.float(), data.edge_index, data.weight.float()
        # h = self.norm(h)
        h = self.fc1(h)
        h = F.elu(h)
        h = self.fc2(h)
        h = F.elu(h)
        h = self.fc3(h)
        return h
    
class GL_MLP_3l_512h(torch.nn.Module):
    def __init__(self, data):
        super().__init__()
        # self.norm = BatchNorm1d(3 * int(data.num_classes))
        self.fc1 = Linear(3 * int(data.num_classes), 512)
        self.fc2 = Linear(512, 512)
        self.fc3 = Linear(512, int(data.num_classes))

    def forward(self, data):
        h, edge_index, edge_weight = data.x.float(), data.edge_index, data.weight.float()
        # h = self.norm(h)
        h = self.fc1(h)
        h = F.elu(h)
        h = self.fc2(h)
        h = F.elu(h)
        h = self.fc3(h)
        return h
    
class GL_MLP_9l_128h(torch.nn.Module):
    def __init__(self, data):
        super().__init__()
        # print('NUM CLASSES', data.num_classes)
        # self.norm = BatchNorm1d(3 * int(data.num_classes))
        self.fc1 = Linear(3 * int(data.num_classes), 128)
        self.fc2 = Linear(128, 128)
        self.fc3 = Linear(128, 128)
        self.fc4 = Linear(128, 128)
        self.fc5 = Linear(128, 128)
        self.fc6 = Linear(128, 128)
        self.fc7 = Linear(128, 128)
        self.fc8 = Linear(128, 128)
        self.fc9 = Linear(128, int(data.num_classes))

    def forward(self, data):
        h, edge_index, edge_weight = data.x.float(), data.edge_index, data.weight.float()
        # h = self.norm(h)
        h = self.fc1(h)
        h = F.elu(h)
        h = self.fc2(h)
        h = F.elu(h)
        h = self.fc3(h)
        h = F.elu(h)
        h = self.fc4(h)
        h = F.elu(h)
        h = self.fc5(h)
        h = F.elu(h)
        h = self.fc6(h)
        h = F.elu(h)
        h = self.fc7(h)
        h = F.elu(h)
        h = self.fc8(h)
        h = F.elu(h)
        h = self.fc9(h)
        return h
    
class GL_MLP_9l_512h(torch.nn.Module):
    def __init__(self, data):
        super().__init__()
        # self.norm = BatchNorm1d(3 * int(data.num_classes))
        self.fc1 = Linear(3 * int(data.num_classes), 512)
        self.fc2 = Linear(512, 512)
        self.fc3 = Linear(512, 512)
        self.fc4 = Linear(512, 512)
        self.fc5 = Linear(512, 512)
        self.fc6 = Linear(512, 512)
        self.fc7 = Linear(512, 512)
        self.fc8 = Linear(512, 512)
        self.fc9 = Linear(512, int(data.num_classes))

    def forward(self, data):
        h, edge_index, edge_weight = data.x.float(), data.edge_index, data.weight.float()
        # h = self.norm(h)
        h = self.fc1(h)
        h = F.elu(h)
        h = self.fc2(h)
        h = F.elu(h)
        h = self.fc3(h)
        h = F.elu(h)
        h = self.fc4(h)
        h = F.elu(h)
        h = self.fc5(h)
        h = F.elu(h)
        h = self.fc6(h)
        h = F.elu(h)
        h = self.fc7(h)
        h = F.elu(h)
        h = self.fc8(h)
        h = F.elu(h)
        h = self.fc9(h)
        return h

class GL_TAGConv_3l_512h_w_k3_gnorm(torch.nn.Module):
    def __init__(self, data):
        super(GL_TAGConv_3l_512h_w_k3_gnorm, self).__init__()
        self.conv1 = TAGConv(int(data.num_features), 512)
        self.conv2 = TAGConv(512, 512)
        self.conv3 = TAGConv(512, int(data.num_classes))
        self.n1 = GraphNorm(512)
        self.n2 = GraphNorm(512)

    def forward(self, data):
        x, edge_index, edge_attr = data.x.float(), data.edge_index, data.weight.float()
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = self.n1(x)
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = self.n2(x)
        x = self.conv3(x, edge_index, edge_attr)
        return x
    
class GL_TAGConv_3l_512h_w_k3_gnorm_gelu(torch.nn.Module):
    def __init__(self, data):
        super(GL_TAGConv_3l_512h_w_k3_gnorm_gelu, self).__init__()
        self.conv1 = TAGConv(int(data.num_features), 512)
        self.conv2 = TAGConv(512, 512)
        self.conv3 = TAGConv(512, int(data.num_classes))
        self.n1 = GraphNorm(512)
        self.n2 = GraphNorm(512)

    def forward(self, data):
        x, edge_index, edge_attr = data.x.float(), data.edge_index, data.weight.float()
        x = F.gelu(self.conv1(x, edge_index, edge_attr))
        x = self.n1(x)
        x = F.gelu(self.conv2(x, edge_index, edge_attr))
        x = self.n2(x)
        x = self.conv3(x, edge_index, edge_attr)
        return x
    

class GL_TAGConv_3l_512h_w_k3_gnorm_leaky_relu(torch.nn.Module):
    def __init__(self, data):
        super(GL_TAGConv_3l_512h_w_k3_gnorm_leaky_relu, self).__init__()
        self.conv1 = TAGConv(int(data.num_features), 512)
        self.conv2 = TAGConv(512, 512)
        self.conv3 = TAGConv(512, int(data.num_classes))
        self.n1 = GraphNorm(512)
        self.n2 = GraphNorm(512)

    def forward(self, data):
        x, edge_index, edge_attr = data.x.float(), data.edge_index, data.weight.float()
        x = F.leaky_relu(self.conv1(x, edge_index, edge_attr))
        x = self.n1(x)
        x = F.leaky_relu(self.conv2(x, edge_index, edge_attr))
        x = self.n2(x)
        x = self.conv3(x, edge_index, edge_attr)
        return x
    

class GL_TAGConv_3l_512h_w_k3_gnorm_relu(torch.nn.Module):
    def __init__(self, data):
        super(GL_TAGConv_3l_512h_w_k3_gnorm_relu, self).__init__()
        self.conv1 = TAGConv(int(data.num_features), 512)
        self.conv2 = TAGConv(512, 512)
        self.conv3 = TAGConv(512, int(data.num_classes))
        self.n1 = GraphNorm(512)
        self.n2 = GraphNorm(512)

    def forward(self, data):
        x, edge_index, edge_attr = data.x.float(), data.edge_index, data.weight.float()
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = self.n1(x)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = self.n2(x)
        x = self.conv3(x, edge_index, edge_attr)
        return x
    

class GL_TAGConv_3l_512h_nw_k3_gnorm(torch.nn.Module):
    def __init__(self, data):
        super(GL_TAGConv_3l_512h_nw_k3_gnorm, self).__init__()
        self.conv1 = TAGConv(int(data.num_features), 512)
        self.conv2 = TAGConv(512, 512)
        self.conv3 = TAGConv(512, int(data.num_classes))
        self.n1 = GraphNorm(512)
        self.n2 = GraphNorm(512)

    def forward(self, data):
        x, edge_index, edge_attr = data.x.float(), data.edge_index, data.weight.float()
        x = F.elu(self.conv1(x, edge_index))
        x = self.n1(x)
        x = F.elu(self.conv2(x, edge_index))
        x = self.n2(x)
        x = self.conv3(x, edge_index)
        return x
    
class GL_TAGConv_3l_512h_nw_k3_gnorm_gelu(torch.nn.Module):
    def __init__(self, data):
        super(GL_TAGConv_3l_512h_nw_k3_gnorm_gelu, self).__init__()
        self.conv1 = TAGConv(int(data.num_features), 512)
        self.conv2 = TAGConv(512, 512)
        self.conv3 = TAGConv(512, int(data.num_classes))
        self.n1 = GraphNorm(512)
        self.n2 = GraphNorm(512)

    def forward(self, data):
        x, edge_index, edge_attr = data.x.float(), data.edge_index, data.weight.float()
        x = F.gelu(self.conv1(x, edge_index))
        x = self.n1(x)
        x = F.gelu(self.conv2(x, edge_index))
        x = self.n2(x)
        x = self.conv3(x, edge_index)
        return x
    

class GL_TAGConv_3l_512h_nw_k3_gnorm_leaky_relu(torch.nn.Module):
    def __init__(self, data):
        super(GL_TAGConv_3l_512h_nw_k3_gnorm_leaky_relu, self).__init__()
        self.conv1 = TAGConv(int(data.num_features), 512)
        self.conv2 = TAGConv(512, 512)
        self.conv3 = TAGConv(512, int(data.num_classes))
        self.n1 = GraphNorm(512)
        self.n2 = GraphNorm(512)

    def forward(self, data):
        x, edge_index, edge_attr = data.x.float(), data.edge_index, data.weight.float()
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = self.n1(x)
        x = F.leaky_relu(self.conv2(x, edge_index))
        x = self.n2(x)
        x = self.conv3(x, edge_index)
        return x
    

class GL_TAGConv_3l_512h_nw_k3_gnorm_relu(torch.nn.Module):
    def __init__(self, data):
        super(GL_TAGConv_3l_512h_nw_k3_gnorm_relu, self).__init__()
        self.conv1 = TAGConv(int(data.num_features), 512)
        self.conv2 = TAGConv(512, 512)
        self.conv3 = TAGConv(512, int(data.num_classes))
        self.n1 = GraphNorm(512)
        self.n2 = GraphNorm(512)

    def forward(self, data):
        x, edge_index, edge_attr = data.x.float(), data.edge_index, data.weight.float()
        x = F.relu(self.conv1(x, edge_index))
        x = self.n1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.n2(x)
        x = self.conv3(x, edge_index)
        return x
    
    
class GL_TAGConv_3l_512h_wnw_k3(torch.nn.Module):
    def __init__(self, data):
        super(GL_TAGConv_3l_512h_wnw_k3, self).__init__()
        self.conv1w = TAGConv(data.num_features, 512)
        self.conv2w = TAGConv(512, 512)
        self.conv3w = TAGConv(512, 512)
        
        self.conv1nw = TAGConv(data.num_features, 512)
        self.conv2nw = TAGConv(512, 512)
        self.conv3nw = TAGConv(512, 512)
        
        self.classifier = Linear(1024, int(data.num_classes))

    def forward(self, data):
        x, edge_index, edge_attr = data.x.float(), data.edge_index, data.weight.float()
        xw = F.elu(self.conv1w(x, edge_index, edge_attr))
        xw = F.elu(self.conv2w(xw, edge_index, edge_attr))
        xw = F.elu(self.conv3w(xw, edge_index, edge_attr))
        
        xnw = F.elu(self.conv1nw(x, edge_index))
        xnw = F.elu(self.conv2nw(xnw, edge_index))
        xnw = F.elu(self.conv3nw(xnw, edge_index))
        
        x_all = torch.cat((xw, xnw), 1)
        x_all = self.classifier(x_all)
        
        return x_all
    
    
class GL_TAGConv_3l_512h_w_k5(torch.nn.Module):
    def __init__(self, data):
        super(GL_TAGConv_3l_512h_w_k5, self).__init__()
        self.conv1 = TAGConv(data.num_features, 512, K=5)
        self.conv2 = TAGConv(512, 512, K=5)
        self.conv3 = TAGConv(512, int(data.num_classes), K=5)

    def forward(self, data):
        x, edge_index, edge_attr = data.x.float(), data.edge_index, data.weight.float()
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = self.conv3(x, edge_index, edge_attr)
        return x
    
class GL_TAGConv_3l_512h_w_k3(torch.nn.Module):
    def __init__(self, data):
        super(GL_TAGConv_3l_512h_w_k3, self).__init__()
        self.conv1 = TAGConv(data.num_features, 512)
        self.conv2 = TAGConv(512, 512)
        self.conv3 = TAGConv(512, data.num_classes)

    def forward(self, data):
        x_init, edge_index, edge_attr = data.x.float(), data.edge_index, data.weight.float()
        x = F.elu(self.conv1(x_init, edge_index, edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = self.conv3(x, edge_index, edge_attr)
        return x
    
    
class GL_TAGConv_3l_1024h_w_k3_gnorm_meanaggr(torch.nn.Module):
    def __init__(self, data):
        super(GL_TAGConv_3l_1024h_w_k3_gnorm_meanaggr, self).__init__()
        self.conv1 = TAGConv(data.num_features, 1024, aggr='mean')
        self.n1 = GraphNorm(1024)
        self.conv2 = TAGConv(1024, 1024, aggr='mean')
        self.n2 = GraphNorm(1024)
        self.conv3 = TAGConv(1024, int(data.num_classes), aggr='mean')

    def forward(self, data):
        x, edge_index, edge_attr = data.x.float(), data.edge_index, data.weight.float()
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = self.n1(x)
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = self.n2(x)
        x = self.conv3(x, edge_index, edge_attr)
        return x
    
    
class GL_TAGConv_3l_512h_nw_k3(torch.nn.Module):
    def __init__(self, data):
        super(GL_TAGConv_3l_512h_nw_k3, self).__init__()
        self.conv1 = TAGConv(data.num_features, 512)
        self.conv2 = TAGConv(512, 512)
        self.conv3 = TAGConv(512, int(data.num_classes))

    def forward(self, data):
        x, edge_index, edge_attr = data.x.float(), data.edge_index, data.weight.float()
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x
    
class GL_TAGConv_3l_128h_nw_k3(torch.nn.Module):
    def __init__(self, data):
        super(GL_TAGConv_3l_128h_nw_k3, self).__init__()
        self.conv1 = TAGConv(data.num_features, 128)
        self.conv2 = TAGConv(128, 128)
        self.conv3 = TAGConv(128, int(data.num_classes))

    def forward(self, data):
        x, edge_index, edge_attr = data.x.float(), data.edge_index, data.weight.float()
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x
    
class GL_TAGConv_9l_128h_w_k3(torch.nn.Module):
    def __init__(self, data):
        super(GL_TAGConv_9l_128h_w_k3, self).__init__()
        self.conv1 = TAGConv(data.num_features, 128)
        self.conv2 = TAGConv(128, 128)
        self.conv3 = TAGConv(128, 128)
        self.conv4 = TAGConv(128, 128)
        self.conv5 = TAGConv(128, 128)
        self.conv6 = TAGConv(128, 128)
        self.conv7 = TAGConv(128, 128)
        self.conv8 = TAGConv(128, 128)
        self.conv9 = TAGConv(128, int(data.num_classes))

    def forward(self, data):
        x, edge_index, edge_attr = data.x.float(), data.edge_index, data.weight.float()
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = F.elu(self.conv3(x, edge_index, edge_attr))
        x = F.elu(self.conv4(x, edge_index, edge_attr))
        x = F.elu(self.conv5(x, edge_index, edge_attr))
        x = F.elu(self.conv6(x, edge_index, edge_attr))
        x = F.elu(self.conv7(x, edge_index, edge_attr))
        x = F.elu(self.conv8(x, edge_index, edge_attr))
        x = self.conv9(x, edge_index, edge_attr)
        return x
    

class GL_TAGConv_9l_512h_w_k3(torch.nn.Module):
    def __init__(self, data):
        super(GL_TAGConv_9l_512h_w_k3, self).__init__()
        self.conv1 = TAGConv(data.num_features, 512)
        self.conv2 = TAGConv(512, 512)
        self.conv3 = TAGConv(512, 512)
        self.conv4 = TAGConv(512, 512)
        self.conv5 = TAGConv(512, 512)
        self.conv6 = TAGConv(512, 512)
        self.conv7 = TAGConv(512, 512)
        self.conv8 = TAGConv(512, 512)
        self.conv9 = TAGConv(512, int(data.num_classes))

    def forward(self, data):
        x, edge_index, edge_attr = data.x.float(), data.edge_index, data.weight.float()
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = F.elu(self.conv3(x, edge_index, edge_attr))
        x = F.elu(self.conv4(x, edge_index, edge_attr))
        x = F.elu(self.conv5(x, edge_index, edge_attr))
        x = F.elu(self.conv6(x, edge_index, edge_attr))
        x = F.elu(self.conv7(x, edge_index, edge_attr))
        x = F.elu(self.conv8(x, edge_index, edge_attr))
        x = self.conv9(x, edge_index, edge_attr)
        return x

    
class GL_TAGConv_9l_512h_nw_k3(torch.nn.Module):
    def __init__(self, data):
        super(GL_TAGConv_9l_512h_nw_k3, self).__init__()
        self.conv1 = TAGConv(data.num_features, 512)
        self.conv2 = TAGConv(512, 512)
        self.conv3 = TAGConv(512, 512)
        self.conv4 = TAGConv(512, 512)
        self.conv5 = TAGConv(512, 512)
        self.conv6 = TAGConv(512, 512)
        self.conv7 = TAGConv(512, 512)
        self.conv8 = TAGConv(512, 512)
        self.conv9 = TAGConv(512, int(data.num_classes))

    def forward(self, data):
        x, edge_index, edge_attr = data.x.float(), data.edge_index, data.weight.float()
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = F.elu(self.conv3(x, edge_index))
        x = F.elu(self.conv4(x, edge_index))
        x = F.elu(self.conv5(x, edge_index))
        x = F.elu(self.conv6(x, edge_index))
        x = F.elu(self.conv7(x, edge_index))
        x = F.elu(self.conv8(x, edge_index))
        x = self.conv9(x, edge_index)
        return x
    
class GL_GCNConv_3l_32h_nw(torch.nn.Module):
    def __init__(self, data):
        super(GL_GCNConv_3l_32h_nw, self).__init__()
        self.conv1 = GCNConv(data.num_features, 32)
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, int(data.num_classes))

    def forward(self, data):
        x, edge_index, edge_attr = data.x.float(), data.edge_index, data.weight.float()
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x
    
class GL_GCNConv_3l_128h_w(torch.nn.Module):
    def __init__(self, data):
        super(GL_GCNConv_3l_128h_w, self).__init__()
        self.conv1 = GCNConv(data.num_features, 128)
        self.conv2 = GCNConv(128, 128)
        self.conv3 = GCNConv(128, int(data.num_classes))

    def forward(self, data):
        x, edge_index, edge_attr = data.x.float(), data.edge_index, data.weight.float()
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = self.conv3(x, edge_index, edge_attr)
        return x
    
class GL_GCNConv_3l_512h_w(torch.nn.Module):
    def __init__(self, data):
        super(GL_GCNConv_3l_512h_w, self).__init__()
        self.conv1 = GCNConv(data.num_features, 512)
        self.conv2 = GCNConv(512, 512)
        self.conv3 = GCNConv(512, int(data.num_classes))

    def forward(self, data):
        x, edge_index, edge_attr = data.x.float(), data.edge_index, data.weight.float()
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = self.conv3(x, edge_index, edge_attr)
        return x 
     
class GL_GCNConv_9l_128h_w(torch.nn.Module):
    def __init__(self, data):
        super(GL_GCNConv_9l_128h_w, self).__init__()
        self.conv1 = GCNConv(data.num_features, 128)
        self.conv2 = GCNConv(128, 128)
        self.conv3 = GCNConv(128, 128)
        self.conv4 = GCNConv(128, 128)
        self.conv5 = GCNConv(128, 128)
        self.conv6 = GCNConv(128, 128)
        self.conv7 = GCNConv(128, 128)
        self.conv8 = GCNConv(128, 128)
        self.conv9 = GCNConv(128, int(data.num_classes))

    def forward(self, data):
        x, edge_index, edge_attr = data.x.float(), data.edge_index, data.weight.float()
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = F.elu(self.conv3(x, edge_index, edge_attr))
        x = F.elu(self.conv4(x, edge_index, edge_attr))
        x = F.elu(self.conv5(x, edge_index, edge_attr))
        x = F.elu(self.conv6(x, edge_index, edge_attr))
        x = F.elu(self.conv7(x, edge_index, edge_attr))
        x = F.elu(self.conv8(x, edge_index, edge_attr))
        x = self.conv9(x, edge_index, edge_attr)
        return x 


class GL_GCNConv_9l_512h_w(torch.nn.Module):
    def __init__(self, data):
        super(GL_GCNConv_9l_512h_w, self).__init__()
        self.conv1 = GCNConv(data.num_features, 512)
        self.conv2 = GCNConv(512, 512)
        self.conv3 = GCNConv(512, 512)
        self.conv4 = GCNConv(512, 512)
        self.conv5 = GCNConv(512, 512)
        self.conv6 = GCNConv(512, 512)
        self.conv7 = GCNConv(512, 512)
        self.conv8 = GCNConv(512, 512)
        self.conv9 = GCNConv(512, int(data.num_classes))

    def forward(self, data):
        x, edge_index, edge_attr = data.x.float(), data.edge_index, data.weight.float()
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = F.elu(self.conv3(x, edge_index, edge_attr))
        x = F.elu(self.conv4(x, edge_index, edge_attr))
        x = F.elu(self.conv5(x, edge_index, edge_attr))
        x = F.elu(self.conv6(x, edge_index, edge_attr))
        x = F.elu(self.conv7(x, edge_index, edge_attr))
        x = F.elu(self.conv8(x, edge_index, edge_attr))
        x = self.conv9(x, edge_index, edge_attr)
        return x  
    
    

class GL_GCNConv_3l_128h_nw(torch.nn.Module):
    def __init__(self, data):
        super(GL_GCNConv_3l_128h_nw, self).__init__()
        self.conv1 = GCNConv(data.num_features, 128)
        self.conv2 = GCNConv(128, 128)
        self.conv3 = GCNConv(128, int(data.num_classes))

    def forward(self, data):
        x, edge_index, edge_attr = data.x.float(), data.edge_index, data.weight.float()
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x
    
class GL_GCNConv_9l_128h_nw(torch.nn.Module):
    def __init__(self, data):
        super(GL_GCNConv_9l_128h_nw, self).__init__()
        self.conv1 = GCNConv(data.num_features, 128)
        self.conv2 = GCNConv(128, 128)
        self.conv3 = GCNConv(128, 128)
        self.conv4 = GCNConv(128, 128)
        self.conv5 = GCNConv(128, 128)
        self.conv6 = GCNConv(128, 128)
        self.conv7 = GCNConv(128, 128)
        self.conv8 = GCNConv(128, 128)
        self.conv9 = GCNConv(128, int(data.num_classes))

    def forward(self, data):
        x, edge_index, edge_attr = data.x.float(), data.edge_index, data.weight.float()
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = F.elu(self.conv3(x, edge_index))
        x = F.elu(self.conv4(x, edge_index))
        x = F.elu(self.conv5(x, edge_index))
        x = F.elu(self.conv6(x, edge_index))
        x = F.elu(self.conv7(x, edge_index))
        x = F.elu(self.conv8(x, edge_index))
        x = self.conv9(x, edge_index)
        return x
    
class GL_GCNConv_3l_512h_nw(torch.nn.Module):
    def __init__(self, data):
        super(GL_GCNConv_3l_512h_nw, self).__init__()
        self.conv1 = GCNConv(data.num_features, 512)
        self.conv2 = GCNConv(512, 512)
        self.conv3 = GCNConv(512, int(data.num_classes))

    def forward(self, data):
        x, edge_index, edge_attr = data.x.float(), data.edge_index, data.weight.float()
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x
    
class GL_SSGConv_3l_128h_w_a05_k1(torch.nn.Module):
    def __init__(self, data):
        super(GL_SSGConv_3l_128h_w_a05_k1, self).__init__()
        self.conv1 = SSGConv(data.num_features, 128, alpha=0.5)
        self.conv2 = SSGConv(128, 128, alpha=0.5)
        self.conv3 = SSGConv(128, int(data.num_classes), alpha=0.5)

    def forward(self, d):
        x, edge_index, edge_attr = d.x.float(), d.edge_index, d.weight.float()
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = self.conv3(x, edge_index, edge_attr)
        return x
    
class GL_SSGConv_3l_128h_w_a09_k1(torch.nn.Module):
    def __init__(self, data):
        super(GL_SSGConv_3l_128h_w_a09_k1, self).__init__()
        self.conv1 = SSGConv(data.num_features, 128, alpha=0.9)
        self.conv2 = SSGConv(128, 128, alpha=0.9)
        self.conv3 = SSGConv(128, int(data.num_classes), alpha=0.9)

    def forward(self, data):
        x, edge_index, edge_attr = data.x.float(), data.edge_index, data.weight.float()
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = self.conv3(x, edge_index, edge_attr)
        return x
    
class GL_SAGEConv_3l_128h(torch.nn.Module):
    def __init__(self, data):
        super(GL_SAGEConv_3l_128h, self).__init__()
        self.conv1 = SAGEConv(data.num_features, 128)
        self.conv2 = SAGEConv(128, 128)
        self.conv3 = SAGEConv(128, int(data.num_classes))

    def forward(self, data):
        x, edge_index, edge_attr = data.x.float(), data.edge_index, data.weight.float()
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x
    
    
class GL_SAGEConv_3l_512h(torch.nn.Module):
    def __init__(self, data):
        super(GL_SAGEConv_3l_512h, self).__init__()
        self.conv1 = SAGEConv(data.num_features, 512)
        self.conv2 = SAGEConv(512, 512)
        self.conv3 = SAGEConv(512, int(data.num_classes))

    def forward(self, data):
        x, edge_index, edge_attr = data.x.float(), data.edge_index, data.weight.float()
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        
        return x
    
    
class GL_SAGEConv_9l_512h(torch.nn.Module):
    def __init__(self, data):
        super(GL_SAGEConv_9l_512h, self).__init__()
        self.conv1 = SAGEConv(data.num_features, 512)
        self.conv2 = SAGEConv(512, 512)
        self.conv3 = SAGEConv(512, 512)
        self.conv4 = SAGEConv(512, 512)
        self.conv5 = SAGEConv(512, 512)
        self.conv6 = SAGEConv(512, 512)
        self.conv7 = SAGEConv(512, 512)
        self.conv8 = SAGEConv(512, 512)
        self.conv9 = SAGEConv(512, int(data.num_classes))

    def forward(self, data):
        x, edge_index, edge_attr = data.x.float(), data.edge_index, data.weight.float()
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = F.elu(self.conv3(x, edge_index))
        x = F.elu(self.conv4(x, edge_index))
        x = F.elu(self.conv5(x, edge_index))
        x = F.elu(self.conv6(x, edge_index))
        x = F.elu(self.conv7(x, edge_index))
        x = F.elu(self.conv8(x, edge_index))
        x = self.conv9(x, edge_index)
        return x
    
    
class GL_SAGEConv_9l_128h(torch.nn.Module):
    def __init__(self, data):
        super(GL_SAGEConv_9l_128h, self).__init__()
        self.conv1 = SAGEConv(data.num_features, 128)
        self.conv2 = SAGEConv(128, 128)
        self.conv3 = SAGEConv(128, 128)
        self.conv4 = SAGEConv(128, 128)
        self.conv5 = SAGEConv(128, 128)
        self.conv6 = SAGEConv(128, 128)
        self.conv7 = SAGEConv(128, 128)
        self.conv8 = SAGEConv(128, 128)
        self.conv9 = SAGEConv(128, int(data.num_classes))

    def forward(self, data):
        x, edge_index, edge_attr = data.x.float(), data.edge_index, data.weight.float()
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = F.elu(self.conv3(x, edge_index))
        x = F.elu(self.conv4(x, edge_index))
        x = F.elu(self.conv5(x, edge_index))
        x = F.elu(self.conv6(x, edge_index))
        x = F.elu(self.conv7(x, edge_index))
        x = F.elu(self.conv8(x, edge_index))
        x = self.conv9(x, edge_index)
        return x
    

class GL_ChebConv_3l_128h_w_k3(torch.nn.Module):
    def __init__(self, data):
        super(GL_ChebConv_3l_128h_w_k3, self).__init__()
        self.conv1 = ChebConv(data.num_features, 128, K=3)
        self.conv2 = ChebConv(128, 128, K=3)
        self.conv3 = ChebConv(128, int(data.num_classes), K=3)

    def forward(self, data):
        x, edge_index, edge_attr = data.x.float(), data.edge_index, data.weight.float()
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = self.conv3(x, edge_index, edge_attr)
        return x    

####################################################################################################################################################
    
class GL_SequentialMultiplierArgs(nn.Sequential):
    def forward(self, *inputs):
        x, y = inputs
        for module in self._modules.values():
            x = module(x, y)
        return x
    
class GL_SequentialMultiplierArgsNext(nn.Sequential):
    def forward(self, *inputs):
        x, y, z = inputs
        for name, module in self._modules.items():
            if name[0] == 'c':
                x = module(x, y, z)
            else:
                x = module(x)
        return x
    
    
class GL_GINConv_3l_128h(torch.nn.Module): 
    def __init__(self, data, num_layers=3, hidden_dim=128):
        super(GL_GINConv_3l_128h, self).__init__()
        
        n_class = int(data.num_classes)
        init_dim = data.num_features
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        convs = [GINConv(
            nn=torch.nn.Sequential(
                torch.nn.Linear(init_dim, hidden_dim),
                # torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.ELU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                # torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.ELU()
            ),
            eps=0.0  # Add a small value to the denominator for numerical stability
        )]

        for i in range(self.num_layers-1):
            convs.append(
                GINConv(
                    nn=torch.nn.Sequential(
                        torch.nn.Linear(hidden_dim, hidden_dim),
                        # torch.nn.BatchNorm1d(hidden_dim),
                        torch.nn.ELU(),
                        torch.nn.Linear(hidden_dim, hidden_dim),
                        # torch.nn.BatchNorm1d(hidden_dim),
                        torch.nn.ELU()
                    ),
                    eps=0.0
                )
            )
        self.convs = GL_SequentialMultiplierArgs(*convs)
        self.fc = torch.nn.Linear(hidden_dim, n_class)

    def forward(self, data):
        x, edge_index, edge_weight = data.x.float(), data.edge_index, data.weight.float()
        h = self.convs(x, edge_index)
        h = self.fc(h)
        return h
    
    
    
class GL_GINConv_3l_512h(torch.nn.Module):
    def __init__(self, data, num_layers=3, hidden_dim=512):
        super(GL_GINConv_3l_512h, self).__init__()
        
        n_class = int(data.num_classes)
        init_dim = data.num_features
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        convs = [GINConv(
            nn=torch.nn.Sequential(
                torch.nn.Linear(init_dim, hidden_dim),
                # torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.ELU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                # torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.ELU()
            ),
            eps=0.0  # Add a small value to the denominator for numerical stability
        )]

        for i in range(self.num_layers-1):
            convs.append(
                GINConv(
                    nn=torch.nn.Sequential(
                        torch.nn.Linear(hidden_dim, hidden_dim),
                        # torch.nn.BatchNorm1d(hidden_dim),
                        torch.nn.ELU(),
                        torch.nn.Linear(hidden_dim, hidden_dim),
                        # torch.nn.BatchNorm1d(hidden_dim),
                        torch.nn.ELU()
                    ),
                    eps=0.0
                )
            )
        self.convs = GL_SequentialMultiplierArgs(*convs)
        self.fc = torch.nn.Linear(hidden_dim, n_class)

    def forward(self, data):
        x, edge_index, edge_weight = data.x.float(), data.edge_index, data.weight.float()
        h = self.convs(x, edge_index)
        h = self.fc(h)
        return h
    
    
    
class GL_GINConv_9l_128h(torch.nn.Module):
    def __init__(self, data, num_layers=9, hidden_dim=128):
        super(GL_GINConv_9l_128h, self).__init__()
        
        n_class = int(data.num_classes)
        init_dim = data.num_features
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        convs = [GINConv(
            nn=torch.nn.Sequential(
                torch.nn.Linear(init_dim, hidden_dim),
                # torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.ELU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                # torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.ELU()
            ),
            eps=0.0  # Add a small value to the denominator for numerical stability
        )]

        for i in range(self.num_layers-1):
            convs.append(
                GINConv(
                    nn=torch.nn.Sequential(
                        torch.nn.Linear(hidden_dim, hidden_dim),
                        # torch.nn.BatchNorm1d(hidden_dim),
                        torch.nn.ELU(),
                        torch.nn.Linear(hidden_dim, hidden_dim),
                        # torch.nn.BatchNorm1d(hidden_dim),
                        torch.nn.ELU()
                    ),
                    eps=0.0
                )
            )
        self.convs = GL_SequentialMultiplierArgs(*convs)
        self.fc = torch.nn.Linear(hidden_dim, n_class)

    def forward(self, data):
        x, edge_index, edge_weight = data.x.float(), data.edge_index, data.weight.float()
        h = self.convs(x, edge_index)
        h = self.fc(h)
        return h
    
    
    
class GL_GINConv_9l_512h(torch.nn.Module):
    def __init__(self, data, num_layers=9, hidden_dim=512):
        super(GL_GINConv_9l_512h, self).__init__()
        
        n_class = int(data.num_classes)
        init_dim = data.num_features
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        convs = [GINConv(
            nn=torch.nn.Sequential(
                torch.nn.Linear(init_dim, hidden_dim),
                # torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.ELU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                # torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.ELU()
            ),
            eps=0.0  # Add a small value to the denominator for numerical stability
        )]

        for i in range(self.num_layers-1):
            convs.append(
                GINConv(
                    nn=torch.nn.Sequential(
                        torch.nn.Linear(hidden_dim, hidden_dim),
                        # torch.nn.BatchNorm1d(hidden_dim),
                        torch.nn.ELU(),
                        torch.nn.Linear(hidden_dim, hidden_dim),
                        # torch.nn.BatchNorm1d(hidden_dim),
                        torch.nn.ELU()
                    ),
                    eps=0.0
                )
            )
        self.convs = GL_SequentialMultiplierArgs(*convs)
        self.fc = torch.nn.Linear(hidden_dim, n_class)

    def forward(self, data):
        x, edge_index, edge_weight = data.x.float(), data.edge_index, data.weight.float()
        h = self.convs(x, edge_index)
        h = self.fc(h)
        return h    



class GL_GATConv_3l_128h(torch.nn.Module):
    def __init__(self, data, num_layers=3, hidden_channels=128, n_heads=2, dropout=0.2):
        super().__init__()
        self.dp = dropout

        n_class = int(data.num_classes)
        init_dim = data.num_features

        # Create the first GATv2Conv layer
        self.layers = [
            GL_SequentialMultiplierArgsNext(OrderedDict([
                (f'conv0', GATv2Conv(in_channels=init_dim,
                                     out_channels=hidden_channels,
                                     heads=n_heads,
                                     edge_dim=1,
                                     aggr="add",
                                     concat=True,
                                     share_weights=False,
                                     add_self_loops=False)),
                # (f'norm0', BatchNorm1d(hidden_channels * n_heads))
            ]))
        ]

        # Create intermediate GATv2Conv layers
        for i in range(1, num_layers):
            self.layers.append(
                GL_SequentialMultiplierArgsNext(OrderedDict([
                    (f'conv{i}', GATv2Conv(in_channels=hidden_channels * n_heads,
                                           out_channels=hidden_channels,
                                           heads=n_heads,
                                           edge_dim=1,
                                           aggr="add",
                                           concat=True,
                                           share_weights=False,
                                           add_self_loops=True)),
                    # (f'norm{i}', BatchNorm1d(hidden_channels * n_heads))
                ]))
            )

        # Convert the list of layers to a torch.nn.Sequential
        self.layers = GL_SequentialMultiplierArgsNext(*self.layers)

        # Output layer
        self.fc = Linear(hidden_channels * n_heads, n_class)

    def forward(self, data):
        h, edge_index, edge_weight = data.x.float(), data.edge_index, data.weight.float()

        for layer in self.layers:
            h = layer(h, edge_index, edge_weight)
            h = F.elu(h)
            h = F.dropout(h, p=self.dp, training=self.training)

        h = self.fc(h)
        return h
    
    

    
    
class GL_GATConv_3l_512h(torch.nn.Module):
    def __init__(self, data, num_layers=3, hidden_channels=512, n_heads=2, dropout=0.2):
        super().__init__()
        self.dp = dropout

        n_class = int(data.num_classes)
        init_dim = data.num_features

        # Create the first GATv2Conv layer
        self.layers = [
            GL_SequentialMultiplierArgsNext(OrderedDict([
                (f'conv0', GATv2Conv(in_channels=init_dim,
                                     out_channels=hidden_channels,
                                     heads=n_heads,
                                     edge_dim=1,
                                     aggr="add",
                                     concat=True,
                                     share_weights=False,
                                     add_self_loops=False)),
                # (f'norm0', BatchNorm1d(hidden_channels * n_heads))
            ]))
        ]

        # Create intermediate GATv2Conv layers
        for i in range(1, num_layers):
            self.layers.append(
                GL_SequentialMultiplierArgsNext(OrderedDict([
                    (f'conv{i}', GATv2Conv(in_channels=hidden_channels * n_heads,
                                           out_channels=hidden_channels,
                                           heads=n_heads,
                                           edge_dim=1,
                                           aggr="add",
                                           concat=True,
                                           share_weights=False,
                                           add_self_loops=True)),
                    # (f'norm{i}', BatchNorm1d(hidden_channels * n_heads))
                ]))
            )

        # Convert the list of layers to a torch.nn.Sequential
        self.layers = GL_SequentialMultiplierArgsNext(*self.layers)

        # Output layer
        self.fc = Linear(hidden_channels * n_heads, n_class)

    def forward(self, data):
        h, edge_index, edge_weight = data.x.float(), data.edge_index, data.weight.float()

        for layer in self.layers:
            h = layer(h, edge_index, edge_weight)
            h = F.elu(h)
            h = F.dropout(h, p=self.dp, training=self.training)

        h = self.fc(h)
        return h
    
    
    
    

    
class GL_GATConv_9l_128h(torch.nn.Module):
    def __init__(self, data, num_layers=9, hidden_channels=128, n_heads=2, dropout=0.2):
        super().__init__()
        self.dp = dropout

        n_class = int(data.num_classes)
        init_dim = data.num_features

        # Create the first GATv2Conv layer
        self.layers = [
            GL_SequentialMultiplierArgsNext(OrderedDict([
                (f'conv0', GATv2Conv(in_channels=init_dim,
                                     out_channels=hidden_channels,
                                     heads=n_heads,
                                     edge_dim=1,
                                     aggr="add",
                                     concat=True,
                                     share_weights=False,
                                     add_self_loops=False)),
                # (f'norm0', BatchNorm1d(hidden_channels * n_heads))
            ]))
        ]

        # Create intermediate GATv2Conv layers
        for i in range(1, num_layers):
            self.layers.append(
                GL_SequentialMultiplierArgsNext(OrderedDict([
                    (f'conv{i}', GATv2Conv(in_channels=hidden_channels * n_heads,
                                           out_channels=hidden_channels,
                                           heads=n_heads,
                                           edge_dim=1,
                                           aggr="add",
                                           concat=True,
                                           share_weights=False,
                                           add_self_loops=True)),
                    # (f'norm{i}', BatchNorm1d(hidden_channels * n_heads))
                ]))
            )

        # Convert the list of layers to a torch.nn.Sequential
        self.layers = GL_SequentialMultiplierArgsNext(*self.layers)

        # Output layer
        self.fc = Linear(hidden_channels * n_heads, n_class)

    def forward(self, data):
        h, edge_index, edge_weight = data.x.float(), data.edge_index, data.weight.float()

        for layer in self.layers:
            h = layer(h, edge_index, edge_weight)
            h = F.elu(h)
            h = F.dropout(h, p=self.dp, training=self.training)

        h = self.fc(h)
        return h
    
    
    
class GL_GATConv_9l_512h(torch.nn.Module):
    def __init__(self, data, num_layers=9, hidden_channels=512, n_heads=2, dropout=0.2):
        super().__init__()
        self.dp = dropout

        n_class = int(data.num_classes)
        init_dim = data.num_features

        # Create the first GATv2Conv layer
        self.layers = [
            GL_SequentialMultiplierArgsNext(OrderedDict([
                (f'conv0', GATv2Conv(in_channels=init_dim,
                                     out_channels=hidden_channels,
                                     heads=n_heads,
                                     edge_dim=1,
                                     aggr="add",
                                     concat=True,
                                     share_weights=False,
                                     add_self_loops=False)),
                # (f'norm0', BatchNorm1d(hidden_channels * n_heads))
            ]))
        ]

        # Create intermediate GATv2Conv layers
        for i in range(1, num_layers):
            self.layers.append(
                GL_SequentialMultiplierArgsNext(OrderedDict([
                    (f'conv{i}', GATv2Conv(in_channels=hidden_channels * n_heads,
                                           out_channels=hidden_channels,
                                           heads=n_heads,
                                           edge_dim=1,
                                           aggr="add",
                                           concat=True,
                                           share_weights=False,
                                           add_self_loops=True)),
                    # (f'norm{i}', BatchNorm1d(hidden_channels * n_heads))
                ]))
            )

        # Convert the list of layers to a torch.nn.Sequential
        self.layers = GL_SequentialMultiplierArgsNext(*self.layers)

        # Output layer
        self.fc = Linear(hidden_channels * n_heads, n_class)

    def forward(self, data):
        h, edge_index, edge_weight = data.x.float(), data.edge_index, data.weight.float()

        for layer in self.layers:
            h = layer(h, edge_index, edge_weight)
            h = F.elu(h)
            h = F.dropout(h, p=self.dp, training=self.training)

        h = self.fc(h)
        return h