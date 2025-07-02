import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import TAGConv, GraphNorm
from torch.amp import autocast

import time

# 1. Set default dtype to bf16 (affects new floating-point tensors)
torch.set_default_dtype(torch.float16)

# 2. Graph parameters
num_nodes = 230_888
num_node_features = 4
num_edges = 13_605_820
num_classes = 4

# 3. Generate random data in bf16 (on GPU)
x = torch.randn((num_nodes, num_node_features), device='cuda')
edge_index = torch.randint(0, num_nodes, (2, num_edges), device='cuda')
mask = torch.randint(0, 2, (num_nodes,), dtype=torch.bool, device='cuda')
weight = torch.rand((num_edges,), device='cuda')
y = torch.randint(0, num_classes, (num_nodes,), device='cuda')

# 4. Construct the Data object
data = Data(
    x=x,
    edge_index=edge_index,
    mask=mask,
    weight=weight,
    num_classes=num_classes,
    y=y
)

print(data)
print(f"x dtype: {data.x.dtype}, weight dtype: {data.weight.dtype}")

# 5. Define the model (no changes)
class GL_TAGConv_3l_512h_w_k3_gnorm(torch.nn.Module):
    def __init__(self, data):
        super(GL_TAGConv_3l_512h_w_k3_gnorm, self).__init__()
        self.conv1 = TAGConv(int(data.num_features), 512)
        self.conv2 = TAGConv(512, 512)
        self.conv3 = TAGConv(512, int(data.num_classes))
        self.n1 = GraphNorm(512)
        self.n2 = GraphNorm(512)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.weight
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = self.n1(x)
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = self.n2(x)
        x = self.conv3(x, edge_index, edge_attr)
        return x

# 6. Move model to GPU and run one validation iteration with autocast
model = GL_TAGConv_3l_512h_w_k3_gnorm(data).cuda().half()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.1)
# model.eval()
model.train()

# with torch.no_grad():
    # with autocast(device_type='cuda', dtype=torch.bfloat16):
out = model(data)
print(f"Output shape: {out.shape}, dtype: {out.dtype}")

# 7. Pause for 30 seconds so you can check GPU RAM usage
print("Sleeping for 30 seconds. Check GPU memory now...")
time.sleep(30)
print("Done.")
