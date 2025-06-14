#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys
from torch.nn import init
#import feather
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
from tqdm import tqdm
torch.set_default_tensor_type(torch.FloatTensor)

import wandb
import time

#sys.path.append("../pytorch_study")
#import d2lzh_pytorch as d2l
print(torch.__version__)

# 定义训练的设备
dev = input('>>> Select cuda device (0~3):')
device = torch.device(f"cuda:{dev}") if torch.cuda.is_available() else torch.device("cpu")

# 这个设置可以把每个单元格所有结果显示出来
from IPython.core.interactiveshell import InteractiveShell 
InteractiveShell.ast_node_interactivity = "all" 




# In[2]:

 # # MLP
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(NeuralNet, self).__init__()
        self.model = nn.Sequential(
            #第1层
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(p=dropout),
            #第2层
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(p=dropout),
            #第3层
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, output_size),
        )
    def forward(self, x):
        x = self.model(x)
        return x


 # ResNet MLP

# ResNet MLP
class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(channels, channels)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        identity = x
        out = self.fc1(x)       #fully connected layer 1
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out) #dropout也为hyperparameter
        out = self.fc2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ResNetMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout, num_layers=3):
        super(ResNetMLP, self).__init__()
        self.num_layers = num_layers
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.layer1 = ResidualBlock(hidden_size, dropout)
        self.layer2 = ResidualBlock(hidden_size, dropout)
        self.layer3 = ResidualBlock(hidden_size, dropout)
        if num_layers > 3:
            self.layer4 = ResidualBlock(hidden_size, dropout)
        if num_layers > 4:
            self.layer5 = ResidualBlock(hidden_size, dropout)
        if num_layers > 5:
            self.layer6 = ResidualBlock(hidden_size, dropout)
        if num_layers > 6:
            self.layer7 = ResidualBlock(hidden_size, dropout)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.input_layer(x)   #nn.Linear(input_size, hidden_size)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)     #dropout也为hyperparameter
        # Residual blocks
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        if self.num_layers > 3:
            out = self.layer4(out)
        if self.num_layers > 4:
            out = self.layer5(out)
        if self.num_layers > 5:
            out = self.layer6(out)
        if self.num_layers > 6:
            out = self.layer7(out)
        out = self.output_layer(out)
        return out


# DenseNet MLP
class DenseNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(DenseNet, self).__init__()
        self.layers = nn.Sequential(
            # input projection
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            # hidden dense layers (5 dense layers, matching ResNet MLP complexity)
            #1
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            #2
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            #3
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            #4
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            #5
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            # output projection
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)

# EfficientNet MLP

class EfficientNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(EfficientNet, self).__init__()
        # initial projection
        self.fc_in = nn.Linear(input_size, hidden_size)
        self.bn_in = nn.BatchNorm1d(hidden_size)
        self.act = nn.SiLU()  # efficient activation
        self.dropout = nn.Dropout(p=dropout)
        # MBConv-like blocks with expansion factor 4
        self.block1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.SiLU(),
            nn.Dropout(p=dropout)
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.SiLU(),
            nn.Dropout(p=dropout)
        )
        self.block3 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.SiLU(),
            nn.Dropout(p=dropout)
        )
        # output projection
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc_in(x)
        x = self.bn_in(x)
        x = self.act(x)
        x = self.dropout(x)
        # MBConv-like residual blocks
        x = x + self.block1(x)
        x = x + self.block2(x)
        x = x + self.block3(x)
        out = self.fc_out(x)
        return out

# Transformer MLP with Self‑Attention
class SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_size, dropout, nhead=16):
        super(SelfAttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(p=dropout),
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # Self‑attention + residual
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        # Feed‑forward network + residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class TransformerMLP(nn.Module):
    """
    Transformer‑style MLP that matches the parameter scale of ResNetMLP
    when hidden_size = 4096*2.  It treats the projected feature vector
    as a single token, applies self‑attention blocks, and projects to
    the output dimension.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 dropout,
                 num_layers=3,
                 nhead=16):
        super(TransformerMLP, self).__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.blocks = nn.ModuleList(
            [SelfAttentionBlock(hidden_size, dropout, nhead)
             for _ in range(num_layers)]
        )
        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch_size, input_size)
        x = self.input_proj(x)        # (batch_size, hidden_size)
        x = x.unsqueeze(1)            # (batch_size, seq_len=1, hidden_size)
        for blk in self.blocks:
            x = blk(x)
        x = x.squeeze(1)              # (batch_size, hidden_size)
        x = self.dropout(x)
        out = self.output_proj(x)     # (batch_size, output_size)
        return out

# 损失函数
loss_fn = torch.nn.MSELoss().to(device)

def log_rmse(net, features, labels):
    with torch.no_grad():
        clipped_preds = torch.max(net(features), torch.tensor(1.0))
        rmse = torch.sqrt(2 * loss_fn(clipped_preds.log(),labels.log()).mean())
    return rmse.item()
#log_rmse=log_rmse.to(device)

# Pearson correlation coefficient
def pearson_corr(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return the Pearson correlation coefficient between prediction and target tensors."""
    preds_flat = preds.flatten()
    targets_flat = targets.flatten()

    # Center the variables
    vx = preds_flat - preds_flat.mean()
    vy = targets_flat - targets_flat.mean()

    # Standard Pearson correlation components
    numerator = (vx * vy).sum()
    denominator = torch.sqrt((vx ** 2).sum()) * torch.sqrt((vy ** 2).sum()) + 1e-8  # prevent divide‑by‑zero

    return numerator / denominator


# # 训练网络

# In[3]:


X_train_P=pd.read_csv("./data/sicmirkan/TCGA_L1000_mRNA_train_zscore.csv", index_col=0)
X_test_P=pd.read_csv("./data/sicmirkan/TCGA_L1000_mRNA_test_zscore.csv",index_col=0) 
y_train_P=pd.read_csv("./data/sicmirkan/TCGA_miRNA_train_zscore.csv", index_col=0)
y_test_P=pd.read_csv("./data/sicmirkan/TCGA_miRNA_test_zscore.csv", index_col=0) 

# filter步骤-调准顺序
f=open('./data/sicmirkan/top414.txt','r') #414miRNA的list
lines = f.readlines()
check_list = []
for line in lines:
    line = line.split('\n')[0]
    check_list.append(line)

# check_list = check_list[1:]   ### Xiaoxuan师姐之前多写了这行，去掉看看
print('After filter num col ——', len(y_test_P.columns.intersection(check_list)))    # 只剩414
#filter出414的表达矩阵
y_train_P = y_train_P[y_train_P.columns.intersection(check_list)]
y_test_P = y_test_P[y_test_P.columns.intersection(check_list)]

#dataloading
X_train_Tensor = torch.tensor(X_train_P.values,dtype=torch.float)
X_test_Tensor  = torch.tensor(X_test_P.values,dtype=torch.float)
y_train_Tensor = torch.tensor(y_train_P.values,dtype=torch.float)
y_test_Tensor  = torch.tensor(y_test_P.values,dtype=torch.float)#.to(device)

##print("训练数据集的长度为：{}".format(len(X_train_Tensor)))
print("测试数据集的长度为：{}".format(len(X_test_Tensor)))
print("测试数据集的长度为：{}".format(len(X_train_Tensor)))
print("测试数据集的长度为：{}".format(len(y_test_Tensor)))
print("测试数据集的长度为：{}".format(len(y_train_Tensor)))


# In[5]:
print(torch.__version__)
print(torch.cuda.is_available())
import datetime as dt










### ====================================== 设置训练参数 ===============================
start = dt.datetime.now()
batch_size = 64     #batch_size = 64
hidden_size = 4096*2          #512 for Transformer
input_size, output_size = X_train_Tensor.shape[1] ,y_train_Tensor.shape[1] #参数
learning_rate = 0.4    #0.00068
eta_min_param = 0.0    #0.9 #Cos Anneal for Transformer
eta_min = learning_rate * eta_min_param
weight_decay = 2e-4
epoch = 3000  # total number of training epochs
dropout = 0.35
net_name = "ResNetMLP" #选择模型
num_layers = 3 # number of layers in the model
nheads = 8 # number of attention heads in the Transformer model

if net_name == "MLP":
    net = NeuralNet(input_size, hidden_size, output_size, dropout)
elif net_name == "ResNetMLP":
    net = ResNetMLP(input_size, hidden_size, output_size, dropout, num_layers)
elif net_name == "DenseNet":
    net = DenseNet(input_size, hidden_size, output_size, dropout)
elif net_name == "EfficientNet":
    net = EfficientNet(input_size, hidden_size, output_size, dropout)
elif net_name == "TransformerMLP":
    net = TransformerMLP(input_size, hidden_size, output_size, dropout, num_layers, nheads)
else:
    raise ValueError("Invalid model name. Choose from 'MLP', 'ResNetMLP', 'DenseNet', or 'EfficientNet'.")
net = net.to(device)
print(f"Model Architecture:\n{net}")

# Optimizer and scheduler
optimizer_name = "SGD"  # Choose from 'SGD', 'Adam', 'AdamW', 'Adamax', 'RMSprop', 'Adagrad'
if optimizer_name == "SGD":
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
elif optimizer_name == "Adam":
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
elif optimizer_name == "AdamW":
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
elif optimizer_name == "Adamax":
    optimizer = torch.optim.Adamax(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
elif optimizer_name == "RMSprop":
    optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
elif optimizer_name == "Adagrad":
    optimizer = torch.optim.Adagrad(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
else:
    raise ValueError("Invalid optimizer name. Choose from 'SGD', 'Adam', 'AdamW', 'Adamax', 'RMSprop', or 'Adagrad'.")

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=eta_min)

num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"Number of parameters: {num_params}")

print(f"Hyperparameters: \n"
      f"  - Batch size: {batch_size}\n"
      f"  - Hidden size: {hidden_size}\n"
      f"  - Learning rate: {learning_rate}\n"
      f"  - Eta min param: {eta_min_param}\n"
      f"  - Weight decay: {weight_decay}\n"
      f"  - Epochs: {epoch}\n"
      f"  - Dropout: {dropout}\n"
      f"  - Model: {net_name}\n"
      f"  - Optimizer: {optimizer_name}\n"
      f"  - Num layers: {num_layers}\n"
      f"  - Num heads: {nheads}\n"
)

wandb.init(project="sicmirkan_414", 
           entity="xincao02-chinese-university-of-hong-kong-shenzhen", 
           name=time.strftime('%m%d_%H:%M:%S'), config={
    "learning_rate": learning_rate,
    "epochs": epoch,
    "hidden_size": hidden_size,
    "weight_decay": weight_decay,
    "dropout": dropout,
})
### ====================================== 设置训练参数 ===============================











for params in net.parameters():
    params=init.normal_(params, mean=0, std=0.01)
    #init.constant_(net.bias, val=0)

# He Kaiming initialization
# for params in net.parameters():
#     if params.dim() > 1:  # Ensure it's a weight tensor, not a bias
#         init.kaiming_normal_(params, mode='fan_in', nonlinearity='relu')

    
# 利用 DataLoader 来加载数据集
train_dataset = torch.utils.data.TensorDataset(X_train_Tensor,y_train_Tensor)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True)

test_dataset = torch.utils.data.TensorDataset(X_test_Tensor,y_test_Tensor)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=True)
print(len(test_dataloader))

# 设置训练网络的一些参数
total_train_step = 0 # 记录训练的次数
total_test_step = 0 # 记录测试的次数
# 训练的轮数
# epoch = 800
train_loss_list = []
test_loss_list = []
train_pcc_list = []
test_pcc_list = []

best_train_loss = float('inf')
best_test_loss = float('inf')
best_train_epoch = -1
best_test_epoch = -1
best_train_pcc = -float('inf')
best_test_pcc = -float('inf')
best_train_pcc_epoch = -1
best_test_pcc_epoch = -1

for epoch_idx in tqdm(range(epoch), desc="Training Epochs"):
    # Training
    net.train()
    epoch_train_loss = 0.0
    train_preds = []
    train_targets = []
    for mRNAs, miRNAs in train_dataloader:
        mRNAs, miRNAs = mRNAs.to(device), miRNAs.to(device)
        outputs = net(mRNAs)
        loss = loss_fn(outputs, miRNAs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
        train_preds.append(outputs.detach().cpu())
        train_targets.append(miRNAs.detach().cpu())
    avg_train_loss = epoch_train_loss / len(train_dataloader)
    train_preds_tensor = torch.cat(train_preds, dim=0)
    train_targets_tensor = torch.cat(train_targets, dim=0)
    train_pcc = pearson_corr(train_preds_tensor, train_targets_tensor).item()
    train_pcc_list.append(train_pcc)
    if train_pcc > best_train_pcc:
        best_train_pcc = train_pcc
        best_train_pcc_epoch = epoch_idx + 1
    if avg_train_loss < best_train_loss:
        best_train_loss = avg_train_loss
        best_train_epoch = epoch_idx + 1

    # Validation
    net.eval()
    epoch_test_loss = 0.0
    test_preds = []
    test_targets = []
    with torch.no_grad():
        for mRNAs, miRNAs in test_dataloader:
            mRNAs, miRNAs = mRNAs.to(device), miRNAs.to(device)
            outputs = net(mRNAs)
            loss = loss_fn(outputs, miRNAs)
            epoch_test_loss += loss.item()
            test_preds.append(outputs.detach().cpu())
            test_targets.append(miRNAs.detach().cpu())
    avg_test_loss = epoch_test_loss / len(test_dataloader)
    test_preds_tensor = torch.cat(test_preds, dim=0)
    test_targets_tensor = torch.cat(test_targets, dim=0)
    test_pcc = pearson_corr(test_preds_tensor, test_targets_tensor).item()
    test_pcc_list.append(test_pcc)
    if test_pcc > best_test_pcc:
        best_test_pcc = test_pcc
        best_test_pcc_epoch = epoch_idx + 1
    if avg_test_loss < best_test_loss:
        best_test_loss = avg_test_loss
        best_test_epoch = epoch_idx + 1

    # Get current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    if epoch_idx % 100 == 0:
        tqdm.write(
            f"{epoch_idx+1}: LR={current_lr:.5f}, "
            f"TrLoss={avg_train_loss:.4f}, TrPCC={train_pcc:.4f}, "
            f"TeLoss={avg_test_loss:.4f}, TePCC={test_pcc:.4f}, "
            f"Best TrLoss={best_train_loss:.4f} (at {best_train_epoch}), "
            f"Best TeLoss={best_test_loss:.4f} (at {best_test_epoch}), "
            f"Best TrPCC={best_train_pcc:.4f} (at {best_train_pcc_epoch}), "
            f"Best TePCC={best_test_pcc:.4f} (at {best_test_pcc_epoch})"
        )
    train_loss_list.append(avg_train_loss)
    test_loss_list.append(avg_test_loss)
    scheduler.step()
    # WandB logging
    wandb.log({
        "Loss/Train": avg_train_loss,
        "Loss/Test": avg_test_loss,
        "Min Loss/Train": best_train_loss,
        "Min Loss/Test": best_test_loss,
        "LearningRate": current_lr,
        "Parameters/HiddenSize": hidden_size,
        "Parameters/Dropout": dropout,
        "Parameters/CosAnneal_EtaMin": eta_min,
        "Parameters/WeightDecay": weight_decay,
        "Parameters/Epochs": epoch,
        "Parameters/BatchSize": batch_size,
        "Parameters/Model": net,
        "Parameters/Optimizer": optimizer_name,
        "Parameters/NumLayers": num_layers,
        "Parameters/NumHeads": nheads,
        "Parameters/NumParams": num_params,
        "PCC/Train": train_pcc,
        "PCC/Test": test_pcc,
        "BestPCC/Train": best_train_pcc,
        "BestPCC/Test": best_test_pcc,
    }, step=epoch_idx + 1)
    # tqdm.write(f"Best TrLoss={best_train_loss:.4f}(at epoch {best_train_epoch}), Best TeLoss={best_test_loss:.4f}(at epoch {best_test_epoch})")

print(f"Best TrLoss={best_train_loss:.4f} (at {best_train_epoch}), Best TeLoss={best_test_loss:.4f} (at {best_test_epoch})")
# Save final model
torch.save(net, f"./data/sicmirkan/414/414_hkm_cuda{dev}.pth")
print(f'>>>>> Model saved to:\t./data/sicmirkan/414/414_hkm_cuda{dev}.pth !!!')
wandb.finish()


# In[65]:


import pickle
with open("train_loss_list500.pkl", "wb") as f:
    pickle.dump(train_loss_list, f)
with open("test_loss_list500.pkl", "wb") as f:
    pickle.dump(test_loss_list, f)


# # In[64]:

# #记录训练画图
# import plotly.graph_objects as go

# # 示例数据
# x = list(range(len(train_loss_list)))
# y1 = train_loss_list
# y2 = test_loss_list

# # 创建图表对象
# fig = go.Figure()

# # 添加第一条线（绿色实线）
# fig.add_trace(go.Scatter(
#     x=x, y=y1, mode='lines+markers',
#     name='Train Loss',
#     marker=dict(size=2),
#     line=dict(color='green', dash='solid', width=1)
# ))

# # 添加第二条线（紫色虚线）
# fig.add_trace(go.Scatter(
#     x=x, y=y2, mode='lines+markers',
#     name='Test Loss',
#     marker=dict(size=2),
#     line=dict(color='purple', dash='dash', width=1)
# ))

# # 更新布局
# fig.update_layout(
#     title="Loss Curve",
#     xaxis_title="Epochs",
#     yaxis_title="Loss",
#     yaxis=dict(range=[0.09, 0.30]),
#     legend=dict(title="Legend"),
# )

# # 显示图表
# fig.write_image(f'./data/sicmirkan/414/result_hkm_plot.pdf')


class Animator:
    def __init__(self, xlabel=None, ylabel=None,legend=None, xlim=None, ylim=None, xscale = 'linear',
                 yscale='linear',fmts=('-','m--','g-.','r:'), nrows=1,ncols=1, figsize=(3.5,2.5)):
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize = figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes,]
        self.config_axes = lambda: d2l.set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)

        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a,b) in enumerate(zip(x,y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x,y, fmt in zip(self.X,self.Y,self.fmts):
            self.axes[0].plot(x,y,fmt)
        self.config_axes()

        plt.draw()
        plt.pause(0.001)
        display.display(self.fig)
        display.clear_output(wait=True)

    def show(self):
        display.display(self.fig)

# #结果读取
# net=torch.load("/data/home/grp-huangxd/caixiaoxuan/106/Results/414/414_targeted_512_DNN_miRNA199.pth",map_location=torch.device('cpu')) #保存模型
# net.eval()
# #net = net.to("cpu")
# filtered_y_train_out = pd.DataFrame(net(X_train_Tensor).cpu().detach().numpy())  #y_train_Tensor
# filtered_y_test_out  = pd.DataFrame(net(X_test_Tensor).detach().numpy())   #y_test_Tensor
# #X_train_Tensor
# filtered_y_train_out.columns=filtered_y_train_P.columns
# filtered_y_train_out.index=filtered_y_train_P.index
# filtered_y_train_out

# filtered_y_test_out.columns=filtered_y_test_P.columns
# filtered_y_test_out.index=filtered_y_test_P.index
# filtered_y_test_out
