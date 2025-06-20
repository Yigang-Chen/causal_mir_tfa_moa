#!/usr/bin/env python
# coding: utf-8

print(1)

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from kan import KAN
from tqdm import tqdm

# 定义训练的设备
device = torch.device("cuda")

# 创建KAN网络模型
class KANModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, grid_size=5, spline_order=3):
        super(KANModel, self).__init__()
        self.model = KAN(
            [input_size, hidden_size, hidden_size, output_size],
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            grid_eps=0.02
        )
    
    def forward(self, x, update_grid=False):
        return self.model(x, update_grid=update_grid)
    
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return self.model.regularization_loss(regularize_activation, regularize_entropy)

# 损失函数
def get_loss_fn():
    return torch.nn.MSELoss()

# 计算log_rmse指标
def log_rmse(net, features, labels):
    with torch.no_grad():
        clipped_preds = torch.max(net(features), torch.tensor(1.0))
        rmse = torch.sqrt(2 * torch.nn.MSELoss()(clipped_preds.log(), labels.log()).mean())
    return rmse.item()

# 训练函数
def train_model(X_train, y_train, X_test, y_test, hidden_size=1024, learning_rate=0.4, 
                batch_size=64, epochs=500, grid_size=5, spline_order=3, log_dir=None,
                milestones=None, gamma=0.1):  # 新增参数
    
    # 转换为tensor
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float).to(device)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float).to(device)
    
    input_size, output_size = X_train_tensor.shape[1], y_train_tensor.shape[1]
    
    # 创建KAN模型
    net = KANModel(input_size, hidden_size, output_size, grid_size, spline_order).to(device)
    
    # 优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    
    # 学习率调度器
    if milestones is not None:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    else:
        scheduler = None
    
    # 损失函数
    loss_fn = get_loss_fn()
    
    # 数据加载器
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    # 记录训练进度
    total_train_step = 0
    total_test_step = 0
    
    # 添加tensorboard
    writer = None
    if log_dir:
        writer = SummaryWriter(log_dir)
    
    train_loss_list = []
    test_loss_list = []
    
    pbar = tqdm(range(epochs), desc="")
    for epoch in pbar:
        # 训练步骤开始
        net.train()
        for data in train_dataloader:
            mRNAs, miRNAs = data
            outputs = net(mRNAs.to(device))
            loss = loss_fn(outputs, miRNAs)
            
            # 添加正则化损失
            reg_loss = net.regularization_loss(regularize_activation=0.01, regularize_entropy=0.01)
            loss = loss + reg_loss
            
            # 优化器优化模型
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_step += 1
            if total_train_step % 100 == 0:
                train_loss_list.append(loss.item())
                train_loss = loss.item()
                if writer:
                    writer.add_scalar("train_loss", loss.item(), total_train_step)
        
        # 测试步骤开始
        net.eval()
        total_test_loss = 0
        with torch.no_grad():
            for data in test_dataloader:
                mRNAs, miRNAs = data
                outputs = net(mRNAs.to(device))
                loss = loss_fn(outputs, miRNAs)
                total_test_loss += loss.item()
        
        if writer:
            writer.add_scalar("test_loss", total_test_loss, total_test_step)
        test_loss_list.append(total_test_loss)
        total_test_step += 1

        # 记录学习率
        current_lr = optimizer.param_groups[0]['lr']
        if writer:
            writer.add_scalar('learning_rate', current_lr, epoch)
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step()

        # get best
        best_train = min(train_loss_list)
        train_index = train_loss_list.index(best_train)
        train_best = f'{best_train:.2f}({int(train_index)+1})'
        best_test = min(test_loss_list)
        test_index = test_loss_list.index(best_test)
        test_best = f'{best_test:.2f}/({int(test_index)+1})'

        # 更新进度条
        pbar.set_postfix(
            train_loss=train_loss,
            test_loss=total_test_loss,
            lr=current_lr,  # 显示当前学习率
            Tr=train_best,
            Tt=test_best,
        )
        
        # 保存最后一轮的模型
        if epoch == (epochs-1):
            model_path = f"kan_model_miRNA{epoch}.pth"
            torch.save(net, model_path)
            print(f"模型已保存到 {model_path}")
    
    if writer:
        writer.close()
    
    return net, train_loss_list, test_loss_list

# 结果可视化函数
def plot_loss_curves(train_loss_list, test_loss_list, output_file=None):
    try:
        import plotly.graph_objects as go
        
        # 创建图表对象
        fig = go.Figure()
        
        # 添加训练损失线（绿色实线）
        fig.add_trace(go.Scatter(x=list(range(len(train_loss_list))), 
                                y=train_loss_list, 
                                mode='lines+markers',
                                name='train_loss',
                                line=dict(color='green', dash='solid')))
        
        # 添加测试损失线（紫色虚线）
        fig.add_trace(go.Scatter(x=list(range(len(test_loss_list))), 
                                y=test_loss_list, 
                                mode='lines+markers',
                                name='valid_loss',
                                line=dict(color='purple', dash='dash')))
        
        # 更新布局
        fig.update_layout(
            title="KAN Model Loss Curve",
            xaxis_title="Epochs",
            yaxis_title="Loss",
            legend=dict(title="Legend"),
        )
        
        # 保存图表
        if output_file:
            fig.write_image(output_file)
            print(f"损失曲线已保存到 {output_file}")
        
        return fig
    except ImportError:
        print("需要安装plotly库来可视化损失曲线")
        return None

# 使用示例
if __name__ == "__main__":
    # 假设已有数据

    print('Device chosen:', device)

    X_train_P=pd.read_csv("./data/sicmirkan/TCGA_L1000_mRNA_train_zscore.csv", index_col=0)
    X_test_P=pd.read_csv("./data/sicmirkan/TCGA_L1000_mRNA_test_zscore.csv",index_col=0) 
    y_train_P=pd.read_csv("./data/sicmirkan/TCGA_miRNA_train_zscore.csv", index_col=0)
    y_test_P=pd.read_csv("./data/sicmirkan/TCGA_miRNA_test_zscore.csv", index_col=0) 

    f=open('./data/sicmirkan/top414.txt','r') #414miRNA的list
    lines = f.readlines()
    check_list = []
    for line in lines:
        line = line.split('\n')[0]
        check_list.append(line)

    #filter出414的表达矩阵
    filtered_y_train_P = y_train_P[y_train_P.columns.intersection(check_list)]
    filtered_y_test_P = y_test_P[y_test_P.columns.intersection(check_list)]
    
    # 假设我们已经有数据，那么训练模型可以这样调用:
    model, train_losses, test_losses = train_model(
        X_train_P, filtered_y_train_P, X_test_P, filtered_y_test_P,
        hidden_size=1024*4,
        learning_rate=0.4,
        batch_size=64,
        epochs=500,
        grid_size=5,
        spline_order=3,
        log_dir="logs/kan_model",
        milestones=[120, 140, 160, 180],  # 在第250和400轮调整学习率
        gamma=0.2  # 每次调整学习率乘以0.1
    )
    
    # 可视化训练结果
    # plot_loss_curves(train_losses, test_losses, "kan_model_loss.pdf")
    
    print("请导入你的数据并使用train_model函数来训练KAN模型")
