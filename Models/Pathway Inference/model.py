### 命令行：
"""
python model.py \
  --genes dataset/PROGENy_gene.csv \
  --tf dataset/PROGENy_tf.csv \
  --mirna dataset/PROGENy_mirna.csv \
  --labels dataset/PROGENy_labels.csv \
  --fusion concat \
  --select 1 0 0  \
  --hidden 1024 \
  --blocks 3 \
  --batch 32 \
  --epochs 50 \
  --lr 1e-3 \
  --seed 14
"""


import argparse
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# --------- Dataset Definition ------------------------------------------------
class MultimodalDataset(Dataset):
    def __init__(self, x_paths, y_path):
        X0 = pd.read_csv(x_paths['genes'], index_col=0).values.astype(np.float32)
        TF = pd.read_csv(x_paths['tf'],    index_col=0).T.values.astype(np.float32)
        MIR = pd.read_csv(x_paths['mirna'], index_col=0).values.astype(np.float32)

        y_df = pd.read_csv(y_path, index_col=0)
        le = LabelEncoder()
        y = le.fit_transform(y_df.values.ravel())

        self.X0 = torch.from_numpy(X0)
        self.TF = torch.from_numpy(TF)
        self.MIR = torch.from_numpy(MIR)
        self.y = torch.from_numpy(y).long()
        self.num_classes = len(le.classes_)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return (self.X0[idx], self.TF[idx], self.MIR[idx], self.y[idx])

# --------- Fusion Module -----------------------------------------------------
class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, embeddings):
        # embeddings: list of (batch, embed_dim)
        Q = embeddings[0].unsqueeze(0)                # (1, batch, dim)
        KV = torch.stack(embeddings[1:], dim=0)       # (seq-1, batch, dim)
        attn_out, _ = self.attn(Q, KV, KV)            # (1, batch, dim)
        return attn_out.squeeze(0)                    # (batch, dim)

# --------- ResNet Blocks & Models ---------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += identity
        return F.relu(out)

class ResNetMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, n_blocks=3):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.bn0 = nn.BatchNorm1d(hidden_dim)
        self.layers = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(n_blocks)])
        self.classifier = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        out = F.relu(self.bn0(self.input_fc(x)))
        out = self.layers(out)
        return self.classifier(out)

class AttentionResNet(nn.Module):
    def __init__(self, feat_dims, hidden_dim, num_classes, n_blocks, select):
        super().__init__()
        self.select = select
        # build projection for each selected modality
        dims = [feat_dims[i] for i in range(3) if select[i]]
        self.proj = nn.ModuleList([nn.Linear(d, hidden_dim) for d in dims])
        self.fusion = CrossAttentionFusion(hidden_dim)
        self.layers = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(n_blocks)])
        self.classifier = nn.Linear(hidden_dim, num_classes)
    def forward(self, x0, tf, mir):
        inputs = [x0, tf, mir]
        selected = [inp for flag, inp in zip(self.select, inputs) if flag]
        # project
        emb = [F.relu(p(inp)) for p, inp in zip(self.proj, selected)]
        # fuse
        if len(emb) == 1:
            fused = emb[0]
        else:
            fused = self.fusion(emb)
        out = self.layers(fused)
        return self.classifier(out)

# --------- Training & Evaluation ---------------------------------------------
def train_epoch(model, loader, criterion, optimizer, device, fusion, select):
    model.train()
    ys, ps = [], []
    total_loss = 0
    for batch in loader:
        x0, tf, mir, y = [b.to(device) for b in batch]
        if fusion == 'concat':
            parts = [x for x,f in zip([x0, tf, mir], select) if f]
            inp = torch.cat(parts, dim=1)
            logits = model(inp)
        else:
            logits = model(x0, tf, mir)
        loss = criterion(logits, y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        ys.append(y.cpu().numpy()); ps.append(preds.cpu().numpy())
    ys = np.concatenate(ys); ps = np.concatenate(ps)
    acc = 100.0 * (ys == ps).mean()
    f1  = 100.0 * f1_score(ys, ps, average='macro')
    return total_loss/len(ys), acc, f1


def eval_epoch(model, loader, criterion, device, fusion, select):
    model.eval()
    ys, ps = [], []
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            x0, tf, mir, y = [b.to(device) for b in batch]
            if fusion == 'concat':
                parts = [x for x,f in zip([x0, tf, mir], select) if f]
                inp = torch.cat(parts, dim=1)
                logits = model(inp)
            else:
                logits = model(x0, tf, mir)
            total_loss += criterion(logits, y).item() * y.size(0)
            preds = logits.argmax(dim=1)
            ys.append(y.cpu().numpy()); ps.append(preds.cpu().numpy())
    ys = np.concatenate(ys); ps = np.concatenate(ps)
    # print(ys.shape, ps.shape)
    acc = 100.0 * (ys == ps).mean()
    f1  = 100.0 * f1_score(ys, ps, average='macro')
    return total_loss/len(ys), acc, f1

# --------- Main ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser("Multimodal ResNet with Selection & Attention")
    parser.add_argument('--genes',  type=str,   required=True)
    parser.add_argument('--tf',     type=str,   required=True)
    parser.add_argument('--mirna',  type=str,   required=True)
    parser.add_argument('--labels', type=str,   required=True)
    parser.add_argument('--fusion', choices=['concat','attention'], default='concat')
    parser.add_argument('--select', nargs=3, type=int, default=[1,1,1],
                        help="Select modalities: gene TF miRNA (1 or 0)")
    parser.add_argument('--hidden', type=int,   default=1024)
    parser.add_argument('--blocks', type=int,   default=3)
    parser.add_argument('--batch',  type=int,   default=32)
    parser.add_argument('--epochs', type=int,   default=50)
    parser.add_argument('--lr',     type=float, default=1e-3)
    parser.add_argument('--seed',   type=int,   default=42)
    args = parser.parse_args()

    # validate selection
    select = args.select
    if any(f not in (0,1) for f in select):
        raise ValueError("--select flags must be 0 or 1")
    if sum(select) == 0:
        raise ValueError("Must select at least one modality")
    if sum(select) == 1 and args.fusion == 'attention':
        raise ValueError("Attention fusion requires at least two modalities")

    torch.manual_seed(args.seed)
    device = torch.device('cpu')

    ds = MultimodalDataset(
        x_paths={'genes':args.genes,'tf':args.tf,'mirna':args.mirna},
        y_path=args.labels
    )
    N = len(ds)
    feat_dims = (ds.X0.shape[1], ds.TF.shape[1], ds.MIR.shape[1])

    # split 70/15/15
    idx = np.arange(N)
    train_i, temp_i, y_tr, y_tmp = train_test_split(
        idx, ds.y.numpy(), test_size=0.3, random_state=args.seed, stratify=ds.y.numpy())
    val_i, test_i = train_test_split(
        temp_i, test_size=0.5, random_state=args.seed, stratify=ds.y.numpy()[temp_i])

    train_loader = DataLoader(torch.utils.data.Subset(ds,train_i), batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(torch.utils.data.Subset(ds,val_i),   batch_size=args.batch)
    test_loader  = DataLoader(torch.utils.data.Subset(ds,test_i),  batch_size=args.batch)

    # model init
    if args.fusion == 'concat':
        in_dim = sum(d for d,f in zip(feat_dims, select) if f)
        model = ResNetMLP(in_dim, args.hidden, ds.num_classes, args.blocks)
    else:
        model = AttentionResNet(feat_dims, args.hidden, ds.num_classes, args.blocks, select)
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    best_val = 0.0
    best_train_loss = float('inf')
    best_train_acc = 0.0
    best_train_f1 = 0.0
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_val_f1 = 0.0

    for e in range(1, args.epochs+1):
        tr_l, tr_acc, tr_f1 = train_epoch(model, train_loader, crit, opt, device, args.fusion, select)
        va_l, va_acc, va_f1 = eval_epoch(model, val_loader, crit, device, args.fusion, select)
        if tr_l < best_train_loss:
            best_train_loss = tr_l
            best_train_acc = tr_acc
            best_train_f1 = tr_f1
        if va_l < best_val_loss:
            best_val_loss = va_l
            best_val_acc = va_acc
            best_val_f1 = va_f1
        print('bestvalacc', best_val_acc)
        print(f"Epoch {e}/{args.epochs} | Train loss={tr_l:.4f}, acc={tr_acc:.1f}%, f1={tr_f1:.1f}% | "
              f"Val   loss={va_l:.4f}, acc={va_acc:.1f}%, f1={va_f1:.1f}%")
        if va_acc > best_val:
            best_val = va_acc
            torch.save(model.state_dict(), 'best_model.pth')

    print(f"\nBest Train -> loss= {best_train_loss:.4f}, acc= {best_train_acc:.1f} %, f1= {best_train_f1:.1f} %")
    print(f"Best Val   -> loss= {best_val_loss:.4f}, acc= {best_val_acc:.1f} %, f1= {best_val_f1:.1f} %")

    te_l, te_acc, te_f1 = eval_epoch(model, test_loader, crit, device, args.fusion, select)
    print(f"\nTest loss= {te_l:.4f}, acc= {te_acc:.1f} %, f1= {te_f1:.1f} %")

if __name__ == '__main__':
    main()