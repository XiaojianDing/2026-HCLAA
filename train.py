import os
import torch
import numpy as np
import random
import argparse
from network import Network
from metric import valid
from loss import ContrastiveLoss
from dataloader import load_data
import torch.nn.functional as F

# 选择数据集
Dataname = 'Caltech-5V'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--pre_epochs", default=200)
parser.add_argument("--con_epochs", default=50)
parser.add_argument("--feature_dim", default=64)
parser.add_argument("--high_feature_dim", default=20)
parser.add_argument("--temperature", default=1)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 加载数据
if args.dataset == "Caltech-5V":
    args.con_epochs = 100
    seed = 1000000

dataset, dims, view, data_size, class_num = load_data(args.dataset)
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
)


# 计算视图的注意力权重
def compute_view_value(rs, H, view):
    """
    使用注意力机制计算视图权重
    :param rs: 视图特征列表 (list of tensor) -> 每个视图的形状: (N, d_v)
    :param H: 全局特征 (N, d)
    :param view: 视图数 (int)
    :return: 归一化视图权重 (tensor) -> (view,)
    """
    device = H.device  # 获取H所在的设备 (CPU or CUDA)
    N, d = H.shape  # 样本数, 全局特征维度
    d_v = rs[0].shape[1]  # 视图特征维度

    # 将映射参数放到同一个设备上
    W_q = torch.nn.Linear(d, d_v, bias=False).to(device)  # 查询变换
    W_k = torch.nn.Linear(d_v, d_v, bias=False).to(device)  # 键变换

    # 计算全局特征的 Query (N, d_v)
    Q = W_q(H)

    # 计算每个视图的 Key (view, N, d_v)，确保在同一设备上
    K = torch.stack([W_k(rs[v].to(device)) for v in range(view)])

    # 计算注意力分数 (view, N, N)
    attention_scores = torch.matmul(K, Q.T) / torch.sqrt(torch.tensor(d_v, dtype=torch.float32, device=device))

    # 计算每个视图的权重 (view,)
    w = torch.mean(attention_scores, dim=(1, 2))  # 取均值，得到单个数值

    # 归一化权重
    w = F.softmax(w, dim=0)

    return w

# 预训练
def pretrain(epoch):
    tot_loss = 0.
    criterion = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs, _, _, _,_= model(xs)
        loss_list = [criterion(xs[v], xrs[v]) for v in range(view)]
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print(f'Epoch {epoch}, Loss: {tot_loss / len(data_loader):.6f}')


# 对比训练
def contrastive_train(epoch):
    tot_loss = 0.
    mse = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs, zs, rs, H,new_rs = model(xs)


        loss_list = []

        with torch.no_grad():
            w = compute_view_value(new_rs, H, len(new_rs))

        for v in range(len(new_rs)):  # Iterate over fused views
            # Self-weighted contrastive learning loss
            loss_list.append(contrastiveloss(H, new_rs[v], w[v].item()))


        for v in range(view):
            # 计算视图权重
            with torch.no_grad():
                w = compute_view_value(rs, H, view)
            loss_list.append(contrastiveloss(H, rs[v], w[v])) # 对比损失
            loss_list.append(mse(xs[v], xrs[v]))  # 重构损失
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print(f'Epoch {epoch}, Loss: {tot_loss / len(data_loader):.6f}')
    return tot_loss / len(data_loader)


# 训练与评估
accs, nmis, purs, losses = [], [], [], []
epoch_accuracies, epoch_nmis, epoch_losses = [], [], []

if not os.path.exists('./models'):
    os.makedirs('./models')

T = 1  # 训练轮次
for i in range(T):
    print(f"ROUND: {i + 1}")
    setup_seed(seed)
    model = Network(view, dims, args.feature_dim, args.high_feature_dim, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    contrastiveloss = ContrastiveLoss(args.batch_size, args.temperature, device).to(device)

    best_acc, best_nmi, best_pur = 0, 0, 0
    epoch = 1

    # 预训练
    while epoch <= args.pre_epochs:
        pretrain(epoch)
        epoch += 1

    # 对比训练
    while epoch <= args.pre_epochs + args.con_epochs:
        loss = contrastive_train(epoch)
        acc, nmi, pur = valid(model, device, dataset, view, data_size, class_num, eval_h=False, epoch=epoch)
        epoch_accuracies.append(acc)
        epoch_nmis.append(nmi)
        epoch_losses.append(loss)

        if acc > best_acc:
            best_acc, best_nmi, best_pur = acc, nmi, pur
            torch.save(model.state_dict(), f'./models/{args.dataset}.pth')
        epoch += 1

    accs.append(best_acc)
    nmis.append(best_nmi)
    purs.append(best_pur)
    print(f'The best clustering performance: ACC = {best_acc:.4f}, NMI = {best_nmi:.4f}, PUR = {best_pur:.4f}')

