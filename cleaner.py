import torch
import numpy as np
import faiss
import scipy.sparse as sp
import torch.optim as optim
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
from ignite.utils import manual_seed
manual_seed(123)


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GCNcleaner(nn.Module):
    def __init__(self, input_dim, hidden_dim = 16, dropout = 0.5):
        super(GCNcleaner, self).__init__()
        self.gc_input = GraphConvolution(input_dim, hidden_dim)
        self.gc_output = GraphConvolution(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc_input(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc_output(x, adj)
        return x


def run_clean(embedding_cnn, labels, loss_sample, ratio, device, gpu_id):
    label_set = np.unique(labels)
    weights = torch.ones(embedding_cnn.shape[0], device=device).detach()
    noisy_idx_all = []
    clean_idx_all = []
    middle_idx_all = []
    clean_ratio = 0.05
    noisy_ratio = 0.05
    class_weight = torch.zeros(label_set.shape[0], device=device)
    # 按类找噪音数据
    for label in label_set:
        idx = np.where(labels == label)[0]
        class_weight[label] = idx.shape[0]
        loss_sorted = torch.argsort(loss_sample[idx], dim=0).cpu()
        noisy_idx_all.extend(idx[loss_sorted[-int(noisy_ratio * idx.shape[0]):]])
        middle_idx_all.extend(idx[loss_sorted[-int(ratio * idx.shape[0]):-int(noisy_ratio * idx.shape[0])]])
        clean_idx_all.extend(idx[loss_sorted[:int(clean_ratio * idx.shape[0])]])

    noisy_prob_all = torch.zeros((embedding_cnn.shape[0], len(label_set)), device=device)
    # 按类构图训练
    for label in label_set:
        idx = np.where(labels == label)[0]
        loss_sorted = torch.argsort(loss_sample[idx], dim=0).cpu()
        train_idx = idx[loss_sorted[:int(clean_ratio * idx.shape[0])]]
        noisy_idx = idx[loss_sorted[int(-noisy_ratio * idx.shape[0]):]]
        noisy_other_idx = [i for i in noisy_idx_all if i not in noisy_idx]

        clean_feature = embedding_cnn[train_idx, :]
        noisy_feature = embedding_cnn[noisy_idx, :]
        choose_feature = embedding_cnn[middle_idx_all, :]
        noisy_other_feature = embedding_cnn[noisy_other_idx, :]

        cur_features = torch.cat((clean_feature, noisy_feature, choose_feature, noisy_other_feature)).to(device)
        pos_idx = np.arange(clean_feature.shape[0])
        neg_idx = np.arange(noisy_feature.shape[0]) + clean_feature.shape[0]
        choose_idx_1 = np.arange(choose_feature.shape[0] + noisy_other_feature.shape[0]) + clean_feature.shape[0] + noisy_feature.shape[0]
        choose_idx = np.arange(choose_feature.shape[0]) + clean_feature.shape[0] + noisy_feature.shape[0]
        other_idx = np.arange(noisy_other_feature.shape[0]) + clean_feature.shape[0] + noisy_feature.shape[0] + choose_feature.shape[0]
        # graph creation
        affinitymat = features2affinitymax(cur_features.data.cpu().numpy(), k = 25, gpu_id=gpu_id)
        affinitymat = affinitymat.minimum(affinitymat.T)
        affinitymat = graph_normalize(affinitymat + sp.eye(affinitymat.shape[0]))
        affinitymat = sparse_mx_to_torch_sparse_tensor(affinitymat).to(device)

        model = train_gcn(cur_features, affinitymat, pos_idx, neg_idx, choose_idx_1, gcn_lambda=1, device=device)
        model.eval()
        output = torch.sigmoid(model(cur_features, affinitymat))
        cur_weights = output[neg_idx]
        choose_weights = output[choose_idx]
        other_weights = output[other_idx]
        # weights[train_idx] += torch.squeeze(train_weight)
        noisy_prob_all[noisy_idx, label] = torch.squeeze(cur_weights)
        noisy_prob_all[middle_idx_all, label] = torch.squeeze(choose_weights)
        noisy_prob_all[noisy_other_idx, label] = torch.squeeze(other_weights)
    return weights, noisy_prob_all, noisy_idx_all, middle_idx_all


def features2affinitymax(features, k = 50, gpu_id = 0):
    # 排除掉自己，自己肯定是最高的
    knn, sim = knn_faiss(features, features, k = k + 1, gpu_id = gpu_id)
    aff = knn2affinitymat(knn[:, 1:], sim[:, 1:])  # skip self-matches
    return aff


def knn_faiss(X, Q, k, gpu_id = 0):
    D = X.shape[1]

    # CPU search if gpu_id = -1. GPU search otherwise.
    if gpu_id == -1:
        # 内积索引
        index = faiss.IndexFlatIP(D)
    else:
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = gpu_id
        index = faiss.GpuIndexFlatIP(res, D, flat_config)
    index.add(X)
    # Q:query k搜索个数
    # sim:距离每个query最近的k个数据的距离  knn:距离每个query最近的k个数据的id
    sim, knn = index.search(Q, min(k, X.shape[0]))
    index.reset()
    del index

    return knn, sim


def knn2affinitymat(knn, sim):
    N, k = knn.shape[0], knn.shape[1]
    # 平铺，沿Y轴扩大k倍，x轴一倍 扩成x行 （N*k）
    row_idx_rep = np.tile(np.arange(N), (k, 1)).T
    # 按列降维 一维数组
    sim_flatten = sim.flatten('F')
    row_flatten = row_idx_rep.flatten('F')
    knn_flatten = knn.flatten('F')

    # # Handle the cases where FAISS returns -1 as knn indices - FIX
    # invalid_idx = np.where(knn_flatten<0)[0]
    # if len(invalid_idx):
    #     sim_flatten = np.delete(sim_flatten, invalid_idx)
    #     row_flatten = np.delete(row_flatten, invalid_idx)
    #     knn_flatten = np.delete(knn_flatten, invalid_idx)

    W = sp.csr_matrix((sim_flatten, (row_flatten, knn_flatten)), shape=(N, N))
    return W


def graph_normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def adjust_learning_rate(optimizer, epoch, lr):
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def ramp_up(epoch, max_epochs, max_val, mult):
    if epoch == 0:
        return 0.
    elif epoch >= max_epochs:
        return max_val
    return max_val * np.exp(mult * (1. - float(epoch) / max_epochs) ** 2)


def weight_schedule(epoch, max_epochs, max_val, mult, n_labeled, n_samples):
    max_val = max_val * (float(n_labeled) / n_samples)
    return ramp_up(epoch, max_epochs, max_val, mult)


def train_gcn(features, affinitymat, pos_idx, neg_idx, unlabel_idx, gcn_lambda, device):
    lr = 0.1
    gcniter = 100
    eps = 1e-6
    ntrain = features.shape[0]
    Z = torch.zeros(ntrain, 1).float().to(device)
    z = torch.zeros(ntrain, 1).float().to(device)
    outputs = torch.zeros(ntrain, 1).float().to(device)

    model = GCNcleaner(input_dim=features.shape[1])
    model = model.to(device)
    model.train()
    alpha = 0.6
    params_set = [dict(params=model.parameters())]
    optimizer = optim.Adam(params_set, lr=lr, weight_decay=5e-4)
    for epoch in range(gcniter):
        adjust_learning_rate(optimizer, epoch, lr)
        w = weight_schedule(epoch, 20, 10, -1, len(pos_idx) + len(neg_idx), ntrain)
        # w = torch.autograd.Variable(torch.FloatTensor([w]).to(device), requires_grad=False)
        optimizer.zero_grad()
        output = torch.sigmoid(model(features, affinitymat))
        zcomp = z[unlabel_idx].detach()
        mse_loss = torch.sum((output[unlabel_idx] - torch.sigmoid(zcomp)) ** 2) / ntrain
        loss_train = -(output.squeeze()[pos_idx] + eps).log().mean()  # loss for clean
        loss_train += -gcn_lambda * (1 - output[neg_idx] + eps).log().mean()   # loss for noisy, treated as negative
        loss_train += w * mse_loss
        loss_train.backward()
        optimizer.step()

        Z = alpha * Z + (1. - alpha) * outputs
        z = Z * (1. / (1. - alpha ** (epoch + 1)))

    return model
