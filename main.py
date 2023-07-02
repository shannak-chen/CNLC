from __future__ import print_function

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from model.ConvNet import CNN
import Generate_noisy_labels as GN
from ignite.utils import manual_seed
from weight_loss import CrossEntropyLoss as CE
from sklearn.mixture import GaussianMixture
import cleaner as cleaner
from nihcc import NIHCC
from sklearn.metrics import roc_auc_score, f1_score, recall_score,  confusion_matrix
manual_seed(123)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
parser = argparse.ArgumentParser(description='PyTorch CoadjutantHashing Training')
parser.add_argument('-d', '--data', metavar='DIR', help='path to dataset (default: ./data)', default='./dataset')
parser.add_argument('--gpu-id', default='2', type=int,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('-ch', '--checkpoint', metavar='DIR', help='path to checkpoint (default: ./checkpoint)', default='./checkpoint')
parser.add_argument('-ds', '--dataset', metavar='FILE', help='dataset to use [nihcc, isic, breakhis] (default: cifar10)', default='nihcc')
parser.add_argument('-e', '--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-st', '--step_size', default=10, type=int, metavar='N', help='step size to decay the learning rate (default: 10)')
parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('-w', '--workers', default=4, type=int, metavar='N', help='number of workers for data processing (default: 4)')
parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float, metavar='LR', help='initial learning rate (default: 0.001)')
parser.add_argument('-sz', '--image_size', default=32, type=int, metavar='N', help='Size of input to use (default: 32)')
parser.add_argument('-c', '--channels', default=3, type=int, metavar='N', help='Number of channels of the input, which could be different for sentences (default: 3)')
parser.add_argument('-nb', '--num_class', default=2, type=int, metavar='N', help='Number of binary bits to train (default: 2)')
parser.add_argument('-gamma', '--gamma', default=0.1, type=float, metavar='F', help='initialize parameter lambda (default: 0.1)')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('-r', default=0.4, help='noisy rate. Default=0.4')

args = parser.parse_args()


def rampup(global_step, rampup_length=200):
    if global_step < rampup_length:
        global_step = np.float(global_step)
        rampup_length = np.float(rampup_length)
        phase = 1.0 - np.maximum(0.0, global_step) / rampup_length
    else:
        phase = 0.0
    return np.exp(-5.0 * phase * phase)


# 调整cnn学习率
def adjust_learning_rate(optimizer, epoch, alpha_plan, beta1_plan):
    for param_group in optimizer.param_groups:
        param_group['lr'] = alpha_plan[epoch]
        param_group['betas'] = (beta1_plan[epoch], 0.999)


# cnn训练
def train_cnn(epoch, model, optimizer, dataloader, weight_cnn, change, p_label, train_dataset, trainlabel, stop):
    model.train()
    count = 0
    correct = 0
    weight_cnn = weight_cnn.detach()
    trainlabel = torch.tensor(trainLabel).to(args.device)
    for _, data in enumerate(dataloader):
        inputs, _, index = data[0].to(args.device), data[1].to(args.device), data[2]
        _, logits = model(inputs)
        _, predict = torch.max(logits, dim=1)
        count += inputs.size(0)
        optimizer.zero_grad()
        train_label = trainlabel[index]
        correct += (train_label == predict).sum().item()
        optimizer.zero_grad()
        b_comp = p_label[index, :].detach()
        if change is True:
            loss = (F.cross_entropy(logits, train_label, reduction='none') * weight_cnn[index]).mean()
        else:
            um = rampup(epoch - stop, rampup_length=args.epochs - stop)
            loss = (F.cross_entropy(logits, train_label, reduction='none') * weight_cnn[index]).mean() + 20 * um * weight_criterion(logits, args.num_class, b_comp) / b_comp.sum()
        p_label[index, :] = F.softmax(logits, dim=1)
        loss.backward()
        optimizer.step()
    acc = float(correct / count * 100)
    print(f"epoch:{epoch} cnn_train_acc: {acc: .2f}")
    return acc, p_label


def get_loss(model, dataloader, trainlabel):
    model.eval()
    trainlabel = torch.tensor(trainLabel).to(args.device)
    loss_sample = torch.zeros(len(trainlabel), device=args.device).detach()

    with torch.no_grad():
        for _, data in enumerate(dataloader):
            inputs, _, index = data[0].to(args.device), data[1].to(args.device), data[2]
            _, logits = model(inputs)
            train_label = trainlabel[index]
            loss = F.cross_entropy(logits, train_label, reduction='none')
            loss_sample[index] = loss
    input_loss = ((loss_sample - loss_sample.min()) / (loss_sample.max() - loss_sample.min())).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(input_loss.cpu())
    prob = gmm.predict_proba(input_loss.cpu())
    prob = prob[:, gmm.means_.argmin()]
    return prob, loss_sample


# 用cnn提取特征
def embedding(model, dataloader, embedding_dim):
    model.eval()
    embedding_cnn = torch.zeros((len(dataloader.dataset), embedding_dim), device = args.device)
    with torch.no_grad():
        for _, data in enumerate(dataloader):
            inputs, _, index = data[0].to(args.device), data[1].to(args.device), data[2]
            embedding_cnn[index], _  = model(inputs)
    return embedding_cnn


def test_cnn(epoch, model, dataloader):
    model.eval()
    count = 0
    correct = 0
    predict_label = []
    true_label = []
    for _, data in enumerate(dataloader):
        inputs, labels = data[0].to(args.device), data[1].to(args.device)
        _, logits = model(inputs)
        _, predict = torch.max(logits, dim=1)
        count += inputs.size(0)
        correct += (labels == predict).sum().item()
        true_label.extend(np.array(labels.cpu()).tolist())
        predict_label.extend(np.array(predict.cpu()).tolist())

    c = confusion_matrix(true_label, predict_label)
    acc = float(correct / count * 100)
    sen = recall_score(true_label, predict_label) * 100
    auc = roc_auc_score(true_label, predict_label) * 100
    f1 = f1_score(true_label, predict_label) * 100
    spe = c[0][0] / (c[0][0] + c[0][1]) * 100
    print(f"epoch:{epoch} cnn_test_acc: {acc: .2f}")
    return acc, sen, auc, f1, spe, c


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


if __name__ == '__main__':

    embedding_dim = 128
    output_dim = 10
    top = 20

    # learning rate
    learning_rate = 1e-4
    weight_decay = 5e-4
    epoch_decay_start = 80
    mom1 = 0.9
    mom2 = 0.1
    alpha_plan = [learning_rate] * args.epochs
    beta1_plan = [mom1] * args.epochs
    for i in range(epoch_decay_start, args.epochs):
        alpha_plan[i] = float(args.epochs - i) / (args.epochs - epoch_decay_start) * learning_rate
        beta1_plan[i] = mom2
    # early-stop
    patience = 20
    min_loss = np.inf
    converge = patience
    cnn_dir = 'model_pretrain/cnn_nihcc_0.4.pth'
    # data transformer
    transforms_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    transforms_test = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

    # create results file
    txtfile = './result/main_nihcc_result.txt'
    # create model  and optimizer
    cnn = CNN(num_class=args.num_class, gpu_id=args.gpu_id, embedding_dim=embedding_dim)
    optimizer_cnn = torch.optim.Adam(cnn.parameters(), lr=1e-4)
    weight_criterion = CE(aggregate='sum')

    # device
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        args.device = torch.device('cuda', args.gpu_id)
    else:
        args.device = torch.device('cpu')
    cnn = cnn.to(args.device)
    # load data
    train_dataset = NIHCC(csv='./data/train.csv',
                          file='./data/images_1024x1024/',
                          transform=transforms_train)
    test_dataset = NIHCC(csv='./data/val.csv',
                         file='./data/images_1024x1024/',
                         transform=transforms_test)

    trainLabel = np.array(train_dataset.targets)
    true_label = np.squeeze(trainLabel).copy()
    trainLabel, actual_noise_rate = GN.noisify(nb_classes=args.num_class, train_labels=np.squeeze(trainLabel), noise_type='pairflip', noise_rate=args.r)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    weight_cnn_all = torch.zeros(len(train_dataset), device=args.device, requires_grad=False)
    train_idx_all = []

    # 加载预训练模型
    checkpoint = torch.load(cnn_dir, map_location=args.device)
    start_epoch = checkpoint['epoch'] + 1
    cnn.load_state_dict(checkpoint['model'])
    optimizer_cnn.load_state_dict(checkpoint['optimizer'])

    cnn = cnn.to(args.device)
    optimizer_to(optimizer_cnn, args.device)

    last_mean = np.inf
    change = True
    patience = 20
    stop = 0
    p_label = torch.zeros((len(train_dataset), args.num_class), device=args.device, requires_grad=False)
    weights = torch.ones(len(train_dataset), device=args.device, requires_grad=False)
    count = 0
    estimated_ratio = 0
    for epoch in range(start_epoch, args.epochs):
        if epoch % 5 == 0:
            if change:
                prob, loss_sample = get_loss(cnn, train_loader, trainLabel)
                pred = (prob > 0.5)
                pred_idx = pred.nonzero()
                estimated_ratio = pred_idx[0].shape[0] / loss_sample.shape[0]
                adjust_learning_rate(optimizer_cnn, epoch, alpha_plan, beta1_plan)
                embedding_cnn = embedding(cnn, train_loader, embedding_dim)
                if 1 - estimated_ratio < 0.15:
                    count = count + 1
                if count < 3:
                    weights, noisy_prob_all, noisy_idx_all, middle_idx_all = cleaner.run_clean(embedding_cnn, trainLabel, loss_sample, 1 - estimated_ratio, device=args.device, gpu_id=args.gpu_id)
                    prob_cleaner, change_label = torch.max(noisy_prob_all, dim=1)
                    prob_sorted = torch.argsort(prob_cleaner, dim=0)
                    top5 = prob_sorted[-int((1 - estimated_ratio) * 0.4 * len(trainLabel)):]
                    trainLabel[top5.cpu()] = change_label[top5].cpu()
                else:
                    change = False
                    stop = epoch

        train_cnn_acc, p_label = train_cnn(epoch, cnn, optimizer_cnn, train_loader, weights, change, p_label, train_dataset, trainLabel, start_epoch)
        test_cnn_acc, sen, auc, f1, spe, c = test_cnn(epoch, cnn, test_loader)
