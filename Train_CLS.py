
import argparse
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from model.cls_model import Illumination_classifier


def init_seeds(seed, use_cuda=False):
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)


def test(model, test_loader, use_cuda=False):

    model.eval()
    test_loss = 0
    correct = 0
    device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(test_loader.dataset)
    precision = correct / float(len(test_loader.dataset))
    print('\nTest set: Avg loss: {:.4f}, Acc.: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * precision))
    return precision


def adjust_learning_rate(optimizer, base_lr, epoch, total_epochs):
    if epoch < total_epochs // 2:
        lr = base_lr
    else:
        lr = base_lr * (total_epochs - epoch) / (total_epochs - total_epochs // 2)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(model, train_loader, test_loader, args):
    best_prec1 = 0.0
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, args.lr, epoch, args.epochs)

        model.train()
        train_tqdm = tqdm(train_loader, total=len(train_loader))
        device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'

        for images, labels in train_tqdm:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            output, _ = model(images)
            loss = F.cross_entropy(output, labels)

            train_tqdm.set_postfix(epoch=epoch, loss_total=loss.item())
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            prec1 = test(model, test_loader, use_cuda=args.cuda)

            if best_prec1 < prec1:
                save_path = os.path.join(args.save_path, 'best_cls.pth')
                torch.save(model.state_dict(), save_path)
                best_prec1 = prec1


def main():

    parser = argparse.ArgumentParser(description='PyTorch Illumination Classifier Training')
    parser.add_argument('--dataset_path', metavar='DIR', required=True, help='Path to dataset')
    parser.add_argument('--arch', metavar='ARCH', default='cls_model', choices=['cls_model', 'fusion_model'])
    parser.add_argument('--save_path', default='runs')
    parser.add_argument('--workers', '-j', default=1, type=int, help='Number of data loader workers')
    parser.add_argument('--epochs', default=300, type=int, help='Number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, help='Manual epoch number (for restart)')
    parser.add_argument('--batch_size', '-b', default=8, type=int, help='Mini-batch size per GPU')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Optimizer momentum')
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float, help='Weight decay for optimizer')
    parser.add_argument('--image_size', default=64, type=int, help='Input image size')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--cuda', action='store_true', help='Use GPU if available')
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    init_seeds(args.seed, use_cuda=args.cuda)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    train_dataset = datasets.ImageFolder(args.dataset_path, transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    if args.arch == 'cls_model':
        model = Illumination_classifier(input_channels=3).to(device)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

    train(model, train_loader, test_loader, args)


if __name__ == '__main__':
    main()
