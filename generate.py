# -*- coding: utf-8 -*-

import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from torchvision import datasets
from torchvision import transforms

from tqdm import tqdm

from models import MnistCNN, CifarCNN
from utils import accuracy, fgsm


def load_dataset(args):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.data == "mnist":
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(os.path.expanduser("data/mnist"), train=True, download=True,
                           transform=transform_train),
            batch_size=args.batch, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(os.path.expanduser("data/mnist"), train=False, download=False,
                           transform=transform_test),
            batch_size=args.batch, shuffle=False)
    elif args.data == "cifar":
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(os.path.expanduser("data/cifar10"), train=True, download=True,
                             transform=transform_train),
            batch_size=args.batch, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(os.path.expanduser("data/cifar10"), train=False, download=False,
                             transform=transform_test),
            batch_size=args.batch, shuffle=False)
    return train_loader, test_loader


def load_cnn(args):
    if args.data == "mnist":
        return MnistCNN
    elif args.data == "cifar":
        return CifarCNN


def main(args):
    print("Generating Model ...")
    print("-" * 30)

    train_loader, test_loader = load_dataset(args)
    CNN = load_cnn(args)
    model = CNN().cuda()
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    loss_func = nn.CrossEntropyLoss().cuda()

    epochs = args.epochs
    print_str = "\t".join(["{}"] + ["{:.6f}"] * 4)
    print("\t".join(["{:}"] * 5).format("Epoch", "TrainLoss", "TestLoss", "TrainAcc.", "TestAcc."))
    for e in range(epochs):
        train_loss, train_acc, train_n = 0, 0, 0
        test_loss, test_acc, test_n = 0, 0, 0

        model.train()
        for x, t in tqdm(train_loader, total=len(train_loader), leave=False):
            x, t = Variable(x.cuda()), Variable(t.cuda())
            y = model(x)
            loss = loss_func(y, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss * t.size(0)
            train_acc += accuracy(y, t)
            train_n += t.size(0)

        model.eval()
        for x, t in tqdm(test_loader, total=len(test_loader), leave=False):
            x, t = Variable(x.cuda()), Variable(t.cuda())
            y = model(x)
            loss = loss_func(y, t)

            test_loss += loss.data * t.size(0)
            test_acc += accuracy(y, t)
            test_n += t.size(0)
        scheduler.step()
        print(print_str.format(e, train_loss / train_n, test_loss / test_n,
                               train_acc / train_n * 100, test_acc / test_n * 100))

    # Generate Adversarial Examples
    print("-" * 30)
    print("Genrating Adversarial Examples ...")
    eps = args.eps
    train_acc, adv_acc, train_n = 0, 0, 0
    normal_data, adv_data = None, None
    for x, t in tqdm(train_loader, total=len(train_loader), leave=False):
        x, t = Variable(x.cuda()), Variable(t.cuda())
        y = model(x)
        train_acc += accuracy(y, t)

        x_adv = fgsm(model, x, t, loss_func, eps)
        y_adv = model(x_adv)
        adv_acc += accuracy(y_adv, t)
        train_n += t.size(0)

        x, x_adv = x.data, x_adv.data
        if normal_data is None:
            normal_data, adv_data = x, x_adv
        else:
            normal_data = torch.cat((normal_data, x))
            adv_data = torch.cat((adv_data, x_adv))

    print("Accuracy(normal) {:.6f}, Accuracy(FGSM) {:.6f}".format(train_acc / train_n * 100, adv_acc / train_n * 100))
    torch.save({"normal": normal_data, "adv": adv_data}, "data.tar")
    torch.save({"state_dict": model.state_dict()}, "cnn.tar")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="cifar")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--milestones", type=list, default=[50, 75])
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--eps", type=float, default=0.15)
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()
    main(args)
