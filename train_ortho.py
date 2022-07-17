# -*- coding: utf-8 -*-

import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from models import Generator, Discriminator
from models import MnistCNN, CifarCNN
from test_model import load_dataset
import neptune.new as neptune
from utils import accuracy

def show_images(e, x, x_adv, x_fake, save_dir,neptune,run):
    fig, axes = plt.subplots(3, 5, figsize=(10, 6))
    mean = torch.tensor((0.4914, 0.4822, 0.4465))
    std = torch.tensor((0.247, 0.243, 0.261))
    unnormalize = transforms.Normalize((-mean / std), (1.0 / std))
    for i in range(5):
        axes[0, i].axis("off"), axes[1, i].axis("off"), axes[2, i].axis("off")
        im = unnormalize(x[i])
        axes[0, i].imshow(im.cpu().numpy().transpose((1, 2, 0)))
        # axes[0, i].imshow(x[i, 0].cpu().numpy(), cmap="gray")
        axes[0, i].set_title("Normal")

        axes[1, i].imshow(x_adv[i].cpu().numpy().transpose((1, 2, 0)))
        # axes[1, i].imshow(x_adv[i, 0].cpu().numpy(), cmap="gray")
        axes[1, i].set_title("Adv")

        im_fake = unnormalize(x_fake[i])
        axes[2, i].imshow(im_fake.cpu().numpy().transpose((1, 2, 0)))
        # axes[2, i].imshow(x_fake[i, 0].cpu().numpy(), cmap="gray")
        axes[2, i].set_title("APE-GAN")
    plt.axis("off")
    plt.savefig(os.path.join(save_dir, "result_{}.png".format(e)))
    if e%5 ==0 and neptune:
        run['images'].log(fig)


def main(args):
    print("main running")
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch
    xi1, xi2 = args.xi1, args.xi2
    nept = args.neptune
    run = None

    check_path = args.checkpoint
    os.makedirs(check_path, exist_ok=True)

    train_data = torch.load("data.tar")
    x_tmp = train_data["normal"][:5]
    x_adv_tmp = train_data["adv"][:5]

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    transform_mid = transforms.Compose([
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=1024, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=512, shuffle=False, num_workers=2)

    in_ch = 1 if args.data == "mnist" else 3
    G = Generator(in_ch).cuda()
    G = nn.DataParallel(G)
    D = Discriminator(in_ch).cuda()
    D = nn.DataParallel(D)

    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.9, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    loss_bce = nn.BCELoss()
    loss_mse = nn.MSELoss()
    loss_kl = nn.KLDivLoss(reduction='batchmean')

    cudnn.benchmark = True

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=200)


    #the model
    model = CifarCNN().cuda()
    model = nn.DataParallel(model)
    model_point = torch.load("cnn.tar")
    model.load_state_dict(model_point["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    if nept:
        run = neptune.init(
            project="woojin.jeon337/encrypt-gan",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3MjY5MDI4ZC03ZmQ0LTRhMDEtYmMyMC03MDZhNzYxZTQ4ODIifQ==",
        )  # your credentials

        run['dataset'] = args.data
        params = {"title": args.title, "batch": batch_size, "lr": lr, "ratio":"{}/{}".format(xi1,xi2)}
        run['params'] = params
        

    print_str = "\t".join(["{}"] + ["{:.6f}"] * 2)
    print("\t".join(["{:}"] * 3).format("Epoch", "Gen_Loss", "Dis_Loss"))
    for e in range(epochs):
        G.eval()
        x_fake = G(Variable(x_tmp.cuda())).data
        show_images(e, x_tmp, x_adv_tmp, x_fake, check_path,nept,run)
        G.train()
        gen_loss, dis_loss, n = 0, 0, 0
        for x, _ in tqdm(train_loader, total=len(train_loader), leave=False):

            current_size = x.size(0)
            x = Variable(x.cuda())
            # Train D
            t_real = Variable(torch.ones(current_size).cuda())
            t_fake = Variable(torch.zeros(current_size).cuda())


            x_fake = G(x)
            x_fake = x_fake
            x_fake = transform_mid(x_fake)
            out_fake = model(x_fake).squeeze()
            out_real = model(x).squeeze()

            # y_real = D(x).squeeze()
            # y_fake = D(x_fake).squeeze()
            #
            # loss_D = loss_bce(y_real, t_real) + loss_bce(y_fake, t_fake)
            # opt_D.zero_grad()
            # loss_D.backward(retain_graph = True)
            # opt_D.step()

            # Train G
            for _ in range(1):
                x_fake = G(x)
                # y_fake = D(x_fake).squeeze()

                loss_G = xi1 * loss_mse(x_fake, x)*-1 + xi2 * loss_kl(out_fake, out_real.exp())
                opt_G.zero_grad()
                loss_G.backward()
                opt_G.step()

            # dis_loss += loss_D.data * x.size(0)
            dis_loss = 0
            gen_loss += loss_G.data * x.size(0)
            n += x.size(0)
        scheduler.step()
        
        print(print_str.format(e, gen_loss / n, dis_loss / n))
        torch.save({"generator": G.state_dict(), "discriminator": D.state_dict()},
                   os.path.join(check_path, "{}.tar".format(e + 1)))
        if nept:
            # run['dis/loss'].log(dis_loss/n)
            run['gan/loss'].log(gen_loss/n)
            run['epochs']=e
        if e%5 == 0:
            acc_gen, acc, n = 0, 0, 0
            for x, t in tqdm(test_loader, total=len(test_loader), leave=False):
                x, t = Variable(x.cuda()), Variable(t.cuda())

                x_gen = G(x)
                y_gen = model(x_gen)
                # y_true = model(x)

                acc_gen += accuracy(y_gen,t)
                # acc += accuracy(y_true,t)
                n += t.size(0)

            
            acc_gen = acc_gen / n *100
            acc = acc / n *100
            print("encrypt accuracy: {}, true accuracy: {}".format(acc_gen,acc))

            if nept:
                run["gen acc"].log(acc_gen)
                # run["true acc"].log(acc)
                

    G.eval()
    x_fake = G(Variable(x_adv_tmp.cuda())).data
    show_images(epochs, x_tmp, x_adv_tmp, x_fake, check_path, nept,run)
    G.train()
    if nept:
        run.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="cifar")
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--xi1", type=float, default=0.5)
    parser.add_argument("--xi2", type=float, default=0.5)
    parser.add_argument("--checkpoint", type=str, default="./checkpoint/test")
    parser.add_argument("--neptune", action='store_true')
    parser.add_argument("--title", type=str, default="untitled")
    parser.add_argument("--batch", type=int, default=512)
    args = parser.parse_args()
    main(args)

    # args.data = 'cifar'
    # args.eps = 0.01
    # args.gan_path = "checkpoint/cifar/10.tar"
