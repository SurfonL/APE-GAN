#!/bin/bash



#SBATCH -J untitled
#SBATCH -p part1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -o run/results/%x_p_%j.txt
#SBATCH -e run/results//%x_e_%j.txt
#SBATCH --gres=gpu



CUDA_VISIBLE_DEVICES=0
#python train.py --data cifar --epochs 30 --checkpoint ./checkpoint/cifar
#python generate.py --data cifar --eps 0.01 --lr 0.005 --batch 1024 --epochs 50
python train_ortho.py\
 --title firstrun\
 --data cifar\
 --epochs 100000\
 --xi1 0.\
 --xi2 1.\
 --batch 256\
 --lr 0.0001\
 --neptune