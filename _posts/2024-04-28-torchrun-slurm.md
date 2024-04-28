---
layout:     post
title:      "在Slurm集群里进行Pytorch DDP训练" 
author:     "lili" 
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - pytorch
    - slurm 
---

本文是在slurm管理的集群里通过sbatch提交任务进行Pytorch DDP训练的例子。我们这里只用cpu。

<!--more-->

**目录**
* TOC
{:toc}

## 训练代码

```python
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from socket import gethostname

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(args, model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = False

    torch.manual_seed(args.seed)
    
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}


    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('data',
                              train=True,
                              download=False,
                              transform=transform)
    dataset2 = datasets.MNIST('data',
                              train=False,
                              download=False,
                              transform=transform)

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["SLURM_PROCID"])
    print(f"Hello from rank {rank} of {world_size} on {gethostname()}" \
          , flush=True)

    setup(rank, world_size)
    if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)

    
    print(f"host: {gethostname()}, rank: {rank}")

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset1,
                                                                    num_replicas=world_size,
                                                                    rank=rank)
    train_loader = torch.utils.data.DataLoader(dataset1,
                                               batch_size=args.batch_size,
                                               sampler=train_sampler,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset2,
                                              **test_kwargs)
    
    model = Net()
    ddp_model = DDP(model)
    optimizer = optim.Adadelta(ddp_model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, ddp_model, train_loader, optimizer, epoch)
        if rank == 0: test(ddp_model, test_loader)
        scheduler.step()

    if args.save_model and rank == 0:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
```

这是一个非常简单的ddp训练代码，如果要在slurm集群上运行，有这么几点：

* word_size需要通过环境变量WORLD_SIZE读取，后面slurm脚本会介绍怎么计算
* rank由slurm的环境变量SLURM_PROCID提供

## slurm脚本

```
#!/bin/bash
#SBATCH --job-name=ddp-torch     # create a short name for your job
#SBATCH -p debug
#SBATCH --nodes=2                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-task=32        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=32G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --time=1:00:00          # total run time limit (HH:MM:SS)
#SBATCH -o logs/out%j.log
#SBATCH -e logs/err%j.log

export MASTER_PORT=29500
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

module purge
source /public/home/lili/.bashrc
conda activate torch22cpu

srun python mnist_classify_ddp.py --epochs=2
```

* 使用debug分区
* 申请两个节点
* 每个节点起一个任务/进程(我们这里是cpu，所以设置为1就可以了，如果是gpu那么通常每个卡起一个进程)
* 每个进程申请32核(blas会充分利用多核)
* 每个节点内存32G
* 运行最长时间为1个小时
* 标准输出和错误输出保存到logs目录下，%j的意思是任务id，这样避免多次运行覆盖
* 使用29500作为master的端口
* WORLD_SIZE是节点总数(SLURM_NNODES)乘以每个节点上的任务数(SLURM_NTASKS_PER_NODE)
* MASTER_ADDR是SLURM_JOB_NODELIST里的第一个节点

我们发现在slurm里运行ddp程序最大的特点就是我们不能提前知道程序会跑到哪些机器上，因此需要用SLURM_相关的环境变量(这些环境变量在我们用sbatch提交任务后被自动设置好)。由于WORLD_SIZE由SLURM_NNODES和SLURM_NTASKS_PER_NODE动态计算得到，如果我们想增加节点，我们只需要修改"SBATCH --nodes=xxx"就可以了。

注意：只有srun的命令会在每个节点上运行，其它命令(比如module purge)只会在提交任务的节点上运行。这些命令通常用于设置环境变量或者初始化环境(mpi通常用module或者python用conda activate或者source)，其它计算节点会继承这些环境变量。

## 提交任务

```shell
sbatch ddp.job
```


