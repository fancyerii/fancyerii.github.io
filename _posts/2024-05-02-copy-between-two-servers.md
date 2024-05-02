---
layout:     post
title:      "在两个远程服务器之间复制文件"
author:     "lili"
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - ssh
    - scp
    - rsync
---



 <!--more-->
 
 
## 问题

假设我们有两台服务器server1和server2，另外我们一台本地的集群local。local可以通过ssh连接到server1和server2，但是server1和server2之间无法直连。我们现在需要把一些文件或者目录从server1复制到server2。

当然最简单的办法就是先用scp或者rsync把文件或者目录从server1复制到local，再由local复制到server2。但是这存在一些问题，比如说文件特别大，local放不下。或者说复制两次，时间就要翻倍，对于特别大的数据来说，一次复制的时间可能是几天甚至几周。

## 解决办法

### 使用scp

根据[这个帖子](https://superuser.com/questions/686394/scp-between-two-remote-hosts-from-my-third-pc)，我们可以这样实现：

```shell
local$ scp -3 -r server1:~/models/Meta-Llama-3-8B server2:/data/models/
```

注意参数-3，这个 -3 选项指示 scp 使用命令所在机器(local)来进行路由流量，也就是server1 -> local -> server。
 
使用这个方法有两个问题：
* scp没有进度信息(不知道是否有什么参数设置不对)
* 无法resume(断点续传)

第一个问题倒不是很大(只是感觉不爽) ，第二个问题对于大文件传输来说是很关键的，传输一个几百GB的文件中断了需要重新开始，这可受不了。

### 使用rsync

参考[这个帖子](https://unix.stackexchange.com/questions/48298/can-rsync-resume-after-being-interrupted)，rsync命令可以使用\-\-partial或者\-\-partial-dir来实现断点续传。但是rsync不能支持source和target都是远程机器，因此下面的命令是不行的：

```shell
rsync server1:~/models/Meta-Llama-3-8B server2:/data/models/
```
 
### sshfs加rsync

rsync只能从本地复制到远程或者从远程复制到本地，但是不能远程到远程。那怎么办呢？我们可以使用sshfs把其中一个远程(网络)目录挂着到本地，这样就可以使用rsync的断点续传能力了！

```shell
local$ sshfs server1:~/models /mnt/models 
local$ rsync --bwlimit=2000 --partial -av --progress /mnt/models/Meta-Llama-3-70B-Instruct server2:/data/models/
```

为了避免网络带宽全部被占用，我们可以使用rsync的\-\-bwlimit进行限速。
