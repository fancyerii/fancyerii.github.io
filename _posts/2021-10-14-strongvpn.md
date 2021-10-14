---
layout:     post
title:      "Ubuntu 16.04 StrongVPN设置" 
author:     "lili" 
mathjax: true
excerpt_separator: <!--more-->
tags:
    - VPN
    - 翻墙
    - Ubuntu
---

本文记录在Ubuntu下设置Strong VPN的过程，仅供参考。本文假设读者的机器不能翻墙，但是需要能访问github，如果不能访问github，可以尝试用ssh协议clone相应的repo，比如我们打不开https://github.com/vpncn/vpncn.github.io，那么可以尝试git clone git@github.com:vpncn/vpncn.github.io.git。

<!--more-->

**目录**
* TOC
{:toc}

## 选择

最近翻墙越来越困难，根据[翻墙软件VPN推荐](https://github.com/vpncn/vpncn.github.io)，目前好用的只有Strong VPN和Express VPN。比较了一下，Strong VPN性价比更高一点。而且如果当前没有梯子，根本打不开Express VPN的官网，用代理打开了去购买时它也有可能提示使用了代理。虽然文章说Strong VPN稍微慢一点，但是我的需求就是普通的上网，不需要看视频，应该够用。另外Strong VPN的客服的相应速度也挺快的，遇到问题发个工单(ticket)，可以直接用中文提问题。

## 设置hosts

默认情况下是无法访问Strong VPN的，需要修改hosts文件，从[这里](http://linkv.org/download/hosts)下载，然后把内容加到/etc/hosts下。

## 购买
打开[优惠链接](https://linkv.org/strongcn/)进行购买，当前的优惠价格是31.99刀/年，用支付宝可以支付。

## 安装

对于Ubuntu 16.10之后的版本有图形界面的支持，但是老的版本(包括16.04)只能用命令行。当然好处就是它哪里都能用，这样如果想在某台没有UI的服务器上临时用用也是可以的。我是按照[General Linux Command Line OpenVPN Setup Tutorial](https://strongvpn.com/setup-linux-openvpn/)进行设置的。分为如下步骤：

### 安装openvpn

文档是这样安装的：
```
sudo apt-get -y install openvpn
```

但是后来我碰到了问题，客服说“在中国境内使用我们的OpenVPN，需要支持scramble 功能。”，可以去[这里](https://app.blackhole.run/#7628c8d9a51GjDabREpchRGfUsSnzHveYZLoEffSsa3x)下载openvpn_2.4.8-bionic0_amd64.deb安装。

不过我在使用sudo dpkg -i openvpn_2.4.8-bionic0_amd64.deb命令时提示没有libssl。搜索后找到[这个文章](https://stackoverflow.com/questions/68148246/broken-php-and-libssl1-1-installation-on-ubuntu-16-04)的解决办法：

```
wget http://archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.0g-2ubuntu4_amd64.deb
sudo dpkg -i libssl1.1_1.1.0g-2ubuntu4_amd64.deb
```

### 下载OPENVPN Config文件

可以参考[这篇文章](https://strongvpn.com/view-greeting/)，把下载的文件重命名放置到/etc/openvpn/strongvpn.conf。注意：如果下载的文件名是str-xxx.ovpn，则说明是新的的server。我因为是刚买的，所以是新的。

```
sudo mv vpn-XXX_ovpnXXX_account.ovpn /etc/openvpn/strongvpn.conf
```

### 设置用户名密码

用喜欢的编辑器创建/etc/openvpn/auth.txt文件，在这个文件增加两行：第一行是用户名，第二行是密码。用户名和密码参考下图去找：

<a>![](/img/strongvpn/1.jpg)</a>
*用户名密码*

然后设置文件权限：

```
sudo chmod 400 /etc/openvpn/auth.txt
```

然后修改/etc/openvpn/strongvpn.conf使用auth.txt：
```
把auth-user-pass改成auth-user-pass /etc/openvpn/auth.txt
```

### 启动服务

```
sudo service openvpn start
```

### 修改DNS resolver 

注意：这一步很重要，官方的教程没有(因为人家不是用来翻墙的)。详细的步骤在[这里](https://github.com/alfredopalhares/openvpn-update-resolv-conf)，下面是我的操作。

#### 安装openresolv

```
sudo apt-get install openresolv
```

#### 下载update-resolv-conf.sh

从github下载这个文件放到/etc/openvpn/下

#### 修改strongvpn.conf
在文件最开头增加如下内容：

```
# This updates the resolvconf with dns settings
setenv PATH /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
script-security 2
up /etc/openvpn/update-resolv-conf.sh
down /etc/openvpn/update-resolv-conf.sh
down-pre
```

### 连接VPN

```
$ cd /etc/openvpn
$ sudo openvpn strongvpn.conf
```

打开百度搜索ip看看显示的ip地址是不是在墙外了。

### 分流设置

可以找客服要分类的设置程序。我试了一下没有问题。


