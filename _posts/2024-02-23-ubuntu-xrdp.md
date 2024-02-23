---
layout:     post
title:      "Ubuntu安装RDP Server" 
author:     "lili" 
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - rdp
    - xrdp
    - gui
---

通常我们通过ssh登录到远程服务器。但是有的时候我们想运行一些有GUI的程序，比如chrome浏览器。这个时候我们就可以在服务器上安装RDP Server，然后本地客户端就可以通过RDP协议通过GUI的方式登录服务器了。RDP是微软开发的协议，最早用于Windows远程桌面。我们在服务器上安装RDP Server之后，本地不管是Linux还是Windows都可以方便的登录了。

<!--more-->

**目录**
* TOC
{:toc}


 

## 安装Xrdp Server

### 安装桌面环境

通常服务器是没有安装桌面环境的，因此我们需要安装。Ubuntu默认使用GNOME，但是比较重。我们这里安装轻量级的xfce4：

```shell
sudo apt update
sudo apt install xfce4 xfce4-goodies xorg dbus-x11 x11-xserver-utils
```

### 安装Xrdp

```shell
sudo apt install xrdp 
```

查看安装是否成功：

```shell
sudo systemctl status xrdp
```

如果看到类似下面的输出就表示安装并且启动成功了：

```
● xrdp.service - xrdp daemon
   Loaded: loaded (/lib/systemd/system/xrdp.service; enabled; vendor preset: enabled)
   Active: active (running) since Sun 2019-07-28 22:40:53 UTC; 4min 21s ago
     Docs: man:xrdp(8)
           man:xrdp.ini(5)
  ...

```

Xrdp使用/etc/ssl/private/ssl-cert-snakeoil.key这个证书，只有ssl-cert这个组才能访问。因此我们需要把xrdp用户加入这个组：

```shell
sudo adduser xrdp ssl-cert  
```

如果服务器有防火墙，需要打开(没有就可以跳过)：

```shell
sudo ufw allow from 192.168.1.0/24 to any port 3389
# 或者
sudo ufw allow 3389
```

ufw的用法请读者自行网上搜索。

## 客户端连接

不同的系统使用不同的RDP客户端，比如Windows自带了客户端。我这里用的是Ubuntu，所以介绍ubuntu下的客户端。

### 安装remmina

Linux下有很多RDP客户端，这里我使用[remmina](https://remmina.org/)。因为是很久以前安装的，我这次没有测试。不过下面的步骤应该是work的，如果不行，请参考[官方文档](https://remmina.org/how-to-install-remmina/)安装。

```shell
sudo add-apt-repository ppa:remmina-ppa-team/remmina-next
sudo apt update -y
sudo apt install remmina remmina-plugin-rdp remmina-plugin-secret
```


## 安装chrome

```shell
sudo apt-get install libxss1 libappindicator1 libindicator7
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo apt install ./google-chrome*.deb
sudo apt-get install -f
```

启动：

```shell
google-chrome-stable
```

## 解决非常卡的问题

参考[](https://askubuntu.com/questions/1323601/xrdp-is-quite-slow)。

```shell
xfconf-query --channel=xfwm4 --property=/general/use_compositing --type=bool --set=false --create
```

如果要持久化，需要修改xorg.conf：

```
Section "Extensions"
    Option "Composite" "Disable"
EndSection
```

或者使用图形工具xfwm4-tweaks-settings配置。



## 安装中文环境和字体

如果启动chrome，我们会发现中文网站都是口口口口口...。因此需要安装中文环境和字体。

```shell
sudo apt install locales
sudo dpkg-reconfigure locales
```
在上面的配置时选择en_US.UTF8，zh_CN GB2312，zh_CN GBK GBK，zh_CN UTF-8 UTF-8。

安装中文字体：

```shell
sudo apt install fonts-wqy-zenhei
```

