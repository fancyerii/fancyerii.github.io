---
layout:     post
title:      "常见Linux工具用法" 
author:     "lili" 
excerpt_separator: <!--more-->
tags:
    - Linux
    - 命令
---

记录一下常见工具的用法。目前包括：命令行设置各种代理服务器的方法；ssh的各种用法；git上传大文件的方法。

<!--more-->

**目录**
* TOC
{:toc}

## 命令行使用代理的方法(不修改配置文件)

### docker build的代理
```
docker build --build-arg http_proxy=http://myproxy:1080 --build-arg https_proxy=http://myproxy:1080 .
```

### maven
```
mvn install -Dhttp.proxyHost=[PROXY_SERVER] -Dhttp.proxyPort=[PROXY_PORT] -Dhttp.nonProxyHosts=[PROXY_BYPASS_IP] -Dhttps.proxyHost= -Dhttps.proxyPort= 
```

### wget
```
wget -e use_proxy=yes -e http_proxy=$proxy http://www.google.com.hk

http_proxy=myproxy:1080 wget http://www.google.com.hk

```

### pip
```
pip install --proxy http://user:password@proxyserver:port numpy
```

### git
```
git -c "http.proxy=address:port" clone http://github.com/xxxx.git
```

### apt-get
```
sudo http_proxy=http://yourserver apt-get update
```

### svn

```
svn co --config-option servers:global:http-proxy-host=ai-dev --config-option servers:global:http-proxy-port=1080 ...
```

## git

### git上传大文件到github

1. 安装git lfs，参考[官网](https://git-lfs.github.com/)。
2. 进入repo执行
```
git lfs install
```
3. track大文件
```
git lfs track my-large-file.gz
git add .gitattributes
```

4. 用普通的git命令add大文件
```
git add my-large-file.gz
git commit -m "add larg file"
git push
```

## ssh

### 无密码登录
```
ssh-keygen -t rsa
# 然后一路回车
```
设置，假设用户名为b，服务器为B：
```
ssh b@B mkdir -p .ssh
```
把公钥加到的服务器：
```
cat .ssh/id_rsa.pub | ssh b@B 'cat >> .ssh/authorized_keys'
```

测试：
```
ssh b@B
```
如果还是要输入密码，很可能是服务器上文件的权限不对：
```
chmod 700 .ssh
chmod 640 .ssh/authorized_keys
```

### 基本用法
```
ssh -l lili ai-dev
ssh lili@ai-dev
```

### 用identity文件登录
```
ssh -i id_file
```

### 用scp复制文件
```
scp abc lili@ai-dev:~/; scp lili@ai-dev:~/abc ~/abc.bak
# 直接从远程的ai-dev复制到spark2，不需要登录服务器
scp lili@ai-dev:~/abc lili@spark2:~/
```
### 远程执行命令
```
ssh lili@ai-dev "ls ~/"
```

### 使用22之外的端口
```
ssh -p 22222 
scp -P 22222
```

### 保持连接(某些服务器会定期踢掉没有活动的客户端）

修改.ssh/config
```
Host myhostshortcut
     HostName myhost.com
     User lili
     ServerAliveInterval 30
```

### 通过ssh连接两台内网机器
比如想在家里访问公司的内网机器，但是没有VPN，则可以使用Port Forwarding。要求：
* 一台有外网ip的服务器(比如阿里云服务器ai-dev)
* 两台内网机器都可以访问这台外网服务器

在被ssh的内网机器上：
```
ssh -f -N -T -R33333:localhost:22 lili@ai-dev
```

在家里的机器：
```
ssh lili@ai-dev -t ssh -p 33333 lili@127.0.0.1
```


### 用浏览器访问内网的HTTP服务

要求:
* 浏览器所在机器可以ssh某内网服务器(ai-dev)
* 内网服务器(ai-dev)可以访问HTTP服务

```
ssh -D 12345 lili@ai-dev
```
设置浏览器使用socks5代理,ip是127.0.0.1,端口是12345。

### 家里用浏览器访问公司内网的HTTP服务

要求：
* 一台有外网ip的服务器(比如阿里云服务器ai-dev)
* 家里和内网服务器机器都可以访问这台外网服务器

公司内网机器：
```
ssh -f -N -T -R33333:localhost:22 lili@ai-dev
```
家里：
```
ssh -L 8001:localhost:8002 lili@ai-dev -t ssh -D 8002 -p 33333 lili@127.0.0.1
```
家里浏览器使用socks5代理,localhost 8001。

## iptables

### 某个端口只开放给某个ip
```
iptables -A INPUT -p tcp --dport 7071 -s 1.2.3.4 -j ACCEPT
iptables -A INPUT -p tcp --dport 7071 -j DROP
```

注意：上面的命令重启后会失效。

```
sudo apt-get install iptables-persistent
```

然后每次修改iptables执行(Ubuntu 14.04)：
```
sudo /etc/init.d/iptables-persistent save 
sudo /etc/init.d/iptables-persistent reload
```
Ubuntu 16.04：
```
sudo netfilter-persistent save
sudo netfilter-persistent reload
```


