---
layout:     post
title:      "Wireguard简介" 
author:     "lili" 
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - VPN
    - wireguard
---

本文介绍[wireguard](https://www.wireguard.com/)。

<!--more-->

**目录**
* TOC
{:toc}

## 基本概念

### Conceptual Overview
 

如果你想了解WireGuard的一般概念概述，请继续阅读这里。然后，您可以进行[安装](https://www.wireguard.com/install/)并阅读有关如何使用WireGuard的[快速入门](https://www.wireguard.com/quickstart/)说明。

如果您对内部工作原理感兴趣，您可能会对[协议](https://www.wireguard.com/protocol/)的简要摘要感兴趣，或者通过阅读[技术白皮书](https://www.wireguard.com/papers/wireguard.pdf)深入了解协议、密码学和基本原理。如果您打算在新平台上实施WireGuard，请阅读[跨平台注意事项](https://www.wireguard.com/xplatform/)。

WireGuard通过UDP安全地封装IP数据包。您添加一个WireGuard接口，使用您的私钥和对等方的公钥进行配置，然后通过它发送数据包。所有与密钥分发和推送配置相关的问题都不属于WireGuard的范围；这些问题更适合留给其他层处理，以免最终出现IKE或OpenVPN的臃肿。相反，它更像是SSH和Mosh的模型；双方都拥有对方的公钥，然后它们可以通过接口开始交换数据包。

### Simple Network Interface

WireGuard的工作原理是通过添加一个网络接口（或多个），类似eth0或wlan0，称为wg0（或wg1、wg2、wg3等）。可以使用ifconfig(8)或ip-address(8)正常配置此网络接口，使用route(8)或ip-route(8)添加和删除其路由，以及使用所有常规的网络工具进行其他配置。使用wg(8)工具配置接口的特定WireGuard方面。此接口充当隧道接口。

WireGuard将隧道IP地址与公钥和远程端点关联起来。当接口向对等方(peer)发送数据包时，它执行以下操作：

1. 此数据包是为192.168.30.8的。这是哪个对等方？让我查一下...好的，它是为对等方ABCDEFGH的。 （或者如果它不是为任何配置的对等方，丢弃数据包。）
2. 使用对等方ABCDEFGH的公钥加密整个IP数据包。
3. 对等方ABCDEFGH的远程端点(remote endpoint)是什么？让我查一下...好的，端点是主机216.58.211.110上的UDP端口53133。
4. 使用UDP将第2步中的加密字节通过互联网发送到216.58.211.110:53133。

当接口接收到一个数据包时，发生以下情况：

1. 我刚刚从主机98.139.183.24的UDP端口7361收到一个数据包。让我们解密它！
2. 它正确解密和验证为对等方LMNOPQRS。好的，让我们记住对等方LMNOPQRS的最新互联网端点是98.139.183.24:7361，使用UDP。
3. 解密后，纯文本数据包来自192.168.43.89。对等方LMNOPQRS是否被允许作为192.168.43.89发送数据包给我们？
4. 如果是，则在接口上接受数据包。如果不是，则丢弃它。

在幕后，有许多工作以提供适当的隐私、真实性和完美前向保密性，使用最先进的密码学技术。

### Cryptokey Routing

在WireGuard的核心是一种称为Cryptokey Routing的概念，它通过将公钥与允许在隧道内的隧道IP地址列表关联起来来实现。每个网络接口都有一个私钥和一组对等方。每个对等方都有一个公钥。公钥短而简单，用于对等方相互进行身份验证。它们可以通过任何带外方法（类似于将SSH公钥发送给朋友以获取对shell服务器的访问权限）在配置文件中传递。

例如，服务器计算机可能具有以下配置：

```
[Interface]
PrivateKey = yAnz5TF+lXXJte14tji3zlMNq+hd2rYUIgJBgB3fBmk=
ListenPort = 51820

[Peer]
PublicKey = xTIBA5rboUvnH4htodjb6e697QjLERt1NAB4mZqp8Dg=
AllowedIPs = 10.192.122.3/32, 10.192.124.1/24

[Peer]
PublicKey = TrMvSoP4jYQlY6RIzBgbssQqY3vxI2Pi+y71lOWWXX0=
AllowedIPs = 10.192.122.4/32, 192.168.0.0/16

[Peer]
PublicKey = gN65BkIKy1eCE9pP1wdc8ROUtkHLF2PfAqYdyYBz6EA=
AllowedIPs = 10.10.10.230/32
```


而客户端计算机可能具有以下更简单的配置：

```
[Interface]
PrivateKey = gI6EdUSYvn8ugXOt8QQD6Yc+JyiZxIhp3GInSWRfWGE=
ListenPort = 21841

[Peer]
PublicKey = HIgo9xNzJMWLKASShiTqIybxZ0U3wGLiUeJ1PKf8ykw=
Endpoint = 192.95.5.69:51820
AllowedIPs = 0.0.0.0/0
```

在服务器配置中，每个对等方（客户端）将能够向网络接口发送具有与其相应的允许IP地址列表匹配的源IP的数据包。例如，当服务器从对等方gN65BkIK...接收到一个数据包后，经过解密和身份验证后，如果其源IP为10.10.10.230，则允许其进入接口；否则将被丢弃。

在服务器配置中，当网络接口希望将数据包发送到对等方（客户端）时，它将查看该数据包的目标IP，并将其与每个对等方的允许IP列表进行比较，以确定要将其发送到哪个对等方。例如，如果网络接口被要求发送一个目标IP为10.10.10.230的数据包，它将使用对等方gN65BkIK...的公钥对其进行加密，然后将其发送到该对等方的最近的互联网端点。

在客户端配置中，其单个对等方（服务器）将能够向网络接口发送任何源IP的数据包（因为0.0.0.0/0是通配符）。例如，当从对等方HIgo9xNz...接收到一个数据包时，如果解密和身份验证正确，带有任何源IP，那么它将被允许进入接口；否则将被丢弃。

在客户端配置中，当网络接口希望将数据包发送到其单个对等方（服务器）时，它将为单个对等方加密具有任何目标IP地址的数据包（因为0.0.0.0/0是通配符）。例如，如果网络接口被要求发送一个带有任何目标IP的数据包，它将使用单个对等方HIgo9xNz...的公钥对其进行加密，然后将其发送到单个对等方的最近的互联网端点。

换句话说，在发送数据包时，允许IP列表的行为类似于路由表，在接收数据包时，允许IP列表的行为类似于访问控制列表。

这就是我们所称的Cryptokey Routing Table：公钥和允许IP的简单关联。

任何IPv4和IPv6的组合都可以用于任何字段。如果需要，WireGuard完全能够将其中一个封装在另一个内部。

由于在WireGuard接口上发送的所有数据包都经过加密和身份验证，而且对等方的身份和对等方的允许IP地址之间存在紧密的耦合，系统管理员不需要复杂的防火墙扩展（例如IPsec），而只需简单地匹配“它是来自此IP吗？在这个接口上？”并确保它是一个安全且真实的数据包。这极大地简化了网络管理和访问控制，并提供了更大的保证，确保您的iptables规则实际上正在执行您打算执行的操作。


### Built-in Roaming

客户端配置包含其单个对等方（服务器）的初始端点，以便在接收加密数据之前就知道将加密数据发送到何处。服务器配置不包含其对等方（客户端）的任何初始端点。这是因为服务器通过检查正确身份验证的数据的来源来发现其对等方的端点。如果服务器本身更改其自己的端点并向客户端发送数据，客户端将发现新的服务器端点并更新配置。客户端和服务器都向它们身份验证解密数据的最近的IP端点发送加密数据。因此，双方都可以完全进行IP漫游(roaming)。

### Ready for Containers

WireGuard[使用创建WireGuard接口的网络命名空间发送和接收加密的数据包](https://www.wireguard.com/netns/)。这意味着您可以在主网络命名空间中创建WireGuard接口，该网络命名空间具有对互联网的访问权限，然后将其移动到属于Docker容器的网络命名空间作为该容器的唯一接口。这确保容器能够访问网络的唯一可能方式是通过安全加密的WireGuard隧道。

## 使用

### 安装

可以参考[安装文档](https://www.wireguard.com/install/)进行安装。

比如Ubuntu:

```
sudo apt install wireguard
```

如果是mac，请使用brew安装，默认的情况大陆的应用商店是不允许的(你懂的)。

```
brew install wireguard-tools
```

### Ubuntu/mac客户端使用

可以参考[这个视频](https://www.wireguard.com/talks/talk-demo-screencast.mp4)，非常直观。

我们这里假设服务器已经配置好了，我们有服务器的public key以及它的endpoint地址(ip:port)。同时我们也有一个服务器允许的ip地址(没有的话请咨询vpn的提供者)。对服务器配置感兴趣的读者可以参考[How To Set Up WireGuard on Ubuntu 20.04](https://www.digitalocean.com/community/tutorials/how-to-set-up-wireguard-on-ubuntu-20-04)。



#### 生成私钥和公钥：

```
wg genkey | tee private.key | wg pubkey > public.key
```

其实这是两个命令：wg genkey生成私钥，存到private.key文件，然后wg pubkey生成公钥，它需要的参数是私钥。上面的命令通过tee和重定向一次搞定。

把你的公钥给服务器的管理人员，让它加到服务器的peer里，并且给你分配一个ip地址。另外也有可能这个私钥和公钥都是管理人员给你的，包括ip地址。

#### 配置

```
sudo vi /etc/wireguard/wg0.conf
```

内容如下：

```
[Interface]
PrivateKey = [你的私钥]
Address = 分配给你的ip地址，可能是10.8.0.1/24
 
#PostUp = iptables -A FORWARD -i wg0 -j ACCEPT; iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
#PostDown = iptables -D FORWARD -i wg0 -j ACCEPT; iptables -t nat -D POSTROUTING -o eth0 -j MASQUERADE


[Peer]
PublicKey = [服务器公钥]
PresharedKey = [服务器presharekey，可选]
AllowedIPs = 10.8.0.0/24 
Endpoint = [服务器ip:port 这通常是一个外网ip]
```

根据[reddit post](https://www.reddit.com/r/WireGuard/comments/vbxzai/wireguard_preshared_key_purpose/)，PresharedKey是未来防备攻击，目前看不配置也没有关系，这取决于管理员。

#### 启动和关闭

```
sudo wg-quick up wg0
sudo wg-quick down wg0
```

注意，如果wg0.conf的位置不是在/etc/wireguard/wg0.conf，那么可以把wg0替换成这个文件的绝对路径。

#### service(Ubuntu only)

为了开机自启动，我们可以配置service，这要求配置文件是/etc/wireguard/wg0.conf：

```
sudo systemctl enable wg-quick@wg0.service
sudo systemctl start wg-quick@wg0.service
```
 
