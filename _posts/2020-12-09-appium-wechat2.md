---
layout:     post
title:      "使用Appium驱动手机版微信(二)" 
author:     "lili" 
mathjax: true
excerpt_separator: <!--more-->
tags:
    - 微信
    - wechat
    - automation
    - Appium
    - webdriver
---

本文是使用Appium驱动手机版微信的系列文章的第二篇，详细介绍Appium的用法。

<!--more-->

**目录**
* TOC
{:toc}
 

本文要求读者有基本的Android开发能力，能够用Android Studio构建最简单的app。本文介绍Appium的基本用法，主要参考了官方文档和[Step by Step Appium Tutorial for Beginners ](http://www.automationtestinghub.com/appium-tutorial/)。


## Appium简介

Appium是一个开源的工具，开源用于自动化(automating) iOS、Android和Windows平台下的原生的(native)、移动web和混合的应用。原生的应用是使用iOS、Android和Windows SDK开发的应用。移动网页app是使用各个平台的浏览器(iOS上的Safara、android上的Chrome或者其它内置浏览器)实现的web应用。混合应用是在原生应用中引入"webview"这样的控件，从而实现HTML和native配合的应用。混合应用可以利用HTML的跨平台性和native应用更强比如NFC这些HTML不支持的能力。[Apache Cordova](https://cordova.apache.org/)或者[Phonegap](http://phonegap.com/)这样的框架可以让我们使用Web技术开发应用，然后它们把这个应用打包成不同平台上的应用，这通常是混合的应用。


Appium是跨平台的：它允许我们使用相同的API来实现在多个平台(iOS、Android和Windows)上的测试。当然Appium并没有什么神奇的能力来实现跨平台，它只是定义了自己的API，然后针对不同的平台，把自己的API翻译成这些平台的测试工具能够执行的API。不同平台的详细情况参考[platform-support](http://appium.io/docs/en/about-appium/platform-support/index.html)。大体情况如下：

* iOS
    * 使用[XCUITest Driver](http://appium.io/docs/en/drivers/ios-xcuitest/index.html)或者[UIAutomation Driver](http://appium.io/docs/en/drivers/ios-uiautomation/index.html)(deprecated)
* android
    * 4.3版本之后是[UiAutomator2 Driver](http://appium.io/docs/en/drivers/android-uiautomator2/index.html)
    * [UiAutomator Driver](http://appium.io/docs/en/drivers/android-uiautomator/index.html)
* windows
    * [Windows Driver](http://appium.io/docs/en/drivers/windows/index.html)


## 基本概念

### Client/Server架构

Appium的核心是一个提供REST API的web服务器。它接受来自客户端的连接，接收命令，在移动设备上(或者windows系统中)执行命令并且把执行的结果返回给客户端。这样的好处是我们可以使用各种语言的客户端，我们也可以设置专门的服务器来运行Appium。更进一步，我们可以把运行Appium的服务器放到云端。


### Session

自动化的控制总是在一个Session的上下文中执行。不同客户端初始化Session的方法不尽相同，但是最终都是发送一个POST请求到/session，并且在请求体中发送一个"必须的能力"(desired capabilities)的JSON对象。收到请求后服务器会创建一个Session并且把session ID返回，从而后续的命令都可以用这个ID。

### Desired Capabilities

Desired Capabilities是客户端想服务器要求必须具备的"能力"。比如我们可以指定platformName为iOS来告诉Appium我们想创建一个iOS的Session。或者safariAllowPopups来在Safari浏览器中运行JavaScript打开新窗口。完整的Capabilities请参考[这里](http://appium.io/docs/en/writing-running-appium/caps/index.html)。

### Appium服务器

Appium服务器(或者简称Appium)使用Node.js实现的，我们可以从[源代码](http://appium.io/docs/en/contributing-to-appium/appium-from-source/index.html)或者使用npm安装。不会nodejs的同学也不用着急，我们可以使用同时打包了服务器并提供GUI的Appium Desktop，它在主流的系统下都有现成的二进制版本。

如果会npm的话安装非常简单：
```
$ npm install -g appium
```

安装后运行服务也非常简单：
```
$ appium
```

### Appium客户端

Appium支持多种语言(Java、Ruby、Python、PHP、JavaScript和C#)的客户端，完整列表参考[这里](http://appium.io/docs/en/about-appium/appium-clients/index.html)。

### Appium Desktop

[Appium Desktop](https://github.com/appium/appium-desktop)是一个Appium服务器的图形化界面版本。除了Appium服务器之外，它还集成了Inspector，这类似于Chrome里的开发者工具(当然没有那么强大)。利用它我们可以查看元素的信息从而可以方便我们定位。


## Get Started

本节介绍使用nodejs实现一个"Hello World"式的代码，如果对于nodejs完全不了解也不想了解的读者可以跳过本节。

### Node.js简介

根据[官方文档](https://nodejs.org/en/about/)，Node.js是一个异步的事件驱动的JavaScript库，用于构建可扩展的Web服务。和大家的印象中不同，Node.js不是前端代码，而通常是后台服务(前端马龙要抢后端马龙的饭碗?)。它和传统的Web框架(比如Spring或者Flask)很类似，但是不同点是它的事件驱动(异步)编程方式。比如下面的代码就实现了一个HTTP服务器，返回一个包含Hello World的网页：

```
const http = require('http');

const hostname = '127.0.0.1';
const port = 3000;

const server = http.createServer((req, res) => {
  res.statusCode = 200;
  res.setHeader('Content-Type', 'text/plain');
  res.end('Hello World');
});

server.listen(port, hostname, () => {
  console.log(`Server running at http://${hostname}:${port}/`);
});
```

http.createServer函数传入的参数是一个lambda(回调函数)，也就是服务器收到请求时会调用这个函数。

### 安装node和npm

读者可以去[这里](https://nodejs.org/en/download/)下载安装。我这里使用的是nvm来管理和安装node，具体可以参考[How to Install Node.js on Ubuntu and Update npm to the Latest Version](https://www.freecodecamp.org/news/how-to-install-node-js-on-ubuntu-and-update-npm-to-the-latest-version/)。下面是我安装的版本：

```
$ node -v
v14.15.0
$ which node
/home/lili/.nvm/versions/node/v14.15.0/bin/node
$ npm -v
6.14.8
$ which npm
/home/lili/.nvm/versions/node/v14.15.0/bin/npm
```

npm是nodejs的包管理器，类似于Python的pip或者Ruby的gem。安装node后会自动也带上npm。

### 安装appium

```
$ npm install -g appium
```

### 启动服务

```
$ appium
[Appium] Welcome to Appium v1.18.3
[Appium] Appium REST http interface listener started on 0.0.0.0:4723
```

启动后服务器就监听在4723端口了，接下来我们就可以编写客户端代码通过Appium服务器控制手机了。

### 创建nodejs客户端代码

```
$ mkdir ~/testappium
$ npm init -y
# 安装nodejs客户端
$ npm install webdriverio
```

在~/testappium下创建index.js，内容如下：
```
const wdio = require("webdriverio");
const assert = require("assert");
 
const opts = {
  path: '/wd/hub',
  port: 4723,
  capabilities: {
    platformName: "Android",
    platformVersion: "10",
    deviceName: "emulator-5554",
    app: "/home/lili/ApiDemos-debug.apk",
    appPackage: "io.appium.android.apis",
    appActivity: ".view.TextFields",
    automationName: "UiAutomator2"
  }
};
 
async function main () {
  const client = await wdio.remote(opts);
 
  const field = await client.$("android.widget.EditText");
  await field.setValue("Hello World!");
  const value = await field.getText();
  assert.equal(value,"Hello World!");
 
  await client.deleteSession();
}
 
main();
```

这里要控制的是ApiDemos-debug.apk，可以在[这里](https://github.com/appium/appium/tree/master/sample-code/apps)下载。需要修改的是platformVersion和deviceName，我这里使用的是安卓模拟器，如果是真机的话可以修改deviceName，或者删除这个值也行。加入它的目的是为了让Appium知道是哪个设备，因为我们同时连接的设备可能有多个。不知道设备有哪些的话可以用"adb devices"命令查看，如果是真机的话需要进入开发者模式并且运行USB调试。

**模拟器的platformVersion可能是代号，比如"P"，有些同学(比如作者)可以以为它一定要数字的版本号，就上网找到Android Pie对应的是9，结果发现连不上。**

**如果只有一个模拟器，那么可以不填deviceName。要看目前启动了哪些设备可以用"adb devices"命令查看**

### 运行

激动人心的时刻到了，我们就要运行helloworld了：
```
$ node index.js
```
赶紧盯着手机(或者模拟器)。

第一次运行时会在手机上安装Appium的驱动以及uiautomator，请允许安装(有的手机可能只能通过应用市场而不能通过apk安装，请参考网上文章允许apk安装)。

如果是第二次运行就会控制API Demos这个app，并且控制台会输出类似的日志(如果出现红的错误就要想办法解决)：
```
lili@lili-Precision-7720:~/testappium$ node index.js
2020-11-07T10:29:32.459Z INFO webdriverio: Initiate new session using the webdriver protocol
2020-11-07T10:29:32.462Z INFO webdriver: [POST] http://localhost:4723/wd/hub/session
2020-11-07T10:29:32.463Z INFO webdriver: DATA {
capabilities: {
alwaysMatch: {
platformName: 'Android',
platformVersion: '10',
deviceName: 'emulator-5554',
app: '/home/lili/ApiDemos-debug.apk',
appPackage: 'io.appium.android.apis',
appActivity: '.view.TextFields',
automationName: 'UiAutomator2'
},
firstMatch: [ {} ]
},
desiredCapabilities: {
platformName: 'Android',
platformVersion: '10',
deviceName: 'emulator-5554',
app: '/home/lili/ApiDemos-debug.apk',
appPackage: 'io.appium.android.apis',
appActivity: '.view.TextFields',
automationName: 'UiAutomator2'
}
}
2020-11-07T10:29:40.493Z INFO webdriver: COMMAND findElement("class name", "android.widget.EditText")
2020-11-07T10:29:40.493Z INFO webdriver: [POST] http://localhost:4723/wd/hub/session/586a8f49-c7b9-4264-b97c-a66e0b989a9d/element
2020-11-07T10:29:40.493Z INFO webdriver: DATA { using: 'class name', value: 'android.widget.EditText' }
2020-11-07T10:29:40.989Z INFO webdriver: RESULT {
'element-6066-11e4-a52e-4f735466cecf': 'b3846977-b968-41a2-a4de-0653b36ffcc6',
ELEMENT: 'b3846977-b968-41a2-a4de-0653b36ffcc6'
}
2020-11-07T10:29:41.008Z INFO webdriver: COMMAND elementClear("b3846977-b968-41a2-a4de-0653b36ffcc6")
2020-11-07T10:29:41.008Z INFO webdriver: [POST] http://localhost:4723/wd/hub/session/586a8f49-c7b9-4264-b97c-a66e0b989a9d/element/b3846977-b968-41a2-a4de-0653b36ffcc6/clear
2020-11-07T10:29:41.041Z INFO webdriver: COMMAND elementSendKeys("b3846977-b968-41a2-a4de-0653b36ffcc6", "Hello World!")
2020-11-07T10:29:41.041Z INFO webdriver: [POST] http://localhost:4723/wd/hub/session/586a8f49-c7b9-4264-b97c-a66e0b989a9d/element/b3846977-b968-41a2-a4de-0653b36ffcc6/value
2020-11-07T10:29:41.041Z INFO webdriver: DATA { text: 'Hello World!' }
2020-11-07T10:29:42.118Z INFO webdriver: COMMAND getElementText("b3846977-b968-41a2-a4de-0653b36ffcc6")
2020-11-07T10:29:42.120Z INFO webdriver: [GET] http://localhost:4723/wd/hub/session/586a8f49-c7b9-4264-b97c-a66e0b989a9d/element/b3846977-b968-41a2-a4de-0653b36ffcc6/text
2020-11-07T10:29:42.621Z INFO webdriver: RESULT Hello World!
2020-11-07T10:29:42.622Z INFO webdriver: COMMAND deleteSession()
2020-11-07T10:29:42.622Z INFO webdriver: [DELETE] http://localhost:4723/wd/hub/session/586a8f49-c7b9-4264-b97c-a66e0b989a9d
```

