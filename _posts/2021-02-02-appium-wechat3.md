---
layout:     post
title:      "使用Appium驱动手机版微信(三)" 
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

本文是使用Appium驱动手机版微信的系列文章的第三篇，详细介绍使用Appium时需要做的一些设置。

<!--more-->

**目录**
* TOC
{:toc}
 

本文要求读者有基本的Android开发能力，能够用Android Studio构建最简单的app。如果读者不知道什么是开发者选项或者一个app的packageName是什么含义，那么建议读者先找一本安卓开发的入门书阅读。入门书推荐[Head First Android Development: A Brain-Friendly Guide](https://www.amazon.com/Head-First-Android-Development-Brain-Friendly/dp/1449362184)；想深入了解的读者可以阅读[The Busy Coder's Guide to Android Development](https://commonsware.com/Android/)——不要被书名骗了，这本书的pdf有4300页！


## Appium Desktop简介

前面我们介绍了使用命令行启动Appium Server，然后通过客户端的程序与它交互让它控制一个Android app。本文介绍Appium Desktop，它是一个“客户端”的GUI程序。如果我们想要控制一个Android app做一些事情，我们需要做两件事：首先需要启动Appium Server；另外我们得定位这个app的某些空间(Control)，从而可以操控这个app(比如点击，拖放等等)。Appium可以同时帮我们做这两件事情：

* 通过Appium Desktop的GUI程序可以直接控制Appium Server的启动。
* 它提供Inspector来帮助我们定位一个app的控件。

我们实际控制或者测试的时候通常使用编程(脚本)的方式，所以第一个功能没那么重要，重要的是它提供的Inspector。

## 下载和运行Appium Desktop

读者可以去[这里](https://github.com/appium/appium-desktop/releases)下载最新版本的Appium Desktop。不同的操作系统需要下载不同的二进制版本，作者是Linux系统，所以下载的是Appium-linux-{VERSION}.AppImage。下载后增加执行权限就可以运行：
```
$ chmod u+x Appium-linux-{VERSION}.AppImage
$ ./Appium-linux-{VERSION}.AppImage
```
注意：对于Windows用户来说下载的是exe，可能需要安装才能使用。而AppImage是不需要安装的。

运行后如下图所示：

<a name='img1'>![](/img/appium/1.png)</a>
*图：Appium Desktop的启动界面*

点击"Start Server {VERSION}"，就会启动Appium Server，出现如下图的日志信息：


<a name='img2'>![](/img/appium/2.png)</a>
*图：Appium Desktop启动服务*

"Appium REST http interface listener started on 0.0.0.0:4723"的日志说明服务器成功的监听在了4723端口。

要关闭服务器可以点击右侧像两条竖线的图标。会出现"Appium server stopped successfully"的信息，我们点X就可以回到启动界面重新启动或者做一些其它的操作。


## 使用Appium Desktop查看app

为了查看某个app的特点控件的信息，我们可以使用Appium Desktop的Inspector，点击右上角放大镜的按钮，会弹出如下的对话框。我们需要在"Desired Capabilities"里填入一些基本信息，读者可以参考下图根据自己的手机系统版本填写：

<a name='img4'>![](/img/appium/4.png)</a>
*图：Desired Capabilities*

为了避免重复填写，可以"Save As..."保存下来。点击"Start Session"就可以开始Inspect。注意：启动的时间可能很长。另外可以用"adb devices"命令确保开发者选项是正确设置的。关于开发者选项的设置请参考网络上的相关资源，本文不会赘述。

如果一切顺利，会出现一个三分屏，最左边的屏幕就是手机的屏幕的镜像。我们可以操作手机打开微信(后面会介绍怎么编程控制启动一个app)。然后点击像一个圆圈(但是不封闭)的"Refreshing and ScreenShot"按钮，这会花费很长时间，最终得到类似下面的结果：


<a name='img3'>![](/img/appium/3.png)</a>
*图：微信的ScreenShot*

我们可以在手机上选中微信的搜索按钮(像个放大镜)，因为我们后面会模拟人类的操作来搜索一个好友。选中后右边会出现"Selected Element"，显示这个搜索按钮的信息。其中最重要的是ID——"com.tencent.mm:id/f8y"，这是用于控件定位最方便准确的方法。下面也有XPath，但是对于Android app的控制，使用控件id更快更简单。另外往下滚动(不知道是不是Linux的问题，滚动条没显示，只能用键盘的方向键控制滚动)，可以看到这个控件的class是"android.widget.ImageView"。

说明：不知道为什么用Appium Desktop的Inspector会这么慢，所以作者一般会使用Android SDK自带的Automator Viewer工具。后面会介绍这个工具。

## 查找一个app的packageName

很多开发者可能会奇怪，这都算一个问题吗？我们不应该知道app的package名字吗？当然如果某个app是你自己开发的，你当然会知道它的名字。但是对于一个第三方的app，可能就是个问题了。如果我们有这个app的apk，有一些工具可以帮我们分析其packageName。但是更多的时候，我们是通过应用市场下载的，可能不知道apk放在哪里。那怎么办呢？

作者上网搜索了很久，其中一个最常见的方法是：把要查找的app启动并且放置于前台运行，然后执行：
```
adb shell "dumpsys window windows | grep -E 'mCurrentFocus'"
```
但是作者尝试了很多次都不行，也许是早期的android版本可以，但是最新的android X是不行的。通过不断搜索和尝试，目前作者找到了一种可行的方法。

* 使用top找到packageName

可以把这个app运行到前台，用top看cpu，然后猜。比如我们启动微信，然后通过"adb shell"后运行top，可以发现如下的结果：


<a name='img5'>![](/img/appium/5.png)</a>
*图：在adb shell里的top*

通过观察，我们猜测微信的packageName是"com.tencent.mm"。这里需要一些世界知识，比如我们知道腾讯的英文名字叫"tencent"，从而猜测"com.tencent.mm"是微信。但是万一微信非常省cpu呢？top看不出来怎么办呢？你当然可以让微信做一些费cpu的事情，比如扫描个二维码。但是这个方法不能"自动"的获取packageName，因为top排在第一的经常变，而且即使不变我们也不能100%确定最费cpu的就是前台的app。说不定有一个木马app一直在后台消耗你的cpu呢？不过除此之外作者没有找到能够work的其它方法，如果读者有更好的方法请留言告诉我。

* 找到Main Activity

对于android app来说还得找到这个app的"主页面的Activity"，这可以通过如下命令来找到：

```
$ adb shell dumpsys activity activities | sed -En -e '/Running activities/,/Run #0/p'
    Running activities (most recent first):
      TaskRecord{b4cdb56 #31516 A=com.tencent.mm U=0 StackId=124 sz=1}
        Run #0: ActivityRecord{b8d8c41 u0 com.tencent.mm/.ui.LauncherUI t31516}
    Running activities (most recent first):
      TaskRecord{b4cda09 #31392 A=com.huawei.android.launcher U=0 StackId=0 sz=1}
        Run #0: ActivityRecord{b7a5c09 u0 com.huawei.android.launcher/.unihome.UniHomeLauncher t31392}
....

```

我们把微信的启动UI运行在前台，然后执行上面的命令，上面的命令会把任务Stack打印出来，我们就能发现微信的Main Activity是"com.tencent.mm.ui.LauncherUI"。注意：我们在写Manifest.xml时如果ActivityName的前缀是packageName的话，可以用点开头省略。所以"com.tencent.mm/.ui.LauncherUI"表示完整的Java Package是"com.tencent.mm.ui.LauncherUI"。


