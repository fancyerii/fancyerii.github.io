---
layout:     post
title:      "使用WebDriver驱动网页版微信(三)" 
author:     "lili" 
mathjax: true
excerpt_separator: <!--more-->
tags:
    - 微信
    - wechat
    - automation
    - selenium
    - webdriver
---

本文是Selenium WebDriver来自动化微信进行消息收发系列文章之三，介绍用WebDriver来去掉微信网页客户端实现发送消息。

<!--more-->

**目录**
* TOC
{:toc}
 

终于到了正题了！有了前面的基础，正文的内容就会很简单明了。完整的代码在[这里](https://github.com/fancyerii/blog-codes/blob/master/webdriver-wechat/src/main/java/com/github/fancyerii/wechatdriver/WechatDriver.java)下载。


## 配置WebDriver

代码如下，请根据自己Driver的设置合适的路径，如果不清楚的读者请阅读本系列文章第一篇。

```
ChromeOptions options = new ChromeOptions();
WebDriver driver=new ChromeDriver(options);
System.setProperty("webdriver.chrome.driver", "/home/lili/soft/chromedriver");
WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(30).getSeconds());
```

## 扫描登录

打开网页微信后会出现如下登录页面，需要用手机扫描二维码登录。

<a name='img28'>![](/img/wechat/28.png)</a>

## 定位二维码图片

我们首先要用XPath定位这个图片，我们这里使用的是"//DIV[@class='qrcode']/IMG"。如果对于XPath还不熟悉的读者请阅读本系列文章第二篇。找到这个图片后我们可以截屏保存为文件：

```
driver.get("http://wx.qq.com/");
WebElement qrImg=wait.until(presenceOfElementLocated(By.xpath("//DIV[@class='qrcode']/IMG")));

byte[] bytes=qrImg.getScreenshotAs(OutputType.BYTES);
```

但是二维码图片是异步加载的，而且定期会刷新，比如第一次下载的是这个图片：

<a name='img29'>![](/img/wechat/29.png)</a>

我们需要有一种办法来判断这是不是二维码。当然真正要实现一个二维码的识别器比较复杂，但是我们可以用一些非常简单的图像处理方法来识别。通过图像的计算来驱动一个程序，这是非常常见的，人类也是一样的工作原理。不过目前的计算机视觉技术还远远没有那么成熟，人在使用一个软件的过程中会有各种决策，比如网络断了，会有文本或者图标的变化，人可以知道原因并且刷新。后面我们介绍桌面驱动app时也会介绍怎么用简单的图像识别的方法来模拟人的决策。也许有读者会想用更加复杂的深度学习之类的方法，但是实现成本很高，我们这里一般只需要像素级别的匹配算法，因为我们假设同一个软件在同一个系统中渲染出来的结果应该是完全不变的，不需要模糊匹配的算法。


我们仔细分析一下上面的图片，这个图片大部分像素都是灰色的背景。如果我们做一下颜色直方图或者简单用图片编辑软件取色就会发现背景色是rgb(204,204,204)，所以我们可以写一个很简单的工具判断这个图片是背景图还是二维码，完整代码在[这里](https://github.com/fancyerii/blog-codes/blob/master/webdriver-wechat/src/main/java/com/github/fancyerii/wechatdriver/ImageTools.java)。

```
    private static final int BACKGROUND_R = 204;
    private static final int BACKGROUND_G = 204;
    private static final int BACKGROUND_B = 204;
    private static final double THRESHOLD = 0.8;

    public static boolean isQRCodeImage(byte[] bytes) throws IOException {
        InputStream is = new ByteArrayInputStream(bytes);
        BufferedImage image = ImageIO.read(is);
        int width = image.getWidth();
        int height = image.getHeight();
        int bgCount = 0;
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                int color = image.getRGB(col, row);
                int r = (color >> 16) & 0xFF;
                int g = (color >> 8) & 0xFF;
                int b = (color >> 0) & 0xFF;
                if (r == BACKGROUND_R && g == BACKGROUND_G && b == BACKGROUND_B) {
                    bgCount++;
                }
            }
        }
        if (1.0 * bgCount / (width * height) > THRESHOLD) {
            return false;
        } else {
            return true;
        }
    }
```

代码很简单，也就是数一下图中背景色的像素个数，然后和图片总的像素比一下，如果大于阈值就认为不是二维码。

接下来的代码就是等待浏览器加载二维码：

```
    byte[] bytes=qrImg.getScreenshotAs(OutputType.BYTES);
    int tryCount=0;
    for(;tryCount<10;tryCount++){
        if(ImageTools.isQRCodeImage(bytes)){
            break;
        }
        try {
            Thread.sleep(1000);
        } catch (InterruptedException e) {
        }
        bytes=qrImg.getScreenshotAs(OutputType.BYTES);
    }
    if(tryCount == 10){
        throw new RuntimeException("找不到二维码");
    }
```

也就是重试十次，如果十次都不是二维码，则抛出异常。

## 等待登录

有了二维码图片，我们应该把二维码发送给微信号的拥有者去让他扫描登录。如果是一个完整的解决方案，我们应该发送一封邮件给对方，邮件里附上二维码图片。但是二维码会定时失效和更新，所以更好的办法是发送一个链接给对方，然后我们定时更新二维码，对应的链接指向我们的二维码图片。这里演示起见只是在终端输出二维码图片的位置。

那怎么知道用户扫描登录了呢？因为扫描后会跳到新的页面，新的页面没有二维码图片，所以我们可以根据是否有二维码图片来判断是否扫描了。代码如下：

```
    //TODO 把二维码发送给人扫描登录
    //这里只是把二维码保存下来，实际上二维码是有有效期的，需要定期刷新。
    File tmpFile=File.createTempFile("tmp-qr", ".png");
    Files.write(tmpFile.toPath(), bytes);
    System.out.println("请扫描 "+tmpFile.getAbsolutePath()+" 登录");

    //检查是否找不到二维码从而判断已经扫码
    while(true){
        qrImg=findElement(driver, By.xpath("//DIV[@class='qrcode']/IMG"));
        if(qrImg==null) break;
        byte[] newImg=qrImg.getScreenshotAs(OutputType.BYTES);
        if(!Arrays.equals(bytes, newImg)){
            bytes=newImg;
            Files.write(tmpFile.toPath(), bytes);
        }
        System.out.println("请扫描 "+tmpFile.getAbsolutePath()+" 登录");
        try {
            Thread.sleep(30000);
        } catch (InterruptedException e) {
        }
    }
```

## 判断是否登录成功

因为网页版微信只有老的账号才能登录，新的账号登录后会提示：
```
<error><ret>1203</ret><message>为了你的帐号安全，此微信号已不允许登录网页微信。你可以使用Windows微信或Mac微信在电脑端登录。Windows微信下载地址：https://pc.weixin.qq.com  Mac微信下载地址：https://mac.weixin.qq.com</message></error>
```

所以我们可以通过这个来判断这个账号是否运行登录：

```
    //判断是否被禁用
    String body=findElement(driver,By.xpath("//BODY")).getAttribute("innerHTML");
    if(body.contains("为了你的帐号安全，此微信号已不允许登录网页微信。")){
        throw new RuntimeException("您的账号不能用网页版登录");
    }
```

## 点击搜索按钮

为了给某个好友发送消息，我们需要点击搜索按钮，然后输入名字。我们可以通过"//INPUT[@placeholder='搜索']"这个XPath定位到搜索框：

```
WebElement searchInput=wait.until(presenceOfElementLocated(By.xpath("//INPUT[@placeholder='搜索']")));
searchInput.sendKeys("文件传输助手");
```

搜索框下面会弹出一个好友的名字，如下图所示：


<a name='img6'>![](/img/wechat/6.png)</a>

我们也要通过XPath找到这个元素并点击：

```
WebElement searchResult=wait.until(presenceOfElementLocated(By.xpath("//DIV[@class='info']/H4[text()='文件传输助手']")));
searchResult.click();
```

## 找到文本框输入并点击发送按钮

接着我们需要在文本框输入文字，注意前面点击操作之后文本框已经自动获得焦点(active)，如下图所示：

<a name='img7'>![](/img/wechat/7.png)</a>


代码如下：

```
WebElement editDiv=wait.until(presenceOfElementLocated(By.xpath("//PRE[@id='editArea']")));
editDiv.sendKeys("明天有空吗？");

WebElement sendButton=wait.until(presenceOfElementLocated(By.xpath("//A[@class='btn btn_send']")));
sendButton.click();
```

这样就实现了给好友发送消息的功能！大家也可以试一下怎么收取某个好友的消息。

## 调试

看起来代码非常简单？实际并没有那么简单，为了写上面的程序，我起码花了一天的实际。大部分时间都花在调试XPath上面了。比如发送消息的文本框PRE是包含在某个DIV中的，我之前定位到这个DIV，结果每次发送都异常。因为Java没有Python那样的REPL的功能，每次修改代码都得重新启动浏览器和登录微信。这不但麻烦，而且行为异常，有被微信封号的危险(再次申明一下：请勿使用此技术做非法商业活动！)。

为了让调试简单，我写了一个简单的[调试工具](https://github.com/fancyerii/blog-codes/blob/master/webdriver-wechat/src/main/java/com/github/fancyerii/wechatdriver/DebugXPath.java)。

这个代码会启动浏览器，然后你就可以用浏览器访问任何网站。如果你想测试XPath，则可以输入：
```
eval:[XPATH]比如：
eval://PRE[@id='editArea']
```

如果想点击某个元素，可以输入：
```
click://A[@class='btn btn_send']
```

如果想给某个文本框发送文本，可以输入：
```
sendtxt://PRE[@id='editArea']###测试消息
```

如果XPath有异常，也会printstack，结合chrome的开发者工具和chropath等，应该可以让调试更加简单轻松一点。



