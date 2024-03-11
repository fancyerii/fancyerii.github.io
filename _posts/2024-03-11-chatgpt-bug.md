---
layout:     post
title:      "ChatGPT发送消息后没有响应的bug" 
author:     "lili" 
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - openai
    - chatgpt
---

今天使用ChatGPT时突然不工作了，点击发送消息按钮或者回车后右侧图标一直旋转，但是消息并没有发生出去，也没有响应。

<!--more-->

**目录**
* TOC
{:toc}

## 问题

今天使用ChatGPT时突然不工作了，点击发送消息按钮或者回车后右侧图标一直旋转，但是消息并没有发生出去，也没有响应。如下图所示：

<a>![](/img/chatgpt-bug.png)</a>

 
## 解决方法

查看了chrome的控制台错误，出现“Uncaught (in promise)”的js错误，如下图所示：

 <a>![](/img/chatgpt-bug2.png)</a>

上网搜索了一下，找到了三个相关的文章：[It disappears when send a message in the browser](https://community.openai.com/t/it-disappears-when-send-a-message-in-the-browser/671885)，[Cannot send question to chatgpt on web](https://community.openai.com/t/cannot-send-question-to-chatgpt-on-web/674723)和[Can’t sent message to ChatGPT (March 2024)](https://community.openai.com/t/cant-sent-message-to-chatgpt-march-2024/675219)。

根据大家的讨论，我尝试了一下把语言从中文设置为自动检查(auto detect)，重新打开就好了。

 <a>![](/img/chatgpt-bug3.png)</a>
