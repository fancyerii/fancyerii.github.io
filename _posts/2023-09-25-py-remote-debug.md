---
layout:     post
title:      "VSCode远程调试Python" 
author:     "lili" 
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - VS Code
    - 远程调试
    -  debugpy 
---

本文介绍用VSCode和debugpy远程调试Python代码。

<!--more-->

**目录**
* TOC
{:toc}

## 背景

我们在开发时经常需要调试，对Python来说，PyCharm是一个不错的IDE。但是我们在开发深度学习算法时，很多时候我们本地的开发机由于资源的限制，往往在本地加载模型。比如加载Llama-2 7B的模型做Lora的微调都可能需要十多G的显存。这个时候我们只能在服务器上运行代码，这就带来一个问题：怎么远程调试Python代码？

一种方式是使用Jupyter Notebook，使用print，另外也可以使用VSCode对Notebook进行远程调试，读者可以参考[用Docker、Jupyter notebook和VSCode搭建深度学习开发环境——用vscode调试](/2022/10/19/docker-jupyter/#%E7%94%A8vscode%E8%B0%83%E8%AF%95)。

不过Notebook只是适合一些简单的代码片段开发，开发复杂的功能还是需要模块化的复杂的python代码。而且我们找到的很多开源代码也都是这样的代码，我们阅读时想深入理解也希望能够单步调试它们。所以我们还是需要远程调试python代码。本文就是作者需要阅读[llama-recipes](https://github.com/facebookresearch/llama-recipes)的finetuning.py，通过VSCode建立远程调试的过程。


## 操作步骤

本文主要参考[How to debug remote Python script in VS Code](https://stackoverflow.com/questions/73378057/how-to-debug-remote-python-script-in-vs-code)

### 安装Remote ssh extension

在vscode的extenstion marketplace里搜索"remote ssh"即可找到微软官方的插件：

<a name='img1'>![](/img/vscode/1.png)</a>


### ssh连接远程服务器

安装好了插件之后，在vscode的左下角有一个绿色的小按钮(参考下图)，然后点击之后选择"Connect to Host"，之后输入"user@host"，user为用户名，host为服务器域名或者ip地址。为了避免重复输入密码，建议配置[SSH login without password](http://www.linuxproblem.org/art_9.html)。

<a name='img2'>![](/img/vscode/2.png)</a>

### 连接远程服务器

点击连接服务器之后，我们可以通过"Open Folder"打开我们的代码，比如我这里选择打开"llama-recipes/src/"。这样我们就可以浏览和编辑位于远程服务器上的代码了。

<a name='img3'>![](/img/vscode/3.png)</a>

### 安装debugpy

在远程服务器和本地开发机都安装[debugpy](https://github.com/microsoft/debugpy)：

```
pip install debugpy
```

### 服务器启动脚本

接下来是启动要调试的代码，比如我们要调试src/llama_recipes/finetuning.py，在调试之前确保程序可以正常启动。常见的问题是需要把代码的路径通过PYTHONPATH环境变量加到解析器的搜索路径中。

```
python -Xfrozen_modules=off -m debugpy  --listen 0.0.0.0:5678 --wait-for-client src/llama_recipes/finetuning.py --use_peft --peft_method lora --quantization .....
```

我上面是让远程调试服务监听在所有ip的5678端口上，请读者修改，确保本地开发机可以连接到这个ip上。

### vscode远程调试

Run->Start debugging->Python: Remote Attach，填写正确的ip和端口。我的json文件如下：

```
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Remote Attach",
            "type": "python",
            "request": "attach",
            "connect": {
                "host": "xxxxxx",
                "port": 5678
            },
            // "pathMappings": [
            //     {
            //         "localRoot": "${workspaceFolder}",
            //         "remoteRoot": "."
            //     }
            // ],
            "justMyCode": true
        }
    ]
}
```
注意：默认会配置pathMappings。我发现有了这个之后无法调试，把它注释后就好了。

最后就是增加断点，启动远程调试，结果如下图：

<a name='img4'>![](/img/vscode/4.png)</a>

 
