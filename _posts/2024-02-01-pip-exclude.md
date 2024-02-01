---
layout:     post
title:      "pip安装去除某些依赖"
author:     "lili"
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - python
    - pip
    - pipenv
    - conda
---

有的时候我们在pip(或者pipenv)安装某个package时并不想安装某个(某些)依赖，但是pip install只有\-\-no-deps，也就是不按照全部依赖。本文介绍workaround的方法。

 <!--more-->

**目录**
* TOC
{:toc} 

## pip不安装依赖

使用：

```
pip install xxx --no-deps
``` 
就可以安装xxx而不安装其依赖。但有的时候我们只想去掉某一两个依赖。最简单的方法是安装后再自己去uninstall。但是这里存在一个问题，依赖本身也存在依赖，我们很难卸载干净。


## 找到某个package引入的依赖

可以使用pip show查看某个package的依赖，比如：

```
$ pip show accelerate
Name: accelerate
Version: 0.26.1
Summary: Accelerate
Home-page: https://github.com/huggingface/accelerate
Author: The HuggingFace team
Author-email: sylvain@huggingface.co
License: Apache
Location: /home/ubuntu/.local/share/virtualenvs/raydata-dorGUW-M/lib/python3.9/site-packages
Requires: huggingface-hub, numpy, packaging, psutil, pyyaml, safetensors, torch
Required-by: 
```

在Requires里我们可以看到accelerate的依赖。但是这里没有依赖的版本要求，我们可以写一段代码来提取。

```python
def find_dependencies2(package_name):
    from pip._vendor import pkg_resources
    package = pkg_resources.working_set.by_key[package_name]
    for dependency in package.requires():
        print(str(dependency)) 

```

比如运行调用find_dependencies2("accelerate")将会输出：

```
numpy>=1.17
packaging>=20.0
psutil
pyyaml
torch>=1.10.0
huggingface-hub
safetensors>=0.3.1
```

我们可以把这些内容保存在requirements.txt里，去掉我们不需要的依赖，然后用pip安装。

## 去掉警告

如果我们运行上面的代码，会输出如下警告：

```
DeprecationWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html
```

搜索了一下，我们应该使用更新的importlib.metadata，修改一下代码：

```python
def find_dependencies(package_name, exclude_extra=True):
    from importlib.metadata import requires
    from packaging.requirements import Requirement
    reqs = requires(package_name)
    for req in reqs:
        res = Requirement(req)
        if exclude_extra and res.marker:
            continue
        print(res)
```

## 关于Requirement

我们这里只关注package的使用，也就是[setup.py](https://github.com/huggingface/accelerate/blob/main/setup.py)里的install_requires，如果我们要开发的信息，可以调用find_dependencies("accelerate", False)，结果为：

```
numpy>=1.17
packaging>=20.0
psutil
pyyaml
torch>=1.10.0
huggingface-hub
safetensors>=0.3.1
black~=23.1; extra == "dev"
ruff>=0.0.241; extra == "dev"
hf-doc-builder>=0.3.0; extra == "dev"
urllib3<2.0.0; extra == "dev"
pytest; extra == "dev"
pytest-xdist; extra == "dev"
pytest-subtests; extra == "dev"
parameterized; extra == "dev"
datasets; extra == "dev"
evaluate; extra == "dev"
transformers; extra == "dev"
scipy; extra == "dev"
scikit-learn; extra == "dev"
deepspeed; extra == "dev"
tqdm; extra == "dev"
bitsandbytes; extra == "dev"
timm; extra == "dev"
rich; extra == "dev"
black~=23.1; extra == "quality"
ruff>=0.0.241; extra == "quality"
hf-doc-builder>=0.3.0; extra == "quality"
urllib3<2.0.0; extra == "quality"
rich; extra == "rich"
sagemaker; extra == "sagemaker"
datasets; extra == "test-dev"
evaluate; extra == "test-dev"
transformers; extra == "test-dev"
scipy; extra == "test-dev"
scikit-learn; extra == "test-dev"
deepspeed; extra == "test-dev"
tqdm; extra == "test-dev"
bitsandbytes; extra == "test-dev"
timm; extra == "test-dev"
pytest; extra == "test-prod"
pytest-xdist; extra == "test-prod"
pytest-subtests; extra == "test-prod"
parameterized; extra == "test-prod"
wandb; extra == "test-trackers"
comet-ml; extra == "test-trackers"
tensorboard; extra == "test-trackers"
dvclive; extra == "test-trackers"
pytest; extra == "testing"
pytest-xdist; extra == "testing"
pytest-subtests; extra == "testing"
parameterized; extra == "testing"
datasets; extra == "testing"
evaluate; extra == "testing"
transformers; extra == "testing"
scipy; extra == "testing"
scikit-learn; extra == "testing"
deepspeed; extra == "testing"
tqdm; extra == "testing"
bitsandbytes; extra == "testing"
timm; extra == "testing"
```

关于依赖的更多介绍可以参考[PEP 508 – Dependency specification for Python Software Packages](https://peps.python.org/pep-0508/)。


## pipenv不安装依赖

参考[pipenv-ignore-sub-dependency](https://stackoverflow.com/questions/53479677/pipenv-ignore-sub-dependency)。

设置.env，增加PIP_NO_DEPS=1，然后自己用pipenv安装依赖。


