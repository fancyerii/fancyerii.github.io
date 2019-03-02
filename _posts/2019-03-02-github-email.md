---
layout:     post
title:      "获取github账号的email的工具 "
author:     "lili"
excerpt_separator: <!--more-->
tags:
    - github
    - 工具
    - git
---


用时候我们需要联系github上某个开源软件的作者，但是我们是看不到作者的email地址。某些开发者会在简介里写上自己的email地址，但是更多的是没有。上网搜索到一个好的工具，能够方便的找到开发者的email。

<!--more-->

它的原理是提取作者向github提交代码时候用的email，也就是我们设置的邮箱。当然如果作者没有公开的commit(那联系他们有什么用呢？)，那么是查询不到的。
```
# 设置邮箱
$ git config --global user.email "email@example.com"
# 查看邮箱
$ git config --global user.email
```

github提供公开的api来查询某个用户的commit，我们可以自己去调用，但是不用自己造轮子了，[github-email](https://github.com/paulirish/github-email)就是这样一个项目。如果想自己调用也很简单，这个项目就是一个[脚本](https://github.com/paulirish/github-email/blob/master/github-email.sh)

**安装**
```
npm install --global github-email
```

**使用**

```
$ github-email fancyerii

Email on GitHub

Email on npm
null

Emails from recent commits
fancyerii@gmail.com

Emails from owned-repo recent activity
GitHub				    noreply@github.com
lili				    fancyerii@gmail.com
李理				    fancyerii@gmail.com
```
