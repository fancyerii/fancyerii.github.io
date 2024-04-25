---
layout:     post
title:      "Introducing Rust" 
author:     "lili" 
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - Rust
---

这一章涵盖了以下内容：

* 介绍 Rust 的特性和目标
* 揭示 Rust 的语法
* 讨论何时使用 Rust 以及何时避免使用它
* 构建你的第一个 Rust 程序
* 解释 Rust 与面向对象和更广泛语言的比较

<!--more-->

**Contents**
* TOC
{:toc}

欢迎来到 Rust —— 一门具有赋能性的编程语言。一旦你深入了解，你会发现这不仅是一门速度和安全性无与伦比的编程语言，更是一门每天都愉快使用的语言。

当你开始用 Rust 编程时，你很可能会想继续这样做。而这本书，《Rust 实战》，将会建立你作为 Rust 程序员的信心。但它并不会教你如何从零开始编程。这本书是为那些考虑将 Rust 作为下一门语言的人阅读的，也适合那些喜欢实现实用工作示例的人。以下是本书包含的一些较大的示例列表：

* Mandelbrot 集合渲染器
* 一个 grep 克隆
* CPU 模拟器
* 生成艺术
* 一个数据库
* HTTP、NTP 和十六进制转储客户端
* LOGO 语言解释器
* 操作系统内核

通过浏览这个列表，你可以了解到本书不仅仅教你 Rust，还介绍了系统编程和低级编程。在《Rust 实战》中，你将学习到操作系统的作用，CPU 的工作原理，计算机如何计时，什么是指针，以及什么是数据类型。你将获得对计算机内部系统如何相互配合的理解。除了语法之外，你还将了解 Rust 的诞生原因以及它解决的挑战。

## 1.1 Rust 被用在哪里？

Rust 自 2016 年至 2020 年连续五年获得了 Stack Overflow 年度开发者调查中的“最受喜爱编程语言”奖项。或许这就是为什么一些大型科技领导者采用了 Rust：

* 亚马逊网络服务（AWS）自 2017 年起就在其无服务器计算产品 AWS Lambda 和 AWS Fargate 中使用 Rust。通过这样做，Rust 已经在更多领域取得了进展。该公司编写了 Bottlerocket 操作系统和 AWS Nitro 系统，以提供其弹性计算云（EC2）服务。

* Cloudflare 开发了许多服务，包括其公共 DNS、无服务器计算和数据包检测，都是用 Rust 实现的。

* Dropbox 用 Rust 重新构建了管理数百亿存储的后端仓库。

* 谷歌使用 Rust 开发了 Android 的部分组件，比如其蓝牙模块。Rust 还用于 Chrome OS 的 crosvm 组件，并在谷歌的新操作系统 Fuchsia 中发挥着重要作用。

* Facebook 使用 Rust 来支持 Facebook 的网络、移动和 API 服务，以及 Hack 编程语言所使用的 HHVM（HipHop 虚拟机）的部分组件。

* 微软在 Rust 中编写了 Azure 平台的一些组件，包括为其物联网（IoT）服务编写的安全守护程序。

* Mozilla 使用 Rust 来增强 Firefox 网页浏览器，该浏览器包含 1500 万行代码。Mozilla 的前两个 Rust-in-Firefox 项目，其 MP4 元数据解析器和文本编码/解码器，带来了整体性能和稳定性的改进。

* GitHub 的 npm 公司使用 Rust 每天提供“超过 13 亿次软件包下载”。

* Oracle 使用 Rust 开发了一个容器运行时，以解决 Go 参考实现中出现的问题。

* 三星通过其子公司 SmartThings，在其物联网（IoT）服务的固件后端中使用 Rust。

Rust 在快速发展的初创公司中也很受欢迎。以下是一些例子：

* Sourcegraph 使用 Rust 来提供所有语言的语法高亮显示。
* Figma 在其多人游戏服务器的性能关键组件中使用 Rust。
* Parity 使用 Rust 开发其 Ethereum 区块链的客户端。

## 1.2 在工作中倡导 Rust 有什么感受？

在工作中倡导 Rust 是什么感觉？克服了最初的障碍之后，通常会顺利进行。以下是一段重印的 2017 年的讨论，提供了一个很好的插曲。谷歌 Chrome OS 团队的一名成员讨论了向项目引入这种语言的经历：

```
indy on Sept 27, 2017
Rust在Google是否是官方认可的语言？
zaxcellent on Sept 27, 2017
作者在这里：Rust在Google并不是官方认可的，但这里有一些人在使用它。在这个组件中使用Rust的技巧是说服我的同事没有其他语言适合这项工作，我认为在这种情况下是合适的。
话虽如此，在Chrome OS构建环境中使用Rust还需要大量工作。Rust的人员在回答我的问题方面非常有帮助。
ekidd on Sept 27, 2017

在这个组件中使用Rust的技巧是说服我的同事没有其他语言适合这项工作，我认为在这种情况下是合适的。
在我自己的项目中我遇到了类似的用例——一个vobsub字幕解码器，它解析复杂的二进制数据，我希望有一天能够将其作为网络服务运行。所以显然，我希望确保我的代码中没有漏洞。
我用Rust编写了代码，然后使用'cargo fuzz'来寻找漏洞。在运行了10亿次模糊迭代之后，我发现了5个bug（查看'trophy-case'中的'vobsub'部分以查看列表https://github.com/rust-fuzz/trophy-case）。
令人高兴的是，这些bug中没有一个实际上可以升级为实际的利用漏洞。在每种情况下，Rust的各种运行时检查都成功地捕捉到了问题并将其转化为受控的panic。（在实践中，这将会干净地重新启动web服务器。）
所以我的结论是，每当我需要一种（1）没有GC，但是（2）在安全关键环境中可以信任的语言时，Rust是一个很好的选择。像Go一样，我可以静态链接Linux二进制文件的事实是一个不错的加分项。
Manishearth on Sept 27, 2017
令人高兴的是，这些bug中没有一个实际上可以升级为实际的利用漏洞。在每种情况下，Rust的各种运行时检查都成功地捕捉到了问题并将其转化为受控的panic。
这在我们在firefox中对Rust代码进行模糊测试的经验中基本上是我们的体验，顺便说一下。模糊测试发现了很多panic（和调试断言/“安全”溢出断言）。在一个案例中，它实际上发现了一个bug，在类似的Gecko代码中已经存在了大约十年。
```


从这段摘录中，我们可以看到语言的采用是由工程师们自下而上地寻求在相对小型项目中克服技术挑战。从这些成功中获得的经验然后被用作证据，来证明承担更加雄心勃勃的工作是合理的。

自 2017 年末以来，Rust 一直在不断成熟和加强。它已经成为谷歌技术领域的一个被接受的部分，并且现在是 Android 和 Fuchsia 操作系统中的官方语言。
 

## 1.3 体验这种语言

这一部分让你有机会第一手体验 Rust。它演示了如何使用编译器，然后进入编写一个快速程序的步骤。我们将在后面的章节中处理完整的项目。

注意：要安装 Rust，请使用官方提供的安装程序 https://rustup.rs/。 
 
 
### 1.3.1 作弊式实现“Hello, world！”

```shell
$ cargo new hello
$ cd hello
$ cargo run
```
如果你已经成功运行到这一步，太棒了！你已经在不需要编写任何Rust代码的情况下运行了你的第一个Rust代码。让我们看看刚刚发生了什么。
Rust的cargo工具既提供了构建系统，也提供了包管理器。这意味着cargo知道如何将你的Rust代码转换为可执行的二进制文件，并且还可以管理下载和编译项目依赖的过程。

cargo new为你创建了一个遵循标准模板的项目。tree命令可以展示默认项目结构以及在执行cargo new后创建的文件。

```
$ tree hello
hello
├── Cargo.toml
└── src
└── main.rs
```

所有使用cargo创建的Rust项目都具有相同的结构。在基本目录中，一个名为Cargo.toml的文件描述了项目的元数据，如项目的名称、版本和依赖项。源代码位于src目录中。Rust源代码文件使用.rs文件扩展名。要查看cargo new创建的文件，请使用tree命令。
接下来执行的命令是cargo run。这行命令更容易理解，但实际上cargo做了比你意识到的更多的工作。你要求cargo运行项目。由于在调用命令时实际上没有什么可运行的内容，它决定代表你以调试模式编译代码，以提供最大的错误信息。恰巧，src/main.rs文件始终包含一个“Hello, world!”的桩代码。编译的结果是一个名为hello（或hello.exe）的文件。hello文件被执行，并且结果打印到屏幕上。
执行cargo run还向项目中添加了新文件。我们现在在项目的根目录下有一个Cargo.lock文件和一个target/目录。这两个文件和目录都由cargo管理。由于这些是编译过程的产物，我们不需要触碰它们。Cargo.lock是一个文件，它指定了所有依赖项的确切版本号，以便将来的构建可靠地按照相同的方式构建，直到修改Cargo.toml文件。
再次运行tree命令可以查看通过调用cargo run来编译hello项目所创建的新结构：

```
$ tree --dirsfirst hello
hello
├── src
│   └── main.rs
├── target
│   ├── debug
│   │   ├── build
│   │   ├── deps
│   │   │   ├── hello-679618e7e46e5378
│   │   │   └── hello-679618e7e46e5378.d
│   │   ├── examples
│   │   ├── incremental
│   │   │   └── hello-o432id62x3tf
│   │   │       ├── s-guv1p80hv8-6ruaz6-b3x0jm9rvmumlibf75se6xr78
│   │   │       │   ├── 15hr9fzzjnx6yscm.o
│   │   │       │   ├── 19m7nsq1z931d55e.o
│   │   │       │   ├── 43ucpmwzd68dp7ys.o
│   │   │       │   ├── 4gzefv9jd74gbjjp.o
│   │   │       │   ├── 5dhsmm86nsbshyuv.o
│   │   │       │   ├── cfqeecaiqm2tw6m.o
│   │   │       │   ├── dep-graph.bin
│   │   │       │   ├── query-cache.bin
│   │   │       │   └── work-products.bin
│   │   │       └── s-guv1p80hv8-6ruaz6.lock
│   │   ├── hello
│   │   └── hello.d
│   └── CACHEDIR.TAG
├── Cargo.lock
└── Cargo.toml
```

恭喜你成功启动项目！现在我们通过一种更长的方式来实现“Hello, World!”。

### 1.3.2 你的第一个Rust程序
对于我们的第一个程序，我们想要编写一个能够以多种语言输出以下文本的程序：

```
Hello, world!
Grüß Gott!
ハロー・ワールド
```

你可能在旅途中见过第一行。其他两行是为了突出Rust的一些特性：简单的迭代和内置的Unicode支持。对于这个程序，我们将像以前一样使用cargo来创建它。以下是要遵循的步骤：

* 打开控制台提示符。
* 在MS Windows上运行cd %TMP%; 在其他操作系统上运行cd $TMP。
* 运行cargo new hello2来创建一个新项目。
* 运行cd hello2来进入项目的根目录。
* 在文本编辑器中打开src/main.rs文件。
* 将该文件中的文本替换为列表1.1中的文本。

以下列表中的代码在源代码库中。打开ch1/ch1-hello2/src/hello2.rs。

```rust
fn greet_world() {
    println!("Hello, world!");     // <1>

    let southern_germany = "Grüß Gott!";         // <2>
    let japan = "ハロー・ワールド";                // <3>

    let regions = [southern_germany, japan];     // <4>

    for region in regions.iter() {               // <5>
            println!("{}", &region);             // <6>
    }
}

fn main() {
    greet_world();                               // <7>
}
```

现在src/main.rs已经更新，请在hello2/目录中执行cargo run。在一些由cargo自身生成的输出之后，您应该会看到三个问候语出现：

```bash
$ cargo run
Compiling hello2 v0.1.0 (/path/to/ch1/ch1-hello2)
Finished dev [unoptimized + debuginfo] target(s) in 0.95s
Running `target/debug/hello2`
Hello, world!
Grüß Gott!
ハロー・ワールド
```

让我们花点时间来介绍一下列表1.2中Rust的一些有趣元素。

您可能首先注意到的一件事是，Rust中的字符串可以包含各种字符。字符串保证以UTF-8编码。这意味着您可以相对轻松地使用非英语语言。
可能看起来不合适的一个字符是println后面的感叹号。如果您有Ruby编程经验，可能会习惯于认为这表示破坏性操作。在Rust中，它表示使用宏。宏现在可以被看作是一种高级函数。它们提供了避免样板代码的能力。在println!的情况下，底层进行了大量的类型检测，以便将任意数据类型打印到屏幕上。

## 1.4  下载本书的源代码

为了跟随本书中的示例，您可能想要访问列表的源代码。为了您的方便，每个示例的源代码都可以从以下两个来源获取：

* https://manning.com/books/rust-in-action
* https://github.com/rust-in-action/code

## 1.5 Rust的外观和感觉是怎样的？

Rust是一种让Haskell和Java程序员感到舒适的编程语言。Rust在接近高级、富有表现力的动态语言（如Haskell和Java）的同时，实现了低级、裸金属的性能。

我们在1.3节看了一些“Hello, world！”的例子，所以让我们尝试一些稍微复杂一点的东西，以更好地感受Rust的特性。列表1.2快速展示了Rust在基本文本处理方面的能力。这个列表的源代码在ch1/ch1-penguins/src/main.rs文件中。一些需要注意的特性包括：

* 分割记录为字段
* 构建字段的集合
* 常见的控制流机制 — 包括for循环和continue关键字。
* 方法语法 — 虽然Rust不是面向对象的，因为它不支持继承，但它继承了面向对象语言的这个特性。
* 高阶编程 — 函数既可以接受函数，也可以返回函数。例如，第19行（.map(\|field\| field.trim())）包含一个闭包，也称为匿名函数或lambda函数。
* 类型注解 — 虽然相对较少，但有时需要作为对编译器的提示（例如，看到第27行以 if let Ok(length)开头的地方）。
* 条件编译 — 在列表中，21-24行（如果 cfg!(...);）在程序的发布版本中不包括。
* 隐式返回 — Rust提供了return关键字，但通常会省略。Rust是一种基于表达式的语言。

```rust
fn main() {                 // <1> <2>
  let penguin_data = "\
  common name,length (cm)
  Little penguin,33
  Yellow-eyed penguin,65
  Fiordland penguin,60
  Invalid,data
  ";

  let records = penguin_data.lines();

  for (i, record) in records.enumerate() {
    if i == 0 || record.trim().len() == 0 {  // <3>
      continue;
    }

    let fields: Vec<_> = record     // <4>
      .split(',')                   // <5>
      .map(|field| field.trim())    // <6>
      .collect();                   // <7>

    if cfg!(debug_assertions) {              // <8>
      eprintln!("debug: {:?} -> {:?}",
	             record, fields);            // <9>
    }

    let name = fields[0];
    if let Ok(length) = fields[1].parse::<f32>() { // <10>
        println!("{}, {}cm", name, length);        // <11>
    }
  }
}
```

列表1.2可能对一些读者来说有些混乱，特别是那些之前没有接触过Rust的读者。在继续之前，这里有一些简要说明：

* 在第17行，fields变量使用Vec\<>类型注解。Vec是_vector_的简写，是一种可以动态扩展的集合类型。下划线()告诉Rust推断元素的类型。
* 在第22行和第28行，我们指示Rust将信息打印到控制台。println!宏将其参数打印到标准输出(stdout)，而eprintln!则打印到标准错误(stderr)。
* 宏类似于函数，不同之处在于宏返回代码而不是数据。宏通常用于简化常见模式。
* eprintln!和println!都在第一个参数中使用字符串字面量以及嵌入的小型语言来控制其输出。{}占位符告诉Rust使用程序员定义的方法将值表示为字符串，而不是使用默认的表示形式{:?}。
* 第27行包含一些新颖的特性。if let Ok(length) = fields[1].parse::\<f32>()读作“尝试将fields[1]解析为32位浮点数，如果成功，则将数字分配给length变量”。if let构造是一种有条件地处理数据的简洁方法，同时还提供一个分配给该数据的局部变量。当parse()方法成功解析字符串时，它返回Ok(T)（其中T代表任何类型）；否则，它返回Err(E)（其中E代表错误类型）。if let Ok(T)的效果是跳过任何错误情况，比如在处理行Invalid,data时遇到的情况。
* 当Rust无法从上下文中推断出类型时，它会要求您指定这些类型。调用parse()包含内联类型注释，如parse::\<f32>()。

将源代码转换为可执行文件称为编译。要编译Rust代码，我们需要安装Rust编译器并对源代码运行它。要编译列表1.2，请按照以下步骤进行：
* 打开控制台提示符（如cmd.exe、PowerShell、终端或Alacritty）。
* 移动到您在1.4节下载的源代码中的ch1/ch1-penguins目录（不是ch1/ch1-penguins/src）。
* 执行cargo run。其输出如下代码片段所示：

```
$ cargo run
Compiling ch1-penguins v0.1.0 (../code/ch1/ch1-penguins)
Finished dev [unoptimized + debuginfo] target(s) in 0.40s
Running `target/debug/ch1-penguins`
dbg: " Little penguin,33" -> ["Little penguin", "33"]
Little penguin, 33cm
dbg: " Yellow-eyed penguin,65" -> ["Yellow-eyed penguin", "65"]
Yellow-eyed penguin, 65cm
dbg: " Fiordland penguin,60" -> ["Fiordland penguin", "60"]
Fiordland penguin, 60cm
dbg: " Invalid,data" -> ["Invalid", "data"]
```


您可能已经注意到了以dbg:开头的干扰性行。我们可以通过使用cargo的--release标志编译发布版本来消除这些行。列表1.2中的第22-24行中的cfg!(debug_assertions) { ... }块提供了这种条件编译的功能。发布版本在运行时要快得多，但编译时间较长：


```
$ cargo run --release
Compiling ch1-penguins v0.1.0 (.../code/ch1/ch1-penguins)
Finished release [optimized] target(s) in 0.34s
Running `target/release/ch1-penguins`
Little penguin, 33cm
Yellow-eyed penguin, 65cm
Fiordland penguin, 60cm
```

通过在cargo命令中添加-q标志，可以进一步减少输出。-q是quiet的简写。以下片段展示了这样的效果：


```
$ cargo run -q --release
Little penguin, 33cm
Yellow-eyed penguin, 65cm
Fiordland penguin, 60cm
```


选择列表1.1和列表1.2是为了将Rust的许多代表性特性打包成易于理解的示例。希望这些示例表明了Rust程序具有高级的感觉，同时又具有低级的性能。现在让我们从具体的语言特性中退后一步，考虑一下语言背后的一些思考以及它在编程语言生态系统中的定位。

## 1.6 Rust是什么？

作为一种编程语言，Rust的独特特性在于它能够在编译时防止无效数据访问。微软安全响应中心和Chromium浏览器项目的研究项目都表明，与无效数据访问相关的问题约占严重安全漏洞的70%。Rust消除了这类bug。它保证了您的程序在没有引入任何运行时成本的情况下是内存安全的。

其他语言也可以提供这种安全级别，但这些语言需要在程序运行时添加执行检查，从而减慢程序运行速度。Rust成功地打破了这种连续性，创造了自己的空间，如图1.1所示。

<a>![](/img/rustinaction/ch1/1.png)</a>

作为一个专业社区，Rust的独特特性在于它愿意明确地将价值观纳入决策过程中。这种包容的理念是普遍存在的。公共信息传递是友好的。Rust社区内的所有互动都受其行为准则的约束。甚至Rust编译器的错误消息都非常有帮助。

直到2018年末，访问Rust主页的访客会看到（技术性很重的）消息：“Rust是一种系统编程语言，运行速度非常快，可以防止段错误，并保证线程安全。”在那时，社区对措辞进行了改变，将其用户（以及潜在用户）置于中心位置（见表1.1）。

<a>![](/img/rustinaction/ch1/2.png)</a>

Rust被标记为一种系统编程语言，通常被视为编程的一种相当专业化、几乎是神秘的分支。然而，许多Rust程序员发现该语言适用于许多其他领域。安全性、生产力和控制对所有软件工程项目都是有用的。此外，Rust社区的包容性意味着该语言受益于具有多样化兴趣的新声音的持续涌现。


让我们详细了解这三个目标：安全性、生产力和控制。这些是什么，为什么这些很重要？

### 1.6.1 Rust的目标：安全性

Rust程序不会出现以下问题：

* 悬垂指针 — 在程序运行过程中，对已经失效的数据保持活动引用（参见列表1.3）
* 数据竞争 — 由于外部因素变化，导致程序无法确定在每次运行中的行为（参见列表1.4）
* 缓冲区溢出 — 尝试访问数组的第12个元素，而该数组只有6个元素（参见列表1.5）
* 迭代器失效 — 在被更改后继续迭代的问题（参见列表1.6）
* 当以调试模式编译程序时，Rust还能防止整数溢出。什么是整数溢出？整数只能表示有限的一组数字；它们在内存中具有固定的宽度。整数溢出是指当整数达到其极限时，流回到起始位置发生的情况。

下面的列表展示了一个悬垂指针。请注意，您可以在ch1/ch1-cereals/src/main.rs文件中找到这个源代码。

```rust
#[derive(Debug)]    // <1>
enum Cereal {       // <2>
    Barley, Millet, Rice,
    Rye, Spelt, Wheat,
}

fn main() {
    let mut grains: Vec<Cereal> = vec![];   // <3>
    grains.push(Cereal::Rye);               // <4>
    drop(grains);                           // <5>

    println!("{:?}", grains);               // <6>
}
```

列表1.3包含了一个在第8行创建的grains内部指针。Vec\<Cereal>实现了对基础数组的内部指针。但是，该列表无法编译。尝试编译会触发一个错误消息，指出尝试“借用”一个“移动”值。学习如何解释这个错误消息并修复底层错误是接下来页面的主题。以下是尝试编译列表1.4代码的输出：

<a>![](/img/rustinaction/ch1/3.png)</a>


列表1.4展示了一个数据竞争条件的示例。如果您记得，这种条件是由于无法确定程序由于外部因素的变化而在每次运行中的行为如何而导致的。您可以在ch1/ch1-race/src/main.rs文件中找到这段代码。

```rust
use std::thread;                          // <1>

fn main() {
    let mut data = 100;

    thread::spawn(|| { data = 500; });    // <2>
    thread::spawn(|| { data = 1000; });   // <2>

    println!("{}", data);
}
```


如果您对线程这个术语不熟悉，总结一下就是这段代码不确定性。当main()退出时，无法知道data变量将会持有什么值。在列表中的第6行和第7行，通过调用thread::spawn()创建了两个线程。每次调用都以一个闭包作为参数，用竖线和大括号表示（例如，\|\| {...}）。第5行创建的线程尝试将data变量设置为500，而第6行创建的线程尝试将其设置为1,000。由于线程的调度由操作系统而不是程序决定，因此无法确定哪个线程会首先运行。

尝试编译列表1.5将导致一系列的错误消息。Rust不允许应用程序中的多个位置具有对数据的写访问权限。代码尝试在三个位置允许这样做：一次是在运行main()的主线程中，另外两次是在由thread::spawn()创建的每个子线程中。这里是编译器的消息：

```
error[E0373]: closure may outlive the current function, but it borrows `data`, which is owned by the current function
 --> src/main.rs:6:19
  |
6 |     thread::spawn(|| { data = 500; });    // <2>
  |                   ^^   ---- `data` is borrowed here
  |                   |
  |                   may outlive borrowed value `data`
  |
note: function requires argument type to outlive `'static`
 --> src/main.rs:6:5
  |
6 |     thread::spawn(|| { data = 500; });    // <2>
  |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
help: to force the closure to take ownership of `data` (and any other referenced variables), use the `move` keyword
  |
6 |     thread::spawn(move || { data = 500; });    // <2>
  |                   ++++

error[E0499]: cannot borrow `data` as mutable more than once at a time
 --> src/main.rs:7:19
  |
6 |     thread::spawn(|| { data = 500; });    // <2>
  |     ---------------------------------
  |     |             |    |
  |     |             |    first borrow occurs due to use of `data` in closure
  |     |             first mutable borrow occurs here
  |     argument requires that `data` is borrowed for `'static`
7 |     thread::spawn(|| { data = 1000; });   // <2>
  |                   ^^   ---- second borrow occurs due to use of `data` in closure
  |                   |
  |                   second mutable borrow occurs here

error[E0373]: closure may outlive the current function, but it borrows `data`, which is owned by the current function
 --> src/main.rs:7:19
  |
7 |     thread::spawn(|| { data = 1000; });   // <2>
  |                   ^^   ---- `data` is borrowed here
  |                   |
  |                   may outlive borrowed value `data`
  |
note: function requires argument type to outlive `'static`
 --> src/main.rs:7:5
  |
7 |     thread::spawn(|| { data = 1000; });   // <2>
  |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
help: to force the closure to take ownership of `data` (and any other referenced variables), use the `move` keyword
  |
7 |     thread::spawn(move || { data = 1000; });   // <2>
  |                   ++++

error[E0502]: cannot borrow `data` as immutable because it is also borrowed as mutable
 --> src/main.rs:9:20
  |
6 |     thread::spawn(|| { data = 500; });    // <2>
  |     ---------------------------------
  |     |             |    |
  |     |             |    first borrow occurs due to use of `data` in closure
  |     |             mutable borrow occurs here
  |     argument requires that `data` is borrowed for `'static`
...
9 |     println!("{}", data);
  |                    ^^^^ immutable borrow occurs here
  |
  = note: this error originates in the macro `$crate::format_args_nl` which comes from the expansion of the macro `println` (in Nightly builds, run with -Z macro-backtrace for more info)

Some errors have detailed explanations: E0373, E0499, E0502.
For more information about an error, try `rustc --explain E0373`.
error: could not compile `ch1-race` (bin "ch1-race") due to 4 previous errors
```


列表1.5提供了一个缓冲区溢出的示例。缓冲区溢出描述了试图访问内存中不存在或非法的项的情况。在我们的例子中，试图访问fruit[4]导致程序崩溃，因为fruit变量只包含三种水果。此列表的源代码位于文件ch1/ch1-fruit/src/main.rs中。

```rust
fn main() {
  let fruit = vec!['🥝', '🍌', '🍇'];

  let buffer_overflow = fruit[4];    // <1>

  assert_eq!(buffer_overflow, '🍉')  // <2>
}
```


当编译并执行列表1.5时，您将遇到以下错误消息：

```
  Compiling ch1-fruit v0.1.0 (/media/lili/mydisk/codes/rust-in-action/ch1/ch1-fruit)
    Finished dev [unoptimized + debuginfo] target(s) in 0.22s
     Running `target/debug/ch1-fruit`
thread 'main' panicked at src/main.rs:4:30:
index out of bounds: the len is 3 but the index is 4
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
```


下一个列表展示了迭代器失效的示例，其中问题是由于在中途被更改后继续迭代的内容引起的。这个列表的源代码在ch1/ch1-letters/src/main.rs中。

```rust
fn main() {
  let mut letters = vec![            // <1>
      "a", "b", "c"
  ];

  for letter in letters {
      println!("{}", letter);
      letters.push(letter.clone());  // <2>
  }
}
```

列表1.6无法编译，因为Rust不允许在迭代块内修改letters变量。以下是错误消息：

```
   Compiling ch1-letters v0.1.0 (/media/lili/mydisk/codes/rust-in-action/ch1/ch1-letters)
error[E0382]: borrow of moved value: `letters`
 --> src/main.rs:8:7
  |
2 |   let mut letters = vec![            // <1>
  |       ----------- move occurs because `letters` has type `Vec<&str>`, which does not implement the `Copy` trait
...
6 |   for letter in letters {
  |                 ------- `letters` moved due to this implicit call to `.into_iter()`
7 |       println!("{}", letter);
8 |       letters.push(letter.clone());  // <2>
  |       ^^^^^^^ value borrowed here after move
  |
note: `into_iter` takes ownership of the receiver `self`, which moves `letters`
 --> /rustc/07dca489ac2d933c78d3c5158e3f43beefeb02ce/library/core/src/iter/traits/collect.rs:268:18
help: consider iterating over a slice of the `Vec<&str>`'s content to avoid moving into the `for` loop
  |
6 |   for letter in &letters {
  |                 +

For more information about this error, try `rustc --explain E0382`.
error: could not compile `ch1-letters` (bin "ch1-letters") due to 1 previous error
```


尽管错误消息中充斥着术语（borrow、move、trait等等），但Rust已经保护了程序员免于陷入许多其他人常遇到的陷阱中。不用担心，随着您在本书的前几章中的学习，这些术语会变得更容易理解。

知道一种语言是安全的，给程序员提供了一定程度的自由。因为他们知道自己的程序不会崩溃，所以他们变得更愿意进行实验。在Rust社区中，这种自由产生了“无畏并发”的说法。

### 1.6.2 Rust的目标：生产力

在选择时，Rust更倾向于对开发者来说最简单的选项。它许多更微妙的特性都是提高生产力的因素。但通过书中的例子来演示程序员的生产力是一个难以展示的概念。让我们从一个可能会让初学者困扰的问题开始——在应该使用相等性（==）测试的表达式中使用赋值（=）：

```
fn main() {
 let a = 10;

 if a = 10 { 
   println!("a equals ten");
 }
}
```

在Rust中，上述代码无法编译通过。Rust编译器生成了以下消息：

```
error[E0308]: mismatched types
--> src/main.rs:4:8
|
4 |
 if a = 10 {
|
 ^^^^^^
|
 |
|
 expected `bool`, found `()`
|
 help: try comparing for equality: `a == 10`
error: aborting due to previous error
For more information about this error, try `rustc --explain E0308`.
error: could not compile `playground`.
To learn more, run the command again with --verbose.
```


起初，“类型不匹配”可能会感觉是一个奇怪的错误消息。我们当然可以将变量与整数进行相等性测试。经过一番思考，为什么 if 测试收到了错误的类型就变得显而易见了。

if 不是接收一个整数，而是接收了一个赋值的结果。在Rust中，这是一个空类型：()。()被读作“单元”。

当没有其他有意义的返回值时，表达式返回 ()。如下所示，在第4行添加第二个等号会导致一个工作正常的程序，打印 a 等于十：

```
fn main() {
 let a = 10;

 if a == 10 { 
   println!("a equals ten");
 }
}
```


Rust具有许多人性化的特性。它提供了泛型、复杂的数据类型、模式匹配和闭包。那些曾经使用其他提前编译语言的人可能会欣赏Rust的构建系统和其全面的包管理器：cargo。
乍一看，我们可以看到cargo是rustc（Rust编译器）的一个前端，但cargo提供了一些额外的工具，包括以下内容：

* cargo new 在新目录中创建一个Rust项目的框架（cargo init 用于当前目录）。
* cargo build 下载依赖并编译代码。
* cargo run 执行cargo build，然后运行生成的可执行文件。
* cargo doc 为当前项目中的每个依赖项构建HTML文档。

### 1.6.3 Rust的目标：控制

Rust为程序员提供了对数据结构在内存中布局方式和访问模式的精细控制。虽然Rust使用符合其“零成本抽象”哲学的合理默认设置，但这些默认设置并不适用于所有情况。

有时，管理应用程序的性能是至关重要的。对于您来说，数据存储在栈上而不是堆上可能很重要。也许，添加引用计数以创建对值的共享引用是有意义的。偶尔，为特定的访问模式创建自己的指针类型可能很有用。设计空间很大，Rust提供了工具，让您实现您喜欢的解决方案。

**注意** 如果“栈”、“堆”和“引用计数”等术语是新的，请不要放弃阅读这本书！我们将花很多时间来解释它们以及它们如何在整本书中相互配合。

列表1.7打印出行a: 10, b: 20, c: 30, d: Mutex { data: 40 }。每个表示都是存储整数的另一种方式。随着我们在接下来的几章中的进展，与每个级别相关的权衡将变得明显起来。目前，重要的是记住类型菜单是全面的。您可以选择适合您特定用例的类型。

列表1.7还展示了创建整数的多种方式。每种形式提供不同的语义和运行时特性。但程序员保留了他们想要做出的权衡的完全控制权。

<a>![](/img/rustinaction/ch1/4.png)</a>


要理解为什么Rust以某种方式执行某些操作，可以参考以下三个原则可能会有所帮助：

* 语言的首要任务是保证安全。
* Rust中的数据默认是不可变的。
* 编译时检查是强烈推荐的。安全应该是“零成本抽象”。

## 1.7 Rust的重要特性

我们的工具塑造了我们相信自己可以创建的东西。Rust使您能够构建您想要创建的软件，但您曾经害怕尝试。Rust是什么样的工具？从上一节讨论的三个原则中得出的是语言的三个主要特性：

* 性能
* 并发
* 内存效率


### 1.7.1 性能

Rust提供了您计算机的所有可用性能。众所周知，Rust不依赖垃圾收集器来确保内存安全。

不幸的是，承诺为您提供更快速的程序存在一个问题：您的CPU速度是固定的。因此，要使软件运行得更快，它需要做更少的事情。然而，语言很大。为了解决这种冲突，Rust将负担转移到了编译器上。

Rust社区更喜欢拥有一个功能更多的编译器的更大的语言，而不是编译器功能较少的简单语言。Rust编译器会积极优化程序的大小和速度。Rust还有一些不太明显的技巧：

* 默认提供高速缓存友好的数据结构。数组通常在Rust程序中保存数据，而不是通过指针创建的深层嵌套树结构。这被称为面向数据的编程。
* 现代包管理器（cargo）的可用性使得轻松受益于成千上万的开源软件包。相比之下，C和C++在这方面的一致性较少，构建具有许多依赖项的大型项目通常很困难。
* 方法始终静态分派，除非您显式请求动态分派。这使得编译器能够大幅优化代码，有时甚至完全消除函数调用的成本。

### 1.7.2 并发性

要求计算机同时做多件事情对软件工程师来说是困难的。就操作系统而言，如果程序员犯了严重错误，两个独立的执行线程可能会互相破坏。然而，Rust催生了无畏的并发性这一说法。它强调安全性超越了独立线程的界限。没有全局解释器锁（GIL）来限制线程的速度。我们将在第二部分中探讨这种情况的一些影响。

### 1.7.3 内存效率

Rust使您能够创建需要最小内存的程序。在需要时，您可以使用固定大小的结构，并准确知道每个字节的管理方式。高级构造，如迭代和泛型类型，产生最小的运行时开销。

## 1.8 Rust的缺点

谈论这种语言时，很容易把它看作是所有软件工程问题的灵丹妙药。
例如：

* "高级语法，低级性能！"
* "并发而不崩溃！"
* "C语言的完美安全性！"

这些标语（有时言过其实）很棒。但是尽管Rust具有诸多优点，它也有一些缺点。

### 1.8.1 循环数据结构

在Rust中，建模循环数据结构，比如任意图结构，是比较困难的。实现一个双向链表是一个大学本科水平的计算机科学问题。然而，Rust的安全检查在这里确实会阻碍进展。如果您是新手，请避免在熟悉Rust之前实现这类数据结构。

### 1.8.2 编译时间

与其同行语言相比，Rust在编译代码时速度较慢。它拥有复杂的编译器工具链，接收多个中间表示，并向LLVM编译器发送大量代码。Rust程序的编译单位不是单个文件，而是整个包（亲切地称为crate）。由于crate可以包含多个模块，这些模块可能是非常大的编译单元。尽管这样可以实现整个crate的优化，但也需要整个crate的编译。

### 1.8.3 严格性

在使用Rust进行编程时，要想懒惰是不可能的，嗯，很难。直到一切都完全正确，程序才会编译通过。编译器很严格，但也很有帮助。随着时间的推移，您可能会开始欣赏这个特性。如果您曾经使用过动态语言进行编程，那么您可能曾经因为变量命名错误导致程序崩溃而感到沮丧。Rust提前展示了这种沮丧，这样您的用户就不必经历事物崩溃的沮丧。

### 1.8.4 语言规模

Rust非常庞大！它拥有丰富的类型系统、几十个关键字，并包含一些其他语言中无法获得的特性。所有这些因素结合起来构成了一个陡峭的学习曲线。为了让学习变得可管理，我建议逐步学习Rust。从语言的最小子集开始，并给自己时间在需要时学习细节。这本书采取了这种方法。高级概念被推迟到更晚的时间。

### 1.8.5 炒作

Rust社区对快速增长并被炒作所吞噬持谨慎态度。然而，许多软件项目在收件箱中都会遇到这样的问题：“您是否考虑过用Rust重写这个项目？”不幸的是，用Rust编写的软件仍然是软件。它并不免疫安全问题，也不能为软件工程的所有问题提供灵丹妙药。

## 1.9 TLS安全案例研究

为了证明Rust无法消除所有错误，让我们来看两个严重的漏洞，这些漏洞威胁了几乎所有面向互联网的设备，并考虑一下Rust是否可以阻止这些漏洞。

到了2015年，随着Rust的日益突出，SSL/TLS的实现（即OpenSSL和苹果的分支）被发现存在严重的安全漏洞。Heartbleed（俗称的CVE-2014-0160）和goto fail;，这两种漏洞都提供了测试Rust内存安全性的机会。Rust很可能在这两种情况下都有所帮助，但仍然可能编写出具有类似问题的Rust代码。

### 1.9.1 Heartbleed

Heartbleed，正式编号为CVE-2014-0160，是由于错误地重复使用缓冲区而导致的。缓冲区是内存中预留的用于接收输入的空间。如果在写入之间不清除缓冲区的内容，数据可能会从一次读取泄漏到下一次读取。

为什么会发生这种情况呢？程序员追求性能。为了减少内存应用程序向操作系统请求内存的频率，缓冲区被重用。

想象一下，我们想要从多个用户那里处理一些机密信息。出于某种原因，我们决定在程序运行过程中重用一个单一的缓冲区。如果我们在使用缓冲区后没有重置它，那么早期调用的信息将泄漏到后来的调用中。以下是一个可能遇到此错误的程序的摘要：

<a>![](/img/rustinaction/ch1/5.png)</a>

Rust无法保护您免受逻辑错误的影响。它确保您的数据永远不会同时写入两个地方。但它并不能确保您的程序没有任何安全问题。

### 1.9.2 goto fail;

对于goto fail;漏洞，官方编号为CVE-2014-1266，它是由程序员的错误以及C设计问题（以及潜在的编译器没有指出这个缺陷）共同导致的。一个本来设计用来验证加密密钥对的函数最终跳过了所有检查。以下是原始SSLVerifySignedServerKeyExchange函数的一部分提取，保留了相当多的混淆语法：

<a>![](/img/rustinaction/ch1/6.png)</a>
<a>![](/img/rustinaction/ch1/7.png)</a>



在示例代码中，问题出现在第15行和第17行之间。在C语言中，逻辑测试不需要花括号。C编译器解释这三行代码如下：


```
if ((err = SSLHashSHA1.update(&hashCtx, &signedParams)) != 0) {
  goto fail;
}
goto fail;
```

Rust是否会有所帮助？可能会。在这种特定情况下，Rust的语法会捕捉到这个错误。它不允许在逻辑测试中没有花括号。Rust还会在代码不可达时发出警告。但这并不意味着在Rust中这个错误变得不可能。在紧迫的截止日期下，焦虑的程序员会犯错误。一般来说，类似的代码会编译并运行。
 

## 1.10 Rust最适合的领域

尽管Rust被设计为系统编程语言，但它是一种通用语言。它已经成功地应用于许多领域，我们接下来讨论一下这些领域。

### 1.10.1 命令行工具

Rust为创建命令行工具的程序员提供了三个主要优势：最小的启动时间、低内存使用和易于部署。由于Rust不需要初始化解释器（如Python、Ruby等）或虚拟机（如Java、C#等），因此程序可以快速启动工作。

作为一种裸金属语言，Rust产生内存效率高的程序。正如你将在本书中看到的，许多类型是零大小的。也就是说，它们只是编译器的提示，不会在运行程序时占用任何内存。

用Rust编写的实用程序默认编译为静态二进制文件。这种编译方法避免了依赖于必须在程序运行前安装的共享库。创建可以在不需要安装步骤的情况下运行的程序使得这些程序易于分发。

### 1.10.2 数据处理

Rust在文本处理和其他形式的数据处理方面表现出色。程序员可以控制内存使用和快速启动时间。截至2017年中期，Rust宣称拥有世界上最快的正则表达式引擎。2019年，Apache Arrow数据处理项目——对Python和R数据科学生态系统至关重要——接受了基于Rust的DataFusion项目。

Rust还作为多个搜索引擎、数据处理引擎和日志解析系统的实现基础。其类型系统和内存控制使您能够创建具有低稳定内存占用的高吞吐量数据管道。通过Apache Storm、Apache Kafka或Apache Hadoop流，可以轻松将小型过滤器程序嵌入到较大的框架中。

### 1.10.3 扩展应用程序

Rust非常适合扩展用动态语言编写的程序。这使得JNI（Java Native Interface）扩展、C扩展或Rust中的Erlang/Elixir NIFs（本地实现的函数）成为可能。C扩展通常是一个令人恐惧的建议。它们往往与运行时紧密集成。一旦犯错，就可能因为内存泄漏或完全崩溃而导致内存消耗失控。Rust消除了这种焦虑。

* Sentry，一个处理应用程序错误的公司，发现Rust非常适合重写其Python系统的CPU密集组件。
* Dropbox使用Rust重写了其客户端应用程序的文件同步引擎：“与性能相比，Rust的人体工程学和对正确性的关注帮助我们控制了同步的复杂性。”

### 1.10.4资源受限环境

长达数十年，C一直占据着微控制器领域。然而，物联网（IoT）正在兴起。这可能意味着许多十亿不安全设备暴露在网络中。任何输入解析代码都将定期检查是否存在弱点。考虑到这些设备的固件更新频率很低，从一开始就尽可能地确保安全至关重要。Rust可以在不带来运行时成本的情况下增加一层安全性。

### 1.10.5 服务器端应用程序

大多数用Rust编写的应用程序都在服务器上运行。这些应用程序可以用于提供网络流量或支持企业运营。还有一层服务位于操作系统和应用程序之间。Rust用于编写数据库、监控系统、搜索应用和消息系统。例如：

* JavaScript和node.js社区的npm软件包注册表是用Rust编写的。
* 一个嵌入式数据库sled能够在16核机器上处理包括5%写入的10亿操作负载，并在不到一分钟内完成。
* 一个全文搜索引擎Tantivy在4核桌面机器上能够在大约100秒内索引8GB的英文维基百科。

### 1.10.6 桌面应用程序

Rust的设计本质上并不妨碍其部署用于开发面向用户的软件。Servo是一款作为Rust早期发展的孵化器的网络浏览器引擎，是一个面向用户的应用程序。自然地，游戏也是如此。

### 1.10.7 桌面

人们的计算机上仍然需要编写应用程序。桌面应用程序通常很复杂，难以工程化和难以支持。Rust的部署人体工程学方法和严谨性使其有望成为许多应用程序的秘密武器。首先，这些将由小型独立开发者构建。随着Rust的成熟，生态系统也将成长。

### 1.10.8 移动端

Android、iOS和其他智能手机操作系统通常为开发人员提供一个指定的路径。对于Android来说，这个路径是Java。对于macOS来说，开发人员通常使用Swift进行编程。然而，还有另一种方法。

这两个平台都提供了本机应用程序在其上运行的能力。这通常是为了让用C++编写的应用程序（如游戏）能够部署到用户的手机上。Rust能够通过相同的接口与手机进行通信，而没有额外的运行时成本。

### 1.10.9 Web

您可能已经意识到，JavaScript是网络的语言。然而，随着时间的推移，情况将会改变。浏览器供应商正在开发一个名为WebAssembly（Wasm）的标准，该标准承诺成为许多语言的编译目标。Rust是其中之一。将Rust项目移植到浏览器只需要两个额外的命令行命令。一些公司正在探索通过Wasm在浏览器中使用Rust，尤其是CloudFlare和Fastly等公司。

### 1.10.10 系统编程

在某种意义上，系统编程是Rust的存在理由。许多大型程序已经在Rust中实现，包括编译器（Rust本身）、视频游戏引擎和操作系统。Rust社区包括解析器生成器、数据库和文件格式的作者。

对于那些与Rust共享目标的程序员来说，Rust已被证明是一个高效的环境。这个领域的三个突出项目包括以下内容：

* 谷歌正在赞助开发Fuchsia OS，这是一个设备操作系统。
* 微软正在积极探索在Windows上用Rust编写低级组件。
* 亚马逊网络服务（AWS）正在构建Bottlerocket，这是一个专门用于在云中托管容器的操作系统。

## 1.11 Rust的隐含特点：其社区
要发展一门编程语言，需要的不仅仅是软件。Rust团队做得非常出色的一件事是在语言周围建立了一个积极和友好的社区。无论您在Rust世界的哪个角落，您都会发现会受到礼貌和尊重对待。

## 1.12 Rust术语手册

当您与Rust社区成员互动时，您很快就会遇到一些具有特殊意义的术语。理解以下术语将更容易理解Rust为什么会发展成现在这样，并且它试图解决的问题：

* 赋予权力给每个人—所有的程序员，无论能力或背景如何，都可以参与。编程，特别是系统编程，不应该只限于一小部分人。
* 极速编程—Rust是一种快速的编程语言。您将能够编写性能与或超过其同行语言的程序，但您将拥有更多的安全保障。
* 无惧并发—并发和并行编程一直被视为困难的事情。Rust让您摆脱了一直困扰其同行语言的整个类别的错误。
* 没有Rust 2.0—今天编写的Rust代码将始终可以使用未来的Rust编译器进行编译。Rust旨在成为一种可靠的编程语言，可以在未来数十年依赖。根据语义化版本控制，Rust永远不会向后不兼容，因此它永远不会发布新的主要版本。
* 零成本抽象—您从Rust获得的功能不会增加运行时成本。当您在Rust中编程时，安全性不会牺牲速度。

## 总结

* 许多公司已成功在Rust中构建大型软件项目。
* 使用Rust编写的软件可以编译为PC、浏览器、服务器，以及移动设备和物联网设备。
* Rust语言受到软件开发人员的喜爱。它多次获得Stack Overflow“最受喜爱的编程语言”称号。
* Rust让您可以毫无顾虑地进行实验。它提供了其他工具无法在不增加运行时成本的情况下提供的正确性保证。

* 使用Rust，您需要学习三个主要的命令行工具：
    * cargo，管理整个crate
    * rustup，管理Rust安装
    * rustc，管理Rust源代码的编译

* Rust项目并非免受所有错误。
* Rust代码稳定、快速
