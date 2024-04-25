---
layout:     post
title:      "Slurm文档翻译" 
author:     "lili" 
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - pytorch
    - extension 
---

本文是[slurm文档](https://slurm.schedmd.com/)的部分翻译。

<!--more-->

**目录**
* TOC
{:toc}

## Overview


Slurm 是一个开源、容错性强且高度可扩展的集群管理(cluster management)和作业调度(job scheduling)系统，适用于大型和小型 Linux 集群。Slurm 在操作上不需要对内核进行修改，并且相对自包含。作为集群工作负载管理器，Slurm 具有三个关键功能。首先，它为用户分配独占和/或非独占的资源（计算节点），供其一段时间内执行工作。其次，它提供了一个框架，用于在分配的节点集上启动、执行和监控工作（通常是并行作业）。最后，它通过管理挂起工作的队列来调解资源争用。可选的插件可用于[会计](https://slurm.schedmd.com/accounting.html)、[高级预约](https://slurm.schedmd.com/reservations.html)、[团队调度](https://slurm.schedmd.com/gang_scheduling.html)（并行作业的时间共享）、回填调度、[拓扑优化的资源选择](https://slurm.schedmd.com/topology.html)、按用户或账户[限制资源](https://slurm.schedmd.com/resource_limits.html)、以及复杂的[多因素作业优先级](https://slurm.schedmd.com/priority_multifactor.html)算法。

### Architecture

Slurm有一个集中式管理器**slurmctld**，用于监控资源和工作。在故障发生时，可能还会有一个备份管理器来承担这些责任。每个计算服务器（节点）都有一个**slurmd**守护进程，可以将其比作远程shell：它等待工作，执行工作，返回状态，并等待更多工作。slurmd守护进程提供容错性强的分层通信。还有一个可选的**slurmdbd**（Slurm DataBase Daemon），用于在单个数据库中记录多个由Slurm管理的集群的会计信息。还有一个可选的**[slurmrestd](https://slurm.schedmd.com/rest.html)**（Slurm REST API Daemon），可用于通过其REST API与Slurm交互。用户工具包括**srun**用于启动作业，**scancel**用于终止排队或正在运行的作业，**sinfo**用于报告系统状态，**squeue**用于报告作业状态，以及**sacct**用于获取正在运行或已完成的作业和作业步骤的信息。**sview**命令以图形方式报告系统和作业状态，包括网络拓扑。还有一个名为**scontrol**的管理工具，用于监视和/或修改集群上的配置和状态信息。用于管理数据库的管理工具是**sacctmgr**。它可用于识别集群、有效用户、有效银行账户等。所有功能都提供了API。

<a>![](/img/slurm/1.png)</a>
*Slurm的组件*

Slurm具有通用插件机制，可以轻松支持各种基础设施。这允许使用构建块方法实现各种Slurm配置。这些插件目前包括：

* 会计存储(Accounting Storage)：主要用于存储作业的历史数据。当与SlurmDBD（Slurm数据库守护程序）一起使用时，还可以提供基于限制的系统以及历史系统状态。
* 能源采集(Account Gather Energy)：采集系统中每个作业或节点的能源消耗数据。该插件与会计存储和作业帐户采集插件集成。
* 通信认证(Authentication of communications)：提供Slurm各个组件之间的认证机制。
* [容器](https://slurm.schedmd.com/containers.html)：支持和实现HPC工作负载容器。
* 凭证（数字签名生成）：用于生成数字签名的机制，用于验证作业步骤是否被授权在特定节点上执行。这与用于认证的插件不同，因为作业步骤请求是从用户的srun命令发送的，而不是直接从生成作业步骤凭证和数字签名的slurmctld守护进程发送的。
* 通用资源：提供控制通用资源（包括图形处理单元（GPU））的接口。
* 作业提交：自定义插件，允许在提交和更新时对作业要求进行站点特定控制。
* 作业会计采集：采集作业步骤资源利用数据。
* 作业完成日志记录：记录作业的终止数据。这通常是由会计存储插件存储的数据的子集。
* 启动器：控制'srun'命令用于启动任务的机制。
* MPI：为各种MPI实现提供不同的挂接。例如，这可以设置MPI特定的环境变量。
* 抢占：确定哪些作业可以抢占其他作业以及要使用的抢占机制。
* 优先级：在提交时为作业分配优先级，并且在持续基础上进行（例如，作业老化）。
* 进程跟踪（用于信号发送）：提供识别与每个作业关联的进程的机制。用于作业会计和信号发送。
* 调度器：插件确定Slurm如何以及何时调度作业。
* 节点选择：用于确定作业分配中使用的资源的插件。
* 站点因素（优先级）：为作业的多因素优先级的特定站点因素组件分配优先级，并且在持续基础上进行（例如，作业老化）。
* 开关或互连：用于与开关或互连进行接口的插件。对于大多数系统（以太网或InfiniBand），这是不需要的。
* 任务亲和性：提供将作业及其各个任务绑定到特定处理器的机制。
* 网络拓扑：基于网络拓扑优化资源选择。用于作业分配和高级预约。


Slurm管理的实体，如图2所示，包括**节点**，Slurm中的计算资源，**分区**，将节点分组到逻辑集合中，**作业**，或为用户分配的资源的分配，以及**作业步骤**，即作业内（可能是并行的）任务集合。分区可以视为作业队列，每个队列都有一系列约束，例如作业大小限制、作业时间限制、允许使用它的用户等。按优先级排序的作业会在分区内分配节点，直到该分区内的资源（节点、处理器、内存等）耗尽。一旦作业分配了一组节点，用户就能够在分配内以任何配置启动并行工作的作业步骤。例如，可以启动一个单一作业步骤，该作业步骤利用分配给作业的所有节点，或者可以独立使用一部分分配的多个作业步骤。Slurm为作业分配的处理器提供资源管理，因此可以同时提交多个作业步骤，并将其排队，直到作业分配内有可用资源。

<a>![](/img/slurm/1.png)</a>
*Slurm实体*

### 可配置性

监控的节点状态包括：处理器数量、实际内存大小、临时磁盘空间大小以及状态（UP、DOWN等）。附加的节点信息包括权重（被分配工作的优先级）和特征（任意信息，如处理器速度或类型）。节点被分组到分区中，分区可能包含重叠的节点，因此最好将它们视为作业队列。分区信息包括：名称、关联节点列表、状态（UP或DOWN）、最大作业时间限制、每个作业的最大节点数、组访问列表、优先级（如果节点位于多个分区中则很重要）以及共享节点访问策略，可选的超额订购级别用于团队调度（例如：YES、NO或FORCE:2）。使用位图表示节点，并且可以通过执行少量比较和一系列快速位图操作来进行调度决策。以下是一个示例（部分）Slurm配置文件。

```
#
# Sample /etc/slurm.conf
#
SlurmctldHost=linux0001  # Primary server
SlurmctldHost=linux0002  # Backup server
#
AuthType=auth/munge
Epilog=/usr/local/slurm/sbin/epilog
PluginDir=/usr/local/slurm/lib
Prolog=/usr/local/slurm/sbin/prolog
SlurmctldPort=7002
SlurmctldTimeout=120
SlurmdPort=7003
SlurmdSpoolDir=/var/tmp/slurmd.spool
SlurmdTimeout=120
StateSaveLocation=/usr/local/slurm/slurm.state
TmpFS=/tmp
#
# Node Configurations
#
NodeName=DEFAULT CPUs=4 TmpDisk=16384 State=IDLE
NodeName=lx[0001-0002] State=DRAINED
NodeName=lx[0003-8000] RealMemory=2048 Weight=2
NodeName=lx[8001-9999] RealMemory=4096 Weight=6 Feature=video
#
# Partition Configurations
#
PartitionName=DEFAULT MaxTime=30 MaxNodes=2
PartitionName=login Nodes=lx[0001-0002] State=DOWN
PartitionName=debug Nodes=lx[0003-0030] State=UP Default=YES
PartitionName=class Nodes=lx[0031-0040] AllowGroups=students
PartitionName=DEFAULT MaxTime=UNLIMITED MaxNodes=4096
PartitionName=batch Nodes=lx[0041-9999]
```

## Commands

所有Slurm守护进程、命令和API函数都有man页面。命令选项\-\-help也提供了选项的简要摘要。请注意，命令选项区分大小写。

* **sacct**用于报告有关正在运行或已完成作业的作业或作业步骤会计信息。

* **salloc**用于实时为作业分配资源。通常用于分配资源并生成一个shell。然后使用shell来执行srun命令来启动并行任务。

* **sattach**用于将标准输入、输出和错误以及信号功能附加到当前运行的作业或作业步骤。可以多次附加和分离作业。

* **sbatch**用于提交一个作业脚本以供以后执行。脚本通常包含一个或多个srun命令来启动并行任务。

* **sbcast**用于将文件从本地磁盘传输到作业分配的节点上的本地磁盘。这可用于有效使用无磁盘的计算节点或相对于共享文件系统提供更好的性能。

* **scancel**用于取消挂起或正在运行的作业或作业步骤。也可以用于向与正在运行的作业或作业步骤关联的所有进程发送任意信号。

* **scontrol**是用于查看和/或修改Slurm状态的管理工具。请注意，许多scontrol命令只能由root用户执行。

* **sinfo**报告由Slurm管理的分区和节点的状态。它具有各种过滤、排序和格式化选项。

* **sprio**用于显示影响作业优先级的组件的详细视图。

* **squeue**报告作业或作业步骤的状态。它具有各种过滤、排序和格式化选项。默认情况下，它按优先级顺序报告运行的作业，然后按优先级顺序报告挂起的作业。

* **srun**用于提交作业以供执行或实时启动作业步骤。srun具有各种选项来指定资源要求，包括：最小和最大节点数、处理器数、要使用或不使用的特定节点以及特定节点特征（如内存、磁盘空间、某些必需特性等）。作业可以包含在作业的节点分配内依次或并行执行的多个作业步骤，这些作业步骤在作业的节点分配内使用独立或共享资源。

* **sshare**显示关于集群中fairshare使用的详细信息。请注意，只有在使用优先级/多因素插件时，这才是可行的。

* **sstat**用于获取正在运行的作业或作业步骤使用的资源的信息。

* **strigger**用于设置、获取或查看事件触发器。事件触发器包括节点宕机或作业接近其时间限制等内容。

* **sview**是用于获取和更新由Slurm管理的作业、分区和节点的状态信息的图形用户界面。


## 示例


首先，我们确定系统上存在哪些分区，它们包含哪些节点，以及系统的一般状态。这些信息由sinfo命令提供。在下面的示例中，我们发现有两个分区：debug和batch。名称debug后面的\*表示这是提交作业的默认分区。我们看到两个分区都处于UP状态。某些配置可能包括用于较大作业的分区，这些分区在周末或夜间除外都处于DOWN状态。有关每个分区的信息可能会分成多行，以便识别处于不同状态的节点。在本例中，两个节点adev[1-2]处于down状态。状态down后面的表示节点没有响应。请注意使用简洁的表达式来指定节点名称，其中包含公共前缀adev和数字范围或特定数字。这种格式可以轻松管理非常大的集群。sinfo命令具有许多选项，可以让您轻松查看您感兴趣的信息，并以您喜欢的格式显示。有关更多信息，请参阅man页面。

```
adev0: sinfo
PARTITION AVAIL  TIMELIMIT NODES  STATE NODELIST
debug*       up      30:00     2  down* adev[1-2]
debug*       up      30:00     3   idle adev[3-5]
batch        up      30:00     3  down* adev[6,13,15]
batch        up      30:00     3  alloc adev[7-8,14]
batch        up      30:00     4   idle adev[9-12]
```

接下来，我们使用squeue命令确定系统上存在哪些作业。ST字段表示作业状态。有两个作业处于运行状态（R是Running的缩写），而一个作业处于挂起状态（PD是Pending的缩写）。TIME字段显示作业已运行的时间，格式为天-小时:分钟:秒。NODELIST(REASON)字段指示作业正在运行的位置或仍然挂起的原因。作业挂起的典型原因是资源(Resources)（等待资源变为可用）和优先级(Priority)（排在更高优先级作业后面）。squeue命令具有许多选项，可以让您轻松查看您感兴趣的信息，并以您喜欢的格式显示。有关更多信息，请参阅man页面。

```
adev0: squeue
JOBID PARTITION  NAME  USER ST  TIME NODES NODELIST(REASON)
65646     batch  chem  mike  R 24:19     2 adev[7-8]
65647     batch   bio  joan  R  0:09     1 adev14
65648     batch  math  phil PD  0:00     6 (Resources)
```



scontrol命令可用于报告有关节点、分区、作业、作业步骤和配置的更详细信息。系统管理员还可以使用它来进行配置更改。以下是一些示例。有关更多信息，请参阅man页面。


```
adev0: scontrol show partition
PartitionName=debug TotalNodes=5 TotalCPUs=40 RootOnly=NO
   Default=YES OverSubscribe=FORCE:4 PriorityTier=1 State=UP
   MaxTime=00:30:00 Hidden=NO
   MinNodes=1 MaxNodes=26 DisableRootJobs=NO AllowGroups=ALL
   Nodes=adev[1-5] NodeIndices=0-4

PartitionName=batch TotalNodes=10 TotalCPUs=80 RootOnly=NO
   Default=NO OverSubscribe=FORCE:4 PriorityTier=1 State=UP
   MaxTime=16:00:00 Hidden=NO
   MinNodes=1 MaxNodes=26 DisableRootJobs=NO AllowGroups=ALL
   Nodes=adev[6-15] NodeIndices=5-14


adev0: scontrol show node adev1
NodeName=adev1 State=DOWN* CPUs=8 AllocCPUs=0
   RealMemory=4000 TmpDisk=0
   Sockets=2 Cores=4 Threads=1 Weight=1 Features=intel
   Reason=Not responding [slurm@06/02-14:01:24]

65648     batch  math  phil PD  0:00     6 (Resources)
adev0: scontrol show job
JobId=65672 UserId=phil(5136) GroupId=phil(5136)
   Name=math
   Priority=4294901603 Partition=batch BatchFlag=1
   AllocNode:Sid=adev0:16726 TimeLimit=00:10:00 ExitCode=0:0
   StartTime=06/02-15:27:11 EndTime=06/02-15:37:11
   JobState=PENDING NodeList=(null) NodeListIndices=
   NumCPUs=24 ReqNodes=1 ReqS:C:T=1-65535:1-65535:1-65535
   OverSubscribe=1 Contiguous=0 CPUs/task=0 Licenses=(null)
   MinCPUs=1 MinSockets=1 MinCores=1 MinThreads=1
   MinMemory=0 MinTmpDisk=0 Features=(null)
   Dependency=(null) Account=(null) Requeue=1
   Reason=None Network=(null)
   ReqNodeList=(null) ReqNodeListIndices=
   ExcNodeList=(null) ExcNodeListIndices=
   SubmitTime=06/02-15:27:11 SuspendTime=None PreSusTime=0
   Command=/home/phil/math
   WorkDir=/home/phil
```



可以使用srun命令在单个命令行中创建资源分配并启动作业步骤的任务。根据所使用的MPI实现，MPI作业也可以以这种方式启动。有关更多特定于MPI的信息，请参阅MPI部分。在此示例中，我们在三个节点上执行/bin/hostname（-N3），并在输出中包含任务编号（-l）。将使用默认分区。默认情况下，每个节点将使用一个任务。请注意，srun命令有许多选项可用于控制分配的资源以及如何在这些资源之间分配任务。

```shell
adev0: srun -N3 -l /bin/hostname
0: adev3
1: adev4
2: adev5
```


这个基于之前示例的变种在四个任务（-n4）中执行/bin/hostname。默认情况下，每个任务将使用一个处理器（请注意我们没有指定节点数量）。

```shell
adev0: srun -n4 -l /bin/hostname
0: adev3
1: adev3
2: adev3
3: adev3
```


常见的操作模式之一是提交一个脚本以供以后执行。在这个示例中，脚本名称是my.script，我们明确使用节点adev9和adev10（-w "adev[9-10]"，注意使用节点范围表达式）。我们还明确说明后续作业步骤将每个生成四个任务，这将确保我们的分配至少包含四个处理器（每个任务启动一个处理器）。输出将出现在文件my.stdout中（"-o my.stdout"）。这个脚本包含了嵌入其中的作业时间限制。其他选项可以通过在脚本开头（在脚本中执行的任何命令之前）使用“\#SBATCH”前缀加上选项来提供。在命令行上提供的选项将覆盖脚本中指定的任何选项。请注意，my.script包含了在分配的第一个节点上执行的命令/bin/hostname，以及使用srun命令启动并依次执行的两个作业步骤。

```
adev0: cat my.script
#!/bin/sh
#SBATCH --time=1
/bin/hostname
srun -l /bin/hostname
srun -l /bin/pwd

adev0: sbatch -n4 -w "adev[9-10]" -o my.stdout my.script
sbatch: Submitted batch job 469

adev0: cat my.stdout
adev9
0: adev9
1: adev9
2: adev10
3: adev10
0: /home/jette
1: /home/jette
2: /home/jette
3: /home/jette
```



最后一种操作模式是创建资源分配并在该分配中启动作业步骤。salloc命令用于创建资源分配，并通常在该分配中启动一个shell。一个或多个作业步骤通常将在该分配中使用srun命令来启动任务（根据使用的MPI类型，启动机制可能会有所不同，请参阅下面的MPI详细信息）。最后，由salloc创建的shell将使用exit命令终止。Slurm不会自动将可执行文件或数据文件迁移到分配给作业的节点上。文件必须存在于本地磁盘上或某个全局文件系统中（例如NFS或Lustre）。我们提供了工具sbcast来使用Slurm的分层通信将文件传输到分配节点的本地存储中。在这个例子中，我们使用sbcast将可执行程序a.out传输到分配节点的本地存储中的/tmp/joe.a.out。在执行程序后，我们从本地存储中删除它。

```
tux0: salloc -N1024 bash
$ sbcast a.out /tmp/joe.a.out
Granted job allocation 471
$ srun /tmp/joe.a.out
Result is 3.14159
$ srun rm /tmp/joe.a.out
$ exit
salloc: Relinquishing job allocation 471
```


在这个例子中，我们提交一个批处理作业，获取它的状态，并取消它。

```
adev0: sbatch test
srun: jobid 473 submitted

adev0: squeue
JOBID PARTITION NAME USER ST TIME  NODES NODELIST(REASON)
  473 batch     test jill R  00:00 1     adev9

adev0: scancel 473

adev0: squeue
JOBID PARTITION NAME USER ST TIME  NODES NODELIST(REASON)
```

## 最佳实践，大规模作业计数

考虑将相关工作放入单个Slurm作业中，并使用多个作业步骤，这样做既有助于性能优化，也便于管理。每个Slurm作业可以包含多个作业步骤，而在Slurm中管理作业步骤的开销要比管理单个作业要低得多。

[作业数组](https://slurm.schedmd.com/job_array.html)是管理具有相同资源需求的一系列批处理作业的有效机制。大多数Slurm命令都可以管理作业数组，可以将其视为单个实体（例如，使用单个命令删除整个作业数组）或作为单独的元素（任务）。

## MPI

MPI的使用取决于所使用的MPI类型。这些不同的MPI实现有三种基本不同的操作模式。

* Slurm直接启动任务，并通过PMI2或PMIx API执行通信的初始化（受大多数现代MPI实现支持）。
* Slurm为作业创建资源分配，然后mpirun使用Slurm的基础设施启动任务（旧版本的OpenMPI）。
* Slurm为作业创建资源分配，然后mpirun使用Slurm以外的某种机制启动任务，例如SSH或RSH。这些任务是在Slurm的监控或控制之外启动的。建议配置Slurm的epilog以在作业的分配被释放时清除这些任务。强烈推荐使用pam_slurm_adopt。

以下是使用Slurm的几种MPI变体的说明链接。

* [Intel MPI](https://slurm.schedmd.com/mpi_guide.html#intel_mpi)
* [MPICH2](https://slurm.schedmd.com/mpi_guide.html#mpich2)
* [MVAPICH2](https://slurm.schedmd.com/mpi_guide.html#mvapich2)
* [Open MPI](https://slurm.schedmd.com/mpi_guide.html#open_mpi)


