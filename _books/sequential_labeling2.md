---
layout:     post
title:      "序列标注算法(二)"
author:     "lili"
mathjax: true
excerpt_separator: <!--more-->
tags:
    - 自然语言处理
    - 序列标注
    - LSTM
    - LSTM-CRF
---

本文介绍序列标注算法，主要是LSTM-CRF模型。更多文章请点击[深度学习理论与实战：提高篇]({{ site.baseurl }}{% post_url 2019-03-14-dl-book %})。
<div class='zz'>转载请联系作者(fancyerii at gmail dot com)！</div>
 <!--more-->
 
**目录**
* TOC
{:toc}


## LSTM

我们也可以使用RNN/LSTM来实现序列标注，比如词性标注的例子，输入是词的序列，输出是对于的词性序列。损失函数是每个时刻的损失函数相加，而每个时刻的损失函数是softmax输出(概率)和真实标签的交叉熵。它的缺点是输出的tag是相互独立的，不管t-1时刻的tag是B-NP，RNN/LSTM在计算t时刻的时候都不会考虑t-1时刻的输出。

## LSTM-CRF
那怎么把LSTM和CRF结合呢？CRF是一个对数线性模型，它的形式为如下：

$$
p(y|x;w)=p(y_1,...,y_m|x_1,...,x_m;w)=\frac{exp(w \cdot \Phi(x,y))}{\sum_{y' \in \mathcal{Y}^m}exp(w \cdot \Phi(x,y'))}
$$

LSTM-CRF的思路就是用LSTM来为CRF提供特征。公式为：

$$
w \cdot \Phi(x,y) \equiv \sum_ih_i[y_i] + P_{y_i,y_{i-1}}
$$

其中$P_{y_i,y_{i-1}}$表示i-1时刻到i时刻的状态跳转特征。

之前我们在CRF里有大量的特征，而在LSTM-CRF里只有两个特征，一个是每个时刻LSTM的输出概率；另一个特征就是前一个tag和当前tag的“跳转概率”（注意这里是加了引号的，因为$P_{y_i,y_{i-1}}$可以大于1甚至是负数）。

我们可以这样来解读：LSTM-CRF里的CRF是根据什么特征来计算输出序列的概率$p(y=y_1,...,y_m \vert x=x_1,...,x_m)$的呢？一是根据LSTM的判断(LSTM输出的$h_i[y_i]$越大，那么i时刻的标签是$h_i$的可能性越大)；二是根据整个序列的跳转概率是否较大，一条路径的$\sum_iP_{y_i,y_{i-1}}$的跳转概率越大，那么LSTM-CRF选择这条路径的概率就越大。当然它是会综合考虑这两个因素。而普通的LSTM只是考虑$h_i[y_i]$，在i时刻选择$h_i$中最大的那个tag。

细心的读者可能会问，既然是两个特征$\sum_ih_i[y_i]$和$\sum_iP_{y_i,y_{i-1}}$，那按照CRF的定义应该是用参数把它们加权求和啊——$w_1 \, \sum_ih_i[y_i] + w_2 \, \sum_iP_{y_i,y_{i-1}}$。这两个参数哪去了呢？答案是不需要了，因为在CRF里，特征一般都是0-1的indicator特征，所以需要加权求和。但是在LSTM-CRF里，跳转矩阵P本身就是参数，而$h_i[y_i]$也是有LSTM产生的，LSTM里就有大量参数，因此没有必要再加两个参数了。

所以LSTM-CRF最大的特点就是：由LSTM提供特征，而且特征是有参数的，是可以学习的！因此它可能根据不同问题学到各种合适的底层特征；而CRF的特征是人工定义出来的，不可变的，我们最多改改这个特征的参数。另外在实际计算中，为了防止下溢，实际使用的是log域的值：

$$
\begin{split}
\Phi(x,y)=\sum_ilog\phi_i(x,y) \\
=\sum_ilog\phi_{EMIT}(y_i \rightarrow x_i)+log\phi_{TRANS}(y_{i-1} \rightarrow y_i) \\
=\sum_ih_i[y_i]+P_{y_i,y_{i-1}}
\end{split}
$$

也就是说，这里的跳转概率矩阵P和发射概率$h_i$都是已经转换到log域的了。后面在代码分析时会更加具体介绍到。

## LSTM-CRF的PyTorch代码



原理明白了，下面我们通过简单的PyTorch代码来了解更多的细节。我们用LSTM-CRF来做命名实体识别(NER)。

### 训练主流程

```
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4

# 准备训练数据
training_data = [(
	"the wall street journal reported today that apple corporation made money".split(),
	"B I I I O O O B I O O".split()
	), (
	"georgia tech is a university in georgia".split(),
	"B I O O O O B".split()
	)]

word_to_ix = {}
for sentence, tags in training_data:
	for word in sentence:
		if word not in word_to_ix:
			word_to_ix[word] = len(word_to_ix)

tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
 
# 训练
for epoch in range(300):
	for sentence, tags in training_data: 
		# 清除梯度
		model.zero_grad()
		
		# 准备一个batch(batch_size=1)的训练数据 
		sentence_in = prepare_sequence(sentence, word_to_ix)
		targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
		
		# forward计算 
		loss = model.neg_log_likelihood(sentence_in, targets)
		
		# 计算梯度，更新参数 
		loss.backward()
		optimizer.step()

# 预测
with torch.no_grad():
	precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
	print(model(precheck_sent))
```
为了演示，这里只有两个训练数据。代码首先是对数据的预处理，主要是word_to_ix，用于实现从词到id的映射；tag_to_ix，用于实现从tag到id的映射。和前面介绍的HMM类似，我们加入了特殊的开始状态(START_TAG)和结束状态(STOP_TAG)。接着定义BiLSTM_CRF，这个类是LSTM-CRF算法的实现，我们后面会重点介绍。然后定义optimizer，接着进行训练，最后是预测。

### prepare_sequence

这个函数很简单，用于把词序列变成id序列。
```
def prepare_sequence(seq, to_ix):
	idxs = [to_ix[w] for w in seq]
	return torch.tensor(idxs, dtype=torch.long)
```

### argmax
```
def argmax(vec):
	_, idx = torch.max(vec, 1)
	return idx.item()
```
这个函数的输入是(batch=1, num_tags=5)，然后寻找用torch.max对第2维求最大，从而得到batch里每一行最大的下标，idx是长度为batch的向量。这里我们的输入batch=1，因此用idx.item得到这一个最大的下标。

### log_sum_exp
```
def log_sum_exp(vec):
	max_score = vec[0, argmax(vec)]
	max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
	return max_score + 
		torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
```
这个函数实现$log(\sum_ie^{x_i})$，为了帮助计算的stable，防止指数溢出，使用了下面的小技巧：

$$
log(\sum_i e^{x_i})=log(e^{\underset{j}{max}x_j}\sum_i e^{x_i-\sum_jx_j})=\underset{j}{max}x_j+log(\sum_i e^{x_i-\underset{j}{max}x_j})
$$

由于$x_i-\underset{j}{max}x_j \le 0$，因此不会存在$e^x_i$太大导致的溢出。

代码实现有些繁琐，注意因为vec是(batch=1,5)，然后通过前面的argmax找到最大的哪个数的下标，然后用vec[0,下标]找到这个最大的数(为什么不直接实现一个max而是argmax？)。然后用expand把最大的数变成(1, 5)，这样它的shape和vec是一样的就可以减了。其实PyTorch是支持Braodcasting的，没有必要这么复杂。这个函数的作用后面会讲到。


### BiLSTM_CRF的构造函数
```
class BiLSTM_CRF(nn.Module):
	def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
		super(BiLSTM_CRF, self).__init__()
		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.vocab_size = vocab_size
		self.tag_to_ix = tag_to_ix
		self.tagset_size = len(tag_to_ix)
		
		self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
		self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
		num_layers=1, bidirectional=True)
		
		# Maps the output of the LSTM into tag space.
		self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
		
		# Matrix of transition parameters.  Entry i,j is the score of
		# transitioning *to* i *from* j.
		self.transitions = nn.Parameter(
			torch.randn(self.tagset_size, self.tagset_size))
		
		# These two statements enforce the constraint that we never transfer
		# to the start tag and we never transfer from the stop tag
		self.transitions.data[tag_to_ix[START_TAG], :] = -10000
		self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000
		
		self.hidden = self.init_hidden()
```
BiLSTM_CRF的构造函数的参数为：

* vocab_size 词典大小，这里是17
* tag_to_ix tag到id的映射
* embedding_dim embedding后的大小，这里是5
* hidden_dim LSTM输出隐状态的大小，这里是4


重要的代码如下，请读者仔细阅读注释：
```
# 创建Embedding层，shape是(vocab_size, embedding_dim)
self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
# 因为是双向LSTM，为了输出hidden_dim，我们定义LSTM的hidden_size是hidden_dim/2，这样双向的输出就是hidden_dim。
self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
	num_layers=1, bidirectional=True)

# 输出的logits
self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

# 跳转矩阵的参数，transitions[i,j]表示从j到i的跳转“概率”，注意下标是反过来的。
self.transitions = nn.Parameter(
	torch.randn(self.tagset_size, self.tagset_size))
 
# 不会有哪个状态跳到开始状态，因此设置为很大分负数
self.transitions.data[tag_to_ix[START_TAG], :] = -10000
# 结束状态也不会跳到其它的状态。
self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000
```

### forward

这个函数实现LSTM-CRF的前向计算，也就是根据输入x计算最可能的y及其概率。
```
def forward(self, sentence):  # dont confuse this with _forward_alg above.
    # 使用LSTM计算特征
    lstm_feats = self._get_lstm_features(sentence)
    
    # viterbi解码寻找最优路径
    score, tag_seq = self._viterbi_decode(lstm_feats)
    return score, tag_seq
```

这个函数首先使用函数_get_lstm_features计算每个时刻的输出$h_t$，它是一个长度为tagset_size的向量，表示输出每一个tag的概率。接着使用_viterbi_decode函数进行viterbi求解最优路径。

### _get_lstm_features
```
def _get_lstm_features(self, sentence):
    self.hidden = self.init_hidden()
    embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
    lstm_out, self.hidden = self.lstm(embeds, self.hidden)
    lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
    lstm_feats = self.hidden2tag(lstm_out)
    return lstm_feats
```
这个函数很简单，输入是句子的id，比如第一个句子有11个词，因此是一个长度为11的一维整数Tensor。代码首先初始化隐状态(包括LSTM的cell和hidden state)，然后对sentence进行embedding。得到(11,5)的Tensor，因为LSTM要求输入是(time,batch,numFeatures)，因此把它变成(11,1,5)的embeds。

然后用lstm计算得到lstm_out，这是一个(11,1,4)的Tensor，然后我们reshape成(11,4)得到lstm_out，然后再用hidden2tag把它变成(11,5)的输出。这个输出表示每个时刻(总共11)输出5个不同tag的"概率"(其实是logits，没有用softmax变成概率)。

### _viterbi_decode
```
def _viterbi_decode(self, feats):
    backpointers = []
    
    # init_vvars是(1, self.tagset_size)，表示初始的概率，除了START_TAG是0
    # 其它状态都是-10000。因此一开始就处于START_TAG状态。
    init_vvars = torch.full((1, self.tagset_size), -10000.)
    init_vvars[0][self.tag_to_ix[START_TAG]] = 0
    
    # forward_var就是类似HMM的alpha，表示前一个时刻的最优路径的概率，forward_var[i]表示最后一个时刻的状态是i的最优路径的概率。
    forward_var = init_vvars
    for feat in feats:
	    bptrs_t = []  # 这一个时刻的backpointers，每个状态占有一个
	    viterbivars_t = []  # 当前时刻的viterbi变量
	    
	    for next_tag in range(self.tagset_size):
		    next_tag_var = forward_var + self.transitions[next_tag]
		    best_tag_id = argmax(next_tag_var)
		    bptrs_t.append(best_tag_id)
		    viterbivars_t.append(next_tag_var[0][best_tag_id].view(1)) 
 
	    forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
	    backpointers.append(bptrs_t)
    
    # 到STOP_TAG的跳转概率
    terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
    best_tag_id = argmax(terminal_var)
    path_score = terminal_var[0][best_tag_id]
    
    # 反向回溯 
    best_path = [best_tag_id]
	for bptrs_t in reversed(backpointers):
	    best_tag_id = bptrs_t[best_tag_id]
	    best_path.append(best_tag_id)
    # 去掉START_TAG 
    start = best_path.pop()
    assert start == self.tag_to_ix[START_TAG]  # Sanity check
    best_path.reverse()
    return path_score, best_path
```

最核心的代码如下，这就是前向算法的递推过程：
```
for feat in feats:
    bptrs_t = []  # 这一个时刻的backpointers，每个状态占有一个
    viterbivars_t = []  # 当前时刻的viterbi变量
    
    for next_tag in range(self.tagset_size):
	    next_tag_var = forward_var + self.transitions[next_tag]
	    best_tag_id = argmax(next_tag_var)
	    bptrs_t.append(best_tag_id)
	    viterbivars_t.append(next_tag_var[0][best_tag_id].view(1)) 
    
    forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
    backpointers.append(bptrs_t)
```

feat是当前时刻LSTM的输出，这里是长度为5的向量，表示输出5种tag的“概率”。forward_var的shape是(1,5)，其实就看成(5,)就好了，表示前一个时刻的5条最优路径。接下来遍历5个tag，逐一计算forward_var + self.transitions[next_tag]。比如forward_var为:
```
tensor([[   -1.3282,     1.6011,     0.2629, -9999.5625,    -0.6535]])
```
表示前一个时刻状态为0的最优路径的概率是-1.3282；前一个时刻状态为1的最优路径是1.6001，。。。。

而transitions是(5,5)的矩阵，transitions[next_tag]是长度为5的向量，表示每个状态跳转到next_tag的概率(还记得下标是反过来的吗，这里就能看出这样做的好处了)，比如next_tag为：
```
tensor([-1.1811e-01, -1.4420e+00, -1.1108e+00, -1.1187e+00, -1.0000e+04])
```
它表示0->next_tag的概率是-1.1811e-01；1->-1.1811e-01的概率是-1.4420e+00。

因此把这两个向量加起来还是一个长度为5的向量：
```
tensor([[-1.4463e+00,  1.5910e-01, -8.4789e-01, -1.0001e+04, -1.0001e+04]])
```
 

注意，标准的前向算法需要再加上当前的发射概率feat[next_tag]，不过我们的目的是求最大值，因此可以先不加。然后我们是用argmax找到最大的值对应的下标，这个例子最大的是1，对应的值是1.5910e-01。它说明最优的路径在上一个时刻经过状态1。我们把最优值1.5910e-01放到变量viterbivars_t里，把backtrace指针1放到bptrs_t里。

变量所有5个next_tag后，我们得到viterbivars_t的是这个时刻的5个最优路径的值，分别表示当前时刻处于状态0,1,2,3,4的最优路径值。但是这个最优路径值没有加上发射概率，因此最后需要用torch.cat(viterbivars_t) + feat得到最终的概率值，然后reshape成新的forward_var。最后把t时刻的backtrace指针都加到backpointers里。把发射概率放到最后计算是为了节省计算量，其实这样也是可以的：
```
forward_var + self.transitions[next_tag]+feat[next_tag]
```
注意forward_var是(1,5)，self.transitions[next_tag]是(5,)，而是一个数。这里的加法会broadcasting，因此feat[next_tag]会加5次，但是我们只是为了求最大值。每个数都加一个相同的值不改变相对大小，因此可以先不加，然后最后统一一起加。前面介绍了forward的计算，也就是根据输入x，计算最优的路径y。接下来看怎么计算loss。


### neg_log_likelihood

在介绍代码之前，我们来分析一下loss怎么定义。前面我们的CRF模型是使用最大似然来估计的，也就是选择参数w使得对数似然L(w)最大。因为我们使用梯度下降，所有可以去负数，也就是求负对数似然最小的参数。我们来回顾一下CRF的公式：

$$
\begin{split}
p(y|x)=\frac{exp(\Phi(x,y))}{\sum_{y'}exp(\Phi(x,y'))} \\
-logp(y|x)=log\sum_{y'}exp(\Phi(x,y')) -\Phi(x,y)
\end{split}
$$

第一项很好计算，而第二项需要使用前向(动态规划)算法。下面我们来看代码：

```
def neg_log_likelihood(self, sentence, tags):
    feats = self._get_lstm_features(sentence)
    forward_score = self._forward_alg(feats)
    gold_score = self._score_sentence(feats, tags)
    return forward_score - gold_score
```
首先是LSTM的计算，这是通过_get_lstm_features函数实现的，前面以及分析过了。forwar_score就是用前向算法计算分母，而gold_score是第一项。

我们先看比较简单的_score_sentence函数，它是用来计算$\Phi(x,y)$。

### _score_sentence

```
def _score_sentence(self, feats, tags): 
    score = torch.zeros(1)
    tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], 
	    dtype=torch.long), 	tags])
    for i, feat in enumerate(feats):
	    score = score + \
		    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
    score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
    return score
```

代码比较简单，首先是把字符串的tags变成id。因为第0个时刻是特殊的START_TAG，因此有"feat[tags[i + 1]]"和"self.transitions[tags[i + 1], tags[i]]"。
最后还要加上跳转到STOP_TAG的概率"self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]"。

### _forward_alg
```
def _forward_alg(self, feats):
    init_alphas = torch.full((1, self.tagset_size), -10000.)
    init_alphas[0][self.tag_to_ix[START_TAG]] = 0.
     
    forward_var = init_alphas
    
    # 遍历每一个词
    for feat in feats:
	    alphas_t = []  # 当前时刻的前向概率
	    for next_tag in range(self.tagset_size):
		    # 发射概率，它和之前的tag无关，需要broadcasting
		    emit_score = feat[next_tag].view(
			    1, -1).expand(1, self.tagset_size)
		    # 跳转概率
		    trans_score = self.transitions[next_tag].view(1, -1)
		    # 计算新的前向概率(没有log-sum-exp)
		    next_tag_var = forward_var + trans_score + emit_score
		    # log-sum-exp
		    alphas_t.append(log_sum_exp(next_tag_var).view(1))
	    forward_var = torch.cat(alphas_t).view(1, -1)
    terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
    alpha = log_sum_exp(terminal_var)
    return alpha
```
前向算法和viterbi算法很类似，只不过是把求最大变成了求和。我们来简单的回归一下前向概率$\alpha$的定义：

$$
\begin{split}
\alpha_t(j)=P(o_1,...,o_t,s_t=j|\lambda) \\
\alpha_t(j)=\sum_{i=1}^{N}\alpha_{t-1}(i)a_{ij}b_j(o_t)
\end{split}
$$

因为发射概率和跳转概率都是很小的值，当序列很长的时候，很多小的数相乘容易导致下溢，因此我们在log域进行计算：

$$
\begin{split}
log\alpha_t(j)=log[\sum_{i=1}^{N}\alpha_{t-1}(i)a_{ij}b_j(o_t)] \\
=log[\sum_{i=1}^{N} e^{log \alpha_{t-1}(i)} e^{log(a_{ij})} e^{log(b_j(o_t))}] \\
=log[\sum_{i=1}^{N} e^{log \alpha_{t-1}(i)+log(a_{ij})+log(b_j(o_t))}]
\end{split}
$$

这就是函数log-sum-exp的作用。细心的读者可能会问，为什么代码里是：
```
next_tag_var = forward_var + trans_score + emit_score
```
而不是：
```
next_tag_var = forward_var + log(trans_score) + log(emit_score)
```

因为我们这里的self.transitions和LSTM输出的feats以及是log域的了。所有前面我们的_score_sentence函数计算logp(y|x)也是直接把跳转概率和发射概率相加就行了，包括再之前的Viterbi算法，所有需要概率相乘的地方都改成加法了。计算出了loss之后，我们就可以让PyTorch自动帮我们计算反向梯度和更新参数了。






