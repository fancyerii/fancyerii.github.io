---
layout:     post
title:      "Huggingface Transformers在padding之后结果差异分析" 
author:     "lili" 
mathjax: true
sticky: false
excerpt_separator: <!--more-->
tags:
    - huggingface
    - transformers
    - padding
---

本文分析了Transformers在padding之后计算的结果的差异原因，对熟悉Transformers源代码以及调试问题有一定帮助。

<!--more-->

**目录**
* TOC
{:toc}


## 问题

我们这里要解决的问题来自[LLaMA2 - tokenizer padding affecting logits (even with attention_mask)](https://discuss.huggingface.co/t/llama2-tokenizer-padding-affecting-logits-even-with-attention-mask/50213/3)。首先我们来了解和复现一下这个问题。

```python
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer 


model_id="/nas/lili/models_hf/Llama-2-7b-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_id, truncation_side='left', padding_side='right')
tokenizer.pad_token = tokenizer.eos_token
model = LlamaForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map={"":0})

input_prompt = "I've got a lovely bunch of coconuts do do do dooo"
input_prompt_tokenized = tokenizer(input_prompt, return_tensors="pt").to('cuda')
print(input_prompt_tokenized)


input_prompt_padded_tokenized = tokenizer(input_prompt, return_tensors="pt", padding="max_length", max_length=50).to('cuda')
print(input_prompt_padded_tokenized)


# Mask labels so that the values corresponding to the padding token do not contribute to the loss

input_ids_masked = torch.zeros(input_prompt_padded_tokenized.input_ids.shape, dtype=torch.int64).to('cuda')
torch.where(input_prompt_padded_tokenized.input_ids == tokenizer.pad_token_id,
            torch.tensor(-100, dtype=torch.int64),
            input_prompt_padded_tokenized.input_ids,
            out=input_ids_masked)
print(input_ids_masked)


# Calculate logits and loss from the model using unpadded input

output1 = model(
    input_prompt_tokenized.input_ids,
    attention_mask=input_prompt_tokenized.attention_mask,
    labels=input_prompt_tokenized.input_ids
)
print(f"Loss (no padding): {output1.loss}")

print(f"Logits (no padding): {output1.logits}")


# Calculate the logits and loss from the model using padded input (should be the same)

output2 = model(
    input_prompt_padded_tokenized.input_ids,
    attention_mask=input_prompt_padded_tokenized.attention_mask,
    labels=input_ids_masked
)

print(f"Loss (padding): {output2.loss}")

print(f"Logits (padding): {output2.logits}")

```

output1和output2唯一的区别就是后者使用了padding，因此前者的输入是18个token，后者是50个token(32个是padding token)。另外值得注意的是第二个输入对应的input_ids_masked是使用torch.where来构造：

<a>![](/img/paddingdebug/1.png)</a>

也就是把padding的位置对应的label设置为-100，这样在计算loss的时候不会计算padding的token。因为即使padding的attention mask是0，仍然会有一些loss。注意，torch.where的input和other参数都可以支持广播，因此这里如果是padding token，那么输入是一个标量torch.tensor(-100, dtype=torch.int64)，否则是长度为50的向量input_prompt_padded_tokenized.input_ids。

上面的输出是：
```
{'input_ids': tensor([[    1,   306, 29915,   345,  2355,   263, 12355,   873, 14928,   310,
          1302,   535,  8842,   437,   437,   437,   437,  3634]],
       device='cuda:0'), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
       device='cuda:0')}
{'input_ids': tensor([[    1,   306, 29915,   345,  2355,   263, 12355,   873, 14928,   310,
          1302,   535,  8842,   437,   437,   437,   437,  3634,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2]],
       device='cuda:0'), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0]], device='cuda:0')}
tensor([[    1,   306, 29915,   345,  2355,   263, 12355,   873, 14928,   310,
          1302,   535,  8842,   437,   437,   437,   437,  3634,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100]],
       device='cuda:0')
Loss (no padding): 2.271912097930908
Logits (no padding): tensor([[[-6.9180e+00, -1.0723e+00,  2.6934e+00,  ..., -4.2578e+00,
          -5.8789e+00, -4.5234e+00],
         [-9.4453e+00, -1.2516e+01, -1.2471e+00,  ..., -5.1328e+00,
          -8.7500e+00, -3.6348e+00],
         [-3.9004e+00,  1.2627e-02,  5.6992e+00,  ..., -3.9551e-01,
          -1.4980e+00,  2.8320e-01],
         ...,
         [-3.2383e+00, -3.9551e+00,  1.2812e+01,  ...,  3.8599e-01,
          -2.6230e+00, -1.0186e+00],
         [-2.7188e+00, -2.7988e+00,  1.3062e+01,  ...,  7.2119e-01,
          -1.9600e+00, -6.4160e-01],
         [-2.5156e+00, -1.4639e+00,  1.2578e+01,  ...,  1.9434e-01,
          -2.3145e+00, -9.1113e-01]]], device='cuda:0',
       grad_fn=<ToCopyBackward0>)
Loss (padding): 2.2886083126068115
Logits (padding): tensor([[[ -6.8242,  -1.2178,   2.7402,  ...,  -4.2617,  -5.8164,  -4.4297],
         [ -9.5312, -12.6875,  -1.3076,  ...,  -5.0625,  -8.7812,  -3.6660],
         [ -3.7285,   0.0786,   5.7227,  ...,  -0.2178,  -1.4248,   0.4294],
         ...,
         [  0.4036,  33.1250,   9.3750,  ...,   0.9189,   1.5947,   1.7139],
         [ -0.6445,  31.4688,  10.5234,  ...,   0.4934,   1.2793,   1.2725],
         [ -1.9326,  27.8594,  11.3984,  ...,   0.1417,   0.7139,   0.7173]]],
       device='cuda:0', grad_fn=<ToCopyBackward0>)

```

通过仔细对比，我们可以发现没有padding的时候loss是2.2719，而有padding时是2.2886，虽然相差不多，但是0.01的差值对于模型来说也不小了。另外如果我们对比logits(output2我们只需要看前18个，因为后面的是padding)，也会发现有不小的差异。

## 排查

### 第一次尝试

为了排查这个问题，我一开始使用了很笨的方法：打开两个vscode，分别单步调试两种代码，然后对比每一步的结果。但是返回的数字的差异实在太小，而且vscode在显示浮点数时默认只显示4位，如下图所示：


<a>![](/img/paddingdebug/2.png)</a>

我们会发现vscode只显示了4位有效数字，实际它们不是完全相同。

### 对比hidden state

既然最终输出的logits不相同，那么第一次两者不同出现在哪一层了？为了输出hidden state，需要增加output_hidden_states=True：

```python
output1 = model(
    input_prompt_tokenized.input_ids,
    attention_mask=input_prompt_tokenized.attention_mask,
    labels=input_prompt_tokenized.input_ids,
    output_hidden_states=True
)
```

另外我写了一段简单的代码对比：

```python
hidden_states1 = output1.hidden_states
hidden_states2 = output2.hidden_states
for layer in range(len(hidden_states1)):
    state1 = hidden_states1[layer][0]
    state2 = hidden_states2[layer][0]
    len, dim = state1.shape
    for i in range(len):
        hidden1 = state1[i]
        hidden2 = state2[i]
        mismatch_idx = -1
        for j in range(dim):
            v1 = hidden1[j]
            v2 = hidden2[j]
            if v1 != v2:
                mismatch_idx = j
                f1 = v1.cpu().detach().numpy().item()
                f2 = v2.cpu().detach().numpy().item()
                f1 = f"{f1:.4f}"
                f2 = f"{f2:.4f}"
                if f1 == f2:
                    print("here")
                #break
        
        if mismatch_idx != -1:
            print(f"layer={layer}, mismatch i={i} j={mismatch_idx}")
            #break

```

输出为：layer=2, mismatch i=0 j=965。这说明在第2层开始出现了不一致。注意：输出总共33个值，其中第0层表示输入embedding层。

### 修改modeling_llama.py

现在我们定位到第二层出现了hidden state不一致，那么具体是哪个module第一次出现不一致呢？这个就需要我们修改代码了。通过调试我们可以发现model(...)的调用路径是：

```
LlamaForCausalLM.forward -> LlamaModel.forward -> LlamaDecoderLayer.forward
```

让我们以相反的顺序来修改代码，增加调试信息。

#### LlamaDecoderLayer.forward

我们需要把每一步的中间结果都保存到debug_outputs里，然后返回到上一层。中间结果包括：

* hidden 输入hidden_states
* input_layernorm layernorm后的结果
* self_attn self_attn后的结果
* add_res 加上残差的结果
* post_attention_layernorm post_attention_layernorm之后的结果
* mlp mlp之后的结果
* final 最终的结果

```python
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        debug_outputs = {"hidden": hidden_states}
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        debug_outputs["input_layernorm"] = hidden_states
        # Self Attention

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
 
        debug_outputs["self_attn"] = hidden_states
        hidden_states = residual + hidden_states
        debug_outputs["add_res"] = hidden_states
        # Fully Connected

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        debug_outputs["post_attention_layernorm"] = hidden_states

        hidden_states = self.mlp(hidden_states)
        debug_outputs["mlp"] = hidden_states
        hidden_states = residual + hidden_states
        debug_outputs["final"] = hidden_states
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        outputs +=(debug_outputs,)
        return outputs
```


#### LlamaModel.forward

它的主要功能是计算attention mask，然后迭代self.layers，这里我们注意的修改就是读取decoder_layer输出tuple的最后一个值(前面的debug_outputs)，把它加到all_debug里。

```python
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
		...
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    debug=debug
                )

            hidden_states = layer_outputs[0]
            ....
            all_debug += (layer_outputs[-1],)
```


#### LlamaForCausalLM.forward

代码如下：

```python
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens

            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism

            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            all_debug=outputs.all_debug
        )
```

它的主要功能是调用self.model()，然后计算loss，最后返回CausalLMOutputWithPast。我们为了返回debug信息，需要修改CausalLMOutputWithPast，给它增加一行：

```
all_debug: Optional[Tuple[torch.FloatTensor]] = None
```

然后在最后的CausalLMOutputWithPast里增加all_debug。

### 对比debug信息

我们写一段简单代码对比两个模型的debug结果：

```python
layer = 0
found_diff = False
for dict1, dict2 in zip(output1.all_debug, output2.all_debug):
    if found_diff:
        break
    print(f"layer={layer}")
    layer += 1
    for key, value in dict1.items():
        value2 = dict2[key]
        print(f"key: {key}, v1: {value.shape}, v2: {value2.shape}")
        dims = value.dim()
        if dims == 3:
            v1 = value[0]
            v2 = value2[0]
            dim1, dim2 = v1.shape
            for i in range(dim1):
                hidden1 = v1[i]
                hidden2 = v2[i]
                mismatch_idx = -1
                for j in range(dim2): 
                    if hidden1[j] != hidden2[j]:
                        mismatch_idx = j
                        break
                
                if mismatch_idx != -1:
                    print(f"mismatch i={i} j ={mismatch_idx}")    
                    found_diff = True
                    break
        elif dims == 4:
            found = False
            v1 = value[0]
            v2 = value2[0]
            d1,d2,d3 = v1.shape
            for i in range(d1):
                if found:
                    break
                for j in range(d2):
                    if found :
                        break
                    for k in range(d3):
                        if v1[i][j][k] != v2[i][j][k]:
                            found = True
                            found_diff = True
                            print(f"mismatch i={i} j={j} k={k}")   
                            break

```

输出为：

```
layer=0
key: hidden, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096]) 
key: input_layernorm, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: attn_input, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: q_proj, v1: torch.Size([4096, 4096]), v2: torch.Size([4096, 4096])
key: query_states, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: key_states, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: value_states, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: attn_weights, v1: torch.Size([1, 32, 18, 18]), v2: torch.Size([1, 32, 50, 50])
key: attn_weights2, v1: torch.Size([1, 32, 18, 18]), v2: torch.Size([1, 32, 50, 50])
key: attn_weights3, v1: torch.Size([1, 32, 18, 18]), v2: torch.Size([1, 32, 50, 50])
key: attn_weight4, v1: torch.Size([1, 32, 18, 18]), v2: torch.Size([1, 32, 50, 50])
key: attn_output, v1: torch.Size([1, 32, 18, 128]), v2: torch.Size([1, 32, 50, 128])
key: attn_output2, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: self_attn, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: add_res, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: post_attention_layernorm, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: mlp, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: final, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
layer=1
key: hidden, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: input_layernorm, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: attn_input, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: q_proj, v1: torch.Size([4096, 4096]), v2: torch.Size([4096, 4096])
key: query_states, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: key_states, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: value_states, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: attn_weights, v1: torch.Size([1, 32, 18, 18]), v2: torch.Size([1, 32, 50, 50])
key: attn_weights2, v1: torch.Size([1, 32, 18, 18]), v2: torch.Size([1, 32, 50, 50])
key: attn_weights3, v1: torch.Size([1, 32, 18, 18]), v2: torch.Size([1, 32, 50, 50])
key: attn_weight4, v1: torch.Size([1, 32, 18, 18]), v2: torch.Size([1, 32, 50, 50])
key: attn_output, v1: torch.Size([1, 32, 18, 128]), v2: torch.Size([1, 32, 50, 128])
key: attn_output2, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: self_attn, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: add_res, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: post_attention_layernorm, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: mlp, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
mismatch i=0 j =965


```

我们发现在第二层(layer==1)时的mlp第一次出现了不一致。于是我们开始调试LlamaMLP类的代码。

### LlamaMLP

调试时突然有了一个发现，LlamaMLP实际的类和\_\_init\_\_的代码不一致！我们看一下代码：

```python
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
```
比如self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)，但是调试时我发现它变成了Linear8bitLt：

<a>![](/img/paddingdebug/3.png)</a>


跟踪代码发现是bitsandbytes/nn/modules.py。回头仔细看来一下前面的代码，果然发现调用LlamaForCausalLM.from_pretrained时传入了参数load_in_8bit=True。int8压缩之前也稍微研究过，读者可以参考[A Gentle Introduction to 8-bit Matrix Multiplication for transformers at scale using Hugging Face Transformers, Accelerate and bitsandbytes](https://huggingface.co/blog/hf-bitsandbytes-integration)和[LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale论文解读](http://fancyerii.github.io/2024/01/16/int8/)。难道是Linear8bitLt引入的不确定性？好像有可能，因为Linear8bitLt依赖离群值(outlier)，而且因为压缩可能会放大微小的差异。

### load_in_8bit=False

因此我们把这个参数设置为False再来跑一下：

```
Loss (no padding): 2.2338905334472656
Logits (no padding): tensor([[[-12.9832,  -7.4134,  -0.4327,  ...,  -6.8297,  -8.0879,  -7.5863],
         [ -9.5230, -12.2163,  -1.1083,  ...,  -5.0527,  -8.9276,  -3.6419],
         [ -3.5179,   0.6801,   6.0908,  ...,  -0.2554,  -1.3332,   0.4213],
         ...,
         [ -3.0661,  -3.5820,  12.6735,  ...,   0.1156,  -2.5818,  -1.0295],
         [ -2.6741,  -2.5679,  12.8021,  ...,   0.4409,  -2.2196,  -0.7664],
         [ -2.1947,  -1.0629,  12.5872,  ...,   0.0837,  -2.1596,  -0.9412]]],
       device='cuda:0', grad_fn=<UnsafeViewBackward0>)
Loss (padding input & masking labels): 2.2338929176330566
Logits (padding input & masking labels): tensor([[[-1.2983e+01, -7.4134e+00, -4.3273e-01,  ..., -6.8297e+00,
          -8.0880e+00, -7.5864e+00],
         [-9.5230e+00, -1.2216e+01, -1.1083e+00,  ..., -5.0527e+00,
          -8.9276e+00, -3.6419e+00],
         [-3.5179e+00,  6.8010e-01,  6.0908e+00,  ..., -2.5537e-01,
          -1.3332e+00,  4.2135e-01],
         ...,
         [ 2.1687e-01,  3.3158e+01,  9.6556e+00,  ...,  5.6584e-01,
           1.4918e+00,  1.4965e+00],
         [-1.6480e+00,  2.9136e+01,  1.1903e+01,  ..., -1.6091e-02,
           8.9055e-01,  6.3661e-01],
         [-3.6790e+00,  2.4334e+01,  1.2093e+01,  ..., -8.4835e-01,
          -2.7381e-01, -5.9110e-01]]], device='cuda:0',
       grad_fn=<UnsafeViewBackward0>)

```

我们发现loss和logits的差距变得非常小了，但是还是不完全相同。下面是对比debug信息的结果：

```
layer=0
key: hidden, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: input_layernorm, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: self_attn, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
mismatch i=0 j =0
key: add_res, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
mismatch i=0 j =0

```

对比hidden_states：

```
layer: 0
layer: 1
layer=1, mismatch i=0 j=0
```

从上面的输出来看，两个输入在第一层就结果不同了，而且是在self_attn之后。

### self_attn

通过跟踪，我们发现它使用的是LlamaSdpaAttention。它最终使用的是torch.nn.functional.scaled_dot_product_attention：

```
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.

            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )
```

但是我通过调试发现进入scaled_dot_product_attention之前的值都是相同的，而这个函数又是无法跟踪的(torch的c++代码)。研究了很久也没法找到线索。后来发现了这段代码：

```python
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
```

顺着它找到：

```python
class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

```
以及：
```python
LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention_2": LlamaFlashAttention2,
    "sdpa": LlamaSdpaAttention,
}
```

看了一下LlamaAttention的代码，发现它是最原始的python实现，非常适合调试。因此修改：

```python
model = LlamaForCausalLM.from_pretrained(model_id, load_in_8bit=False, device_map={"":0}, _attn_implementation="eager")
```

注意：平时开发时不能使用_attn_implementation，因为它是内部参数，随时可能被修改。这里只是调试使用。

### LlamaAttention

我们修改LlamaAttention.forward，增加调试信息：

```python
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        
        bsz, q_len, _ = hidden_states.size()
        debug_info = {"attn_input": hidden_states}
        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            debug_info["q_proj"] = self.q_proj.weight
            debug_info["k_proj"] = self.k_proj.weight
            debug_info["v_proj"] = self.v_proj.weight
            debug_info["query_states"] = query_states
            debug_info["key_states"] = key_states
            debug_info["value_states"] = value_states

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models

            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        debug_info["attn_weights"] = attn_weights
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
        debug_info["attn_weights2"] = attn_weights
        # upcast attention to fp32

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        debug_info["attn_weights3"] = attn_weights
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        debug_info["attn_weight4"] = attn_weights
        attn_output = torch.matmul(attn_weights, value_states)
        debug_info["attn_output"] = attn_output
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)
        debug_info["attn_output2"] = attn_output
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value, debug_info

```

这里多返回了一个debug_info，因此调用的代码LlamaDecoderLayer.forward也需要修改。

```python
        hidden_states, self_attn_weights, present_key_value, *debug_infos = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        if debug_infos:
            debug_outputs.update(debug_infos[0])
```

### 再次比较

```
layer=0
key: hidden, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: input_layernorm, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: attn_input, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: q_proj, v1: torch.Size([4096, 4096]), v2: torch.Size([4096, 4096])
key: k_proj, v1: torch.Size([4096, 4096]), v2: torch.Size([4096, 4096])
key: v_proj, v1: torch.Size([4096, 4096]), v2: torch.Size([4096, 4096])
key: query_states, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
mismatch i=0 j =0
```

通过上面的输出，我们发现query_states = self.q_proj(hidden_states)在相同的输入时结果产生了差异。当然这里我说的输入相同指的是v2[:,:18,:]==v1，而输出query_states的v2[:,:18,:]!=v1。而模型的参数self.q_proj显然是一样的。这是怎么回事呢？

### 保存现场

为了便于复现，我把这些变量保存了下来：

```python
dict1 = output1.all_debug[0]
dict2 = output2.all_debug[0]
q_proj1 = dict1["q_proj"]
q_proj2 = dict2["q_proj"]
query_states1 = dict1["query_states"]
query_states2 = dict2["query_states"]
torch.save(q_proj1, "q_proj1.pt")
torch.save(q_proj2,  "q_proj2.pt")
torch.save(query_states1, "query_states1.pt")
torch.save(query_states2, "query_states2.pt")
torch.save(dict1["attn_input"], "attn_input1.pt")
torch.save(dict2["attn_input"], "attn_input2.pt")
```


### 复现问题

下面我们脱离huggingface transformers来复现不一致的现象：

```python
import torch
from torch import nn
q_proj1 = torch.load("q_proj1.pt")
q_proj2 = torch.load("q_proj2.pt")
query_states1 = torch.load("query_states1.pt")
query_states2 = torch.load("query_states2.pt")
attn_input1 = torch.load("attn_input1.pt")
attn_input2 = torch.load("attn_input2.pt")


print(f"q_proj1: {q_proj1.shape}, dtype: {q_proj1.dtype}, device: {q_proj1.get_device()}")
print(f"q_proj2: {q_proj2.shape}, dtype: {q_proj2.dtype}, device: {q_proj2.get_device()}")
print(f"query_states1: {query_states1.shape}, dtype: {query_states1.dtype}, device: {query_states1.get_device()}")
print(f"query_states2: {query_states2.shape}, dtype: {query_states2.dtype}, device: {query_states2.get_device()}")
print(f"attn_input1: {attn_input1.shape}, dtype: {attn_input1.dtype}, device: {attn_input1.get_device()}")
print(f"attn_input2: {attn_input2.shape}, dtype: {attn_input2.dtype}, device: {attn_input2.get_device()}")


# check q_proj1 == proj2

print(f"q_proj1==q_proj2: {torch.equal(q_proj1, q_proj2)}")
print(f"attn_input1 == attn_input2[:,:18,:]  {torch.equal(attn_input1,attn_input2[:,:18,:])}")

linear = nn.Linear(4096, 4096, bias=None)
print(f"linear in {linear.weight.get_device()}")
linear = linear.to("cuda:0")
with torch.no_grad():
    linear.weight.copy_(q_proj1)
    o1 = linear(attn_input1)
    o2 = linear(attn_input2)
    print(f"o1.shape={o1.shape}")
    print(f"o1.shape={o2.shape}")
    
    print(f"o1==query_states1? {torch.equal(o1, query_states1)}")
    print(f"o2==query_states2? {torch.equal(o2, query_states2)}")
    print(f"o1 == o2? {torch.equal(o1, o2[:,:18,:])}")
    print(f"o1 ~= o2? (1e-5): {torch.allclose(o1, o2[:,:18,:],atol=1e-5)}")  
    print(f"o1 ~= o2? (1e-6): {torch.allclose(o1, o2[:,:18,:],atol=1e-6)}")  
    
    o1 = o1[0]
    o2 = o2[0]
    x1,x2 = o1.shape
    for i in range(x1):
        for j in range(x2):
            if o1[i][j] != o2[i][j]:
                print(f"i={i}, j={j}, o1={o1[i][j]}, o2={o2[i][j]}")
                break
        else:
            continue
        break

linear = linear.to("cpu")
print(f"linear in {linear.weight.get_device()}")
with torch.no_grad():
    attn_input1 = attn_input1.to("cpu")
    attn_input2 = attn_input2.to("cpu")
    query_states1 = query_states1.to("cpu")
    query_states2 = query_states2.to("cpu")
    
    o1 = linear(attn_input1)
    o2 = linear(attn_input2)
    print(f"o1.shape={o1.shape}")
    print(f"o1.shape={o2.shape}")
    
    print(f"o1==query_states1? {torch.equal(o1, query_states1)}")
    print(f"o2==query_states2? {torch.equal(o2, query_states2)}")
    print(f"o1 ~= o2? (1e-6): {torch.allclose(o1, o2[:,:18,:],atol=1e-6)}")  
    print(f"o1 ~= o2? (1e-7): {torch.allclose(o1, o2[:,:18,:],atol=1e-7)}")  
    
```

代码的输出为：

```
q_proj1: torch.Size([4096, 4096]), dtype: torch.float32, device: 0
q_proj2: torch.Size([4096, 4096]), dtype: torch.float32, device: 0
query_states1: torch.Size([1, 18, 4096]), dtype: torch.float32, device: 0
query_states2: torch.Size([1, 50, 4096]), dtype: torch.float32, device: 0
attn_input1: torch.Size([1, 18, 4096]), dtype: torch.float32, device: 0
attn_input2: torch.Size([1, 50, 4096]), dtype: torch.float32, device: 0
q_proj1==q_proj2: True
attn_input1 == attn_input2[:,:18,:]  True
linear in -1
o1.shape=torch.Size([1, 18, 4096])
o1.shape=torch.Size([1, 50, 4096])
o1==query_states1? True
o2==query_states2? True
o1 == o2? False
o1 ~= o2? (1e-5): True
o1 ~= o2? (1e-6): False
i=0, j=0, o1=0.11561134457588196, o2=0.11561146378517151
linear in -1
o1.shape=torch.Size([1, 18, 4096])
o1.shape=torch.Size([1, 50, 4096])
o1==query_states1? False
o2==query_states2? False
o1 ~= o2? (1e-6): True
o1 ~= o2? (1e-7): False

```

上面的pt文件可以在[google drive](https://drive.google.com/drive/folders/1hPwci2Lba41-GaAXYmglPLmdyg0C8G7N)下载。

通过上面的代码我们发现了这是pytroch的问题，而且可以发现cpu的一致性要比gpu高。

### 原因

通过搜索发现了[Batched computations or slice computations](https://pytorch.org/docs/stable/notes/numerical_accuracy.html#batched-computations-or-slice-computations)。

>Many operations in PyTorch support batched computation, where the same operation is performed for the elements of the batches of inputs. An example of this is torch.mm() and torch.bmm(). It is possible to implement batched computation as a loop over batch elements, and apply the necessary math operations to the individual batch elements, for efficiency reasons we are not doing that, and typically perform computation for the whole batch. The mathematical libraries that we are calling, and PyTorch internal implementations of operations can produces slightly different results in this case, compared to non-batched computations. In particular, let A and B be 3D tensors with the dimensions suitable for batched matrix multiplication. Then (A@B)[0] (the first element of the batched result) is not guaranteed to be bitwise identical to A[0]@B[0] (the matrix product of the first elements of the input batches) even though mathematically it’s an identical computation.

>Similarly, an operation applied to a tensor slice is not guaranteed to produce results that are identical to the slice of the result of the same operation applied to the full tensor. E.g. let A be a 2-dimensional tensor. A.sum(-1)[0] is not guaranteed to be bitwise equal to A[:,0].sum().

因此最终的差异就是来自它。

下面是英文版，主要通过ChatGPT翻译。

## Problem

The issue we are addressing here arises from [LLaMA2 - tokenizer padding affecting logits (even with attention_mask)](https://discuss.huggingface.co/t/llama2-tokenizer-padding-affecting-logits-even-with-attention-mask/50213/3). First, let's understand and reproduce this problem.


```python
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer 


model_id="/nas/lili/models_hf/Llama-2-7b-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_id, truncation_side='left', padding_side='right')
tokenizer.pad_token = tokenizer.eos_token
model = LlamaForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map={"":0})

input_prompt = "I've got a lovely bunch of coconuts do do do dooo"
input_prompt_tokenized = tokenizer(input_prompt, return_tensors="pt").to('cuda')
print(input_prompt_tokenized)


input_prompt_padded_tokenized = tokenizer(input_prompt, return_tensors="pt", padding="max_length", max_length=50).to('cuda')
print(input_prompt_padded_tokenized)


# Mask labels so that the values corresponding to the padding token do not contribute to the loss

input_ids_masked = torch.zeros(input_prompt_padded_tokenized.input_ids.shape, dtype=torch.int64).to('cuda')
torch.where(input_prompt_padded_tokenized.input_ids == tokenizer.pad_token_id,
            torch.tensor(-100, dtype=torch.int64),
            input_prompt_padded_tokenized.input_ids,
            out=input_ids_masked)
print(input_ids_masked)


# Calculate logits and loss from the model using unpadded input 

output1 = model(
    input_prompt_tokenized.input_ids,
    attention_mask=input_prompt_tokenized.attention_mask,
    labels=input_prompt_tokenized.input_ids
)
print(f"Loss (no padding): {output1.loss}")

print(f"Logits (no padding): {output1.logits}")


# Calculate the logits and loss from the model using padded input (should be the same) 

output2 = model(
    input_prompt_padded_tokenized.input_ids,
    attention_mask=input_prompt_padded_tokenized.attention_mask,
    labels=input_ids_masked
)

print(f"Loss (padding): {output2.loss}")

print(f"Logits (padding): {output2.logits}")
```

The only difference between output1 and output2 is that the latter utilizes padding. As a result, the input for the former consists of 18 tokens, while the latter has 50 tokens (with 32 being padding tokens). Additionally, it is noteworthy that the masked input_ids corresponding to the second input is constructed using torch.where:

<a>![](/img/paddingdebug/1.png)</a>

Namely, set the labels corresponding to padding positions to -100, so that the loss calculation excludes the padding tokens. Even though the attention mask for padding is 0, there may still be some loss. Note that both the input and other parameters of torch.where support broadcasting. Therefore, if it is a padding token, the input is a scalar torch.tensor(-100, dtype=torch.int64), otherwise, it is a vector of length 50, input_prompt_padded_tokenized.input_ids.

The output above is:

```
{'input_ids': tensor([[    1,   306, 29915,   345,  2355,   263, 12355,   873, 14928,   310,
          1302,   535,  8842,   437,   437,   437,   437,  3634]],
       device='cuda:0'), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
       device='cuda:0')}
{'input_ids': tensor([[    1,   306, 29915,   345,  2355,   263, 12355,   873, 14928,   310,
          1302,   535,  8842,   437,   437,   437,   437,  3634,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2]],
       device='cuda:0'), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0]], device='cuda:0')}
tensor([[    1,   306, 29915,   345,  2355,   263, 12355,   873, 14928,   310,
          1302,   535,  8842,   437,   437,   437,   437,  3634,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100]],
       device='cuda:0')
Loss (no padding): 2.271912097930908
Logits (no padding): tensor([[[-6.9180e+00, -1.0723e+00,  2.6934e+00,  ..., -4.2578e+00,
          -5.8789e+00, -4.5234e+00],
         [-9.4453e+00, -1.2516e+01, -1.2471e+00,  ..., -5.1328e+00,
          -8.7500e+00, -3.6348e+00],
         [-3.9004e+00,  1.2627e-02,  5.6992e+00,  ..., -3.9551e-01,
          -1.4980e+00,  2.8320e-01],
         ...,
         [-3.2383e+00, -3.9551e+00,  1.2812e+01,  ...,  3.8599e-01,
          -2.6230e+00, -1.0186e+00],
         [-2.7188e+00, -2.7988e+00,  1.3062e+01,  ...,  7.2119e-01,
          -1.9600e+00, -6.4160e-01],
         [-2.5156e+00, -1.4639e+00,  1.2578e+01,  ...,  1.9434e-01,
          -2.3145e+00, -9.1113e-01]]], device='cuda:0',
       grad_fn=<ToCopyBackward0>)
Loss (padding): 2.2886083126068115
Logits (padding): tensor([[[ -6.8242,  -1.2178,   2.7402,  ...,  -4.2617,  -5.8164,  -4.4297],
         [ -9.5312, -12.6875,  -1.3076,  ...,  -5.0625,  -8.7812,  -3.6660],
         [ -3.7285,   0.0786,   5.7227,  ...,  -0.2178,  -1.4248,   0.4294],
         ...,
         [  0.4036,  33.1250,   9.3750,  ...,   0.9189,   1.5947,   1.7139],
         [ -0.6445,  31.4688,  10.5234,  ...,   0.4934,   1.2793,   1.2725],
         [ -1.9326,  27.8594,  11.3984,  ...,   0.1417,   0.7139,   0.7173]]],
       device='cuda:0', grad_fn=<ToCopyBackward0>)
```

Upon careful comparison, we can observe that the loss without padding is 2.2719, while with padding it is 2.2886. Although the difference is relatively small, a 0.01 gap can still be significant for the model. Additionally, if we compare the logits (for output2, we only need to examine the first 18, as the rest are padding), we will also notice considerable differences.

## debug

### First Attempt

To troubleshoot this issue, I initially adopted a rather crude method: I opened two VSCode instances, debugged the two versions of code separately, and compared the results at each step. However, the differences in the returned numbers were extremely small, and VSCode, by default, displays only 4 decimal places for floating-point numbers, as shown in the following figure:

<a>![](/img/paddingdebug/2.png)</a>

We can see that VSCode displays only 4 significant digits, and based on these 4 digits, they appear to be the same. However, in reality, they are not exactly the same

### Comparing Hidden States

Since the final output logits are different, at which layer do the differences first appear? To output hidden states, it is necessary to set output_hidden_states=True:

```python
output1 = model(
    input_prompt_tokenized.input_ids,
    attention_mask=input_prompt_tokenized.attention_mask,
    labels=input_prompt_tokenized.input_ids,
    output_hidden_states=True
)
```

I wrote a simple piece of code to compare the hidden states:

```python
hidden_states1 = output1.hidden_states
hidden_states2 = output2.hidden_states
for layer in range(len(hidden_states1)):
    state1 = hidden_states1[layer][0]
    state2 = hidden_states2[layer][0]
    len, dim = state1.shape
    for i in range(len):
        hidden1 = state1[i]
        hidden2 = state2[i]
        mismatch_idx = -1
        for j in range(dim):
            v1 = hidden1[j]
            v2 = hidden2[j]
            if v1 != v2:
                mismatch_idx = j
                f1 = v1.cpu().detach().numpy().item()
                f2 = v2.cpu().detach().numpy().item()
                f1 = f"{f1:.4f}"
                f2 = f"{f2:.4f}"
                if f1 == f2:
                    print("here")
                #break         
        if mismatch_idx != -1:
            print(f"layer={layer}, mismatch i={i} j={mismatch_idx}")
            #break
```

The output is: layer=2, mismatch i=0 j=965. This indicates that the inconsistency starts from layer 2. Note: The output consists of a total of 33 values, where layer 0 represents the input embedding layer.

### Modifying modeling_llama.py

Now that we have identified the inconsistency in the hidden states starting from the second layer, the next step is to pinpoint which specific module first exhibits the inconsistency. This requires us to modify the code. Through debugging, we can find that the call path for model(...) is:

LlamaForCausalLM.forward -> LlamaModel.forward -> LlamaDecoderLayer.forward

Let's modify the code in reverse order, adding debugging information.

### LlamaDecoderLayer.forward
We need to save each intermediate result into debug_outputs at every step and then return it to the upper layer. The intermediate results include:

* hidden: Input--hidden_states
* input_layernorm: Results after layer normalization
* self_attn: Results after self-attention
* add_res: Results after adding the residual
* post_attention_layernorm: Results after post-attention layer normalization
* mlp: Results after the multi-layer perceptron (mlp)
* final: The final result"


```python
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        debug_outputs = {"hidden": hidden_states}
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        debug_outputs["input_layernorm"] = hidden_states
        # Self Attention         

	hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
 
        debug_outputs["self_attn"] = hidden_states
        hidden_states = residual + hidden_states
        debug_outputs["add_res"] = hidden_states
        # Fully Connected         
	
	residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        debug_outputs["post_attention_layernorm"] = hidden_states

        hidden_states = self.mlp(hidden_states)
        debug_outputs["mlp"] = hidden_states
        hidden_states = residual + hidden_states
        debug_outputs["final"] = hidden_states
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        outputs +=(debug_outputs,)
        return outputs
```

### LlamaModel.forward

Its main function is to compute the attention mask and then iterate through self.layers. The key modification here is to read the last value (debug_outputs) of the decoder_layer output tuple and add it to all_debug.

```python
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
		...
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    debug=debug
                )

            hidden_states = layer_outputs[0]
            ....
            all_debug += (layer_outputs[-1],)
```


### LlamaForCausalLM.forward

The code is as follows:

```python
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)         

	outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n             

	    shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens             
	    
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism             

	    shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            all_debug=outputs.all_debug
        )
```

Its main function is to call self.model(), then compute the loss, and finally return CausalLMOutputWithPast. In order to return debug information, we need to modify CausalLMOutputWithPast by adding the following line:

```python
all_debug: Optional[Tuple[torch.FloatTensor]] = None
```


Then, add all_debug to the final CausalLMOutputWithPast.


### Comparing Debug Information

Let's write a simple piece of code to compare the debug results of the two models:


```python
layer = 0
found_diff = False
for dict1, dict2 in zip(output1.all_debug, output2.all_debug):
    if found_diff:
        break
    print(f"layer={layer}")
    layer += 1
    for key, value in dict1.items():
        value2 = dict2[key]
        print(f"key: {key}, v1: {value.shape}, v2: {value2.shape}")
        dims = value.dim()
        if dims == 3:
            v1 = value[0]
            v2 = value2[0]
            dim1, dim2 = v1.shape
            for i in range(dim1):
                hidden1 = v1[i]
                hidden2 = v2[i]
                mismatch_idx = -1
                for j in range(dim2): 
                    if hidden1[j] != hidden2[j]:
                        mismatch_idx = j
                        break
                
                if mismatch_idx != -1:
                    print(f"mismatch i={i} j ={mismatch_idx}")    
                    found_diff = True
                    break
        elif dims == 4:
            found = False
            v1 = value[0]
            v2 = value2[0]
            d1,d2,d3 = v1.shape
            for i in range(d1):
                if found:
                    break
                for j in range(d2):
                    if found :
                        break
                    for k in range(d3):
                        if v1[i][j][k] != v2[i][j][k]:
                            found = True
                            found_diff = True
                            print(f"mismatch i={i} j={j} k={k}")   
                            break
```

The output is:

```
layer=0
key: hidden, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096]) 
key: input_layernorm, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: attn_input, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: q_proj, v1: torch.Size([4096, 4096]), v2: torch.Size([4096, 4096])
key: query_states, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: key_states, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: value_states, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: attn_weights, v1: torch.Size([1, 32, 18, 18]), v2: torch.Size([1, 32, 50, 50])
key: attn_weights2, v1: torch.Size([1, 32, 18, 18]), v2: torch.Size([1, 32, 50, 50])
key: attn_weights3, v1: torch.Size([1, 32, 18, 18]), v2: torch.Size([1, 32, 50, 50])
key: attn_weight4, v1: torch.Size([1, 32, 18, 18]), v2: torch.Size([1, 32, 50, 50])
key: attn_output, v1: torch.Size([1, 32, 18, 128]), v2: torch.Size([1, 32, 50, 128])
key: attn_output2, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: self_attn, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: add_res, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: post_attention_layernorm, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: mlp, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: final, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
layer=1
key: hidden, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: input_layernorm, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: attn_input, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: q_proj, v1: torch.Size([4096, 4096]), v2: torch.Size([4096, 4096])
key: query_states, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: key_states, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: value_states, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: attn_weights, v1: torch.Size([1, 32, 18, 18]), v2: torch.Size([1, 32, 50, 50])
key: attn_weights2, v1: torch.Size([1, 32, 18, 18]), v2: torch.Size([1, 32, 50, 50])
key: attn_weights3, v1: torch.Size([1, 32, 18, 18]), v2: torch.Size([1, 32, 50, 50])
key: attn_weight4, v1: torch.Size([1, 32, 18, 18]), v2: torch.Size([1, 32, 50, 50])
key: attn_output, v1: torch.Size([1, 32, 18, 128]), v2: torch.Size([1, 32, 50, 128])
key: attn_output2, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: self_attn, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: add_res, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: post_attention_layernorm, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: mlp, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
mismatch i=0 j =965
```

We observed the first inconsistency in the mlp at the second layer (layer==1). Thus, we began debugging the code for the LlamaMLP class.

### LlamaMLP

Suddenly, during debugging, we made a discovery: the actual LlamaMLP class is inconsistent with the code in init! Let's take a look at the code:

```python
   def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
```

For example, self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False), but during debugging, I found that it became Linear8bitLt:

<a>![](/img/paddingdebug/3.png)</a>

Tracking the code, I found it in bitsandbytes/nn/modules.py. Looking back at the previous code, I indeed discovered that the parameter load_in_8bit=True was passed when calling LlamaForCausalLM.from_pretrained. I've also delved a bit into int8 compression before; readers can refer to the article [A Gentle Introduction to 8-bit Matrix Multiplication for transformers at scale using Hugging Face Transformers, Accelerate and bitsandbytes](https://huggingface.co/blog/hf-bitsandbytes-integration) and [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale paper reading](http://fancyerii.github.io/2024/01/16/int8/) Could it be the uncertainty introduced by Linear8bitLt? It seems possible, as Linear8bitLt relies on outliers, and compression may amplify tiny differences.

### Setting load_in_8bit=False

Therefore, we set this parameter to False and run it again:

```
Loss (no padding): 2.2338905334472656
Logits (no padding): tensor([[[-12.9832,  -7.4134,  -0.4327,  ...,  -6.8297,  -8.0879,  -7.5863],
         [ -9.5230, -12.2163,  -1.1083,  ...,  -5.0527,  -8.9276,  -3.6419],
         [ -3.5179,   0.6801,   6.0908,  ...,  -0.2554,  -1.3332,   0.4213],
         ...,
         [ -3.0661,  -3.5820,  12.6735,  ...,   0.1156,  -2.5818,  -1.0295],
         [ -2.6741,  -2.5679,  12.8021,  ...,   0.4409,  -2.2196,  -0.7664],
         [ -2.1947,  -1.0629,  12.5872,  ...,   0.0837,  -2.1596,  -0.9412]]],
       device='cuda:0', grad_fn=<UnsafeViewBackward0>)
Loss (padding input & masking labels): 2.2338929176330566
Logits (padding input & masking labels): tensor([[[-1.2983e+01, -7.4134e+00, -4.3273e-01,  ..., -6.8297e+00,
          -8.0880e+00, -7.5864e+00],
         [-9.5230e+00, -1.2216e+01, -1.1083e+00,  ..., -5.0527e+00,
          -8.9276e+00, -3.6419e+00],
         [-3.5179e+00,  6.8010e-01,  6.0908e+00,  ..., -2.5537e-01,
          -1.3332e+00,  4.2135e-01],
         ...,
         [ 2.1687e-01,  3.3158e+01,  9.6556e+00,  ...,  5.6584e-01,
           1.4918e+00,  1.4965e+00],
         [-1.6480e+00,  2.9136e+01,  1.1903e+01,  ..., -1.6091e-02,
           8.9055e-01,  6.3661e-01],
         [-3.6790e+00,  2.4334e+01,  1.2093e+01,  ..., -8.4835e-01,
          -2.7381e-01, -5.9110e-01]]], device='cuda:0',
       grad_fn=<UnsafeViewBackward0>)
```

We observed that the difference in loss and logits has become very small, although not entirely identical. Here are the results of comparing debug information:

```
layer=0
key: hidden, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: input_layernorm, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: self_attn, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
mismatch i=0 j =0
key: add_res, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
mismatch i=0 j =0
```

Comparing hidden_states:

```
layer: 0
layer: 1
layer=1, mismatch i=0 j=0
```

From the above output, it can be seen that the two inputs diverge in the first layer, specifically after the self_attn operation.

### self_attn

Through tracking, we found that it uses LlamaSdpaAttention. Ultimately, it utilizes torch.nn.functional.scaled_dot_product_attention:

```python
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.

            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )
```

But through debugging, I found that the values before entering scaled_dot_product_attention are the same, and this function is untrackable (being Torch's C++ code). After researching for a while, I couldn't find any clues. Later, I came across this piece of code:

```python
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
```

Following it, I found:


```python
class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)
```

and,

```python
LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention_2": LlamaFlashAttention2,
    "sdpa": LlamaSdpaAttention,
}
```
After checking the code of LlamaAttention, I found that it is the most primitive Python implementation, which is very suitable for debugging. Therefore, I made the modification:

```python
model = LlamaForCausalLM.from_pretrained(model_id, load_in_8bit=False, device_map={"":0}, _attn_implementation="eager")
```

**Note**: Normally, _attn_implementation should not be used during development, as it is an internal parameter that may be modified at any time. Here, it is only used for debugging purposes.

```python
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        
        bsz, q_len, _ = hidden_states.size()
        debug_info = {"attn_input": hidden_states}
        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            debug_info["q_proj"] = self.q_proj.weight
            debug_info["k_proj"] = self.k_proj.weight
            debug_info["v_proj"] = self.v_proj.weight
            debug_info["query_states"] = query_states
            debug_info["key_states"] = key_states
            debug_info["value_states"] = value_states

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models             

            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        debug_info["attn_weights"] = attn_weights
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
        debug_info["attn_weights2"] = attn_weights
        # upcast attention to fp32         

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        debug_info["attn_weights3"] = attn_weights
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        debug_info["attn_weight4"] = attn_weights
        attn_output = torch.matmul(attn_weights, value_states)
        debug_info["attn_output"] = attn_output
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)
        debug_info["attn_output2"] = attn_output
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value, debug_info
```

Here, we added an additional debug_info to the return, so the calling code in LlamaDecoderLayer.forward also needs to be modified.

```python
        hidden_states, self_attn_weights, present_key_value, *debug_infos = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        if debug_infos:
            debug_outputs.update(debug_infos[0])
```

### Compare again

```
layer=0
key: hidden, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: input_layernorm, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: attn_input, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
key: q_proj, v1: torch.Size([4096, 4096]), v2: torch.Size([4096, 4096])
key: k_proj, v1: torch.Size([4096, 4096]), v2: torch.Size([4096, 4096])
key: v_proj, v1: torch.Size([4096, 4096]), v2: torch.Size([4096, 4096])
key: query_states, v1: torch.Size([1, 18, 4096]), v2: torch.Size([1, 50, 4096])
mismatch i=0 j =0
```

From the above output, we can see that query_states = self.q_proj(hidden_states) produces differences in results for the same input. Of course, when I say the input is the same, I mean v2[:,:18,:] == v1, while the output query_states is not equal to v1. And the model's parameter self.q_proj is obviously the same. What's going on here?

### Save the State

To facilitate reproduction, I have saved these variables:

```python
dict1 = output1.all_debug[0]
dict2 = output2.all_debug[0]
q_proj1 = dict1["q_proj"]
q_proj2 = dict2["q_proj"]
query_states1 = dict1["query_states"]
query_states2 = dict2["query_states"]
torch.save(q_proj1, "q_proj1.pt")
torch.save(q_proj2,  "q_proj2.pt")
torch.save(query_states1, "query_states1.pt")
torch.save(query_states2, "query_states2.pt")
torch.save(dict1["attn_input"], "attn_input1.pt")
torch.save(dict2["attn_input"], "attn_input2.pt")
```

### Reproduce the Issue

Now, let's deviate from Hugging Face Transformers to reproduce the inconsistency:

```python
import torch
from torch import nn
q_proj1 = torch.load("q_proj1.pt")
q_proj2 = torch.load("q_proj2.pt")
query_states1 = torch.load("query_states1.pt")
query_states2 = torch.load("query_states2.pt")
attn_input1 = torch.load("attn_input1.pt")
attn_input2 = torch.load("attn_input2.pt")


print(f"q_proj1: {q_proj1.shape}, dtype: {q_proj1.dtype}, device: {q_proj1.get_device()}")
print(f"q_proj2: {q_proj2.shape}, dtype: {q_proj2.dtype}, device: {q_proj2.get_device()}")
print(f"query_states1: {query_states1.shape}, dtype: {query_states1.dtype}, device: {query_states1.get_device()}")
print(f"query_states2: {query_states2.shape}, dtype: {query_states2.dtype}, device: {query_states2.get_device()}")
print(f"attn_input1: {attn_input1.shape}, dtype: {attn_input1.dtype}, device: {attn_input1.get_device()}")
print(f"attn_input2: {attn_input2.shape}, dtype: {attn_input2.dtype}, device: {attn_input2.get_device()}")


# check q_proj1 == proj2 

print(f"q_proj1==q_proj2: {torch.equal(q_proj1, q_proj2)}")
print(f"attn_input1 == attn_input2[:,:18,:] {torch.equal(attn_input1,attn_input2[:,:18,:])}")
print(f"linear in {linear.weight.get_device()}")
linear = linear.to("cuda:0")
with torch.no_grad():
    linear.weight.copy_(q_proj1)
    o1 = linear(attn_input1)
    o2 = linear(attn_input2)
    print(f"o1.shape={o1.shape}")
    print(f"o1.shape={o2.shape}")
    
    print(f"o1==query_states1? {torch.equal(o1, query_states1)}")
    print(f"o2==query_states2? {torch.equal(o2, query_states2)}")
    print(f"o1 == o2? {torch.equal(o1, o2[:,:18,:])}")
    print(f"o1 ~= o2? (1e-5): {torch.allclose(o1, o2[:,:18,:],atol=1e-5)}")  
    print(f"o1 ~= o2? (1e-6): {torch.allclose(o1, o2[:,:18,:],atol=1e-6)}")  
    
    o1 = o1[0]
    o2 = o2[0]
    x1,x2 = o1.shape
    for i in range(x1):
        for j in range(x2):
            if o1[i][j] != o2[i][j]:
                print(f"i={i}, j={j}, o1={o1[i][j]}, o2={o2[i][j]}")
                break
        else:
            continue
        break

linear = linear.to("cpu")
print(f"linear in {linear.weight.get_device()}")
with torch.no_grad():
    attn_input1 = attn_input1.to("cpu")
    attn_input2 = attn_input2.to("cpu")
    query_states1 = query_states1.to("cpu")
    query_states2 = query_states2.to("cpu")
    
    o1 = linear(attn_input1)
    o2 = linear(attn_input2)
    print(f"o1.shape={o1.shape}")
    print(f"o1.shape={o2.shape}")
    
    print(f"o1==query_states1? {torch.equal(o1, query_states1)}")
    print(f"o2==query_states2? {torch.equal(o2, query_states2)}")
    print(f"o1 ~= o2? (1e-6): {torch.allclose(o1, o2[:,:18,:],atol=1e-6)}")  
    print(f"o1 ~= o2? (1e-7): {torch.allclose(o1, o2[:,:18,:],atol=1e-7)}")  
```

The output of the code is:

```
q_proj1: torch.Size([4096, 4096]), dtype: torch.float32, device: 0
q_proj2: torch.Size([4096, 4096]), dtype: torch.float32, device: 0
query_states1: torch.Size([1, 18, 4096]), dtype: torch.float32, device: 0
query_states2: torch.Size([1, 50, 4096]), dtype: torch.float32, device: 0
attn_input1: torch.Size([1, 18, 4096]), dtype: torch.float32, device: 0
attn_input2: torch.Size([1, 50, 4096]), dtype: torch.float32, device: 0
q_proj1==q_proj2: True
attn_input1 == attn_input2[:,:18,:]  True
linear in -1
o1.shape=torch.Size([1, 18, 4096])
o1.shape=torch.Size([1, 50, 4096])
o1==query_states1? True
o2==query_states2? True
o1 == o2? False
o1 ~= o2? (1e-5): True
o1 ~= o2? (1e-6): False
i=0, j=0, o1=0.11561134457588196, o2=0.11561146378517151
linear in -1
o1.shape=torch.Size([1, 18, 4096])
o1.shape=torch.Size([1, 50, 4096])
o1==query_states1? False
o2==query_states2? False
o1 ~= o2? (1e-6): True
o1 ~= o2? (1e-7): False
```


The .pt file above can be downloaded from [google drive](https://drive.google.com/drive/folders/1hPwci2Lba41-GaAXYmglPLmdyg0C8G7N).

Through the above code, we found that this is a PyTorch issue, and it can be observed that CPU consistency is higher than GPU.

### Reason

Through searching, I found information about [Batched computations or slice computations](https://pytorch.org/docs/stable/notes/numerical_accuracy.html#batched-computations-or-slice-computations).

>Many operations in PyTorch support batched computation, where the same operation is performed for the elements of the batches of inputs. An example of this is torch.mm() and torch.bmm(). It is possible to implement batched computation as a loop over batch elements, and apply the necessary math operations to the individual batch elements, for efficiency reasons we are not doing that, and typically perform computation for the whole batch. The mathematical libraries that we are calling, and PyTorch internal implementations of operations can produces slightly different results in this case, compared to non-batched computations. In particular, let A and B be 3D tensors with the dimensions suitable for batched matrix multiplication. Then (A@B)[0] (the first element of the batched result) is not guaranteed to be bitwise identical to A[0]@B[0] (the matrix product of the first elements of the input batches) even though mathematically it’s an identical computation.

>Similarly, an operation applied to a tensor slice is not guaranteed to produce results that are identical to the slice of the result of the same operation applied to the full tensor. E.g. let A be a 2-dimensional tensor. A.sum(-1)[0] is not guaranteed to be bitwise equal to A[:,0].sum().

So, the ultimate difference comes from it.


