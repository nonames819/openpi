uv下载不行记得换源

[[tool.uv.index]]
name = "sii"
url = "http://nexus.sii.shaipower.online/repository/pypi/simple"
default = true

[[tool.uv.index]]
name = "tuna"
url = "https://pypi.tuna.tsinghua.edu.cn/simple"
default = true

# train

## data

## model
pi05在ae部分使用adaRMS，不把state token作为输入
原始pi0中state作为输入，但不能attend action，ar_mask可表示为0,0,0,0,1,1,0,0，对于input_mask 11011111, 
对应mask就是
```
Key ID:  0  1  2  3 | 4 | 5  6  7
Query ID: 0    [1, 1, 0, 1,  0,  0, 0, 0]  (第2行、列变为0)
          1    [1, 1, 0, 1,  0,  0, 0, 0]
          2    [0, 0, 0, 0,  0,  0, 0, 0] 
          3    [1, 1, 0, 1,  0,  0, 0, 0]
          ------------------------------
          4    [1, 1, 0, 1,  1,  0, 0, 0]
          ------------------------------
          5    [1, 1, 0, 1,  1,  1, 1, 1]
          6    [1, 1, 0, 1,  1,  1, 1, 1]
          7    [1, 1, 0, 1,  1,  1, 1, 1]
```

# eval

## data transform 
[InjectDefaultPrompt(...ompt=None), 
LiberoInputs(model_t...: 'pi05'>), 
Normalize(norm_stats...ict=False), 
InjectDefaultPrompt(...ompt=None), 
ResizeImages(height=...width=224), 
TokenizePrompt(token...put=False), 
PadStatesAndActions(...on_dim=32)]

## input info
dict_keys(['images', 'image_masks', 'state', 'tokenized_prompt', 'tokenized_prompt_mask', 
'token_ar_mask', 'token_loss_mask']) # None

## infer
img -> bs*256*2048
text: bs*200 -> bs*200*2048
tokens: bs*968*2048
ar_mask: 768+200
input_mask: 根据是否有效来定

## control
记录的数据和输出的action都是delta action，只是一个相对量，需要乘以对应维度上的量纲才能变成真实的变化量


        output_max=(0.05, 0.05, 0.05, 0.5, 0.5, 0.5),
        output_min=(-0.05, -0.05, -0.05, -0.5, -0.5, -0.5),

