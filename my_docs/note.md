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

## training resource 

single gpu bs=64
after embedding prefix/suffix 9287MiB / 143771MiB
after layer0 22653MiB / 143771MiB
after loss 24589MiB / 143771MiB
after backward 36363MiB / 143771MiB
several step 50533MiB / 143771MiB
大概稳定在49803MiB / 143771MiB



single gpu bs=4
34054 / 143771MiB
forward: 0.0802614688873291 backward: 0.24120402336120605 (0.4~0.5s/it)

single gpu bs=8
34054 / 143771MiB
forward: 0.12296175956726074 backward: 0.405914306640625 1.52it/s(0.658s/it)

single gpu bs=16
34446 / 143771MiB
forward: 0.20833396911621094 backward: 0.7435786724090576 1.08s/it

single gpu bs=32
40760MiB / 143771MiB
forward: 0.4006338119506836 backward: 1.4410548210144043 2.02s/it

2gpu global bs=32
39503MiB / 143771MiB 1.12s/it

single gpu bs=64
49803MiB / 143771MiB 30h single step time 3.62

single gpu bs=128
75420MiB / 143771MiB 60h single step time 7.11

single gpu bs=256
127020MiB / 143771MiB 120h

