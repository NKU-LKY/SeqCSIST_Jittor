# 使用介绍

- 本仓库只包含jittor复现涉及的代码，不包含其他已有文件
- 将grokcso_jt文件夹直接放置在根目录下
- test_jittor_derefnet.py文件和train_jittor_derefnet.py文件放置在根目录/tools文件夹中
- 无需改动pytorch版本下的任何文件
- 训练命令为：

```
python tools/train_jittor_derefnet.py
```

- 测试命令为：

```
python tools/test_jittor_derefnet.py --ckpt work_dir/DeRefNet_jittor/final_model.pkl --use_cuda 1
```
