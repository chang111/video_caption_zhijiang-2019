
## 代码目录树如下：

```
|--coco-caption   //coco-caption相关的测试代码
|--data
	|--test   //比赛测试视频
    |--test_i3d_feats   //提取测试视频的i3d特征文件
    |--test_res152_20_feats   //提取测试视频20帧的resnet152特征文件
    |--test_res152_40_feats   //提取测试视频40帧的resnet152特征文件
    |--caption_vatex_msrvtt.json   //vatex数据集和msrvtt数据集的caption文件
    |--caption_vatex_msrvtt_msvd.json  //vatex数据集和msrvtt数据集和msvd数据集的caption文件
    |--info_vatex_msrvtt.json     //vatex数据集和msrvtt数据集的vocab文件
    |--info_vatex_msrvtt_msvd.json   //vatex数据集和msrvtt数据集和msvd数据集的vocab文件
    |--train_vatex_msrvtt.json    //vatex数据集和msrvtt数据集的训练文件
    |--train_vatex_msrvtt_msvd.json   //vatex数据集和msrvtt数据集和msvd数据集的训练文件
|--misc  //loss定义文件
|--models   //模型定义文件
|--save_model   //保存的训练好的模型，用于生成测试结果；我们分别训练了3个不同训练集组合和特征组合
    |--vatex_msrvtt_msvd_res152_i3d   //训练数据集为vatex、msrvtt、msvd，训练特征为res152特征和i3d特征
        |--model_320.pth  //训练320个epoch的模型
        |--model_200.pth  //训练200个epoch的模型
        |--opt_info.json  //训练时的配置文件
    |--vatex_msrvtt_res152   //训练数据集为vatex、msrvtt，训练特征为res152特征
        |--model_280.pth  //训练280个epoch的模型
        |--model_320.pth  //训练320个epoch的模型
        |--opt_info.json  //训练时的配置文件
    |--vatex_msrvtt_res152_i3d   //训练数据集为vatex、msrvtt，训练特征为res152特征和i3d特征
        |--model_300.pth  //训练300个epoch的模型
        |--model_320.pth  //训练320个epoch的模型
        |--opt_info.json  //训练时的配置文件
    |--model_rgb.pth   //提取i3d特征的模型
|--src   //定义i3d模型的文件
|--submit  //测试结果存放的文件夹
|--dataloader.py   //定义Dataset文件
|--extractVideoI3DFeature.py   //提取i3d特征
|--opts.py   //训练配置文件
|--prepro_feats.py   //提取resnet152特征文件
|--prepro_vocab.py   //训练前处理训练集的vacab文件
|--test.py   //用于生成测试结果的文件
|--train.py   //训练文件
|--Readme.md   //本文档
```



## 代码运行方式：

1.测试：生成提交的测试结果，结果会保存在./submit/result.json

	python test.py

2.训练：

首先需要将训练视频下载放在./data/文件夹下

我将我所用训练的所有视频保存在百度云盘上了

下载地址：链接：https://pan.baidu.com/s/1SZk2R2cqq4MtUwtdKO5QOQ 
提取码：phsx 

然后分别提取训练集的各种特征：

提取训练集i3d特征：

```
python extractVideoI3DFeature.py --input_path data/train_all --save_path data/train_all_i3d_feats
```

提取训练集20帧的resnet152特征：

```
python prepro_feats.py --video_path data/train_all --n_frame_steps 20 --output_dir data/train_all_res152_20_feats
```

提取训练集40帧的resnet152特征：

```
python prepro_feats.py --video_path data/train_all --n_frame_steps 40 --output_dir data/train_all_res152_40_feats
```

提取训练集vocab：

提取vatex+msrvtt组成的训练集vocab：

```
python prepro_vocab.py --input_json data/train_vatex_msrvtt.json --info_json data/info_vatex_msrvtt.json --caption_json data/caption_vatex_msrvtt.json
```

提取vatex+msrvtt+msvd组成的训练集vocab：

```
python prepro_vocab.py --input_json data/train_vatex_msrvtt_msvd.json --info_json data/info_vatex_msrvtt_msvd.json --caption_json data/caption_vatex_msrvtt_msvd.json
```

训练数据集为vatex、msrvtt、msvd，训练特征为res152特征和i3d特征进行训练：

```
python train.py --gpu 0 --epochs 1001 --batch_size 256 --checkpoint_path save_model/vatex_msrvtt_msvd_res152_i3d --input_json data/train_vatex_msrvtt_msvd.json --info_json data/info_vatex_msrvtt_msvd.json --caption_json data/caption_vatex_msrvtt_msvd.json --feats_dir data/train_all_res152_20_feats --model S2VTAttModel  --with_c3d 1 --c3d_feats_dir data/train_all_i3d_feats --dim_vid 3072
```

训练数据集为vatex、msrvtt，训练特征为res152特征进行训练：

```
python train.py --gpu 0 --epochs 1001 --batch_size 256 --checkpoint_path save_model/vatex_msrvtt_res152 --input_json data/train_vatex_msrvtt.json --info_json data/info_vatex_msrvtt.json --caption_json data/caption_vatex_msrvtt.json --feats_dir data/train_all_res152_40_feats --model S2VTAttModel  --with_c3d 0 --dim_vid 2048
```

训练数据集为vatex、msrvtt，训练特征为res152特征和i3d特征进行训练：

```
python train.py --gpu 0 --epochs 1001 --batch_size 256 --checkpoint_path save_model/vatex_msrvtt_res152_i3d --input_json data/train_vatex_msrvtt.json --info_json data/info_vatex_msrvtt.json --caption_json data/caption_vatex_msrvtt.json --feats_dir data/train_all_res152_20_feats --model S2VTAttModel  --with_c3d 1 --c3d_feats_dir data/train_all_i3d_feats --dim_vid 3072
```

  

## 程序说明：

本代码采用的所有训练数据集来自vatex数据集、msrvtt数据集、msvd数据集

数据集总共视频数为40185个


## Requirements：

- ubuntu
- cuda
- pytorch
- python3
- ffmpeg 
- tqdm
- pillow
- pretrainedmodels
- nltk
- matplotlib# Video-cpation-zhijiang-2018
