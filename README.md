# 中文命名实体识别（NER）项目

## 🚀 一、项目简介
基于 bert-base-chinese 模型，完成中文实体识别任务
- 数据加载与预处理（BIO标注格式）
- 模型设计与训练流程
- 模型评估(F1)  

最终实验结果得到F1分数约为64.%

## 📊 二、数据集来源

weibo命名实体识别数据集

简介：本数据集包括训练集（1350）、验证集（269）、测试集（270），实体类型包括地缘政治实体(GPE.NAM)、地名(LOC.NAM)、机构名(ORG.NAM)、人名(PER.NAM)及其对应的代指(以NOM为结尾)。

下载地址：📥 [天池数据集](https://tianchi.aliyun.com/dataset/144312)
## 🧠 三、项目结构示例

[注](注)：hfmodel，output等文件没有上传
```
ner
├── README.md          # 项目说明
├── config.json        # 配置文件
├── config_loader.py   # 配置加载工具
├── data               # 数据集目录
│   ├── class.txt
│   ├── dev.txt
│   ├── test.txt
│   └── train.txt
├── data_process.py    # 数据预处理脚本
├── hfmodel            # 本地 HuggingFace 模型缓存（未上传）
│   └── bert-base-chinese
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── vocab.txt
├── main.py            # 主程序入口
├── model.py           # 模型定义
└── output             # 模型输出目录（未上传）
    ├── config.json
    ├── pytorch_model.bin
    ├── special_tokens_map.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── vocab.txt
```
## 🧩 四、环境依赖
- Python 3.9
- torch==2.1.0+cu118
- transformers==4.45.2