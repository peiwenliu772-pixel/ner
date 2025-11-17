# 中文命名实体识别（NER）项目

## 🚀 一、项目简介
基于 bert-base-chinese 和 hfl/chinese-bert-wwm 模型，同时支持 weibo 和 MSRA 两个数据集，完成中文实体识别任务。
- 数据加载与预处理（BIO标注格式）
- 模型设计与训练流程
- 模型评估(F1)  

## 📊 二、数据集来源

1.weibo 命名实体识别数据集

简介：本数据集包括训练集（1350）、验证集（269）、测试集（270），实体类型包括地缘政治实体(GPE.NAM)、地名(LOC.NAM)、机构名(ORG.NAM)、人名(PER.NAM)及其对应的代指(以NOM为结尾)。

下载地址：📥 [天池数据集](https://tianchi.aliyun.com/dataset/144312)


2.MSRA 命名实体识别数据集

简介：MSRA 数据集是面向新闻领域的中文命名实体识别数据集，包括训练集（46364）、测试集（4365），实体类型包括地名 (LOC)、人名 (NAME)、组织名 (ORG)。

下载地址：📥 [天池数据集](https://tianchi.aliyun.com/dataset/144307?spm=a2c22.12282016.0.0.432a4f03K11Mhq)
## 🧠 三、项目结构示例

[注](注)：pre_model，output等文件没有上传
``` 
ner
├── config_loader.py    # 配置加载工具
├── data                # 数据集
│   ├── msra
│   └── weibo
├── data_process.py     # 数据预处理脚本
├── download_model.py   # 下载模型到本地目录
├── main.py             # 主程序
├── model.py            # 模型定义
├── ner_config          # 配置文件
│   ├── msra_bert_base_chinese.json
│   ├── msra_bert_wwm.json
│   ├── weibo_bert_base_chinese.json
│   └── weibo_bert_wwm.json
├── output              # 模型结果输出
│   ├── msra
│   │   ├── bert-base-chinese
│   │   └── hfl
│   │       └── chinese-bert-wwm
│   └── weibo
│       ├── bert-base-chinese
│       └── hfl
│           └── chinese-bert-wwm
├── pre_model           # 预训练模型权重
│   ├── models--bert-base-chinese
│   └── models--hfl--chinese-bert-wwm
└── README.md           # 项目说明
```
## 🧩 四、环境依赖

- Python 3.9
- torch==2.1.0+cu118
- transformers==4.45.2
  
## 🖊️  五、快速开始

### 选择不同模型和数据集

<div style="background-color: #f5f5f5; padding: 12px 16px; border-radius: 6px; position: relative; margin: 10px 0;">
  只需main.py中修改不同的配置文件。
  <button style="position: absolute; right: 12px; top: 50%; transform: translateY(-50%); background: transparent; border: none; cursor: pointer; color: #666;">📋</button>
</div>

### 训练/评估

<div style="background-color: #f5f5f5; padding: 12px 16px; border-radius: 6px; position: relative; margin: 10px 0;">
  直接运行main.py即可。
  <button style="position: absolute; right: 12px; top: 50%; transform: translateY(-50%); background: transparent; border: none; cursor: pointer; color: #666;">📋</button>
</div>

### 测试
<div style="background-color: #f5f5f5; padding: 12px 16px; border-radius: 6px; position: relative; margin: 10px 0;">
  加载已保存的权重文件，进行测试，返回Precision（精确率）、Recall（召回率）、F1-score（F1 分数）。
  <button style="position: absolute; right: 12px; top: 50%; transform: translateY(-50%); background: transparent; border: none; cursor: pointer; color: #666;">📋</button>
</div>

## 📁 六、实验结果

bert-base-chinese 和 chinese-bert-wwm 在 weibo 和 mrsa 数据集上实验结果。

| 数据集  |模型                | F1值  |
|--------|--------------------|-------|
| weibo  | bert-base-chinese  | 64.4% |
| weibo  | chinese-bert-wwm   | 64.1% |
| mrsa   | bert-base-chinese  | 94.9% |
| mrsa   | chinese-bert-wwm   | 94.7% |

