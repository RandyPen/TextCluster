# 短文本聚类

### 项目介绍
短文本聚类是常用的文本预处理步骤，可以用于洞察文本常见模式、分析设计语义解析规范等。本项目实现了内存友好的短文本聚类方法。



### 依赖库

>  pip install tqdm jieba



### 使用方法

```bash
python cluster.py --infile ./data/infile \
--output ./data/output
```

具体参数设置可以参考```cluster.py```文件内```_get_parser()```函数参数说明，包含设置分词词典、停用词、匹配采样数、匹配度阈值等。



### 算法原理

![算法原理](./data/images/Algorithm_cn.png)



### 文件路径

```html
TextCluster
|      README.md
|      LICENSE
|      cluster.py                    主要执行程序
|      
|------utils                         公共功能模块
|    |    __init__.py
|    |    segmentor.py               分词器封装
|    |    similar.py                 相似度计算函数
|    |    utils.py                   文件处理模块
|
|------data
|    |    infile                     默认输入文本路径，用于测试中文模式
|    |    infile_en                  默认输入文本路径，用于测试英文模式
|    |    seg_dict                   默认分词词典
|    |    stop_words                 默认停用词路径
```



注：本方法仅面向短文本，长文本聚类可根据需求选用[SimHash](https://en.wikipedia.org/wiki/SimHash), [LDA](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)等其他算法。



# Text Cluster

### Introduction

Text cluster is a normal preprocess procedure to analysis text feature. This project implements a RAM friendly method only for **short text cluster**. For long text, it is preferable to choose [SimHash](https://en.wikipedia.org/wiki/SimHash) or [LDA](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) or others according to demand.



### Requirements

>  pip install tqdm spacy



### Usage

```bash
python cluster.py --infile ./data/infile_en \
--output ./data/output \
--lang en
```

For more configure arguments description, see ```_get_parser()``` in ```cluster.py```, including stop words setting, sample number.



### Basic Idea

![Algorithm_en](/Users/randypen/Code/TextCluster/data/images/Algorithm_en.png)

### File Structure

```html
TextCluster
|      README.md
|      LICENSE
|      cluster.py                    main excutable function
|      
|------utils                         utilities
|    |    __init__.py
|    |    segmentor.py               tokenizer wrapper
|    |    similar.py                 similarity calculator
|    |    utils.py                   file process module
|
|------data
|    |    infile                     default input file path, to test Chinese mode
|    |    infile_en                  default input file path, to test English mode
|    |    seg_dict                   default tokenizer dict path
|    |    stop_words                 default stop words path
```





# Other Language

For other specific language, modify tokenizer wrapper in ```./utils/segmentor.py```. 

