# EMN(Episode Memory Network)QA系统实现

[TOC]

##1. 关于QA调研简介
在QA系统中，可以明确分为三个部分：

    1.  问题分析
    2.	信息检索
    3.	答案抽取
    
当前的问答系统中，大多对问题分析和答案抽取部分已经做了具有较强鲁棒的方法，比如Facebook 的 DrQA (DanQi Chen et al., 2017)文章中提出的DrQA，对问题分析上，提出了question 去match每一篇文章的段落，匹配出相应有关系的段落。

而在大多数QA的系统中都是对信息检索模块的方法更新，比如insurence QA(Minwei Feng et al., 2015)，但是这些方法都没有注意到，对于真正的阅读理解环境里，一个段落中包含着对于问题有用的信息，也包含了对问题没有用的信息，那么如何去掉冗余的信息，有效的提取有用的信息，是一个需要深度探索的问题。

针对以上问题，我实践出来一种方法， 可以有效的解决信息检索过程中信息冗余的问题，并且具有很强的扩展性，一个端到端的模型，不基于任何外部KB，在准确性上表现较好，并且其核心的模块可以任意移植到其他的NLP问题中，比如文本分类，关系抽取等等。这个方法通过在检索过程中使用fact级别的attention多次迭代，选取出最有价值的k个fact。

## 2.Model Quick Look

```
EMN (
  (embedding): Embedding(66, 80, padding_idx=0)
  (input_gru): GRU(80, 80)
  (question_gru): GRU(80, 80)
  (drop_out): Dropout (p = 0.1)
  (gate): Sequential (
    (0): Linear (320 -> 80)
    (1): ReLU ()
    (2): Linear (80 -> 1)
    (3): Sigmoid ()
  )
  (atten_gru_cell): GRUCell(80, 80)
  (memory_gru_cell): GRUCell(80, 80)
  (answer_gru): GRUCell(160, 80)
  (fc): Linear (80 -> 66)
)
```

## 3. 运行说明
**项目目前使用Python3开发，还未完全适配2.\*，请使用Python3运行本项目**
### 3.1 Quick Start
项目文件中包含了一个已经训练好的模型，可以直接使用，使用方法如下：

```python
$ python3 interact.py

Facts : 
Mary grabbed the milk there </s> <PAD>
Mary gave the milk to Jeff </s>
Mary moved to the garden </s> <PAD>
Fred travelled to the hallway </s> <PAD>
Fred moved to the bathroom </s> <PAD>
Fred moved to the hallway </s> <PAD>
Mary journeyed to the bedroom </s> <PAD>
Jeff put down the milk </s> <PAD>
Bill picked up the football there </s>
Bill handed the football to Mary </s>
Mary passed the football to Bill </s>
Bill travelled to the office </s> <PAD>

Question :  Who gave the football ?

Answer :  Mary </s>
Prediction :  Mary </s>
```
首先，我们需要告诉系统，所有的fact，然后对fact和question，做 预处理，最后送进模型进行预测，模型的预训练数据已经在文件夹里面，也包括了一个很小的数据集，数据样例：

![-c500](media/15122165506878/15124302838158.jpg)


###3.2 Train Model
- 数据解释

上面已经给出了训练数据的样例，每一行看做是答案的一个证据或者叫fact，连续编号的行代表一个样例，在有问号的一行代表，关于前面的fact的内容的提问。这一行的后面给出了答案，和答案所在的fact的id。

训练和测试数据都使用Facebook的`bAbI`的QA数据集，下载地址：https://research.fb.com/downloads/babi/ ，本项目中采用了three args relations data set，数据集中包含了10000个样例。

- 项目目录结构

```
EMNQA
├── __init__.py
├── model.py
├── data_set.py
├── util.py
├── test.py
├── train.py
├── interact.py
├── earlystoping-EMNQA.model
├── data (or $EMNQA_DATA)
    ├── qa5_three-arg-relations_test.txt
    └── qa5_three-arg-relations_train.txt
└── requirements.txt
```

- 开始训练
 
    1.安装必要的Package
    
    ```bash
    cd EMNQA
    pip3 install -r requirements.txt
    ```
    
    2.开始训练
    >注意，一下命令行参数均为可选值，因为每个参数都有default值
    
    - 准备好数据
    - 启动训练`python3 train.py --train-data-file (your train data file) --test-data-file (your test data file)`，训练完成后，会自动接入测试集进行测试。
    
    3.单独测试
    
    - 启动测试 `python3 test.py --data-file (your train data file) --model-file (your model file)`。

- 性能说明

    在训练集上一共有`10000`条问答数据，测试集中有`10000`条。
    
    本机配置：
    
    **CPU：**Core i7 4850u
    **GPU：**NO
    **Mem:**16G
    **Time Spend:** almost 24h
            
    在本机上没有GPU的情况下，跑完整个训练集花费了接近**24h**。
    
### 3.3 Model Test
在训练不被打断的情况下，模型会在训练完毕后自动接入测试。并输出测试结果（准确性）


## 4. Test Result
| | bAbI Train | bAbI Test | 
| :---: | :-------: | :-------------: 
| ACC (%) | 99.6 | 97.4 | 

## 5. 运行截图
![-c400](media/15122165506878/15124478469238.jpg)


## 6. 在工程中遇到的问题、原因及解决方案
###6.1 softmax和log softmax造成的差异
在最后的答案提取过程中，我在第一版项目中使用了softmax函数作为概率预测函数，而在训练过程中发现，训练到第三个Epoch的时候，损失值不会下降了。

为了得到原因，我一步步的从损失函数的计算上开始调试，最后发现，输出的概率矩阵中，基本大多都是0，导致在求max过程中逐渐产生了随机选取max，导致损失函数无法迭代下降。
总结原因是因为softmax损失函数会造成下溢的问题。

最后改为`log_softmax`方法解决。

```
第1轮迭代是softmax输出
1.00000e-02 *
 1.7682  1.6000  1.5325  ...   1.3639  1.4683  1.6326
 1.9087  1.6602  1.6396  ...   1.3151  1.3959  1.5971
 1.7560  1.6217  1.5103  ...   1.3643  1.4741  1.6191
          ...             ⋱             ...          
 1.8685  1.6965  1.5689  ...   1.3149  1.3977  1.5929
 1.7326  1.6201  1.4960  ...   1.3919  1.4430  1.6519
 1.8455  1.6714  1.5876  ...   1.3536  1.3680  1.6259
[torch.FloatTensor of size 128x66]
第3轮
 7.2926e-06  2.2543e-05  7.5913e-06  ...   1.7174e-06  7.5473e-06  2.6475e-06
 5.8383e-06  1.7381e-05  6.0026e-06  ...   1.2274e-06  6.5481e-06  2.0100e-06
 6.8195e-06  2.0560e-05  7.1012e-06  ...   1.5307e-06  7.2134e-06  2.4412e-06
                ...                   ⋱                   ...                
 5.7231e-06  1.6663e-05  5.8476e-06  ...   1.1718e-06  6.4028e-06  1.9603e-06
 6.8745e-06  2.0760e-05  7.1571e-06  ...   1.5508e-06  7.2511e-06  2.4650e-06
 5.7001e-06  1.6459e-05  5.8104e-06  ...   1.1555e-06  6.3587e-06  1.9514e-06
[torch.FloatTensor of size 128x66]
```

可以看到，所有的概率都已经是很小的数字。

![-c400](media/15122165506878/15124478213648.jpg)

###6.2 模型迭代速度过慢
在之前的版本中，我将模型的所有组件的函数均初始化成全1的矩阵，但是实践中发现，大概200轮的迭代才能使模型收敛到损失小于0.01，这使得模型训练时间大约为3天多。

从网上搜查资料，发现模型参数的初始化是有方法的，并不以一味的随意初始化，最后采用了xavier normal的方法进行初始化RNN的参数还有embedding的参数。

初始化参数相当于在一个给一个起点，如果起点离最优值进，那么迭代速度就会快。

xavier normal初始化参数的实质是让所有参数满足均匀分布

![-c300](media/15122165506878/15124476100379.jpg)

其中分母中的$n_j$表示当前这一层$j$的神经元个数。
这个公式是由线性激活函数，同时使每一层的方差相等，且分布均值为0得来的。

