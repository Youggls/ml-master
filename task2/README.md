# Task2
> 概念学习

## 如何运行

### Requirements

- python package:

  ```
  tabulate
  csv
  ```

- python version: python > 3.5

### Shell

在源代码目录下运行 `python main.py` 即可

## FindS

FindS 算法仅对每个正样本进行学习，对冲突的属性进行一般化。

## 候选消除法

候选消除法同时考虑了正负样本。

## 结果

### FindS

| Outlook | Temperature | Humidity | Wind |
| :-----: | :---------: | :------: | :--: |
|    ?    |      ?      |    ?     |  ?   |

### 候选消除法

集合 $G$ 和 $S$ 均为空集

### 解释

FindS 结果均为 ? 表示学习到了最普遍的假设，说明原始数据冲突过多

候选消除法学习到两个空集，说明数据中难以使用该算法归纳学习到相应的概念
