# RecSys
## 数据集
本项目使用到 movielens 中的两种数据集。      
下载地址：[movielens](https://grouplens.org/datasets/movielens/)   
## MyALS
### 模型概述
`MyALS` 手动实现了 ALS 算法，该算法的使用与 `spark` 库中的 ALS算法类似，代码参考了 [ALS.py](https://github.com/tushushu/imylu/blob/master/imylu/recommend/als.py)，在原有代码基础上，添加了**正则化**和**提前停止**机制，并进一步封装模型。同时编写了**测试代码**以及**评估函数**，以对不同的参数组合进行充分评估。

该模型接受的输入见 `fit()` 函数的参数说明

模型的使用方式如下：       

1. 若使用内建数据集，需要运行 `preprocessing.py` 获取 `.npy` 格式的数据集，三个测试通过表示获取成功。

   ```python
   ...
   ============================= test session starts =============================
   ...
   ======================== 3 passed, 1 warning in 10.72s ========================
   ...
   ```

1. 导包后使用 `MyALS()` 初始化模型，可调参数如下：
   
   - rank：矩阵分解秩（隐向量长度），默认为 10
   - max_iter：最大迭代次数，默认为 10
   - reg_param：代价函数的正则系数，默认为 0（即禁用正则化）
   - early_stop_enable：训练提前停止，默认为 `False`
   
2. 使用 `fit()` 训练模型，需要如下参数：
   - X：评分矩阵，类型为 `ndarray` ，第 i 行第 j 列表示第 i 个用户对物品 j 的评分
   
3. 使用 `predict()` 进行预测，需要如下参数：
   - user_ids：包含需要推荐的用户 id 的列表
   - n_items：为每个用户推荐的物品数量
   

关于提前停止，本模型按照如下规则工作：
   - 每一轮迭代时将本次迭代的 rmse 与上一轮（初始为 0）比较
   - 若连续三轮的 rmse 差值的绝对值小于 0.001，认为模型已收敛，提前停止训练
### 模型评估
`ml-latest` 中的 `ratings.csv` 为千万级的数据集
```python
>>> ratings = np.load("ml-latest/ratings.npy", allow_pickle=True)
>>> ratings.shape
(27753444, 4)
```
在手动实现的模型上训练困难， 故使用小型数据集 `ml-latest-small`
```python
>>> ratings = np.load("ml-latest-small/ratings.npy", allow_pickle=True)
>>> ratings.shape
(100836, 4)
```
运行 `test/MyALS_test.py`，对模型进行测，测试的输出见后面的模型测试输出部分：

评估函数见 `utils.py` ，使用准确率（precision）、召回率（recall）和覆盖率（coverage）评估模型。

整理得到如下表格：

|            | rank   | max_iter                 | reg_param       | precision  | recall     | coverage   |
| ---------- | ------ | ------------------------ | --------------- | ---------- | ---------- | ---------- |
| test01     | 10     | 10                       | 3               | 36.42%     | 19.18%     | 14.07%     |
| test02     | 10     | 10                       | 1               | 36.78%     | 19.25%     | 14.00%     |
| test03     | 10     | 10                       | 0.1             | 36.40%     | 18.97%     | 14.04%     |
| test04     | 20     | 10                       | 1               | 37.33%     | 19.03%     | 17.19%     |
| test05     | 5      | 10                       | 1               | 33.10%     | 17.73%     | 9.92%      |
| test06     | 10     | 10                       | 0（不使用正则） | 36.26%     | 19.12%     | 13.95%     |
| **test07** | **20** | **10（不使用提前停止）** | **1**           | **37.86%** | **19.50%** | **17.33%** |

将第二组输出与其他传统推荐算法进行对比，结果如下：

|                    | precision  | recall     | coverage   |
| ------------------ | ---------- | ---------- | ---------- |
| User-CF(n_item=80) | 25.20%     | 12.17%     | 20.29%     |
| Item-CF(n_item=10) | 22.28%     | 10.76%     | 18.84%     |
| LFM(ratio=5)       | 26.94%     | 13.01%     | 44.25%     |
| **ALS(n_item=80)** | **37.86%** | **19.50%** | **17.33%** |

注：三种传统推荐算法的数据来源参考项亮的《推荐系统实践》中表 2-4、表 2-8 和表 2-14 .

综合两表，不同的超参数对模型产生了不同的影响。

1. 在隐向量长度为 20，迭代 10 次且正则系数取 1 时，模型在六轮测试中取得了最佳的准确率和覆盖率，同时也获得了较优的召回率。
2. 横向对比不同的算法，ALS 的准确率与召回率表现出色，覆盖率虽然不高，但与 User-CF 和 Item-CF 相比，差距不大。
3. 正则化技术虽然对结果影响不大，但微小的提升同样展现了正则化对模型的正向影响。
4. 提前停止技术使得训练模型时可以选择较大的迭代次数 `max_iter`，并在模型的 rmse 收敛时及时停止训练，节省训练时间。

### 模型测试输出

```python
Testing started at 16:40 ...
Launching pytest with arguments C:/Users/x50021862/PycharmProjects/RecSys/test/MyALS_test.py --no-header --no-summary -q in C:\Users\x50021862\PycharmProjects\RecSys\test

============================= test session starts =============================
collecting ... collected 6 items

MyALS_test.py::test01 PASSED                                             [ 14%]with reg_params
Iterations: 1, RMSE: 3.387168
Iterations: 2, RMSE: 0.342432
Iterations: 3, RMSE: 0.327447
Iterations: 4, RMSE: 0.321426
Iterations: 5, RMSE: 0.318394
Iterations: 6, RMSE: 0.317453
Iterations: 7, RMSE: 0.316863
Iterations: 8, RMSE: 0.316687
**Early stopped!**
recall:0.19182377049180338
precision:0.3642469914608689
coverage:0.14071900220102715

MyALS_test.py::test02 PASSED                                             [ 28%]with different reg_params
Iterations: 1, RMSE: 3.389359
Iterations: 2, RMSE: 0.342279
Iterations: 3, RMSE: 0.327487
Iterations: 4, RMSE: 0.321661
Iterations: 5, RMSE: 0.318675
Iterations: 6, RMSE: 0.317414
Iterations: 7, RMSE: 0.316618
Iterations: 8, RMSE: 0.316261
Iterations: 9, RMSE: 0.315949
**Early stopped!**
recall:0.19252049180327865
precision:0.3678034252261619
coverage:0.1399853264856933

MyALS_test.py::test03 PASSED                                             [ 43%]with different reg_params
Iterations: 1, RMSE: 3.392162
Iterations: 2, RMSE: 0.344011
Iterations: 3, RMSE: 0.328964
Iterations: 4, RMSE: 0.322572
Iterations: 5, RMSE: 0.319103
Iterations: 6, RMSE: 0.318025
Iterations: 7, RMSE: 0.317248
Iterations: 8, RMSE: 0.316983
Iterations: 9, RMSE: 0.316722
**Early stopped!**
recall:0.18973360655737698
precision:0.36396205845630863
coverage:0.14042553191489363

MyALS_test.py::test04 PASSED                                             [ 57%]with different rank
Iterations: 1, RMSE: 3.333144
Iterations: 2, RMSE: 0.326090
Iterations: 3, RMSE: 0.306246
Iterations: 4, RMSE: 0.300385
Iterations: 5, RMSE: 0.296947
Iterations: 6, RMSE: 0.296178
Iterations: 7, RMSE: 0.295443
Iterations: 8, RMSE: 0.295411
**Early stopped!**
recall:0.19034836065573785
precision:0.3733802273200875
coverage:0.17197358767424797

MyALS_test.py::test05 PASSED                                             [ 72%]with different rank
Iterations: 1, RMSE: 3.424804
Iterations: 2, RMSE: 0.353408
Iterations: 3, RMSE: 0.343831
Iterations: 4, RMSE: 0.339192
Iterations: 5, RMSE: 0.336236
Iterations: 6, RMSE: 0.334922
Iterations: 7, RMSE: 0.334298
Iterations: 8, RMSE: 0.333837
Iterations: 9, RMSE: 0.333599
**Early stopped!**
recall:0.17733606557377043
precision:0.3310650061223571
coverage:0.0991929567131328

MyALS_test.py::test06 PASSED                                             [ 86%]without reg_params
Iterations: 1, RMSE: 3.396352
Iterations: 2, RMSE: 0.344836
Iterations: 3, RMSE: 0.330820
Iterations: 4, RMSE: 0.324948
Iterations: 5, RMSE: 0.321327
Iterations: 6, RMSE: 0.319715
Iterations: 7, RMSE: 0.318565
Iterations: 8, RMSE: 0.317950
Iterations: 9, RMSE: 0.317426
Iterations: 10, RMSE: 0.317104
recall:0.1911885245901639
precision:0.3625556963682326
coverage:0.13954512105649303

MyALS_test.py::test07 PASSED                                             [100%]without early stop
Iterations: 1, RMSE: 3.330642
Iterations: 2, RMSE: 0.325120
Iterations: 3, RMSE: 0.304763
Iterations: 4, RMSE: 0.298848
Iterations: 5, RMSE: 0.295929
Iterations: 6, RMSE: 0.295228
Iterations: 7, RMSE: 0.294563
Iterations: 8, RMSE: 0.294504
Iterations: 9, RMSE: 0.294325
Iterations: 10, RMSE: 0.294372
recall:0.19504098360655733
precision:0.3786692089857688
coverage:0.17329420396184886

================== 7 passed, 1 warning in 1333.41s (0:22:13) ==================

Process finished with exit code 0
```



