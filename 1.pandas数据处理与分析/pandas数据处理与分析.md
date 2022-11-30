## 第一部分  基础知识

### 第1章  预备知识

#### 1.1  Python基础

##### 1.1.1  推导式

1.   反映两个对象之间的映射关系

```python
[i ** 2 for i in range(5)]  # 列表推导式
{i ** 2 for i in range(5)}  # 集合推导式
{i: i ** 2 for i in range(5)}  # 字典推导式
```

2.   **生成器表达式**

当需要把列表等作为中间对象来进行操作时，考虑使用生成器节省内存空间

```python 
sum(i ** 2 for i in range(5))  # 生成器
```

3.   **嵌套**

-   单个推导式存在多个迭代

    ```python
    # 多个迭代，按从左到右决定先后顺序，靠前的在外层
    [(i, j) for i in ['桃', '星', '梅', '方'] for j in range(2, 6)]  # 结果是一个列表
    
    # if在for前面
    [(i, j) if j % 2 == 0 else None for i in ['桃', '星', '梅', '方'] for j in range(2, 6)]

-   推导式嵌套推导式

    ```python
    [[(i, j) for i in ['桃', '星', '梅', '方']] for j in range(2, 6)]  # 结果是列表内嵌列表
    ```

-   带`if`判断

    ```python
    L = [1, 2, 3, 4, 5, 6, 7]
    [i if i <= 5 else 5 for i in L]  # [1, 2, 3, 4, 5, 5, 5]
    [i for i in L if i <= 5]  # [1, 2, 3, 4, 5]

##### 1.1.2  匿名函数

lambda后紧跟形参，只能使用表达式，不能赋值、循环与选择语句，但可以借助推导式和条件赋值来实现

```python
lst = [[1, 2], [3, 4, 5], [6], [7, 8], [9]]
def func(x):
    flag = False
    for i in x:
        if i % 3 == 0:
            flag = True
    return flag

list(filter(func, lst))  # 包含3的整数倍的内层列表
list(filter(lambda x: sum(1 if i % 3 == 0 else 0 for i in x) > 0, lst))
```

##### 1.1.3  打包函数

同时遍历两个迭代器相同位置的元素

```python 
l1 = list('abc')
l2 = ['apple', 'banana', 'cat']
res = list(zip(l1, l2))  # 打包
list(zip(*res))  # 解包

for i, j in enumerate(l1):
    print(i, j)
    
for i, j in zip(range(len(l1)), l1):
    print(i, j)
```

#### 1.2  NumPy基础

##### 1.2.1  NumPy数组的构造

常用：`np.array([1, 2])`

1.   等差数列

     `np.linspace()`、`np.arange()`

2.   特殊矩阵

     `np.zeros()`、`np.ones()`、`np.eye()`、`np.full()`

     `np.zeros_like()`、`np.ones_like()`、`np.full_like()`

3.   随机数组

     -   均匀分布**[low, high)**:  `np.random.uniform(low=0, high=1, size=(2, 3))`
         -   均匀0-1分布**[0, 1)**: `np.random.rand(2, 3, 4)`
     -   正态分布$N[\mu,\sigma^2]$: `np.random.normal(loc=0, scale=1, size=(2, 3))`
         -   标准正态分布$N[0, 1]$: `np.random.randn(2, 3, 4)`
     -   离散均匀分布整数数组: `np.random.randint(low=0, high=1, size=(2, 3))`
     -   有放回抽样: `np.random.choice(a=range(5), size=(2, 3), p=[0.1, 0.2, 0.3, 0.2, 0.2])`
         -   无放回抽样: `np.random.choice(a=range(5), size=(2, 3), replace=False`
     -   打散: `np.random.permutation([1, 2, 3])`
         -   等价于无放回抽样列表所有元素: `np.random.choice([1, 2, 3], 3, replace=False)`

4.   随机种子: `np.random.seed(7)`

##### 1.2.2  NumPy数组的变形

1.   数组元素组织方式变化导致的变形

     -   维度交换

         ```python 
         arr = np.arange(6).reshape(1, 2, 3)
         np.transpose(a=arr, axes=(1, 2, 0))  # 原来的1维放到现在的0维，原来的2维放到1维
         
         arr.T == np.transpose(2, 1, 0) == np.transpose(arr)  # 默认维度逆向交换
         
         # 交换两个维度，np.transpose()需要写出所有维度
         np.swapaxes(a=arr, axis1=1, axis2=0)

     -   维度变换

         ```python 
         # reshape默认以行的顺序来读写，order='C'
         arr = np.arange(6).reshape(1, 2, 3, order='F')  # 列优先
         
         # 维度增加
         arr = np.arange(6).reshape(2, 3)
         expand_arr = np.expand_dims(a=arr, axis=(0, 1, 4))  # (1, 1, 2, 3, 1)
         (arr.reshape(1, 1, 2, 3, 1) == expand_arr).all()
         arr[np.newaxis, np.newaxis, :, :, np.newaxis] == expand_arr
         # 维度缩减
         np.squeeze(expand_arr, axis=(0, 1)).shape  # (2, 3, 1)，默认缩减所有=1的维度

2.   数组合并或拆分导致的变形

     -   合并`np.stack()`、`np.concatenate()`

         ```python
         # np.stack()，拼接的数组必须尺寸相同，且产生新的维度，axis决定新维度在哪产生，维度大小由拼接的数组数量决定
         arr1 = np.arange(12).reshape(4, 3)
         arr2 = np.arange(12, 24).reshape(4, 3)
         np.stack([arr1, arr2], axis=0).shape  # (2, 4, 3)
         np.stack([arr1, arr2], axis=1).shape  # (4, 2, 3)
         np.stack([arr1, arr2], axis=2).shape  # (4, 3, 2)
         
         # np.concatenate只需在拼接的维度上一样即可
         np.concatenate([arr1, arr2], axis=2)  # AxisError: axis 2 is out of bounds for array of dimension 2

     -   拆分`np.split()`

         ```python 
         # np.split()，indices_or_sections为整数表示均分，为一维序列，表示沿着axis用索引切割
         # indices_or_sections=[2, 3]表示：arr[:2]、arr[2:3]、arr[3:]
         np.split(arr, indices_or_sections=[1, 2], axis=0)
         ```

     -   重复`np.repeat()`

         ```python
         # np.repeat()，沿着axis对数组按照给定次数进行重复
         arr = np.arange(6).reshape(2, 3)
         np.repeat(a=arr, repeats=[2, 3], axis=0)
         np.repeat(a=arr, repeats=[2, 3, 1], axis=1)  # repeats列表的长度必须与arr的轴的长度一致

##### 1.2.3  NumPy数组的切片

```python 
# 输入切片，取子数组
arr = np.arange(24).reshape(4, 2, 3)
arr[0:2, 0:2, 0:2]  # 取2*2*2子数组

# 输入长度相同的列表，不是取子数组，而是取元素，输入值表示元素在各个维度的索引
arr[[0, 1], [0, 1]]  # 取出arr[0, 0]和arr[1, 1]
arr[[0, 1], [0, 1], [0, 1]]  # 取出arr[0, 0, 0]和arr[1, 1, 1]

# 输入布尔数组，保留某一维度的若干维数
arr[[True, False, True, False], :, :]

# 最后几个维度的:可以忽略
arr[[True, False, True, False], :, :] == arr[[True, False, True, False]]
arr[[0, 0, 0], [1, 1, 1], :] == arr[[0, 0, 0], [1, 1, 1]] 

# 最初几个维度的:可以用...代替
arr[:, :, 0:2] == arr[..., 0:2]
```

##### 1.2.4  广播机制

数组$A$的维度：$d_p^A\times\cdots\times d_1^A$

数组$B$的维度：$d_q^A\times\cdots\times d_1^A$

设$r=max(p,q)$

首先对数组维度小的数组补充维度：在数组**前面**补充，维数=1

对比两个数组，对维数=1的维度进行复制扩充，维数=另一个数组相同位置的维数

当相同位置的维数不相等，且任一维数都$\neq1$，则报错

1.   标量和数组的广播

     当一个标量和数组进行运算时，标量会自动把大小扩充为数组大小，之后进行逐元素操作

2.   二维数组之间的广播

     除非其中的某个数组的维度是$m×1$或者$1×n$，扩充其具有$1$的维度为另一个数组对应维度的大小，否则报错

3.   一维数组与二维数组的广播

     当一维数组$A_k$与二维数组$B_{m,n}$操作时，等价于把一维数组视作$A_{1,k}$的二维数组，当$k!=n$且$k,n$都不是$1$时报错

     ```python
     np.ones(3) + np.ones((2,3))  # OK
     np.ones(2) + np.ones((2,3))  # 报错
     ```

##### 1.2.5  常用函数



### 第2章  pandas基础

#### 2.1  文件的读取与写入

#### 2.2  基本数据结构

#### 2.3  常用基本函数

#### 2.4  窗口

## 第二部分  4类操作

### 第3章  索引

### 第4章  分组

### 第5章  变形

### 第6章  连接

## 第三部分  4类数据

### 第7章  缺失数据

### 第8章  文本数据

### 第9章  分类数据

### 第10章  时间序列数据

## 第四部分  进阶实践

### 第11章  数据观测

### 第12章  特征工程

### 第13章  性能优化