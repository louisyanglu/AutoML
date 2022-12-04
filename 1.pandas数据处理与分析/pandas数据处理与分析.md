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

-   首先对数组维度小的数组补充维度：在数组**前面**补充，维数=1
-   对比两个数组，对维数=1的维度进行复制扩充，维数=另一个数组相同位置的维数
-   当相同位置的维数不相等，且任一维数都$\neq1$，则报错

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

1.   **聚合函数**

     统计函数包括`max, min, mean, median, std, var, sum, quantile`，其中**分位数计算**是**全局方法**，因此只能通过`np.median()`和`np.quantile()`的方法调用

     如果数组中包含`nan`，结果也为`nan`。若要忽略`nan`进行计算，则使用**全局方法**`np.nanmax()`

2.   **相关性计算**

     -   协方差：`np.cov(arr1, arr2)`
     -   相关性系数：`np.corrcoef(arr1, arr2)`

3.   ufunc**函数**

     逐元素处理，**全局方法**

     `np.cos、sin、tan、arccos、arcsin、arctan、abs、sqrt、power、exp、log、log10、log2、ceil、floor()`

4.   **逻辑函数**

     -   **比较**：`<、>、<=、>=、!=、==`

     -   **内置函数**：`isnan()、isinf()、all()、any()`

     -   **逻辑运算符**：`~、&、|`

         -   优先级：`非not > 与and > 或or`

     -   **填充函数**：`np.where(bool_arr, fill_arr_for_True, fill_arr_for_False)`

         如果后两个待填充的值与第一个数组维度不匹配且符合广播条件，则会被广播，之后使用相应位置的值来填充

         ```python 
         arr = np.arange(4).reshape(-1, 2)
         np.where(arr > 0, arr.sum(0), arr.sum(1))
         ```

     -   **截断函数**：`np.clip(arr, min, max)`

         ```python 
         np.clip(arr, 1, 2)
         # 等价于
         arr1 = np.where(arr > 2, 2, arr)
         np.where(arr1 < 1, 1, arr1)

5.   **返回索引的函数**

     -   返回**非零**、**最大**、**最小**值所在的索引：`np.nonzero()、argmax()、argmin()`

6.   **累计函数**

     -   `cumprod`, `cumsum`分别表示累乘和累加函数，返回同长度的数组
     -   `diff`表示和前一个元素做差，由于第一个元素为缺失值，因此在默认参数情况下，返回长度是原数组减1

7.   **向量内积**

     `a.dot(b)`

8.   **向量、矩阵范数**

     `np.linalg.norm()`

     | ord可选参数 | 矩阵范数 | 向量范数 |
     | :---- | ----: | ----: |
     | None   | Frobenius norm | 2-norm |
     | 'fro'  | Frobenius norm  | / |
     | 'nuc'  | nuclear norm    | / |
     | inf    | max(sum(abs(x), axis=1))   | max(abs(x)) |
     | -inf   | min(sum(abs(x), axis=1))  |  min(abs(x)) |
     | 0      | /   |  sum(x != 0) |
     | 1      | max(sum(abs(x), axis=0))  |  as below |
     | -1     | min(sum(abs(x), axis=0))   |  as below |
     | 2      | 2-norm (largest sing. value) | as below |
     | -2     | smallest singular value    | as below |
     | other  | /   | sum(abs(x)\*\*ord)\*\*(1./ord) |

9.   **矩阵乘法**

     `a @ b`

10.   **卡方统计量**

      设矩阵$A_{m\times n}$，记$B_{ij} = \frac{(\sum_{i=1}^mA_{ij})\times (\sum_{j=1}^nA_{ij})}{\sum_{i=1}^m\sum_{j=1}^nA_{ij}}$，定义卡方值如下：$$\chi^2 = \sum_{i=1}^m\sum_{j=1}^n\frac{(A_{ij}-B_{ij})^2}{B_{ij}}$$

      ```python
      A = np.random.randint(10, 20, (8, 5))
      B = A.sum(0) * A.sum(1)[:, None] / A.sum()
      chi_val = ((A - B) ** 2 / B).sum()

------

### 第2章  pandas基础

#### 2.1  文件的读取与写入

##### 2.1.1  文件读取

一些常用的公共参数，`header=None`表示第一行不作为列名，`index_col`表示把某一列或几列作为索引，`usecols`表示读取列的集合，默认读取所有的列，`parse_dates`表示需要转化为时间的列，`nrows`表示读取的数据行数。这些参数在`read_csv、read_table、read_excel`里都可以使用

在读取`txt`文件时，经常遇到分隔符非空格的情况，`read_table`有一个**分割参数**`sep`，它使得用户可以自定义分割符号，进行`txt`数据的读取。参数`sep`中使用的是**正则表达式**，因此需要对诸如`|`等进行转义变成`\|`，否则无法读取到正确的结果。

##### 2.1.2  数据写入

`pandas`中没有定义`to_table`函数，但是`to_csv`可以保存为`txt`文件，并且允许自定义分隔符，常用制表符`\t`分割。

如果想要把表格快速转换为`markdown`和`latex`语言，可以使用`to_markdown`和`to_latex`函数，此处需要安装`tabulate`包。

#### 2.2  基本数据结构

##### 2.2.1  Series

`Series`一般由四个部分组成，分别是序列的值`data`、索引`index`、存储类型`dtype`、序列的名字`name`。其中，索引也可以指定它的名字，默认为空。

```python
s = pd.Series(data = [100, 'a', {'dic1':5}],
              index = pd.Index(['id1', 20, 'third'], name='my_idx'),  # 指定索引名
              dtype = 'object',
              name = 'my_name')
s.index.name = 'my_idx'  # 显式指定索引名
```

常用的`dtype`还有`int、float、string、category`

`object`代表了一种混合类型，目前`pandas`把纯字符串序列也默认认为是一种`object`类型的序列，但它也可以显式指定`string`类型存储

对于这些属性，可以通过 . 的方式来获取。`.values、index、dtype、name、shape`

##### 2.2.2  DataFrame

`DataFrame`在`Series`的基础上增加了列索引，一个数据框可以由二维的`data`与行列索引来构造。

但一般而言，更多的时候会采用从列索引名到数据的映射来构造数据框，同时再加上行索引。

```python
df = pd.DataFrame(data = {'col_0': [1,2,3],
                          'col_1':list('abc'),
                          'col_2': [1.2, 2.2, 3.2]},
                  index = ['row_0', 'row_0', 'row_0']  # 不报错
                  )
```

如果字典中'col_0'对应的是Series：

-   Series与df索引一致，Series的值直接填入
-   Series与df索引不一致，且Series索引不重复，当前df的行索引在Series未出现则填入nan值
-   Series与df索引不一致，且Series索引重复，报错

在`DataFrame`中可以用`[col_name]`与`[col_list]`来取出相应的列与由多个列组成的表，结果分别为`Series`和`DataFrame`。如`df[col_0]`是**Series**，`df[[col_0]]`是**DataFrame**。

Series转换为DataFrame：`s.to_frame()`

对于df属性，可以通过 . 的方式来获取。`.values、index、columns、dtypes、shape`，与Series相比，没有`name`属性，多了`columns`属性，且dtype是复数带s

-   增加或修改一列时，直接使用`df[col_name]`
-   删除一列时，使用`drop()`。绝大多数方法都不会直接改变原df，而是返回一个拷贝

#### 2.3  常用基本函数

##### 2.3.1  汇总函数

`head, tail`函数分别表示返回表或者序列的前`n`行和后`n`行，其中`n`默认为5

`info, describe`分别返回表的信息概况和表中数值列对应的主要统计量。`info, describe`只能实现较少信息的展示，如果想要对一份数据集进行全面且有效的观察，特别是在列较多的情况下，推荐使用**[pandas-profiling](https://pandas-profiling.github.io/pandas-profiling/docs/)**包

##### 2.3.2  特征统计函数

-   常见的是`sum, mean, median, var, std, max, min`
-   `quantile, count, idxmax, idxmin`，它们分别返回的是分位数、非缺失值个数、最大/最小值对应的索引

上面这些所有的函数，由于操作后返回的是标量，所以又称为**聚合函数**，它们有一个公共参数`axis`，默认为0代表逐列聚合，如果设置为1则表示逐行聚合

##### 2.3.3  频次函数

-   对序列`Series`使用`unique`和`nunique`可以分别得到其唯一值组成的列表和唯一值的个数
-   对序列`Series`使用`value_counts`可以得到唯一值和其对应出现的频数
-   如果想要观察`DataFrame`多个列组合的唯一值，可以使用`drop_duplicates`。其中的关键参数是`keep`，默认值`first`表示每个组合保留第一次出现的所在行，`last`表示保留最后一次出现的所在行，`False`表示把所有重复组合所在的行剔除
-   `duplicated`和`drop_duplicates`的功能类似，但前者返回了是否为唯一值的布尔列表，其`keep`参数与后者一致

##### 2.3.4  替换函数

一般而言，替换操作是针对某一个列进行的，因此下面的例子都以`Series`举例。

`pandas`中的替换函数可以归纳为三类：映射替换、逻辑替换、数值替换

-   在`replace`中，可以通过字典构造，或者传入两个列表来进行替换

    ```python
    df['Gender'].replace({'Female':0, 'Male':1})  # 字典
    df['Gender'].replace(['Female', 'Male'], [0, 1])

-   `replace`还有一种特殊的方向替换，指定`method`参数为`ffill`则为用前面一个最近的未被替换的值进行替换，`bfill`则使用后面最近的未被替换的值进行替换

    ```python
    s = pd.Series(['a', 1, 'b', 2, 1, 1, 'a'])
    s.replace([1, 2], method='ffill')  # 1、2都被替换

-   **逻辑替换**。包括了`where`和`mask`，这两个函数是完全对称的：`where`函数在传入条件为`False`的对应行进行替换，而`mask`在传入条件为`True`的对应行进行替换，当不指定替换值时，替换为缺失值。传入的条件只需是与被调用的`Series`索引一致的**布尔序列**即可

-   **数值替换**。包含了`round, abs, clip`方法，它们分别表示按照给定精度四舍五入、取绝对值和截断

    ```python
    s.clip(0, 2)  # 分别表示上下截断边界
    ```

##### 2.3.5  排序函数

1.   **值排序**`sort_values`

     在排序中，经常遇到多列排序的问题，比如在体重相同的情况下，对身高进行排序，并且保持身高降序排列，体重升序排列

     ```python 
     df_demo.sort_values(['Weight','Height'],ascending=[True,False])

2.   **索引排序**``sort_index`

     索引排序的用法和值排序完全一致，只不过元素的值在索引中，此时需要用参数`level`指定索引层的名字或者层号。另外，需要注意的是字符串的排列顺序由字母顺序决定。

     ```python
     df_demo.sort_index(level=['Grade','Name'],ascending=[True,False])
     ```

3.   **元素排序**`rank()`

     `Series.rank(ascending=True, pct=False, method='average')`

     `pct`是否返回元素对应的分位数

     `method`可能的参数：

     -   'min'/'max'：最小/大的可能排名
     -   'first'：先后排名
     -   'dense'：排名相差1

##### 2.3.6  apply函数

`apply`方法常用于`DataFrame`的行迭代或者列迭代，`apply`的参数往往是一个以序列为输入的函数。

`apply`的自由是牺牲性能换来的，只有当不存在内置函数且迭代次数较少时，才使用`apply`

```python
# 可以利用`lambda`表达式使得书写简洁，这里的`x`就指代被调用的`df_demo`表中逐个输入的序列
%timeit -n 100 -r 7 df_demo.apply(lambda x:x.mean())  # -r表示运行轮数(runs)，-n表示每轮运行次数(loops)
```

#### 2.4  窗口

##### 2.4.1  滑动窗口

要使用滑窗函数，就必须先要对一个序列使用`.rolling`得到滑窗对象，其最重要的参数为窗口大小`window`。需要注意的是窗口包含当前行所在的元素

-   `min_periods`：参与计算的最小样本量，默认=window，必须满足`min_periods`<=`window`

**类滑窗函数**

`shift, diff, pct_change`是一组类滑窗函数，它们的公共参数为`periods=n`，默认为1。它们的功能可以用窗口大小为`n+1`的`rolling`方法等价代替

-   `shift`：取**向前**第`n`个元素的值，**n为负表示向后**

    ```python
    s.shift(n) == s.rolling(n + 1).apply(lambda x: list(x)[0])  # n为正
    s.shift(n) == s[::-1].rolling(-n + 1).apply(lambda x: list(x)[0])[::-1]  # n为负

-   `diff`：与向前第`n`个元素做差

    ```python 
    s.diff(n) == s.rolling(n + 1).apply(lambda x:list(x)[-1]-list(x)[0])

-   `pct_change`：与向前第`n`个元素相比计算增长率

    ```python 
    s.pct_change() == s.rolling(2).apply(lambda x: list(x)[-1] / list(x)[0] - 1)
    ```

##### 2.4.2  扩张窗口

扩张窗口又称累计窗口，可以理解为一个动态长度的窗口，其窗口的大小就是从序列开始处到具体操作的对应位置，其使用的聚合函数会作用于这些逐步扩张的窗口上。具体地说，设序列为a1, a2, a3, a4，则其每个位置对应的窗口即\[a1\]、\[a1, a2\]、\[a1, a2, a3\]、\[a1, a2, a3, a4\]

-   `cummax`：`s.expanding().apply(lambda x: x.max())`
-   `cumsum`：`s.expanding().apply(lambda x: x.sum())`
-   `cumprod`：`s.expanding().apply(lambda x: x.prod())`

##### 2.4.3  指数加权窗口

在扩张窗口中，可以使用各类函数进行历史的累计指标统计，但给窗口中的所有函数赋予了同样的权重。若给窗口中的元素赋予不同的权重，则此时为**指数加权窗口**。

最重要的参数是$\alpha$，窗口权重为$\omega_i=(1-\alpha)^{t-i}$，序列第一个元素$x_0$距当前元素$x_t$最远，其权重最小：$\omega_0=(1-\alpha)^t$

加权并归一化：$$\begin{split}y_t &=\frac{\sum_{i=0}^{t} w_i x_{t-i}}{\sum_{i=0}^{t} w_i} \\&=\frac{x_t + (1 - \alpha)x_{t-1} + (1 - \alpha)^2 x_{t-2} + ...+ (1 - \alpha)^{t} x_{0}}{1 + (1 - \alpha) + (1 - \alpha)^2 + ...(1 - \alpha)^{t}}\\\end{split}$$

```python 
s.ewm(alpha=0.2).mean()  # 调用ewm函数

def ewm_func(x, alpha=0.2):
    win = (1 - alpha) ** np.arange(x.shape[0])
    win = win[::-1]
    res = (win * x).sum() / win.sum()
s.expanding().apply(ewm_func)
```

------

## 第二部分  4类操作

### 第3章  索引

####  3.1  单级索引

##### 3.1.1  DataFrame的列索引

列索引是最常见的索引形式，一般通过`[]`来实现。通过`[列名]`可以从`DataFrame`中取出相应的列，返回值为`Series`。

如果要取出多个列，则可以通过`[列名组成的列表]`，其返回值为一个`DataFrame。`

##### 3.1.2  Series的行索引

1.   以字符串为索引的`Series`，索引可重复
     -   如果取出单个索引的对应元素，则可以使用`[item]`，若`Series`只**有单个值对应**，则返回这个**标量**值，如果有**多个值对应**，则返回一个`Series`
     -   如果取出多个索引的对应元素，则可以使用`[items的列表]`
     -   如果想要取出某两个索引之间的元素，并且这两个索引在整个索引中**唯一出现**，则**可以使用切片**，同时需要注意这里的切片会包含两个端点
     -   如果这两个索引在整个索引中**重复出现**，那么需要经过排序才能使用切片
2.   以整数为索引的`Series`，索引可重复
     -   和字符串一样，如果使用`[int]`或`[int_list]`，则可以取出对应索引元素的值
     -   如果使用整数切片，则会取出对应索引位置的值，注意这里的整数切片同`Python`中的切片一样**不包含右端点**

##### 3.1.3  loc索引器

基于元素的`loc`索引器的一般形式是`loc[*, *]`，其中第一个`*`代表行的选择，第二个`*`代表列的选择，如果省略第二个位置写作`loc[*]`，这个`*`是指行的筛选。

`*`的位置一共有五类合法对象，分别是：单个元素、元素列表、元素切片、布尔列表以及函数

1.   `*`**为单个元素**

     直接取出相应的行或列，如果该元素在索引中重复则结果为`DataFrame`，否则为`Series`

     也可以同时选择行和列，返回`Series`或标量

2.   `*`**为元素列表**

     取出列表中所有元素值对应的行或列

3.   `*`**为元素切片**

     字符串/整数索引，如果是**唯一值**的起点和终点字符，那么就可以使用切片，并且包含两个端点，如果**不唯一则报错**

4.   `*`**为布尔列表**

     传入`loc`的布尔列表与`DataFrame`长度相同，且列表为`True`的位置所对应的行会被选中，`False`则会被剔除

5.   `*`**为函数**

     这里的函数，必须以前面的四种合法形式之一为返回值，并且函数的输入值为`DataFrame`本身

     ```python
     def condition(x):
         condition_1_1 = x.School == 'Fudan University'
         condition_1_2 = x.Grade == 'Senior'
         condition_1_3 = x.Weight > 70
         condition_1 = condition_1_1 & condition_1_2 & condition_1_3
         condition_2_1 = x.School == 'Peking University'
         condition_2_2 = x.Grade == 'Senior'
         condition_2_3 = x.Weight > 80
         condition_2 = condition_2_1 & (~condition_2_2) & condition_2_3
         result = condition_1 | condition_2
         return result
     df_demo.loc[condition]
     ```

     -   由于函数无法返回如`start: end: step`的切片形式，故返回切片时要用`slice`对象进行包装

         ```python
         df_demo.loc[lambda x: slice('Gaojuan You', 'Gaoqiang Qian')]

     -   在对表或者序列赋值时，应当在使用**一层索引器后直接进行赋值**操作，这样做是由于进行**多次索引后赋值是赋在临时返回的`copy`副本上**的，而没有真正修改元素，从而报出`SettingWithCopyWarning`警告

         ```python
         df_chain = pd.DataFrame([[0,0],[1,0],[-1,0]], columns=list('AB'))
         df_chain
         import warnings
         with warnings.catch_warnings():
             warnings.filterwarnings('error')
             try:
                 df_chain[df_chain.A!=0]['B'] = 1 # 使用方括号列索引后，再使用一次列索引，共使用2层索引
             except Warning as w:
                 Warning_Msg = w
         print(Warning_Msg)
         df_chain  # 没有任何变化，因为赋值在临时副本上
         
         df_chain.loc[df_chain.A!=0,'B'] = 1  # 只有一层索引
         ```

##### 3.1.4  iloc索引器

`iloc`的使用与`loc`完全类似，只不过是针对位置进行筛选，在相应的`*`位置处一共也有五类合法对象，分别是：整数、整数列表、整数切片、布尔列表以及函数，函数的返回值必须是前面的四类合法对象中的一个，其输入同样也为`DataFrame`本身

-   与`loc`不同，`iloc`整数切片不包含结束端点
-   在使用布尔列表的时候要特别注意，不能传入`Series`而必须传入`Series`的`values`，否则会报错`ValueError: iLocation based boolean indexing cannot use an indexable as a mask`
-   当仅需索引单个元素时，可使用性能更好的`.at[low, col]`和`.iat[row_pos, col_pos]`

##### 3.1.5  query()函数

1.   把字符串形式的查询表达式传入`query`方法来查询数据，其表达式的执行结果必须返回**布尔列表**。

2.   在`query`表达式中，帮用户注册了所有来自`DataFrame`的列名，所有属于该`Series`的方法都可以被调用，和正常的函数调用并没有区别。对于**含有空格的列名**，需要使用`` `col name` ``的方式进行引用。

3.   同时，在`query`中还注册了若干英语的字面用法，帮助提高可读性，例如：`or, and, or, in, not in`

4.   此外，在字符串中出现与列表的比较时，`==`和`!=`分别表示元素出现在列表和没有出现在列表，等价于`in`和`not in`

5.   对于`query`中的字符串，如果要引用外部变量，只需在变量名前加`@`符号

     ```python
     low, high =70, 80
     df.query('(Weight >= @low) & (Weight <= @high)')
     ```

##### 3.1.6  索引运算

筛选出两个表索引的交集/并集

由于索引的元素可能重复，而集合的元素要求互异，因此**先去重**，再计算

-   交集：`id1.intersection(id2)`
-   并集：`id1.union(id2)`
-   差集：`id1.difference(id2)`
-   异或集：`id1.symmetric_difference(id2)`

#### 3.2  多级索引

##### 3.2.1  多级索引及其表结构

##### 3.2.2  多级索引中的loc索引器

##### 3.2.3  多级索引的构造

#### 3.3  常用索引方法

##### 3.3.1  索引层的交换和删除

##### 3.3.2  索引属性的修改

##### 3.3.3  索引的设置与重置

##### 3.3.4  索引的对齐

------

### 第4章  分组

------

### 第5章  变形

------

### 第6章  连接

------

## 第三部分  4类数据

### 第7章  缺失数据

------

### 第8章  文本数据

------

### 第9章  分类数据

------

### 第10章  时间序列数据

------

## 第四部分  进阶实践

### 第11章  数据观测

------

### 第12章  特征工程

------

### 第13章  性能优化