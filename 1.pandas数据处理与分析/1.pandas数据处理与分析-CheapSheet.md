| 模块            | 功能                                     | 函数                                                         | 备注 |
| --------------- | ---------------------------------------- | ------------------------------------------------------------ | ---- |
| NumPy数组的构造 |                                          |                                                              |      |
|                 | 等差数列                                 | `np.linspace()`、`np.arange()`                               |      |
|                 | 特殊矩阵                                 | `np.zeros()`、`np.ones()`、`np.eye()`、`np.full()`<br>`np.zeros_like()`、`np.ones_like()`、`np.full_like()` |      |
| 随机数组        | 均匀分布**[low, high)**                  | `np.random.uniform(low=0, high=1, size=(2, 3))`              |      |
|                 | 均匀0-1分布**[0, 1)**                    | `np.random.rand(2, 3, 4)`                                    |      |
|                 | 正态分布$N[\mu,\sigma^2]$                | `np.random.normal(loc=0, scale=1, size=(2, 3))`              |      |
|                 | 标准正态分布$N[0, 1]$                    | `np.random.randn(2, 3, 4)`                                   |      |
|                 | 离散均匀分布整数数组                     | `np.random.randint(low=0, high=1, size=(2, 3))`              |      |
|                 | 有放回抽样                               | `np.random.choice(a=range(5), size=(2, 3), p=[0.1, 0.2, 0.3, 0.2, 0.2])` |      |
|                 | 无放回抽样                               | `np.random.choice(a=range(5), size=(2, 3), replace=False`    |      |
|                 | 打散                                     | `np.random.permutation([1, 2, 3])`<br>等价于无放回抽样列表所有元素: `np.random.choice([1, 2, 3], 3, replace=False)` |      |
|                 | 随机种子                                 | `np.random.seed(7)`                                          |      |
| NumPy数组的变形 |                                          |                                                              |      |
|                 | 维度交换                                 | `np.transpose(a=arr, axes=(1, 2, 0))`<br>原来的1维放到现在的0维，原来的2维放到1维<br>`np.swapaxes(a=arr, axis1=1, axis2=0)`<br>交换两个维度 |      |
|                 | 维度增加                                 | `np.expand_dims(a=arr, axis=(0, 1, 4))`<br>`arr[np.newaxis, np.newaxis, :, :, np.newaxis]` |      |
|                 | 维度缩减                                 | `np.squeeze(expand_arr, axis=(0, 1))`<br>默认缩减所有=1的维度 |      |
|                 | 数组合并                                 | `np.stack([arr1, arr2], axis=0)`<br> 拼接的数组必须尺寸相同，且产生新的维度。axis决定新维度在哪产生，维度大小由拼接的数组数量决定<br>`np.concatenate([arr1, arr2], axis=2)`<br> 只需在拼接的维度上一样即可 |      |
|                 | 数组拆分                                 | `np.split(arr, indices_or_sections=[1, 2], axis=0)`<br>indices_or_sections为整数表示均分，为一维序列，表示沿着axis用索引切割。=[2, 3]表示：arr[:2]、arr[2:3]、arr[3:] |      |
|                 | 数组重复                                 | `np.repeat(a=arr, repeats=[2, 3, 1], axis=1)`<br> repeats列表的长度必须与arr的轴的长度一致 |      |
| NumPy数组的切片 |                                          |                                                              |      |
|                 | 输入**切片**，取子数组                   | `arr[0:2, 0:2, 0:2]`<br> 取(2,2,2)子数组                     |      |
|                 | 输入长度相同的**列表**，取元素           | `arr[[0, 1], [0, 1], [0, 1]]`<br> 输入值表示元素在各个维度的索引，取出arr[0, 0, 0]和arr[1, 1, 1] |      |
|                 | 输入**布尔数组**，保留某一维度的若干维数 | `arr[[True, False, True, False], :, :]`                      |      |
|                 | 最后几个维度的:可以忽略                  | `arr[[True, False, True, False], :, :] == arr[[True, False, True, False]]` |      |
|                 | 最初几个维度的:可以用...代替             | `arr[:, :, 0:2] == arr[..., 0:2]`                            |      |
| 广播机制        |                                          |                                                              |      |
| NumPy常用函数   |                                          |                                                              |      |
|                 | 聚合函数                                 | `max, min, mean, median, std, var, sum, quantile`<<br>分位数计算，使用全局方法`np.median/quantile()`<br>如果数组中包含`nan`，使用**全局方法**`np.nanmax()` |      |
|                 | 相关性计算                               | `np.cov(arr1, arr2)`<br> `np.corrcoef(arr1, arr2)`           |      |
|                 | ufunc逐元素处理函数                      | `np.cos、sin、tan、arccos、arcsin、arctan、abs、sqrt、power、exp、log、log10、log2、ceil、floor()`<br> 全局方法 |      |
|                 | 逻辑函数                                 | **比较**：`<、>、<=、>=、!=、==`<br>**内置函数**：`isnan()、isinf()、all()、any()`<br> **逻辑运算符**：`~、&、|`。优先级：`非not > 与and > 或or`<br>**填充函数**：`np.where(bool_arr, fill_arr_for_True, fill_arr_for_False)`<br> **截断函数**：`np.clip(arr, min, max)` |      |
|                 | 返回索引的函数                           | 返回**非零**、**最大**、**最小**值所在的索引：`np.nonzero()、argmax()、argmin()` |      |
|                 | 累计函数                                 | `cumprod`, `cumsum`分别表示累乘和累加函数<br> `diff`表示和前一个元素做差，默认参数下返回长度为原数组减1 |      |
|                 | 向量内积                                 | `a.dot(b)`                                                   |      |
|                 | 向量、矩阵范数                           | `np.linalg.norm()`                                           |      |
|                 | 矩阵乘法                                 | `a @ b`                                                      |      |
|                 | 卡方统计量                               | 设矩阵$A_{m\times n}$，记$B_{ij} = \frac{(\sum_{i=1}^mA_{ij})\times (\sum_{j=1}^nA_{ij})}{\sum_{i=1}^m\sum_{j=1}^nA_{ij}}$，卡方值：$$\chi^2 = \sum_{i=1}^m\sum_{j=1}^n\frac{(A_{ij}-B_{ij})^2}{B_{ij}}$$<br>`B = A.sum(0) * A.sum(1)[:, None] / A.sum()`<br/>`chi_val = ((A - B) ** 2 / B).sum()` |      |
| 文件读取与写入  |                                          |                                                              |      |
|                 | 文件读取                                 | `read_csv、read_table、read_excel`                           |      |
|                 | 数据写入                                 | `to_csv、to_excel、to_markdown、to_latex`                    |      |
| Pandas常用函数  |                                          |                                                              |      |
|                 | 汇总函数                                 | `df.info, describe`                                          |      |
|                 | 聚合函数                                 | `sum, mean, median, var, std, max, min`<br>`quantile, count, idxmax, idxmin` |      |
|                 | 频次函数                                 | `unique、nunique、value_counts`                              |      |
|                 | 替换函数                                 | `Series.replace({'Female':0, 'Male':1})`<br>`s.replace([1, 2], method='ffill')` |      |
|                 | 逻辑替换                                 | 不满足条件，替换：`Series.where(cond, other)`<br> 满足条件，替换：`Series.mask(cond, other)` |      |
|                 | 数值替换                                 | `round, abs, clip`                                           |      |
|                 | 排序函数                                 | **值排序**`sort_values`<br>**索引排序**`sort_index`<br>**元素排序**`rank()` |      |
|                 | apply函数                                | 输入为一个序列                                               |      |
| 窗口函数        |                                          |                                                              |      |
|                 | 滑动窗口                                 | `.rolling`得到滑窗对象，窗口大小`window`。参与计算的最小样本量`min_periods`<=`window` |      |
|                 | 类滑窗函数                               | `shift, diff, pct_change`，公共参数为`periods=n`，默认为1    |      |
|                 | 扩张窗口                                 | `s.expanding()`                                              |      |
|                 | 指数加权窗口                             | `s.ewm(alpha=0.2)`                                           |      |
| 索引            |                                          |                                                              |      |
|                 | 索引运算                                 | 交集：`id1.intersection(id2)` <br>并集：`id1.union(id2)` <br> 差集：`id1.difference(id2)` <br> 异或集：`id1.symmetric_difference(id2)` |      |
|                 | 多级索引的构造                           | `from_tuples, from_arrays, from_product`                     |      |
|                 | 索引层内部的交换                         | `df_ex.swaplevel(0,2,axis=1)`，只能交换两个层<br>`df_ex.reorder_levels([2,0,1],axis=0)`，可以交换任意层 |      |
|                 | 索引层的删除                             | `df_ex.droplevel([0,1],axis=0)`                              |      |
|                 | 修改索引层名字                           | `df_ex.rename_axis(index={'Upper':'Changed_row'}, columns={'Other':'Changed_Col'}) ` |      |
|                 | 修改索引的值                             | `df_ex.rename(columns={'cat':'not_cat'}, level=2)`<br> `df_temp.index.map(lambda x: (x[0], x[1], x[2].upper()))` |      |
|                 | 替换某一层的整个索引元素                 | `df_ex.rename(index=lambda x: next(iter(list('abcdefgh'))), level=2)` |      |
|                 | 索引的设置与重置                         | `df.set_index([], append=False)`<br> `df.reset_index(drop=False)` |      |
|                 | 索引的对齐                               | `df.reindex(index=['1001','1002','1003','1004'], <br/>                   columns=['Weight','Gender'])`<br> `df.reindex_like(df_existed)` |      |
| 分组            |                                          |                                                              |      |
|                 | groupby对象                              | `gb.ngroups`，得到分组个数<br> `gb.groups`，返回从$\color{#FF0000}{组名}$映射到$\color{#FF0000}{组索引列表}$的字典<br> `gb.size`，统计每个组的元素个数<br> `gb.get_group(组名)`，直接获取所在组对应的行 |      |
|                 | **逐列处理**agg                          | `gb.agg(['sum', 'idxmax', 'skew'])`，使用多个函数<br> `gb.agg({'Height':['mean','max'], 'Weight':'count'})`，对特定的列使用特定的聚合函数<br> `gb.agg(lambda x: x.mean() - x.min())`，使用自定义函数<br> `gb.agg([('range', lambda x: x.max() - x.min()), ('my_sum', 'sum')])`，聚合结果重命名 |      |
|                 | **返回序列**transform                    | 返回同长度的序列、标量广播                                   |      |
|                 | **过滤**filter                           | 对于组的过滤，传入参数只允许返回bool值                       |      |
|                 | **多列操作**apply                        | 可返回标量、Series、DataFrame                                |      |
| 变形            |                                          |                                                              |      |
|                 | 长表变宽表                               | `df.pivot(index=, columns=, values=)`，行列所对应的value需**唯一**<br> `df.pivot_table(index = 'Name', columns = 'Subject', values = 'Grade', aggfunc='mean', margins=True)`，行列对应的值可不唯一，为填入数据，要对其做聚合 |      |
|                 | 宽表变长表                               | `df.melt(id_vars=, value_vars, var_name, value_name)`<br> `pd.wide_to_long(df, stubnames=, i=, j=, sep='_', suffix)`，列中包含交叉信息 |      |
|                 | 索引变形                                 | `stack()`、`unstack()`                                       |      |
|                 | 扩张变形                                 | `df.explode()`<br> `pd.get_dummies()`                        |      |
| 连接            |                                          |                                                              |      |
|                 | 列连接                                   | `pd.merge()`                                                 |      |
|                 | 索引连接                                 | `df1.join(df2)`                                              |      |
|                 | 方向连接                                 | `pd.concat([], join=, axis=, keys=)`                         |      |
|                 | 比较与组合                               | `df1.compare(df2, keep_shape=True)` <br> `df1.combine(df2, func, overwrite=False)` |      |
| 缺失数据        |                                          |                                                              |      |
|                 | 缺失信息的统计                           | `isna()、isnull()、notna()、notnull()`搭配`all、any`         |      |
|                 | 缺失信息的删除                           | `dropna()`                                                   |      |
|                 | 缺失值的填充                             | `fillna()`                                                   |      |
|                 | 插值函数                                 | `s.interpolate(limit_direction='backward', limit=1)`         |      |
|                 |                                          |                                                              |      |
|                 |                                          |                                                              |      |

