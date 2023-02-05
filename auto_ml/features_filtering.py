#

#/usr/bin/env python

# -*- codeing:utf-8 -*-

"""自动化批量特征筛选模块

=========================
总共分为四个板块：
Part 1.相关第三方库
Part 2.基本方法实现函数
Part 3.高阶函数辅助函数
Part 4.高阶函数

=========================
使用过程中最常调用高阶函数进行批量自动化特征衍生。
高阶函数能够区分执行训练集和测试集的特征衍生过程,
并且支持测试集特征自动补全、目标编码等额外功能,
具体包括：

特征信息包含量指标：
缺失值比例计算:
MissingValueThreshold
单变量方差大小:
VarThreshold

特征和标签关联度指标:
多项式特征衍生函数：
Polynomial_Features

分组统计特征衍生函数：
Group_Statistics

目标编码函数：
Target_Encode

时序字段特征衍生函数：
timeSeries_Creation

NLP特征衍生函数：
NLP_Group_Stat
"""
#######################################################
## Part 1.相关依赖库

# 基础数据科学运算库
import numpy as np
import pandas as pd

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, SelectPercentile
from scipy.stats import pearsonr
from sklearn.feature_selection import r_regression, f_regression, mutual_info_regression  # 回归
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif  # 分类


#######################################################

def MissingValueThreshold(X_train_temp, X_test_temp, threshold = 0.95, fn = -99999):
    """
    根据比例删除缺失值比例较高的特征
    同时将其他缺失值统一填补为fn的值
    
    :param X_train_temp: 训练集特征
    :param X_test_temp: 测试集特征
    :param threshold: 缺失值比例阈值
    :param fn: 其他缺失值填补数值
    
    :return: 剔除指定特征后的X_train_temp和X_test_temp
    """
    for col in X_train_temp:
        if X_train_temp[col].isnull().sum() / X_train_temp.shape[0] >= threshold:
            del X_train_temp[col]
            del X_test_temp[col]
        else:
            X_train_temp[col] = X_train_temp[col].fillna(fn)
            X_test_temp[col] = X_test_temp[col].fillna(fn)
    return X_train_temp, X_test_temp


def VarThreshold(X_train_temp, X_test_temp, threshold=0):
    """
    根据方差删除方差较低的特征
    【必要流程】
    选择VarianceThreshold的默认参数,剔除那些数据一致的特征(无论是离散变量还是连续变量)
    【可选流程】
    如果需要进一步剔除那些方差不为0,但取值较小的列,则可以通过修改方差阈值的方法,
    利用VarianceThreshold剔除那些方差不满足阈值的列。
    【连续变量】
    我们可以通过df.var(ddof=0)查看每一列的方差,在对连续特征方差分布有一定的了解后,
    设置阈值并剔除方差较小的列。这里需要注意的是,方差大小也会受到特征取值大小影响,
    但连续变量的标准化并不能达到消除量纲影响、同时又保留方差能够衡量特征信息量的功能,
    因此大多数时候我们只会考虑利用VarianceThreshold剔除那些方差为0的连续变量。
    切忌先对连续变量进行标准化再进行方差筛选。
    【离散变量】
    对于二分类离散变量来说,我们会假设其满足伯努利分布,然后通过每一类样本的占比与方差之间的映射关系,
    $$\mathrm{Var}[X] = p(1 - p)$$
    通过少数类占比是否少于某一比例(或者多数类样本是否多于某一比例)来判断是否需要剔除某列。

    当然,如果是对于三分类及以上的离散变量,我们可以将其视作连续变量处理。

    :param X_train_temp: 训练集特征
    :param X_test_temp: 测试集特征
    :param threshold: 缺失值比例阈值

    :return: 剔除指定特征后的X_train_temp和X_test_temp
    """

    select_var = VarianceThreshold(threshold=threshold)
    select_var.fit(X_train_temp)
    selected_var = X_train_temp.columns[select_var.get_support()]
    X_train_temp = X_train_temp[selected_var]
    X_test_temp = X_test_temp[selected_var]
    return X_train_temp, X_test_temp


# 连续变量
def feature_selection_corr(X_train_temp, X_test_temp, y_train, k):
    """
    相关系数：衡量连续变量的同步变化趋势,很多分类问题采用相关系数进行衡量并不能得到一个很好的结果
    在机器学习建模流程中,借助关联度指标进行特征筛选也往往是初筛,并不需要精准的检验结果作为依据,
    例如我们并不会以p值作为特征是否可用的依据(并不只使用那些显著相关的变量),
    而是“简单粗暴”的划定一个范围(例如挑选相关性前100的特征)"""
    # pearsonr返回r,相关性系数、p-value(双尾),p值可粗略表示两者不相关的概率,p越小越相关
    # score_func = lambda X, y: np.array(list(map(lambda x: pearsonr(x, y), X.T))).T[0]  # r值

    # 1.r_regression只会根据输入函数的评分按照由高到低进行筛选,因此输出的特征只是相关系数最大并不是相关系数绝对值最大的特征,
    # 2.r_regression返回结果的特征并未排序
    # KB= SelectKBest(r_regression, k=k)
    
    # 3.f_regression基于F检验,计算F-Score,基于F-Score的排序结果和基于相关系数绝对值的排序结果一致
    KB= SelectKBest(f_regression, k=k)
    KB.fit_transform(X_train_temp, y_train)
    selected_var = X_train_temp.columns[KB.get_support()]
    # 4.SelectPercentile可以按照比例进行筛选
    # SP = SelectPercentile(f_regression, percentile=30)
    X_train_temp = X_train_temp[selected_var]
    X_test_temp = X_test_temp[selected_var]
    return X_train_temp, X_test_temp


# 离散变量
def feature_selection_chi(X, y, threshold=0.05):
    """
    通常来说卡方检验是作用于离散变量之间的独立性检验,卡方值越大，关联性越强
    但sklearn中的卡方检验只需要参与检验的其中标签是离散特征即可,
    只不过对于连续变量来说,无法通过列联表的方式进行频数的汇总统计。
    【离散变量】
    计算列联表,得到实际值O
    根据列连表中,行/列的占比(概率值),得到列联表期望值E
    计算卡方值(自由度=(行数-1)*(列数-1))
    【连续变量】
    首先,求得连续特征总和tol;
    然后,计算标签的不同取值占比t0,t1
    然后,我们用总值tol分别乘以t0和t1,算得在0类用户和1类用户中的期望总值E
    然后,计算0类用户和1类用户的实际总值O
    最后,计算卡方值(自由度是y取值水平-1)
    """
    chi_X = X.copy()
    chi_X = chi_X.where(chi_X >= 0, 0)  # 卡方检验不能非负

    chival, pval = chi2(chi_X, y)
    k_chi = pval.shape[0] - (pval > threshold).sum()  # 原假设：两者独立无关,p越小越拒绝
    select_chi = SelectKBest(chi2, k=k_chi)
    select_chi.fit(chi_X, y)
    selected_chi = X.columns[select_chi.get_support()]
    return selected_chi


# 连续变量
def feature_selection_f(X, y, threshold=0.05):
    """
    只能挖掘线性关系,F值越大,p值越小,关联性越强
    """
    # SelectKBest特征筛选评估器在调用score_func时只能使用默认参数,
    # 即把密集矩阵视作连续变量、稀疏矩阵视作离散变量,并且一个X特征矩阵只能有一种类别判定
    # 因此若要调用特征筛选评估器借助MI进行特征筛选,则只能分连续变量和离散变量分别进行筛选
    X_num = X.select_dtypes(include='number')
    X_cat = X.select_dtypes(include='O')

    if X_num.shape[1] > 0:
        fval_num, pval_num = f_classif(X_num, y)
        k_f_num = pval_num.shape[0] - (pval_num > threshold).sum()
        select_f_num = SelectKBest(f_classif, k=k_f_num)
        select_f_num.fit(X_num, y)
        selected_f_num = X_num.columns[select_f_num.get_support()]

    # 如果要对离散变量进行特征筛选,则需要先将离散变量转化为稀疏矩阵
    # if X_cat.shape[1] > 0:
    #     int_spar = pd.SparseDtype(int, fill_value=0)
    #     X_cat = X_cat.astype(int_spar)
    #     fval_cat, pval_cat = f_classif(X_cat, y)
    #     k_f_cat = pval_cat.shape[0] - (pval_cat > threshold).sum()
    #     select_f_cat = SelectKBest(f_classif, k=k_f_cat)
    #     select_f_cat.fit(X_cat, y)
    #     selected_f_cat = X_cat.columns[select_f_cat.get_support()]

    # selected_f = selected_f_num.append(selected_f_cat)
        selected_f = selected_f_num.append(X_cat.columns)
        return selected_f


# 通用分类
def feature_selection_mic(X, y, discrete_features='auto', threshold_coef=0.1):
    """需要人工输入discrete_features: 也就是离散变量的列"""
    val = mutual_info_classif(X, y, discrete_features=discrete_features, random_state=7)  # 互信息估计值
    threshold = val.mean() * threshold_coef
    k_mic = val.shape[0] - (val <= threshold).sum()
    select_mic = SelectKBest(mutual_info_classif, k=k_mic)
    select_mic.fit(X, y)
    selected_mic = X.columns[select_mic.get_support()]
    return selected_mic
