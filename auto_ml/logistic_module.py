import toad
import matplotlib.pyplot as plt
from toad.plot import bin_plot
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

warnings.filterwarnings('ignore')


def process_data(data):
    data = data.set_index(['uuid', 'cust_no', 'certno', 'occur_month', 'overdue_days', 'his_overdue_date', 'busi_amt', 'bal', 'payout_date'])
    for col in data.columns:
        try:
            data[col] = data[col].astype(float)
        except:
            pass
    data.columns = data.columns.str.lower()

    blanks = [np.nan, None, -77777, -88888, -99999, '(null)', '-77777', '-88888', '-99999']
    frame = data.replace(blanks, np.nan)

    return data, frame


def get_bin_num(data, frame, num_col, n_bins=2, min_samples=0.05):
    num_combiner = toad.transform.Combiner()
    # num_col = 'query_last1yearsloanapproval_instisum'
    data_ob = frame[[num_col, 'target']]
    num_combiner.fit(data_ob, y='target', method='dt', n_bins=n_bins, empty_separate=True, min_samples=min_samples)
    # num_combiner.fit(data_ob, y ='target', method='quantile', n_bins=20, empty_separate=True)
    bins = num_combiner.transform(data_ob, labels=True)
    bin_plot(bins, x=num_col, target='target', annotate_format='.3f')
    toad.metrics.KS_bucket(data[num_col], data['target'], bucket=bins[num_col])[['min', 'max', 'bads', 'goods', 'total', 'bad_rate', 'good_rate', 'odds', 'cum_bads_prop', 'cum_goods_prop', 'ks', 'lift', 'cum_lift']]


def get_bin_cat(data, cat_col, ):
    cat_combiner = toad.transform.Combiner()
    # cat_col = 'pboc_education'
    data_ob = data[[cat_col, 'target']]
    cat_combiner.fit(data_ob, y='target', method='chi', n_bins=10, min_samples=0.05)
    bins = cat_combiner.transform(data_ob, labels=True)
    bin_plot(bins, x=cat_col, target='target', annotate_format='.3f')
    toad.metrics.KS_bucket(data[cat_col], data['target'], bucket=bins[cat_col])[['min', 'max', 'bads', 'goods', 'total', 'bad_rate', 'good_rate', 'odds', 'cum_bads_prop', 'cum_goods_prop', 'ks', 'lift', 'cum_lift']]


def stepwise_selection(X, y, initial_list=[], threshold_in=0.01, threshold_out=0.05, verbose=True):
    included = list(initial_list)

    while True:
        changed = False
        # forward step
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            # best_feature = new_pval.argmin()
            best_feature = new_pval.idxmin()
            # best_feature = new_pval.index[new_pval.argmin()]
            included.append(best_feature)
            changed = True
            if verbose:
                print('Add {:30} with p-value {:6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()
        if worst_pval > threshold_out:
            changed = True
            # worst_feature = pvalues.argmax()
            worst_feature = pvalues.idxmax()
            # worst_feature = pvalues.index[pvalues.argmax()]
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included


def get_vif(X, step_select):
    """
    X: df
    step_select: 逐步回归筛选的变量
    """
    vif_df = add_constant(X[step_select])
    vif_res = pd.Series([variance_inflation_factor(vif_df.values, i) for i in range(
        vif_df.shape[1])], index=vif_df.columns)  # >5剃掉，>2.5考虑剃掉
    return vif_res


def get_coef(clf, step_select):
    res = pd.Series({k: v for k, v in zip(
        step_select, clf.coef_[0])}).to_frame()
    res['feature'] = res.index.str.slice(0, -2)
    res.rename({0: 'coef'}, axis=1, inplace=True)
    result = pd.DataFrame({
        'coef': [clf.intercept_[0]],
        'feature': ['intercept'],
    }, index=['intercept'])
    result = pd.concat([result, res], axis=0)
    return result


def probability_mapping(X, clf):
    scores = pd.DataFrame(clf.predict_proba(X), columns=['好样本p', '坏样本p'])
    scores['odds'] = scores['坏样本p'] / scores['好样本p']
    base_score = 650
    odds = 1 / 20
    pdo = 40
    B = pdo / np.log(2)
    A = base_score + B * np.log(odds)
    scores['score'] = round(A - B * np.log(scores['odds']), 0)
    scores.index = X.index
    return scores


def get_var_score(clf, step_select, X):
    base_score = 650
    odds = 1 / 20
    pdo = 40
    B = pdo / np.log(2)
    A = base_score + B * np.log(odds)
    weights = clf.coef_[0]
    bias = clf.intercept_
    feature_score = pd.DataFrame(
        [{'feature': '基准点', 'score': float(A - B * bias)}])

    for i in range(X[step_select].shape[1]):
        col = X[step_select].columns[i]
        f_score = {}
        f_score['feature'] = [col]
        f_score['score'] = [-B * weights[i]]
        f_score = pd.DataFrame(f_score)
        feature_score = pd.concat([feature_score, f_score], axis=0)
    return feature_score


def calculate_psi(base, test, return_frame=True):
    psi = list()
    frame = list()


    def _psi(base_series, test_series, bins=10, min_sample=10):
        base_list = base_series.values
        test_list = test_series.values
        try:
            base_df = pd.DataFrame(base_list, columns=['score'])
            test_df = pd.DataFrame(test_list, columns=['score'])

            # 1.去除缺失值后，统计两个分布的样本量
            base_notnull_cnt = len(list(base_df['score'].dropna()))
            test_notnull_cnt = len(list(test_df['score'].dropna()))

            # 空分箱
            base_null_cnt = len(base_df) - base_notnull_cnt
            test_null_cnt = len(test_df) - test_notnull_cnt

            # 2.最小分箱数
            q_list = []
            if pd.api.types.is_numeric_dtype(base_series):  # 连续型，按分位数分箱汇总
                if type(bins) == int:
                    bin_num = min(bins, int(base_notnull_cnt / min_sample))
                    q_list = [x / bin_num for x in range(1, bin_num)]
                    break_list = []
                    for q in q_list:
                        bk = base_df['score'].quantile(q)
                        break_list.append(bk)
                    break_list = sorted(list(set(break_list)))  # 去重复后排序
                    score_bin_list = [-np.inf] + break_list + [np.inf]
                else:
                    score_bin_list = bins
            elif pd.api.types.is_object_dtype(base_series):
                score_bin_list = sorted(
                    list(set(list(base_list) + list(test_list))))  # 离散型，直接按值汇总

            # 4.统计各分箱内的样本量
            base_cnt_list = [base_null_cnt]
            test_cnt_list = [test_null_cnt]
            bucket_list = ["MISSING"]
            for i in range(len(score_bin_list)-1):
                left = round(score_bin_list[i+0], 4)
                right = round(score_bin_list[i+1], 4)
                bucket_list.append("(" + str(left) + ',' + str(right) + ']')

                base_cnt = base_df[(base_df.score > left) & (base_df.score <= right)].shape[0]
                base_cnt_list.append(base_cnt)

                test_cnt = test_df[(test_df.score > left) & (test_df.score <= right)].shape[0]
                test_cnt_list.append(test_cnt)

            # 5.汇总统计结果
            stat_df = pd.DataFrame({"variable": base_series.name, "bucket": bucket_list, "base_cnt": base_cnt_list, "test_cnt": test_cnt_list})
            stat_df['base_dist'] = stat_df['base_cnt'] / len(base_df)
            stat_df['test_dist'] = stat_df['test_cnt'] / len(test_df)

            def sub_psi(row):
                # 6.计算PSI
                base_list = row['base_dist']
                test_dist = row['test_dist']
                # 处理某分箱内样本量为0的情况
                if base_list == 0 and test_dist == 0:
                    return 0
                elif base_list == 0 and test_dist > 0:
                    base_list = 1 / base_notnull_cnt
                elif base_list > 0 and test_dist == 0:
                    test_dist = 1 / test_notnull_cnt

                return (test_dist - base_list) * np.log(test_dist / base_list)

            stat_df['psi'] = stat_df.apply(lambda row: sub_psi(row), axis=1)
            stat_df = stat_df[['variable', 'bucket', 'base_cnt', 'base_dist', 'test_cnt', 'test_dist', 'psi']]

            psi = stat_df['psi'].sum()

        except:
            print('error!!!')
            psi = np.nan
            stat_df = None
        return psi, stat_df


    def unpack_tuple(x):
        if len(x) == 1:
            return x[0]
        else:
            return x


    if isinstance(test, pd.DataFrame):
        for col in test:
            p, f = _psi(base[col], test[col])
            psi.append(p)
            frame.append(f)

        psi = pd.Series(psi, index=test.columns)
    else:
        psi, frame = _psi(base, test)

    res = (psi,)
    if return_frame:
        res += (frame,)
    return unpack_tuple(res)
