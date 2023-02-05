import pandas as pd
import numpy as np
import xlwings as xw
import scorecardpy as sc
import toad
import sys
import time
from toad.metrics import AUC, KS
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import lightgbm as lgb
from lightgbm import early_stopping
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.early_stop import no_progress_loss
from sklearn.model_selection import KFold, cross_validate
import statsmodels.api as sm
from sklearn.feature_selection import RFE
import shap
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import pickle
import datetime
import warnings
import seaborn as sns
import matplotlib.pyplot as plt


def get_base_info(train):
    """
    返回 DataFrame
    Args:
        train (_type_): DataFrame
    """
    base_info = []
    for col in train.columns:
        base_info.append((col,  
                        train[col].nunique(), 
                        train[col].isnull().sum() * 100 / train.shape[0], 
                        train[col].value_counts(normalize=True, dropna=False).values[0] * 100, 
                        train[col].dtype))
    base_info_df = pd.DataFrame(
        base_info, columns=['Feature', 'unique值个数', '缺失率', '最大同值率', 'type'])
    base_info_df.sort_values('缺失率', ascending=False)
    return base_info_df


def plot_missing(train):
    """
    缺失值分布
    """
    missing = train.isnull().sum()
    missing = missing[missing > 0]
    if missing.shape[0] > 0:
        missing.sort_values(ascending=False, inplace=True)
        missing.plot.bar()
    else:
        return 


def get_variable_category(dat):
    datetime_cols = dat.apply(pd.to_numeric,errors='ignore').select_dtypes(object).apply(pd.to_datetime,errors='ignore').select_dtypes('datetime64').columns.tolist()
    numeric_cols = dat.select_dtypes('number').columns.tolist()
    category_cols = dat.select_dtypes('object').columns.tolist()
    char_cols_too_many_unique = [i for i in category_cols if len(dat[i].unique()) >= 50]
    res = {
        'datetime_cols': datetime_cols,
        'numeric_cols': numeric_cols,
        'category_cols': category_cols,
        'high_cardinality_features': char_cols_too_many_unique
    }
    return res


def process_blank_cols(dat):
    blank_cols = [i for i in list(dat) if dat[i].astype(str).str.findall(r'^\s*$').apply(lambda x:0 if len(x)==0 else 1).sum()>0]
    if len(blank_cols) > 0:
        warnings.warn(f"There are blank strings in {len(blank_cols)} columns, which are replaced with NaN. \n (ColumnNames: {', '.join(blank_cols)})")
#        dat[dat == [' ','']] = np.nan
#        dat2 = dat.apply(lambda x: x.str.strip()).replace(r'^\s*$', np.nan, regex=True)
        dat.replace(r'^\s*$', np.nan, regex=True)
    return dat


def find_outliers_by_isoforest(data, if_plot=True):
    from sklearn.ensemble import IsolationForest
    
    clf = IsolationForest(contamination=0.01, bootstrap=True, random_state=5)
    outlier_index = clf.fit_predict(data)
    outliers = data[outlier_index == -1]
    if if_plot and data.shape[1] >= 2:
        col1 = data.columns[0]
        col2 = data.columns[1]
        sns.scatterplot(data[col1], data[col2], hue=outlier_index, palette='Set1')
    return outliers


def find_outliers_by_modelpred(model, X, y, sigma=3):
    
    def plot_outliers(y, y_pred, z, outliers):
        plt.figure(figsize=(15, 5))
        ax_131 = plt.subplot(1, 3, 1)
        plt.plot(y, y_pred, '.')
        plt.plot(y.loc[outliers], y_pred.loc[outliers], 'ro')
        plt.legend(['Accepted', 'Outliers'])
        plt.xlabel('y')
        plt.ylabel('y_pred')
        
        ax_132 = plt.subplot(1, 3, 2)
        plt.plot(y, y_pred, '.')
        plt.plot(y.loc[outliers], y.loc[outliers] - y_pred.loc[outliers], 'ro')
        plt.legend(['Accepted', 'Outliers'])
        plt.xlabel('y')
        plt.ylabel('y - y_pred')

        ax_133 = plt.subplot(1, 3, 3)
        z.plot.hist(bins=50, ax=ax_133)
        z.loc[outliers].plot.hist(color='r', bins=50, ax=ax_133)
        plt.legend(['Accepted', 'Outliers'])
        plt.xlabel('z')
        plt.ylabel('Frequency')

    model.fit(X, y)
    y_pred = pd.Series(model.predict(X), index=y.index)
    residuals = y - y_pred
    mean_resid = residuals.mean()
    std_resid = residuals.std()
    # cal z statistic, define outliers to be where |z| > sigma
    z = (residuals - mean_resid) / std_resid
    outliers = z[abs(z) > sigma].index
    plot_outliers(y, y_pred, z, outliers)
    return outliers


def find_outliers_by_feature(dat, numeric_cols):
    outliers = []
    for col in numeric_cols:
        upper_limit = dat[col].mean() + 3 * dat[col].std()
        lower_limit = dat[col].mean() - 3 * dat[col].std()
        ind = dat[(dat[col] > upper_limit) | (dat[col] < lower_limit)].index.tolist()
        outliers.extend(ind)
    return list(set(outliers))


def plot_box(train_data):
    """
    箱形图:主要借助中位数和四分位数来进行计算，以上四分位数+1.5倍四分位距为上界、下四分位数-1.5倍四分位距为下界，超出界限则认为是异常值
    """
    fig = plt.figure(figsize=(60, int(np.ceil(train_data.columns.shape[0] / 6) * 50)), dpi=75)  # 4:3
    i = 0
    for col in train_data.columns:
        i += 1
        ax = plt.subplot(len(train_data.columns), 6, i)
        sns.boxplot(x=train_data[col], width=0.5, ax=ax)
        ax.set_xlabel(col, fontsize=36)
    plt.show()


IND = ['uuid', 'cust_no', 'occur_month', 'overdue_days', 'his_overdue_date', 'busi_amt', 'bal', 'payout_date']

def check_index(data):
    no_contain = []
    for i in IND:
        if i not in data.columns:
            no_contain.append(i)
    if no_contain:
        raise Exception(f"[{', '.join(no_contain)}]\t不在数据集中")


def get_date_cols(dev):
    date_cols = []
    for c in dev.columns:
        try:
            dev[c] = dev[c].astype(np.datetime64)
        except:
            pass
        else:
            date_cols.append(c)
    return date_cols


def variable_quality(data, y_label='target', check=True, iv_limit=0.02):
    if check:
        check_index(data)
        data = data.set_index(IND)
    filter_res = sc.var_filter(data, y_label, return_rm_reason=True, iv_limit=iv_limit)
    n_var_num = filter_res['rm']['rm_reason'].isnull().sum()
    print(f'[{n_var_num}]\t个变量可供筛选')

    blanks = [-77777, -88888, -99999, '(null)', '-77777', '-88888', '-99999']
    eda_num = toad.detect(data.select_dtypes(exclude='O'), blanks=blanks)
    eda_cat = toad.detect(data.select_dtypes(include='O'), blanks=blanks)
    if eda_num.shape[0] > 0:
        eda_num['type'] = eda_num['type'].astype(str)
    if eda_cat.shape[0] > 0:
        eda_cat['type'] = eda_cat['type'].astype(str)

    app = xw.App(visible=False, add_book=False)
    wb = app.books.add()
    wb.sheets.add('变量IV、同值率、缺失率')
    wb.sheets.add('变量分布情况')
    wb.sheets['Sheet1'].delete()
    wb.sheets['变量IV、同值率、缺失率'].range('A1').value = filter_res['rm']
    wb.sheets['变量分布情况'].range('A1').value = pd.concat([eda_num, eda_cat])
    wb.sheets['变量IV、同值率、缺失率'].range('A:Z').font.name = '微软雅黑'
    wb.sheets['变量分布情况'].range('A:Z').font.name = '微软雅黑'
    wb.sheets['变量IV、同值率、缺失率'].range('D:D').api.NumberFormat = "0.000_ "
    wb.sheets['变量IV、同值率、缺失率'].range('E:F').api.NumberFormat = "0.00%"

    wb.sheets['变量IV、同值率、缺失率'].autofit()
    wb.sheets['变量分布情况'].autofit()
    wb.save('2.变量EDA.xlsx')
    wb.close()
    app.kill()

    return filter_res['dt']


def variable_special_value_bin(data, check=True, save_breaks_list='special_value_breaks_list'):
    if check:
        check_index(data)
        data = data.set_index(IND)

    # 哪些值被认为是特殊值（特殊值包含缺失值）
    dat_cat = data[data.select_dtypes(include='O').columns.tolist() + ['target']]
    dic_cat = {i : ['-99999', '-88888', '-77777'] for i in dat_cat.columns}
    dat_num = data.select_dtypes(exclude='O')
    dic_num = {i : [-99999, -88888, -77777, -99] for i in dat_num.columns}  # -99999、-88888、-77777、-99会单独分箱
    special_value_bins_num = sc.woebin(dat_num, 'target', special_values=dic_num, method='chimerge', save_breaks_list=save_breaks_list + '_num')
    special_value_bins = special_value_bins_num.copy()
    if len(dat_cat.columns) > 1:
        special_value_bins_cat = sc.woebin(dat_cat, 'target', special_values=dic_cat, method='chimerge', save_breaks_list=save_breaks_list + '_cat')
        special_value_bins.update(special_value_bins_cat)
    special_value_bins_all = pd.concat(special_value_bins, ignore_index=True)

    def gb_distr(binx):
            binx['good_distr'] = binx['good']/sum(binx['count'])
            binx['bad_distr'] = binx['bad']/sum(binx['count'])
            return binx

    special_value_bins_all = special_value_bins_all.groupby('variable').apply(gb_distr)
    return special_value_bins_all


def set_FormatConditions(Selection):
    Selection.api.FormatConditions.Delete()
    # 添加数据条
    Selection.api.FormatConditions.AddDatabar()
    Selection.api.FormatConditions(Selection.api.FormatConditions.Count).ShowValue = True
    Selection.api.FormatConditions(Selection.api.FormatConditions.Count).SetFirstPriority()
    Selection.api.FormatConditions(1).MinPoint.Modify.newtype = xw.constants.ConditionValueTypes.xlConditionValueAutomaticMin
    Selection.api.FormatConditions(1).MaxPoint.Modify.newtype = xw.constants.ConditionValueTypes.xlConditionValueAutomaticMax
    # 数据条颜色
    Selection.api.FormatConditions(1).BarColor.Color = 13012579
    Selection.api.FormatConditions(1).BarColor.TintAndShade = 0
    # 数据条颜色方向等设置
    Selection.api.FormatConditions(1).BarFillType = xw.constants.DataBarFillType.xlDataBarFillGradient
    Selection.api.FormatConditions(1).Direction = xw.constants.Constants.xlContext
    Selection.api.FormatConditions(1).NegativeBarFormat.ColorType = xw.constants.DataBarNegativeColorType.xlDataBarColor
    Selection.api.FormatConditions(1).BarBorder.Type = xw.constants.DataBarBorderType.xlDataBarBorderSolid
    Selection.api.FormatConditions(1).NegativeBarFormat.BorderColorType = xw.constants.DataBarNegativeColorType.xlDataBarColor
    Selection.api.FormatConditions(1).BarBorder.Color.Color = 13012579
    Selection.api.FormatConditions(1).BarBorder.Color.TintAndShade = 0
    Selection.api.FormatConditions(1).AxisPosition = xw.constants.DataBarAxisPosition.xlDataBarAxisAutomatic
    Selection.api.FormatConditions(1).AxisColor.Color = 0
    Selection.api.FormatConditions(1).AxisColor.TintAndShade = 0
    Selection.api.FormatConditions(1).NegativeBarFormat.Color.Color = 255
    Selection.api.FormatConditions(1).NegativeBarFormat.Color.TintAndShade = 0
    Selection.api.FormatConditions(1).NegativeBarFormat.BorderColor.Color = 255
    Selection.api.FormatConditions(1).NegativeBarFormat.BorderColor.TintAndShade = 0


def export_variable_bin(bins, file_name, show_img=False, show_cond_bar=False):
    app = xw.App(visible=False, add_book=False)
    wb = app.books.add()
    wb.sheets.add('目录')
    wb.sheets['Sheet1'].delete()

    num_cols = sorted(bins['variable'].unique().tolist())

    for i in range(len(num_cols)):
        num_col = num_cols[i]
        col = str(i)
        
        wb.sheets.add(col)

        sheet = wb.sheets['目录']
        sheet.api.Hyperlinks.Add(Anchor=sheet.range(f'C{i + 3}').api, Address="", SubAddress=wb.sheets[col].name+"!A1", TextToDisplay=num_col)

        sheet = wb.sheets[col]
        
        bucket = bins[bins['variable'] == num_col].copy()
        bad_prop_total = bucket['bad'].sum() / bucket['count'].sum()
        bucket['cumsum_good_pct'] = bucket['good'].cumsum() / bucket['good'].sum()
        bucket['cumsum_bad_pct'] = bucket['bad'].cumsum() / bucket['bad'].sum()
        bucket['ks'] = abs(bucket['cumsum_bad_pct'] - bucket['cumsum_good_pct'])
        bucket['lift'] = bucket['badprob'] / bad_prop_total

        ks_value = round(bucket['ks'].max(), 3)  # ks值
        iv_value = round(bucket['total_iv'].unique()[0], 3)  # iv值
        sheet.range('A1:Z40').font.name = '微软雅黑'

        sheet.api.Hyperlinks.Add(Anchor=sheet.range('A1').api, Address="", SubAddress=wb.sheets['目录'].name+f"!C{i + 3}", TextToDisplay="Content")

        sheet.range('A3').value = f'KS={ks_value}'
        sheet.range('A3').font.bold = True
        sheet.range('A3').font.color = (255, 0, 0)
        
        sheet.range('B3').value = f'IV={iv_value}'
        sheet.range('B3').font.bold = True
        sheet.range('B3').font.color = (255, 0, 0)

        sheet.range('A4').value = bucket
        
        sheet.range('E:E').api.NumberFormat = "0.00%"
        sheet.range('H:H').api.NumberFormat = "0.00%"
        sheet.range('N:R').api.NumberFormat = "0.00%"
        sheet.range('I:K').api.NumberFormat = "0.0000_ "
        sheet.range('S:S').api.NumberFormat = "0.0000_ "

        sheet.autofit()

        if show_cond_bar:
            Selection = sheet.range('H:H')
            set_FormatConditions(Selection)

        if show_img:
            fig = sc.woebin_plot(bins, x=[num_col])
            sheet.pictures.add(fig[num_col].figure, name=num_col, scale=1, anchor=sheet.range('F16'))
        wb.sheets['目录'].range(f'D{i + 3}').formula = r"=VLOOKUP(C{},'C:\Users\21080270\Desktop\数据字典\[携程、饿了么字典.xlsx]在用字段'!$A:$B,2,FALSE)".format(i + 3)
        wb.sheets['目录'].range(f'E{i + 3}').value = ks_value
        wb.sheets['目录'].range(f'F{i + 3}').value = iv_value

        print("\r", end="")
        print(f"进度: {round(100 * (i + 1) / len(num_cols), 2)}%  ", '♥' * (i // 2), end="")
        sys.stdout.flush()
        time.sleep(0.05)
        

    sheet = wb.sheets['目录']
    sheet.range('C1').value = 'CONTENTS'
    sheet.range('C2').value = 'Var'
    sheet.range('D2').value = '字段含义'
    sheet.range('E2').value = 'KS'
    sheet.range('F2').value = 'IV'
    sheet.range('C:F').font.name = '微软雅黑'
    sheet.range('C1').font.size = 20
    sheet.autofit()

    wb.save(file_name)
    wb.close()
    app.kill()


def variable_breaks_list_bin(data, breaks_list):
    bins_select = sc.woebin(data, 'target', special_values=None, breaks_list=breaks_list, method='chimerge')
    bins_all_select = pd.concat(bins_select, ignore_index=True)

    def gb_distr(binx):
            binx['good_distr'] = binx['good']/sum(binx['count'])
            binx['bad_distr'] = binx['bad']/sum(binx['count'])
            return binx

    bins_all_select = bins_all_select.groupby('variable').apply(gb_distr)
    return bins_select, bins_all_select


def stepwise_selection(X,y,initial_list=[],
                        threshold_in=0.01,
                        threshold_out=0.05,
                        verbose=True):
    included = list(initial_list)
    
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y,sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            # best_feature = new_pval.argmin()
            best_feature = new_pval.index[new_pval.argmin()]
            included.append(best_feature)   
            changed=True
            if verbose:
                print('Add {:30} with p-value {:6}'.format(best_feature,best_pval))

        # backward step
        model = sm.OLS(y,sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()
        if worst_pval > threshold_out:
            changed=True
            # worst_feature = pvalues.argmax()
            worst_feature = pvalues.index[pvalues.argmax()]
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:6}'.format(worst_feature,worst_pval))
        if not changed:
            break
    return included


def get_data_info(df):
    data = df.copy()
    data['payout_month'] = data['payout_date'].str.slice(0, 7)
    cnt = data.groupby('payout_month').size()
    rat = cnt / data.shape[0]
    bad_cnt = data.groupby('payout_month')['target'].sum()
    bad_rate = bad_cnt / cnt
    doc_des = pd.concat([cnt, rat, bad_cnt, bad_rate], axis=1).reset_index()
    doc_des.columns = ['用信时间', '数量', '占比', '坏样本', '坏账率']
    doc_des_total = pd.DataFrame(
        {'用信时间': ['合计'],
        '数量': doc_des['数量'].sum(),
        '占比': [1],
        '坏样本': doc_des['坏样本'].sum(),
        '坏账率': doc_des['坏样本'].sum() / doc_des['数量'].sum(),}
    )
    return pd.concat([doc_des, doc_des_total], axis=0, ignore_index=True)


def get_psi_info(data, oot):
    c = toad.transform.Combiner()
    c.fit(data, 'target', method='quantile', n_bins=6, empty_separate=True)
    psi_val, psi_frame = toad.metrics.PSI(oot, data, combiner=c, return_frame=True)
    psi_frame['psi'] = psi_frame[['test', 'base']].apply(lambda d: (d['test'] - d['base']) * np.log(d['test'] / d['base']), axis=1)
    psi_val = psi_val.to_frame()
    psi_val.columns = ['psi_all']
    return psi_val, pd.merge(psi_frame, psi_val, how='left', left_on=['columns'], right_index=True)


def get_logit_coef(clf, X_train_woe, step_select):
    res = pd.Series({k: v for k, v in zip(X_train_woe[step_select].columns, clf.coef_[0])}).to_frame()
    res['variable'] = res.index.str.slice(0, -4)
    res.rename({0: 'coef'}, axis=1, inplace=True)
    result = pd.DataFrame({
        'coef': [clf.intercept_[0]],
        'variable': ['intercept'],
    }, index=['intercept'])
    result = pd.concat([result, res], axis=0)
    return result


def get_vif_value(X_dev_woe, step_select):
    vif_df = add_constant(X_dev_woe[step_select])
    vif_res = pd.Series([variance_inflation_factor(vif_df.values, i) for i in range(vif_df.shape[1])], index=vif_df.columns)  # >5剃掉，>2.5考虑剃掉
    return vif_res


def get_logit_card(clf, bins_select, step_select, points0=600, odds0=1/20, pdo=40):
    card = sc.scorecard(bins_select, clf, step_select, points0=points0, odds0=odds0, pdo=pdo)
    
    with open(f'11.base model card {datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d %H%M%S")}.pkl', "wb") as f:
        pickle.dump(card, f)
    return pd.concat(card, ignore_index=True)


def get_model_psi_score(train_score, y_train, oot_score, y_oot):
    score = {'train':train_score, 'test':oot_score}
    label = {'train':y_train, 'test':y_oot}
    score['train'].index = range(score['train'].shape[0])
    score['test'].index = range(score['test'].shape[0])
    label['train'].index = range(label['train'].shape[0])
    label['test'].index = range(label['test'].shape[0])

    res = sc.perf_psi(
        score = score,
        label = label,
        x_tick_break = 20,
        return_distr_dat=True
    )
    return res


def get_score_by_card(X, card):
    """根据评分卡给样本打分

    Args:
        X (_type_): 原始字段, 非woe值
        card (_type_): 评分卡

    Returns:
        _type_: 分值
    """
    scores = sc.scorecard_ply(X, card)
    return scores


def get_init_lgb_param(scale_pos_weight=1, seed=7):
    return {'boosting_type': 'gbdt',
            # 'num_leaves': 8,
            'max_depth': 3,
            'learning_rate': 0.05,
            'objective': 'binary',
            'scale_pos_weight': scale_pos_weight,  # 19
            'min_split_gain': 0,
            # 'min_child_weight': 1e-3,  # Minimum sum of instance weight (hessian) needed in a child (leaf)
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_lambda': 0,
            'reg_alpha': 0,
            'random_state': seed,
            # 'importance_type': 'split',
            'force_col_wise': True,
            'verbose': -1,
        }


def model_eva(clf, X_train, y_train, X_test, y_test, X_oot, y_oot, X_dev=None, y_dev=None, if_dev=False):
    if if_dev:
        clf.fit(X_dev, y_dev)
        dev_pred = clf.predict_proba(X_dev)[:,1]

        train_pred = clf.predict_proba(X_train)[:,1]
        test_pred = clf.predict_proba(X_test)[:,1]
        oot_pred = clf.predict_proba(X_oot)[:,1]

        res = {}
        res['auc_train'] = AUC(train_pred, y_train)
        res['auc_test'] = AUC(test_pred, y_test)
        res['auc_oot'] = AUC(oot_pred, y_oot)
        res['ks_train'] = KS(train_pred, y_train)
        res['ks_test'] = KS(test_pred, y_test)
        res['ks_oot'] = KS(oot_pred, y_oot)

        res['auc_dev'] = AUC(dev_pred, y_dev)
        res['ks_dev'] = KS(dev_pred, y_dev)
    else:
        clf.fit(X_train, y_train)

        train_pred = clf.predict_proba(X_train)[:,1]
        test_pred = clf.predict_proba(X_test)[:,1]
        oot_pred = clf.predict_proba(X_oot)[:,1]

        res = {}
        res['auc_train'] = AUC(train_pred, y_train)
        res['auc_test'] = AUC(test_pred, y_test)
        res['auc_oot'] = AUC(oot_pred, y_oot)
        res['ks_train'] = KS(train_pred, y_train)
        res['ks_test'] = KS(test_pred, y_test)
        res['ks_oot'] = KS(oot_pred, y_oot)
    return res


def get_init_lgb_model(X_train, y_train, X_test, y_test, X_oot, y_oot, scale_pos_weight=1, seed=7):
    callback = [early_stopping(stopping_rounds=50)]
    train_lgb = lgb.Dataset(X_train, y_train)
    init_param = get_init_lgb_param(scale_pos_weight, seed)
    results = lgb.cv(init_param, train_lgb, num_boost_round=500, nfold=5, stratified=True, metrics=('auc'), seed=seed, callbacks=callback)
    init_param['n_estimators'] = len(results['auc-mean'])
    clf = LGBMClassifier(**init_param)
    clf.set_params(importance_type ='split')
    return (clf, model_eva(clf, X_train, y_train, X_test, y_test, X_oot, y_oot))


def lgb_model_exploration(X, y, X_oot, y_oot, seeds=range(0, 10, 1)):
    exploration_results = []
    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, stratify=y, test_size=0.3)
        if y_train.mean() < 0.05:
            scale_pos_weight = round(0.05 * len(y_train) / sum(y_train), 4)
        else:
            scale_pos_weight = 1
        (clf, res) = get_init_lgb_model(X_train, y_train, X_test, y_test, X_oot, y_oot, scale_pos_weight, seed)
        r = [seed, 0.3, clf.get_params()['n_estimators'], res['auc_train'], res['auc_test'], res['auc_oot'], res['ks_train'], res['ks_test'], res['ks_oot']]
        exploration_results.append(r)
    df = pd.DataFrame(exploration_results)
    df.columns = ['seed', 'test_size', 'n_estimators', 'auc_train', 'auc_test', 'auc_oot', 'ks_train', 'ks_test', 'ks_oot']
    return df


def lgb_booster_leaf_attribute(param, X_train, y_train):
    clf = LGBMClassifier(**param)
    clf.fit(X_train, y_train)
    # 所有树节点
    t_df = clf.booster_.trees_to_dataframe()
    # 叶子节点
    leaf_node_df = t_df[t_df['left_child'].isnull()]
    s_samples = leaf_node_df['count'].describe(np.arange(0.1, 1, 0.1))
    s_weight = leaf_node_df['weight'].describe(np.arange(0.1, 1, 0.1))
    s_gain = t_df[t_df['node_depth'] == clf.get_params()['max_depth']]['split_gain'].describe(np.arange(0.1, 1, 0.1))
    res = pd.concat([s_samples, s_weight, s_gain], axis=1)
    return res


def lgb_bayes_hyperopt_process(init_param, param_space, X_train, y_train, X_test, y_test):
    bayes_init_param = init_param.copy()

    #? 1.定义目标函数
    def hyperopt_objective(params):
        
        for key in param_space.keys():
            if key in ['n_estimators', 'max_depth', 'min_child_samples', 'num_leaves']:
                bayes_init_param[key] = int(params[key])
            else:
                bayes_init_param[key] = round(params[key], 4)

        clf = LGBMClassifier(**bayes_init_param)
        #定义评估器
        #需要搜索的参数需要从输入的字典中索引出来
        #不需要搜索的参数，可以是设置好的某个值
        #在需要整数的参数前调整参数类型
        # validation_loss = cross_validate(
        #     clf, X_train, y_train
        #     ,scoring="roc_auc"
        #     ,cv=5
        #     ,verbose=False
        #     ,n_jobs=-1
        #     ,error_score='raise'
        #     )
        # #最终输出结果，由于只能取最小值
        # #要求解最大auc所对应的参数组合，使用1-auc
        # return 1 - np.mean(validation_loss["test_score"])

        train_loss = cross_validate(
            clf, X_train, y_train, scoring="roc_auc", cv=5, verbose=False, n_jobs=-1, error_score='raise')
        cv_score_train = np.mean(train_loss["test_score"])
        test_loss = cross_validate(
            clf, X_test, y_test, scoring="roc_auc", cv=5, verbose=False, n_jobs=-1, error_score='raise')
        cv_score_test = np.mean(test_loss["test_score"])
        return 1 - (cv_score_test - abs(cv_score_train - cv_score_test) * 0.2)  # 最大化test，同时减小test与train之间的差异
    
    #? 3.定义优化目标函数的具体流程
    def param_hyperopt(max_evals=100):
        
        #保存迭代过程
        trials = Trials()
        
        #设置提前停止
        early_stop_fn = no_progress_loss(100)
        
        #定义代理模型
        #algo = partial(tpe.suggest, n_startup_jobs=20, n_EI_candidates=50)
        params_best = fmin(
            hyperopt_objective #目标函数
            , space = param_space #参数空间
            , algo = tpe.suggest #代理模型你要哪个呢？
            , max_evals = max_evals #允许的迭代次数
            , verbose=True
            , trials = trials
            , early_stop_fn = early_stop_fn
            , rstate=np.random.default_rng(7)
            )
        
        #打印最优参数，fmin会自动打印最佳分数
        print("\n", "\n", "best params: ", params_best, "\n")
        return params_best, trials
    
    params_best, trials = param_hyperopt(1000)
    return params_best, trials


def lgb_model_bayes_opt(trials_result, init_param, X_train, y_train, X_test, y_test, X_oot, y_oot):
    param_col = sorted(trials_result[0]['misc']['vals'].keys())
    metric_col = ['auc_train', 'auc_test', 'auc_oot', 'ks_train', 'ks_test', 'ks_oot']
    col = ['id'] + param_col + metric_col
    bayes_param = init_param.copy()
    bayes_df = pd.DataFrame(columns=col)

    for i in range(0, len(trials_result)):
        vals = trials_result[i]['misc']['vals']
        val = {k: v[0] for k, v in vals.items()}
        for key in val.keys():
            if key in ['n_estimators', 'max_depth', 'min_child_samples', 'num_leaves']:
                val[key] = int(val[key])
            else:
                val[key] = round(val[key], 4)
        bayes_param.update(val)
        model = LGBMClassifier(**bayes_param)
        
        res = model_eva(model, X_train, y_train, X_test, y_test, X_oot, y_oot)
        params = []
        for c in param_col:
            params.append(val[c])
        metric_res = []
        for m in metric_col:
            metric_res.append(res[m])
        result = [i] + params + metric_res
        
        bayes_df = pd.concat([bayes_df, pd.Series(dict(zip(col, result))).to_frame().T], axis=0)

    bayes_df['ks_dec1'] = bayes_df['ks_train'] - bayes_df['ks_test']
    bayes_df['ks_dec2'] = bayes_df['ks_train'] - bayes_df['ks_oot']
    return bayes_df


def get_bayes_param_from_trail(i, trials, init_param):
    vals = trials.trials[i]['misc']['vals']
    val = {k: v[0] for k, v in vals.items()}
    for key in val.keys():
        if key in ['n_estimators', 'max_depth', 'min_child_samples', 'num_leaves']:
            val[key] = int(val[key])
        else:
            val[key] = round(val[key], 4)

    bayes_param = init_param.copy()
    bayes_param.update(val)
    return bayes_param


#定义单参数调整函数，贪心算法，选择局部最优参数
def lgb_tune1(fixed_params, param, ranges, X_train, y_train, X_test, y_test, X_oot, y_oot):
    lgb_tune_df = pd.DataFrame(columns=['id', 'ks_train', 'ks_test', 'ks_oot', 'auc_train', 'auc_test', 'auc_oot'] )
    for i in ranges:
        fixed_params[param] = i
        model = LGBMClassifier(**fixed_params)
        res = model_eva(model, X_train, y_train, X_test, y_test, X_oot, y_oot)
        col = ['id', 'ks_train', 'ks_test', 'ks_oot', 'auc_train', 'auc_test', 'auc_oot'] 
        result = [i, res['ks_train'], res['ks_test'], res['ks_oot'], res['auc_train'], res['auc_test'], res['auc_oot']]
        lgb_tune_df = pd.concat([lgb_tune_df, pd.Series(dict(zip(col, result))).to_frame().T], axis=0, ignore_index=True)
    lgb_tune_df['ks_dec1']=lgb_tune_df['ks_train']-lgb_tune_df['ks_test']
    lgb_tune_df['ks_dec2']=lgb_tune_df['ks_train']-lgb_tune_df['ks_oot']
    return lgb_tune_df


#定义双参数调整函数，贪心算法，选择局部最优参数
def lgb_tune2(fixed_params, param1, param2, ranges1, ranges2, X_train, y_train, X_test, y_test, X_oot, y_oot):
    ranges1 = ranges1
    ranges2 = ranges2
    lgb_tune_df=pd.DataFrame(columns=['id1', 'id2', 'ks_train', 'ks_test', 'ks_oot', 'auc_train', 'auc_test', 'auc_oot'] )
    for i in ranges1:
        for j in ranges2:
            fixed_params[param1]=i
            fixed_params[param2]=j
            model=LGBMClassifier(**fixed_params)
            res = model_eva(model, X_train, y_train, X_test, y_test, X_oot, y_oot)
            col = ['id1', 'id2', 'ks_train', 'ks_test', 'ks_oot', 'auc_train', 'auc_test', 'auc_oot'] 
            result = [i, j, res['ks_train'], res['ks_test'], res['ks_oot'], res['auc_train'], res['auc_test'], res['auc_oot']]
            lgb_tune_df = pd.concat([lgb_tune_df, pd.Series(dict(zip(col, result))).to_frame().T], axis=0, ignore_index=True)
    lgb_tune_df['ks_dec1'] = lgb_tune_df['ks_train'] - lgb_tune_df['ks_test']
    lgb_tune_df['ks_dec2'] = lgb_tune_df['ks_train'] - lgb_tune_df['ks_oot']
    return lgb_tune_df


def filtering_feat_RFE(model, X_dev, y_dev, X_train, y_train, X_test, y_test, X_oot, y_oot):
    rfe = RFE(model, n_features_to_select=1, step=1, verbose=0)
    # rfe.fit(X_train, y_train)
    rfe.fit(X_dev, y_dev)
    dic = {k : v for k, v in zip(rfe.ranking_, X_train.columns)}

    feature_ranking = pd.Series(dic).reset_index().rename({'index': 'ranking', 0: 'feature'}, axis=1)
    feature_ranking = feature_ranking.sort_values('ranking', axis=0)

    dev_auc_scores = []
    train_auc_scores = []
    test_auc_scores = []
    oot_auc_scores = []

    dev_ks_scores = []
    train_ks_scores = []
    test_ks_scores = []
    oot_ks_scores = []

    for i in range(1, feature_ranking.shape[0] + 1):
        cols = feature_ranking[feature_ranking['ranking'] <= i]['feature'].values
        res = model_eva(model, X_train[cols], y_train, X_test[cols], y_test, X_oot[cols], y_oot, X_dev[cols], y_dev, if_dev=True)
        dev_auc_scores.append(res['auc_dev'])
        train_auc_scores.append(res['auc_train'])
        test_auc_scores.append(res['auc_test'])
        oot_auc_scores.append(res['auc_oot'])

        dev_ks_scores.append(res['ks_dev'])
        train_ks_scores.append(res['ks_train'])
        test_ks_scores.append(res['ks_test'])
        oot_ks_scores.append(res['ks_oot'])

    scores = pd.DataFrame([])
    scores['dev_ks'] = dev_ks_scores
    scores['train_ks'] = train_ks_scores
    scores['test_ks'] = test_ks_scores
    scores['oot_ks'] = oot_ks_scores

    scores['dev_auc'] = dev_auc_scores
    scores['train_auc'] = train_auc_scores
    scores['test_auc'] = test_auc_scores
    scores['oot_auc'] = oot_auc_scores
    scores['des1'] = scores['train_ks'] - scores['test_ks']
    scores['des2'] = scores['train_ks'] - scores['oot_ks']
    scores.index = range(1, feature_ranking.shape[0] + 1)
    return feature_ranking, scores


def lgb_feature_importance(model):
    feature_df1 = pd.DataFrame(zip(model.feature_name_, model.booster_.feature_importance(importance_type='split')),columns=['feature_name','importance_split'])
    feature_df2 = pd.DataFrame(zip(model.feature_name_, model.booster_.feature_importance(importance_type='gain')),columns=['feature_name','importance_gain'])
    feature_df = pd.merge(feature_df1, feature_df2, how='left', on='feature_name')
    feature_df = feature_df.sort_values(by='importance_split',ascending=False)
    return feature_df


def lgb_shap_importance(model, X_dev, X_oot):

    imp_feature_df1 = pd.DataFrame(zip(model.feature_name_, model.booster_.feature_importance(importance_type='split')),columns=['feature_name','importance_split'])
    imp_feature_df2 = pd.DataFrame(zip(model.feature_name_, model.booster_.feature_importance(importance_type='gain')),columns=['feature_name','importance_gain'])
    imp_feature_df = pd.merge(imp_feature_df1, imp_feature_df2, how='left', on='feature_name')
    imp_feature_df = imp_feature_df.sort_values(by='importance_split',ascending=False)

    explainer = shap.TreeExplainer(model)
    shap_values_dev = explainer(X_dev)[:, :, 1]
    shap_values_oot = explainer(X_oot)[:, :, 1]

    feature_df1 = pd.DataFrame(zip(shap_values_dev.feature_names, np.abs(shap_values_dev.values).mean(0)), columns=['feature_name','dev_importance_shap'])
    feature_df2 = pd.DataFrame(zip(shap_values_oot.feature_names, np.abs(shap_values_oot.values).mean(0)), columns=['feature_name','oot_importance_shap'])
    result = pd.merge(feature_df1, feature_df2, how='left', on='feature_name')
    result = pd.merge(result, imp_feature_df, how='left', on='feature_name')
    result = result.sort_values(by='dev_importance_shap',ascending=False)
    return result


def get_score(X, clf, base_score=600, odds=1/20, pdo=40):
    scores = pd.DataFrame(clf.predict_proba(X), columns=['好样本p', '坏样本p'])
    scores['odds'] = scores['坏样本p'] / scores['好样本p']
    B = pdo / np.log(2)
    A = base_score + B * np.log(odds)
    scores['score'] = round(A - B * np.log(scores['odds']), 0)
    scores.index = X.index
    return scores