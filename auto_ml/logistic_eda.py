import pandas as pd
import numpy as np
import toad
import xlwings as xw
import matplotlib.pyplot as plt
from toad.plot import bin_plot
import sys
import time
from varclushi import VarClusHi
import statsmodels.api as sm
import warnings

warnings.filterwarnings('ignore')


def get_eda(eda_num, eda_cat, iv):
    app = xw.App(visible=False, add_book=False)
    wb = app.books.add()
    wb.sheets.add('EDA_num')
    wb.sheets.add('EDA_cat')
    wb.sheets.add('EDA_iv')
    wb.sheets['Sheet1'].delete()

    wb.sheets['EDA_num'].range('A1').value = eda_num
    wb.sheets['EDA_cat'].range('A1').value = eda_cat
    wb.sheets['EDA_iv'].range('A1').value = iv.sort_values(ascending=False)

    wb.sheets['EDA_num'].range('A:Z').font.name = '微软雅黑'
    wb.sheets['EDA_cat'].range('A:Z').font.name = '微软雅黑'
    wb.sheets['EDA_iv'].range('A:Z').font.name = '微软雅黑'

    wb.sheets['EDA_num'].autofit()
    wb.sheets['EDA_cat'].autofit()
    wb.sheets['EDA_iv'].autofit()
    wb.save('1. EDA.xlsx')
    wb.close()
    app.kill()


def get_bucket(data, frame, selected_iv, file_name='2. var bucket.xlsx'):
    app = xw.App(visible=False, add_book=False)
    wb = app.books.add()
    wb.sheets.add('目录')
    wb.sheets['Sheet1'].delete()

    num_cols = selected_iv.drop('target', axis=1).select_dtypes(exclude='O').columns.sort_values()

    for i in range(len(num_cols)):
        num_col = num_cols[i]

        col = str(i)
        
        wb.sheets.add(col)

        sheet = wb.sheets['目录']
        sheet.api.Hyperlinks.Add(Anchor=sheet.range(f'C{i + 3}').api, Address="", SubAddress=wb.sheets[col].name+"!A1", TextToDisplay=num_col)

        sheet = wb.sheets[col]
        num_combiner = toad.transform.Combiner()
        data_ob = frame[[num_col, 'target']]
        num_combiner.fit(data_ob, y ='target', method='quantile', n_bins=10, empty_separate=True)
        bins = num_combiner.transform(data_ob, labels=True)
        res = toad.metrics.KS_bucket(data[num_col], data['target'], bucket=bins[num_col])[['min', 'max', 'bads', 'goods', 'total', 'bad_rate', 'good_rate', 'odds', 'cum_bads_prop', 'cum_goods_prop', 'ks', 'lift', 'cum_lift']]
        ks_value = round(res['ks'].max(), 2)
        iv_value = round(toad.stats.IV(selected_iv[num_col], selected_iv['target'], method='quantile', n_bins=10), 2)
        sheet.range('A1:Z40').font.name = '微软雅黑'

        # sheet.range('A1').value = 'Content'
        # sheet.api.Hyperlinks.Add(Anchor=sheet.range('A1').api, Address="", SubAddress=wb.sheets['目录'].name+"!A1", TextToDisplay="Content")
        sheet.api.Hyperlinks.Add(Anchor=sheet.range('A1').api, Address="", SubAddress=wb.sheets['目录'].name+f"!C{i + 3}", TextToDisplay="Content")

        sheet.range('C3').value = f'KS={ks_value}'
        sheet.range('C3').font.bold = True
        sheet.range('C3').font.color = (255, 0, 0)
        
        sheet.range('D3').value = f'IV={iv_value}'
        sheet.range('D3').font.bold = True
        sheet.range('D3').font.color = (255, 0, 0)

        sheet.range('C4').value = res
        
        sheet.range('I:M').api.NumberFormat = "0.00%"
        sheet.range('N:P').api.NumberFormat = "0.0000_ "

        # sheet.range('A:Z').columns.autofit()
        sheet.autofit()

        # ax = bin_plot(bins, x=num_col, target='target', annotate_format='.3f')
        # sheet.pictures.add(ax.figure, name=num_col, scale=0.75, anchor=sheet.range('D18'))

        wb.sheets['目录'].range(f'D{i + 3}').value = ks_value
        wb.sheets['目录'].range(f'E{i + 3}').value = iv_value

        print("\r", end="")
        print(f"进度: {round(100 * (i + 1) / len(num_cols), 2)}%  ", '♥' * (i // 2), end="")
        sys.stdout.flush()
        time.sleep(0.05)
        

    sheet = wb.sheets['目录']
    sheet.range('C1').value = 'CONTENTS'
    sheet.range('C2').value = 'Var'
    sheet.range('D2').value = 'KS'
    sheet.range('E2').value = 'IV'
    sheet.range('C:E').font.name = '微软雅黑'
    sheet.range('C1').font.size = 20
    sheet.autofit()

    wb.save(file_name)
    wb.close()
    app.kill()


def get_varclus(selected_iv):
    # clus_df = selected_iv[list(set(dummy_df.columns.str.slice(0, -2).tolist()))]
    clus_df = selected_iv
    var_clus_model = VarClusHi(clus_df, maxeigval2=1)  # 0.7，越小聚类越多
    var_clus_model.varclus()
    r = var_clus_model.rsquare
    # r.to_excel('3. var clus.xlsx')

    app = xw.App(visible=False, add_book=False)
    wb = app.books.add()
    wb.sheets.add('var_clus')
    wb.sheets['Sheet1'].delete()
    wb.sheets['var_clus'].range('A1').value = r
    wb.sheets['var_clus'].range('A:Z').font.name = '微软雅黑'
    wb.sheets['var_clus'].autofit()
    wb.save('3. var clus.xlsx')
    wb.close()
    app.kill()


def eda_flow(data):
    data = data.replace('(null)', np.nan)
    data = data.set_index(['uuid', 'cust_no', 'certno', 'occur_month', 'overdue_days', 'his_overdue_date', 'busi_amt', 'bal', 'payout_date'])
    for col in data.columns:
        try:
            data[col] = data[col].astype(float)
        except:
            pass
    data.columns = data.columns.str.lower()

    blanks = [np.nan, None, -77777, -88888, -99999, '(null)', '-77777', '-88888', '-99999']

    eda_num = toad.detect(data.select_dtypes(exclude='O'), blanks=blanks)
    eda_num['type'] = eda_num['type'].astype(str)
    eda_cat = toad.detect(data.select_dtypes(include='O'), blanks=blanks)
    eda_cat['type'] = eda_cat['type'].astype(str)

    frame  = data.replace(blanks, np.nan)
    selected_empty, dropped_empty = toad.selection.drop_empty(frame, threshold=0.9, nan=blanks, return_drop=True)  # 求缺失值，-99999被处理为np.nan
    selected_var, dropped_var = toad.selection.drop_var(selected_empty, threshold=0, return_drop=True)  # 求方差，np.nan可不处理
    selected_var = selected_var.replace(blanks, -1)  # 求IV、corr, np.nan被处理成-1
    selected_iv, dropped_iv, iv = toad.selection.drop_iv(selected_var, target='target', threshold=0.02, return_drop=True, return_iv=True)  # 默认合并：dt
    selected_corr, dropped_corr = toad.selection.drop_corr(selected_iv, target='target', threshold=0.8, by='IV', return_drop=True)

    get_eda(eda_num, eda_cat, iv)

    get_bucket(data, frame, selected_iv)  # selected_corr

    # get_varclus(selected_iv)

    return selected_iv, selected_corr

