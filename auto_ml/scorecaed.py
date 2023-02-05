
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import re
import warnings
import multiprocessing as mp
import matplotlib.pyplot as plt
import time
from  copy  import deepcopy
import statsmodels.api as sm



#===============================================================#
#----------------------------stepwise---------------------------#
#===============================================================#


def stepwise_selection(X,y,initial_list=[],
                        threshold_in=0.01,
                        threshold_out=0.05,
                        verbose=True):
    """
    perform a forward-backward feature selection based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list_like with the target
        initial_list - list of features to starts with (column names od X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features
    Always set threshold_in < threshold_out to avoid infinte looping
    """
    included = list(initial_list)
    
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.Logit(y,sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)   
            changed=True
            if verbose:
                print('Add {:30} with p-value {:6}'.format(best_feature,best_pval))

        # backward step
        model = sm.Logit(y,sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:6}'.format(worst_feature,worst_pval))
        if not changed:
            break
    return included


#===============================================================#
#-------------------------Condition_func------------------------#
#===============================================================#

# remove columns if len(x.unique()) == 1
def rmcol_datetime_unique1(dat, check_char_num = False): # add more datatime types later
    if check_char_num:
        # character columns with too many unique values
        char_cols = [i for i in list(dat) if not is_numeric_dtype(dat[i])]
        char_cols_too_many_unique = [i for i in char_cols if len(dat[i].unique()) >= 50]
    
        if len(char_cols_too_many_unique) > 0:
            print('>>> There are {} variables have too many unique non-numberic values, which might cause the binning process slow. Please double check the following variables: \n{}'.format(len(char_cols_too_many_unique), ', '.join(char_cols_too_many_unique)))
            print('>>> Continue the binning process?')
            print('1: yes \n2: no \n')
            cont = int(input("Selection: "))
            while cont not in [1, 2]:
                cont = int(input("Selection: "))
            if cont == 2:
                raise SystemExit(0)

    # remove only 1 unique vlaues variable 
    unique1_cols = [i for i in list(dat) if len(dat[i].unique())==1]
    if len(unique1_cols) > 0:
        warnings.warn("There are {} columns have only one unique values, which are removed from input dataset. \n (ColumnNames: {})".format(len(unique1_cols), ', '.join(unique1_cols)))
        dat=dat.drop(unique1_cols, axis=1)
    
    # remove date time variable
    datetime_cols = dat.dtypes[dat.dtypes == 'datetime64[ns]'].index.tolist()
    if len(datetime_cols) > 0:
        warnings.warn("There are {} date/time type columns are removed from input dataset. \n (ColumnNames: {})".format(len(datetime_cols), ', '.join(datetime_cols)))
        dat=dat.drop(datetime_cols, axis=1)
    # return dat
    return dat


def rep_blank_na(dat): # cant replace blank string in categorical value with nan
    # remove duplicated index
    if dat.index.duplicated().any():
        dat = dat.reset_index()
        warnings.warn('There are duplicated index in dataset. The index has been reseted.')
    
    blank_cols = [index for index, x in dat.isin(['', ' ']).sum().iteritems() if x > 0]
    if len(blank_cols) > 0:
        warnings.warn('There are blank strings in {} columns, which are replaced with NaN. \n (ColumnNames: {})'.format(len(blank_cols), ', '.join(blank_cols)))
#        dat[dat == [' ','']] = np.nan
#        dat2 = dat.apply(lambda x: x.str.strip()).replace(r'^\s*$', np.nan, regex=True)
        dat.replace(r'^\s*$', np.nan, regex=True)
    
    return dat

def check_y(dat, y, positive):
    positive = str(positive)
    # ncol of dt
    if isinstance(dat, pd.DataFrame) & (dat.shape[1] <= 1): 
        raise Exception("Incorrect inputs; dat should be a DataFrame with at least two columns.")
    
    # y ------
    if isinstance(y, str):
        y = [y]
    # length of y == 1
    if len(y) != 1:
        raise Exception("Incorrect inputs; the length of y should be one")
    
    y = y[0]
    # y not in dat.columns
    if y not in dat.columns: 
        raise Exception("Incorrect inputs; there is no \'{}\' column in dat.".format(y))
    
    # remove na in y
    if pd.isnull(dat[y]).any():
        warnings.warn("There are NaNs in \'{}\' column. The rows with NaN in \'{}\' were removed from dat.".format(y,y))
        dat = dat.dropna(subset=[y])
        # dat = dat[pd.notna(dat[y])]
    
    
    # numeric y to int
    if is_numeric_dtype(dat[y]):
        dat.loc[:,y] = dat[y].apply(lambda x: x if pd.isnull(x) else int(x)) #dat[y].astype(int)
    # length of unique values in y
    unique_y = np.unique(dat[y].values)
    if len(unique_y) == 2:
        # if [v not in [0,1] for v in unique_y] == [True, True]:
        if True in [bool(re.search(positive, str(v))) for v in unique_y]:
            y1 = dat[y]
            y2 = dat[y].apply(lambda x: 1 if str(x) in re.split('\|', positive) else 0)
            if (y1 != y2).any():
                dat.loc[:,y] = y2 #dat[y] = y2
                warnings.warn("The positive value in \"{}\" was replaced by 1 and negative value by 0.".format(y))
        else:
            raise Exception("Incorrect inputs; the positive value in \"{}\" is not specified".format(y))
    else:
        raise Exception("Incorrect inputs; the length of unique values in y column \'{}\' != 2.".format(y))
    
    return dat

# x variable
def x_variable(dat, y, x):
    if isinstance(y, str):
        y = [y]
    x_all = list(set(dat.columns) - set(y))
    
    if x is None:
        x = x_all
    else:
        if isinstance(x, list) is False:
            raise Exception("Incorrect inputs; x variables should be a list.")
            
        else:
            if all([i in list(x_all) for i in x]) is True:
                x = x
            else:
                x_notin_xall = set(x).difference(x_all)
                if len(x_notin_xall) > 0:
                    warnings.warn("Incorrect parameter; there are {} x variables are not exist in input data, which are removed from x. \n({})".format(len(x_notin_xall), ', '.join(x_notin_xall)))
                    x = list(set(x).intersection(x_all))
            
    return x

def check_print_step(print_step):
    if not isinstance(print_step, (int, float)) or print_step<0:
        warnings.warn("Incorrect inputs; print_step should be a non-negative integer. It was set to 1.")
        print_step = 1
    return print_step

# check breaks_list
def check_breaks_list(breaks_list, xs):
    if breaks_list is not None:
        # is string
        if isinstance(breaks_list, str):
            breaks_list = eval(breaks_list)
        # is not dict
        if not isinstance(breaks_list, dict):
            raise Exception("Incorrect inputs; breaks_list should be a dict.")
    return breaks_list

# check special_values
def check_special_values(special_values, xs):
    if special_values is not None:
        # # is string
        # if isinstance(special_values, str):
        #     special_values = eval(special_values)
        if isinstance(special_values, list):
            warnings.warn("The special_values should be a dict. Make sure special values are exactly the same in all variables if special_values is a list.")
            sv_dict = {}
            for i in xs:
                sv_dict[i] = special_values
            special_values = sv_dict
        elif not isinstance(special_values, dict): 
            raise Exception("Incorrect inputs; special_values should be a list or dict.")
    return special_values

#=============================================================#
#-----------------------woe_bin-------------------------------#
#=============================================================#


# converting vector (breaks & special_values) to dataframe
def split_vec_todf(vec):
    '''
    Create a dataframe based on provided vector. 
    Split the rows that including '%,%' into multiple rows. 
    Replace 'missing' by np.nan.

    Params
    ------
    vec: list

    Returns
    ------
    pandas.DataFrame
        returns a dataframe with three columns
        {'bin_chr':orginal vec, 'rowid':index of vec, 'value':splited vec}
    '''
    if vec is not None:
        vec = [str(i) for i in vec]
        a = pd.DataFrame({'bin_chr':vec}).assign(rowid=lambda x:x.index)
        b = pd.DataFrame([i.split('%,%') for i in vec], index=vec)\
        .stack().replace('missing', np.nan) \
        .reset_index(name='value')\
        .rename(columns={'level_0':'bin_chr'})[['bin_chr','value']]
        return pd.merge(a,b,on='bin_chr')


def add_missing_spl_val(dtm, breaks, spl_val):
    '''
    add missing to spl_val if there is nan in dtm.value and 
    missing is not specified in breaks and spl_val
    
    Params
    ------
    dtm: melt dataframe
    breaks: breaks list
    spl_val: speical values list
    
    Returns
    ------
    list
        returns spl_val list
    '''
    if dtm.value.isnull().any():
        if breaks is None:
            if spl_val is None:
                spl_val=['missing']
            elif any([('missing' in str(i)) for i in spl_val]):
                spl_val=spl_val
            else:
                spl_val=['missing']+spl_val
        elif any([('missing' in str(i)) for i in breaks]):
            spl_val=spl_val
        else:
            if spl_val is None:
                spl_val=['missing']
            elif any([('missing' in i) for i in spl_val]):
                spl_val=spl_val
            else:
                spl_val=['missing']+spl_val
    return spl_val


# count number of good or bad in y
def n0(x): return sum(x==0)
def n1(x): return sum(x==1)

# split dtm into bin_sv and dtm (without speical_values)
def dtm_binning_sv(dtm, breaks, spl_val):
    '''
    Split the orginal dtm (melt dataframe) into 
    binning_sv (binning of special_values) and 
    a new dtm (without special_values).
    
    Params
    ------
    dtm: melt dataframe
    spl_val: speical values list
    
    Returns
    ------
    list
        returns a list with binning_sv and dtm
    '''
    spl_val = add_missing_spl_val(dtm, breaks, spl_val)
    if spl_val is not None:
        # special_values from vector to dataframe
        sv_df = split_vec_todf(spl_val)
        # value 
        if is_numeric_dtype(dtm['value']):
            sv_df['value'] = sv_df['value'].astype(dtm['value'].dtypes)
            # sv_df['bin_chr'] = sv_df['bin_chr'].astype(dtm['value'].dtypes).astype(str)
            sv_df['bin_chr'] = np.where(
              np.isnan(sv_df['value']), sv_df['bin_chr'], 
              sv_df['value'].astype(dtm['value'].dtypes).astype(str))
            # sv_df = sv_df.assign(value = lambda x: x.value.astype(dtm['value'].dtypes))
        # dtm_sv & dtm
        dtm_sv = pd.merge(dtm.fillna("missing"), sv_df[['value']].fillna("missing"), how='inner', on='value', right_index=True)
        dtm = dtm[~dtm.index.isin(dtm_sv.index)].reset_index() if len(dtm_sv.index) < len(dtm.index) else None
        # dtm_sv = dtm.query('value in {}'.format(sv_df['value'].tolist()))
        # dtm    = dtm.query('value not in {}'.format(sv_df['value'].tolist()))
        
        if dtm_sv.shape[0] == 0:
            return {'binning_sv':None, 'dtm':dtm}
        # binning_sv
        binning_sv = pd.merge(
          dtm_sv.fillna('missing').groupby(['variable','value'])['y'].agg([n0, n1])\
          .reset_index().rename(columns={'n0':'good','n1':'bad'}),
          sv_df.fillna('missing'), 
          on='value'
        ).groupby(['variable', 'rowid', 'bin_chr']).agg({'bad':sum,'good':sum})\
        .reset_index().rename(columns={'bin_chr':'bin'})\
        .drop('rowid', axis=1)
    else:
        binning_sv = None
    # return
    return {'binning_sv':binning_sv, 'dtm':dtm}
    
# check empty bins for unmeric variable
def check_empty_bins(dtm, binning):
    bin_list = np.unique(dtm.bin.astype(str)).tolist()
    if 'nan' in bin_list: 
        bin_list.remove('nan')
    binleft = set([re.match(r'\[(.+),(.+)\)', i).group(1) for i in bin_list]).difference(set(['-inf', 'inf']))
    binright = set([re.match(r'\[(.+),(.+)\)', i).group(2) for i in bin_list]).difference(set(['-inf', 'inf']))
    if binleft != binright:
        bstbrks = sorted(list(map(float, ['-inf'] + list(binright) + ['inf'])))
        labels = ['[{},{})'.format(bstbrks[i], bstbrks[i+1]) for i in range(len(bstbrks)-1)]
        dtm.loc[:,'bin'] = pd.cut(dtm['value'], bstbrks, right=False, labels=labels)
        binning = dtm.groupby(['variable','bin'])['y'].agg([n0, n1])\
          .reset_index().rename(columns={'n0':'good','n1':'bad'})
    return binning

# required in woebin2 # return binning if breaks provided
#' @import data.table
def woebin2_breaks(dtm, breaks, spl_val):
    '''
    get binning if breaks is provided
    
    Params
    ------
    dtm: melt dataframe
    breaks: breaks list
    spl_val: speical values list
    
    Returns
    ------
    DataFrame
        returns a binning datafram
    '''
    
    # breaks from vector to dataframe
    bk_df = split_vec_todf(breaks)
    # dtm $ binning_sv
    dtm_binsv_list = dtm_binning_sv(dtm, breaks, spl_val)
    dtm = dtm_binsv_list['dtm']
    binning_sv = dtm_binsv_list['binning_sv']
    if dtm is None: return {'binning_sv':binning_sv, 'binning':None}
    
    # binning
    if is_numeric_dtype(dtm['value']):
        # best breaks
        bstbrks = ['-inf'] + list(set(bk_df.value.tolist()).difference(set([np.nan, '-inf', 'inf', 'Inf', '-Inf']))) + ['inf']
        bstbrks = sorted(list(map(float, bstbrks)))
        # cut
        labels = ['[{},{})'.format(bstbrks[i], bstbrks[i+1]) for i in range(len(bstbrks)-1)]
        dtm.loc[:,'bin'] = pd.cut(dtm['value'], bstbrks, right=False, labels=labels)
        dtm['bin'] = dtm['bin'].astype(str)
        
        binning = dtm.groupby(['variable','bin'])['y'].agg([n0, n1])\
          .reset_index().rename(columns={'n0':'good','n1':'bad'})
        # check empty bins for unmeric variable
        binning = check_empty_bins(dtm, binning)
        
        # sort bin
        binning = pd.merge(
          binning.assign(value=lambda x: [float(re.search(r"^\[(.*),(.*)\)", i).group(2)) if i != 'nan' else np.nan for i in binning['bin']] ),
          bk_df.assign(value=lambda x: x.value.astype(float)), 
          how='left',on='value'
        ).sort_values(by="rowid").reset_index(drop=True)
        # merge binning and bk_df if nan isin value
        if bk_df['value'].isnull().any():
            binning = binning.assign(bin=lambda x: [i if i != 'nan' else 'missing' for i in x['bin']])\
              .fillna('missing').groupby(['variable','rowid'])\
              .agg({'bin':lambda x: '%,%'.join(x), 'good':sum, 'bad':sum})\
              .reset_index()
    else:
        # merge binning with bk_df
        binning = pd.merge(
          dtm, 
          bk_df.assign(bin=lambda x: x.bin_chr),
          how='left', on='value'
        ).fillna('missing').groupby(['variable', 'rowid', 'bin'])['y'].agg([n0,n1])\
        .rename(columns={'n0':'good','n1':'bad'})\
        .reset_index().drop('rowid', axis=1)
    # return
    return {'binning_sv':binning_sv, 'binning':binning}
    
# required in woebin2_init_bin # return pretty breakpoints
def pretty(low, high, n):
    '''
    pretty breakpoints, the same as pretty function in R
    
    Params
    ------
    low: minimal value 
    low: maximal value 
    n: number of intervals
    
    Returns
    ------
    numpy.ndarray
        returns a breakpoints array
    '''
    # nicenumber
    def nicenumber(x):
        exp = np.trunc(np.log10(abs(x)))
        f   = abs(x) / 10**exp
        if f < 1.5:
            nf = 1.
        elif f < 3.:
            nf = 2.
        elif f < 7.:
            nf = 5.
        else:
            nf = 10.
        return np.sign(x) * nf * 10.**exp
    # pretty breakpoints
    d     = abs(nicenumber((high-low)/(n-1)))
    miny  = np.floor(low  / d) * d
    maxy  = np.ceil (high / d) * d
    return np.arange(miny, maxy+0.5*d, d)

# required in woebin2 # return initial binning
def woebin2_init_bin(dtm, min_perc_fine_bin, breaks, spl_val):
    '''
    initial binning
    
    Params
    ------
    dtm: melt dataframe
    min_perc_fine_bin: the minimal precentage in the fine binning process
    breaks: breaks
    breaks: breaks list
    spl_val: speical values list
    
    returns
    ------
    dict
        returns a dict with initial binning and special_value binning
    '''
    
    # dtm $ binning_sv
    dtm_binsv_list = dtm_binning_sv(dtm, breaks, spl_val)
    dtm = dtm_binsv_list['dtm']
    binning_sv = dtm_binsv_list['binning_sv']
    if dtm is None: return {'binning_sv':binning_sv, 'initial_binning':None}
    # binning
    if is_numeric_dtype(dtm['value']): # numeric variable
        xvalue = dtm['value']
        # breaks vector & outlier
        iq = xvalue.quantile([0.25, 0.5, 0.75])
        iqr = iq[0.75] - iq[0.25]
        xvalue_rm_outlier = xvalue if iqr == 0 else xvalue[(xvalue >= iq[0.25]-3*iqr) & (xvalue <= iq[0.75]+3*iqr)]
        # number of initial binning
        n = np.trunc(1/min_perc_fine_bin)
        len_uniq_x = len(np.unique(xvalue_rm_outlier))
        if len_uniq_x < n: n = len_uniq_x
        # initial breaks
        brk = np.unique(xvalue_rm_outlier) if len_uniq_x < 10 else pretty(min(xvalue_rm_outlier), max(xvalue_rm_outlier), n)
        
        brk = list(filter(lambda x: x>np.nanmin(xvalue) and x<np.nanmax(xvalue), brk))
        brk = [float('-inf')] + sorted(brk) + [float('inf')]
        # initial binning datatable
        # cut
        labels = ['[{},{})'.format(brk[i], brk[i+1]) for i in range(len(brk)-1)]
        dtm.loc[:,'bin'] = pd.cut(dtm['value'], brk, right=False, labels=labels)#.astype(str)
        # init_bin
        init_bin = dtm.groupby('bin')['y'].agg([n0, n1])\
        .reset_index().rename(columns={'n0':'good','n1':'bad'})
        # check empty bins for unmeric variable
        init_bin = check_empty_bins(dtm, init_bin)
        
        init_bin = init_bin.assign(
          variable = dtm['variable'].values[0],
          brkp = lambda x: [float(re.match('^\[(.*),.+', i).group(1)) for i in x['bin']],
          badprob = lambda x: x['bad']/(x['bad']+x['good'])
        )[['variable', 'bin', 'brkp', 'good', 'bad', 'badprob']]
    else: # other type variable
        # initial binning datatable
        init_bin = dtm.groupby('value')['y'].agg([n0,n1])\
        .rename(columns={'n0':'good','n1':'bad'})\
        .assign(
          variable = dtm['variable'].values[0],
          badprob = lambda x: x['bad']/(x['bad']+x['good'])
        ).reset_index()
        # order by badprob if is.character
        if dtm.value.dtype.name not in ['category', 'bool']:
            init_bin = init_bin.sort_values(by='badprob').reset_index()
        # add index as brkp column
        init_bin = init_bin.assign(brkp = lambda x: x.index)\
            [['variable', 'value', 'brkp', 'good', 'bad', 'badprob']]\
            .rename(columns={'value':'bin'})
    
    # remove brkp that good == 0 or bad == 0 ------
    while len(init_bin.query('(good==0) or (bad==0)')) > 0:
        # brkp needs to be removed if good==0 or bad==0
        rm_brkp = init_bin.assign(count = lambda x: x['good']+x['bad'])\
        .assign(
          count_lag  = lambda x: x['count'].shift(1).fillna(len(dtm)+1),
          count_lead = lambda x: x['count'].shift(-1).fillna(len(dtm)+1)
        ).assign(merge_tolead = lambda x: x['count_lag'] > x['count_lead'])\
        .query('(good==0) or (bad==0)')\
        .query('count == count.min()').iloc[0,]
        # set brkp to lead's or lag's
        shift_period = -1 if rm_brkp['merge_tolead'] else 1
        init_bin = init_bin.assign(brkp2  = lambda x: x['brkp'].shift(shift_period))\
        .assign(brkp = lambda x:np.where(x['brkp'] == rm_brkp['brkp'], x['brkp2'], x['brkp']))
        # groupby brkp
        init_bin = init_bin.groupby('brkp').agg({
          'variable':lambda x: np.unique(x),
          'bin': lambda x: '%,%'.join(x),
          'good': sum,
          'bad': sum
        }).assign(badprob = lambda x: x['bad']/(x['good']+x['bad']))\
        .reset_index()
    # format init_bin
    if is_numeric_dtype(dtm['value']):
        init_bin = init_bin\
        .assign(bin = lambda x: [re.sub(r'(?<=,).+%,%.+,', '', i) if ('%,%' in i) else i for i in x['bin']])\
        .assign(brkp = lambda x: [float(re.match('^\[(.*),.+', i).group(1)) for i in x['bin']])
    # return 
    return {'binning_sv':binning_sv, 'initial_binning':init_bin}


# required in woebin2_tree # add 1 best break for tree-like binning
def woebin2_tree_add_1brkp(dtm, initial_binning, min_perc_coarse_bin, bestbreaks=None):
    '''
    add a breakpoint into provided bestbreaks
    
    Params
    ------
    dtm
    initial_binning
    min_perc_coarse_bin
    bestbreaks
    
    Returns
    ------
    DataFrame
        a binning dataframe with updated breaks
    '''
    def iv_01(good, bad):
        # iv calculation
        iv_total = pd.DataFrame({'good':good,'bad':bad}) \
        .replace(0, 0.9) \
        .assign(
            DistrBad = lambda x: x.bad/sum(x.bad),
            DistrGood = lambda x: x.good/sum(x.good)
        ) \
        .assign(iv = lambda x: (x.DistrBad-x.DistrGood)*np.log(x.DistrBad/x.DistrGood)) \
        .iv.sum()
        return iv_total

    # dtm removed values in spl_val
    # total_iv for all best breaks
    def total_iv_all_breaks(initial_binning, bestbreaks, dtm_rows):
        # best breaks set
        breaks_set = set(initial_binning.brkp).difference(set(list(map(float, ['-inf', 'inf']))))
        if bestbreaks is not None: breaks_set = breaks_set.difference(set(bestbreaks))
        breaks_set = sorted(breaks_set)
        # loop on breaks_set
        init_bin_all_breaks = initial_binning.copy(deep=True)
        for i in breaks_set:
            # best break + i
            bestbreaks_i = [float('-inf')]+sorted(bestbreaks+[i] if bestbreaks is not None else [i])+[float('inf')]
            # best break datatable
            labels = ['[{},{})'.format(bestbreaks_i[i], bestbreaks_i[i+1]) for i in range(len(bestbreaks_i)-1)]
            init_bin_all_breaks.loc[:,'bstbin'+str(i)] = pd.cut(init_bin_all_breaks['brkp'], bestbreaks_i, right=False, labels=labels)#.astype(str)
        # best break dt
        total_iv_all_brks = pd.melt(
          init_bin_all_breaks, id_vars=["variable", "good", "bad"], var_name='bstbin', 
          value_vars=['bstbin'+str(i) for i in breaks_set])\
          .groupby(['variable', 'bstbin', 'value'])\
          .agg({'good':sum, 'bad':sum}).reset_index()\
          .assign(count=lambda x: x['good']+x['bad'])
          
        total_iv_all_brks['count_distr'] = total_iv_all_brks.groupby(['variable', 'bstbin'])\
          ['count'].apply(lambda x: x/dtm_rows).reset_index(drop=True)
        total_iv_all_brks['min_count_distr'] = total_iv_all_brks.groupby(['variable', 'bstbin'])\
          ['count_distr'].transform(lambda x: min(x))
          
        total_iv_all_brks = total_iv_all_brks\
          .assign(bstbin = lambda x: [float(re.sub('^bstbin', '', i)) for i in x['bstbin']] )\
          .groupby(['variable','bstbin','min_count_distr'])\
          .apply(lambda x: iv_01(x['good'], x['bad'])).reset_index(name='total_iv')
        # return 
        return total_iv_all_brks
    # binning add 1best break
    def binning_add_1bst(initial_binning, bestbreaks):
        if bestbreaks is None:
            bestbreaks_inf = [float('-inf'),float('inf')]
        else:
            bestbreaks_inf = [float('-inf')]+sorted(bestbreaks)+[float('inf')]
        labels = ['[{},{})'.format(bestbreaks_inf[i], bestbreaks_inf[i+1]) for i in range(len(bestbreaks_inf)-1)]
        binning_1bst_brk = initial_binning.assign(
          bstbin = lambda x: pd.cut(x['brkp'], bestbreaks_inf, right=False, labels=labels)
        )
        if is_numeric_dtype(dtm['value']):
            binning_1bst_brk = binning_1bst_brk.groupby(['variable', 'bstbin'])\
            .agg({'good':sum, 'bad':sum}).reset_index().assign(bin=lambda x: x['bstbin'])\
            [['bstbin', 'variable', 'bin', 'good', 'bad']]
        else:
            binning_1bst_brk = binning_1bst_brk.groupby(['variable', 'bstbin'])\
            .agg({'good':sum, 'bad':sum, 'bin':lambda x:'%,%'.join(x)}).reset_index()\
            [['bstbin', 'variable', 'bin', 'good', 'bad']]
        # format
        binning_1bst_brk['total_iv'] = iv_01(binning_1bst_brk.good, binning_1bst_brk.bad)
        binning_1bst_brk['bstbrkp'] = [float(re.match("^\[(.*),.+", i).group(1)) for i in binning_1bst_brk['bstbin']]
        return binning_1bst_brk
    # dtm_rows
    dtm_rows = len(dtm.index)
    # total_iv for all best breaks
    total_iv_all_brks = total_iv_all_breaks(initial_binning, bestbreaks, dtm_rows)
    # bestbreaks: total_iv == max(total_iv) & min(count_distr) >= min_perc_coarse_bin
    bstbrk_maxiv = total_iv_all_brks.loc[lambda x: x['min_count_distr'] >= min_perc_coarse_bin]
    if len(bstbrk_maxiv.index) > 0:
        bstbrk_maxiv = bstbrk_maxiv.loc[lambda x: x['total_iv']==max(x['total_iv'])]
        bstbrk_maxiv = bstbrk_maxiv['bstbin'].tolist()[0]
    else:
        bstbrk_maxiv = None
    # bestbreaks
    if bstbrk_maxiv is not None:
        # add 1best break to bestbreaks
        bestbreaks = bestbreaks+[bstbrk_maxiv] if bestbreaks is not None else [bstbrk_maxiv]
    # binning add 1best break
    bin_add_1bst = binning_add_1bst(initial_binning, bestbreaks)
    return bin_add_1bst
    
    
# required in woebin2 # return tree-like binning
def woebin2_tree(dtm, min_perc_fine_bin=0.02, min_perc_coarse_bin=0.05, 
                 stop_limit=0.1, max_num_bin=8, breaks=None, spl_val=None):
    '''
    binning using tree-like method
    
    Params
    ------
    dtm:
    min_perc_fine_bin:
    min_perc_coarse_bin:
    stop_limit:
    max_num_bin:
    breaks:
    spl_val:
    
    Returns
    ------
    dict
        returns a dict with initial binning and special_value binning
    '''
    # initial binning
    bin_list = woebin2_init_bin(dtm, min_perc_fine_bin=min_perc_fine_bin, breaks=breaks, spl_val=spl_val)
    initial_binning = bin_list['initial_binning']
    binning_sv = bin_list['binning_sv']
    if len(initial_binning.index)==1: 
        return {'binning_sv':binning_sv, 'binning':initial_binning}
    # initialize parameters
    len_brks = len(initial_binning.index)
    bestbreaks = None
    binning_tree = None
    IVt1 = IVt2 = 1e-10
    IVchg = 1 ## IV gain ratio
    step_num = 1
    # best breaks from three to n+1 bins
    while (IVchg >= stop_limit) and (step_num+1 <= min([max_num_bin, len_brks])):
        binning_tree = woebin2_tree_add_1brkp(dtm, initial_binning, min_perc_coarse_bin, bestbreaks)
        # best breaks
        bestbreaks = binning_tree.loc[lambda x: x['bstbrkp'] != float('-inf'), 'bstbrkp'].tolist()
        # information value
        IVt2 = binning_tree['total_iv'].tolist()[0]
        IVchg = IVt2/IVt1-1 ## ratio gain
        IVt1 = IVt2
        # step_num
        step_num = step_num + 1
    if binning_tree is None: binning_tree = initial_binning
    return {'binning_sv':binning_sv, 'binning':binning_tree}
    

# required in woebin2 # return chimerge binning
#' @importFrom stats qchisq
def woebin2_chimerge(dtm, min_perc_fine_bin=0.02, min_perc_coarse_bin=0.05, 
                     stop_limit=0.1, max_num_bin=8, breaks=None, spl_val=None):
    '''
    binning using chimerge method
    
    Params
    ------
    dtm:
    min_perc_fine_bin:
    min_perc_coarse_bin:
    stop_limit:
    max_num_bin:
    breaks:
    spl_val:
    
    Returns
    ------
    dict
        returns a dict with initial binning and special_value binning
    '''
    # [chimerge](http://blog.csdn.net/qunxingvip/article/details/50449376)
    # [ChiMerge:Discretization of numeric attributs](http://www.aaai.org/Papers/AAAI/1992/AAAI92-019.pdf)
    # chisq = function(a11, a12, a21, a22) {
    #   A = list(a1 = c(a11, a12), a2 = c(a21, a22))
    #   Adf = do.call(rbind, A)
    #
    #   Edf =
    #     matrix(rowSums(Adf), ncol = 1) %*%
    #     matrix(colSums(Adf), nrow = 1) /
    #     sum(Adf)
    #
    #   sum((Adf-Edf)^2/Edf)
    # }
    # function to create a chisq column in initial_binning
    def add_chisq(initial_binning):
        chisq_df = pd.melt(initial_binning, 
          id_vars=["brkp", "variable", "bin"], value_vars=["good", "bad"],
          var_name='goodbad', value_name='a')\
        .sort_values(by=['goodbad', 'brkp']).reset_index(drop=True)
        ###
        chisq_df['a_lag'] = chisq_df.groupby('goodbad')['a'].apply(lambda x: x.shift(1))#.reset_index(drop=True)
        chisq_df['a_rowsum'] = chisq_df.groupby('brkp')['a'].transform(lambda x: sum(x))#.reset_index(drop=True)
        chisq_df['a_lag_rowsum'] = chisq_df.groupby('brkp')['a_lag'].transform(lambda x: sum(x))#.reset_index(drop=True)
        ###
        chisq_df = pd.merge(
          chisq_df.assign(a_colsum = lambda df: df.a+df.a_lag), 
          chisq_df.groupby('brkp').apply(lambda df: sum(df.a+df.a_lag)).reset_index(name='a_sum'))\
        .assign(
          e = lambda df: df.a_rowsum*df.a_colsum/df.a_sum,
          e_lag = lambda df: df.a_lag_rowsum*df.a_colsum/df.a_sum
        ).assign(
          ae = lambda df: (df.a-df.e)**2/df.e + (df.a_lag-df.e_lag)**2/df.e_lag
        ).groupby('brkp').apply(lambda x: sum(x.ae)).reset_index(name='chisq')
        return pd.merge(initial_binning.assign(count = lambda x: x['good']+x['bad']), chisq_df, how='left')
    # initial binning
    bin_list = woebin2_init_bin(dtm, min_perc_fine_bin=min_perc_fine_bin, breaks=breaks, spl_val=spl_val)
    initial_binning = bin_list['initial_binning']
    binning_sv = bin_list['binning_sv']
    # dtm_rows
    dtm_rows = len(dtm.index)
    # chisq limit
    from scipy.special import chdtri
    chisq_limit = chdtri(1, stop_limit)
    # binning with chisq column
    binning_chisq = add_chisq(initial_binning)
    # param
    bin_chisq_min = binning_chisq.chisq.min()
    bin_count_distr_min = min(binning_chisq['count']/dtm_rows)
    bin_nrow = len(binning_chisq.index)
    # remove brkp if chisq < chisq_limit
    while bin_chisq_min < chisq_limit or bin_count_distr_min < min_perc_coarse_bin or bin_nrow > max_num_bin:
        # brkp needs to be removed
        if bin_chisq_min < chisq_limit:
            rm_brkp = binning_chisq.assign(merge_tolead = False).sort_values(by=['chisq', 'count']).iloc[0,]
        elif bin_count_distr_min < min_perc_coarse_bin:
            rm_brkp = binning_chisq.assign(
              count_distr = lambda x: x['count']/sum(x['count']),
              chisq_lead = lambda x: x['chisq'].shift(-1).fillna(float('inf'))
            ).assign(merge_tolead = lambda x: x['chisq'] > x['chisq_lead'])
            # replace merge_tolead as True
            rm_brkp.loc[np.isnan(rm_brkp['chisq']), 'merge_tolead']=True
            # order select 1st
            rm_brkp = rm_brkp.sort_values(by=['count_distr']).iloc[0,]
        elif bin_nrow > max_num_bin:
            rm_brkp = binning_chisq.assign(merge_tolead = False).sort_values(by=['chisq', 'count']).iloc[0,]
        # set brkp to lead's or lag's
        shift_period = -1 if rm_brkp['merge_tolead'] else 1
        binning_chisq = binning_chisq.assign(brkp2  = lambda x: x['brkp'].shift(shift_period))\
        .assign(brkp = lambda x:np.where(x['brkp'] == rm_brkp['brkp'], x['brkp2'], x['brkp']))
        # groupby brkp
        binning_chisq = binning_chisq.groupby('brkp').agg({
          'variable':lambda x:np.unique(x),
          'bin': lambda x: '%,%'.join(x),
          'good': sum,
          'bad': sum
        }).assign(badprob = lambda x: x['bad']/(x['good']+x['bad']))\
        .reset_index()
        # update
        ## add chisq to new binning dataframe
        binning_chisq = add_chisq(binning_chisq)
        ## param
        bin_chisq_min = binning_chisq.chisq.min()
        bin_count_distr_min = min(binning_chisq['count']/dtm_rows)
        bin_nrow = len(binning_chisq.index)
    # format init_bin # remove (.+\\)%,%\\[.+,)
    if is_numeric_dtype(dtm['value']):
        binning_chisq = binning_chisq\
        .assign(bin = lambda x: [re.sub(r'(?<=,).+%,%.+,', '', i) if ('%,%' in i) else i for i in x['bin']])\
        .assign(brkp = lambda x: [float(re.match('^\[(.*),.+', i).group(1)) for i in x['bin']])
    # return 
    return {'binning_sv':binning_sv, 'binning':binning_chisq}

# required in woebin2 # # format binning output
def binning_format(binning):
    '''
    format binning dataframe
    
    parameters
    ------
    binning: with columns of variable, bin, good, bad
    
    Returns
    ------
    DataFrame
        binning dataframe with columns of 'variable', 'bin', 
        'count', 'count_distr', 'good', 'bad', 'badprob', 'woe', 
        'bin_iv', 'total_iv',  'breaks', 'is_special_values'
    '''
    def woe_01(good, bad):
        # woe calculation
        woe = pd.DataFrame({'good':good,'bad':bad}) \
        .replace(0, 0.9) \
        .assign(
            DistrBad = lambda x: x.bad/sum(x.bad),
            DistrGood = lambda x: x.good/sum(x.good)
        ) \
        .assign(woe = lambda x: np.log(x.DistrBad/x.DistrGood)) \
        .woe
        # return woe
        return woe

    def miv_01(good, bad):
        # iv calculation
        infovalue = pd.DataFrame({'good':good,'bad':bad}) \
        .replace(0, 0.9) \
        .assign(
            DistrBad = lambda x: x.bad/sum(x.bad),
            DistrGood = lambda x: x.good/sum(x.good)
        ) \
        .assign(iv = lambda x: (x.DistrBad-x.DistrGood)*np.log(x.DistrBad/x.DistrGood)) \
        .iv
        # return iv
        return infovalue

    binning['count'] = binning['good'] + binning['bad']
    binning['count_distr'] = binning['count']/sum(binning['count'])
    binning['badprob'] = binning['bad']/binning['count']
    # binning = binning.assign(
    #   count = lambda x: (x['good']+x['bad']),
    #   count_distr = lambda x: (x['good']+x['bad'])/sum(x['good']+x['bad']),
    #   badprob = lambda x: x['bad']/(x['good']+x['bad']))
    # new columns: woe, iv, breaks, is_sv
    binning['woe'] = woe_01(binning['good'],binning['bad'])
    binning['bin_iv'] = miv_01(binning['good'],binning['bad'])
    binning['total_iv'] = binning['bin_iv'].sum()
    # breaks
    binning['breaks'] = binning['bin']
    if any([r'[' in str(i) for i in binning['bin']]):
        def re_extract_all(x): 
            gp23 = re.match(r"^\[(.*), *(.*)\)((%,%missing)*)", x)
            breaks_string = x if gp23 is None else gp23.group(2)+gp23.group(3)
            return breaks_string
        binning['breaks'] = [re_extract_all(i) for i in binning['bin']]
    # is_sv    
    binning['is_special_values'] = binning['is_sv']
    # return
    return binning[['variable', 'bin', 'count', 'count_distr', 'good', 'bad', 'badprob', 'woe', 'bin_iv', 'total_iv',  'breaks', 'is_special_values']]


# This function provides woe binning for only two columns (one x and one y) dataframe.
def woebin2(dtm, breaks=None, spl_val=None, 
            min_perc_fine_bin=0.02, min_perc_coarse_bin=0.05, 
            stop_limit=0.1, max_num_bin=8, method="tree"):
    '''
    provides woe binning for only two series
    
    Params
    ------
    
    
    Returns
    ------
    DataFrame
        
    '''
    # binning
    if breaks is not None:
        # 1.return binning if breaks provided
        bin_list = woebin2_breaks(dtm=dtm, breaks=breaks, spl_val=spl_val)
    else:
        if stop_limit == 'N':
            # binning of initial & specialvalues
            bin_list = woebin2_init_bin(dtm, min_perc_fine_bin=min_perc_fine_bin, breaks=breaks, spl_val=spl_val)
        else:
            if method == 'tree':
                # 2.tree-like optimal binning
                bin_list = woebin2_tree(
                  dtm, min_perc_fine_bin=min_perc_fine_bin, min_perc_coarse_bin=min_perc_coarse_bin, 
                  stop_limit=stop_limit, max_num_bin=max_num_bin, breaks=breaks, spl_val=spl_val)
            elif method == "chimerge":
                # 2.chimerge optimal binning
                bin_list = woebin2_chimerge(
                  dtm, min_perc_fine_bin=min_perc_fine_bin, min_perc_coarse_bin=min_perc_coarse_bin, 
                  stop_limit=stop_limit, max_num_bin=max_num_bin, breaks=breaks, spl_val=spl_val)
    # rbind binning_sv and binning
    binning = pd.concat(bin_list, keys=bin_list.keys()).reset_index()\
              .assign(is_sv = lambda x: x.level_0 =='binning_sv')
    # return
    return binning_format(binning)


def woebin(dt, y, x=None,method="tree", breaks_list=None, special_values=None, 
           min_perc_fine_bin=0.01, min_perc_coarse_bin=0.05, 
           stop_limit=0.1, max_num_bin=5, 
           positive="bad|1", no_cores=None, print_step=0,n_jobs=-1 ):
    '''
    WOE Binning
    ------
    `woebin` generates optimal binning for numerical, factor and categorical variables 
    using methods including tree-like segmentation or chi-square merge. 
    woebin can also customizing breakpoints if the breaks_list or special_values was provided.
    The default woe is defined as ln(Distr_Bad_i/Distr_Good_i). If you 
    prefer ln(Distr_Good_i/Distr_Bad_i), please set the argument `positive` 
    as negative value, such as '0' or 'good'.
    
    Params
    ------
    dt: A data frame with both x (predictor/feature) and y (response/label) variables.
    y: Name of y variable.
    x: Name of x variables. Default is NULL. If x is NULL, 
      then all variables except y are counted as x variables.
    breaks_list: List of break points, default is NULL. 
      If it is not NULL, variable binning will based on the 
      provided breaks.  【分割点】
    special_values: the values specified in special_values 
      will be in separate bins. Default is NULL.  【单独分箱的值】
    min_perc_fine_bin: The minimum percentage of initial binning 
      class number over total. Accepted range: 0.01-0.2; default 
      is 0.02, which means initial binning into 50 fine bins for 
      continuous variables.  【初始分箱样本数】
    min_perc_coarse_bin: The minimum percentage of final binning 
      class number over total. Accepted range: 0.01-0.2; default 
      is 0.05.  【每箱最小样本数】
    stop_limit: Stop binning segmentation when information value 
      gain ratio less than the stop_limit, or stop binning merge 
      when the minimum of chi-square less than 'qchisq(1-stoplimit, 1)'. 
      Accepted range: 0-0.5; default is 0.1.  【停止分箱条件：最小信息增益比、最小卡方值】
    max_num_bin: Integer. The maximum number of binning.  【最大箱数】
    positive: Value of positive class, default "bad|1".
    no_cores: Number of CPU cores for parallel computation. 
      Defaults NULL. If no_cores is NULL, the no_cores will 
      set as 1 if length of x variables less than 10, and will 
      set as the number of all CPU cores if the length of x variables 
      greater than or equal to 10.
    print_step: A non-negative integer. Default is 1. If print_step>0, 
      print variable names by each print_step-th iteration. 
      If print_step=0 or no_cores>1, no message is print.
    method: Optimal binning method, it should be "tree" or "chimerge". 
      Default is "tree".
    
    Returns
    ------
    dictionary
        Optimal or customized binning dataframe.
    
    Examples
    ------
    import scorecardpy as sc
    import pandas as pd
    
    # load data
    dat = sc.germancredit()
    
    # Example I
    # binning of two variables in germancredit dataset
    bins_2var = sc.woebin(dat, y = "creditability", 
      x = ["credit.amount", "purpose"])
    
    # Example II
    # binning of the germancredit dataset
    bins_germ = sc.woebin(dat, y = "creditability")
    
    # Example III
    # customizing the breakpoints of binning
    dat2 = pd.DataFrame({'creditability':['good','bad']}).sample(50, replace=True)
    dat_nan = pd.concat([dat, dat2], ignore_index=True)
    
    breaks_list = {
      'age.in.years': [26, 35, 37, "Inf%,%missing"],
      'housing': ["own", "for free%,%rent"]
    }
    special_values = {
      'credit.amount': [2600, 9960, "6850%,%missing"],  
      'purpose': ["education", "others%,%missing"]
    }
    
    bins_cus_brk = sc.woebin(dat_nan, y="creditability",
      x=["age.in.years","credit.amount","housing","purpose"],
      breaks_list=breaks_list, special_values=special_values)
    '''
    # start time
    start_time = time.time()
    
    dt = dt.copy(deep=True)
    if isinstance(y, str):
        y = [y]
    if isinstance(x, str) and x is not None:
        x = [x]
    if x is not None:
        x=[i for i in x if i not in y ]
        dt = dt[y+x]
    # remove date/time col
    dt = rmcol_datetime_unique1(dt, check_char_num = True)
    # replace "" by NA
    dt = rep_blank_na(dt)
    # check y
    dt = check_y(dt, y, positive)
    # x variable names
    xs = x_variable(dt,y,x)
    xs_len = len(x)
    # print_step
    print_step = check_print_step(print_step)
    # breaks_list
    breaks_list = check_breaks_list(breaks_list, xs)
    # special_values
    special_values = check_special_values(special_values, xs)
    ### ### 
    # stop_limit range
    if stop_limit<0 or stop_limit>0.5 or not isinstance(stop_limit, (float, int)):
        warnings.warn("Incorrect parameter specification; accepted stop_limit parameter range is 0-0.5. Parameter was set to default (0.1).")
        stop_limit = 0.1
    # min_perc_fine_bin range
    if min_perc_fine_bin<0.01 or min_perc_fine_bin>0.2 or not isinstance(min_perc_fine_bin, (float, int)):
        warnings.warn("Incorrect parameter specification; accepted min_perc_fine_bin parameter range is 0.01-0.2. Parameter was set to default (0.02).")
        stop_limit = 0.02
    # min_perc_coarse_bin
    if min_perc_coarse_bin<0.01 or min_perc_coarse_bin>0.2 or not isinstance(min_perc_coarse_bin, (float, int)):
        warnings.warn("Incorrect parameter specification; accepted min_perc_coarse_bin parameter range is 0.01-0.2. Parameter was set to default (0.05).")
        min_perc_coarse_bin = 0.05
    # max_num_bin
    if not isinstance(max_num_bin, (float, int)):
        warnings.warn("Incorrect inputs; max_num_bin should be numeric variable. Parameter was set to default (5).")
        max_num_bin = 5
    # method
    if method not in ["tree", "chimerge"]:
        warnings.warn("Incorrect inputs; method should be tree or chimerge. Parameter was set to default (tree).")
        method = "tree"
    ### ### 
    # binning for each x variable
    # loop on xs
    if (no_cores is None) or (no_cores < 1):
        no_cores = 1 if xs_len<10 else mp.cpu_count() if n_jobs==-1 else n_jobs if mp.cpu_count()>=n_jobs else mp.cpu_count()
    # ylist to str
    y = y[0]
    # binning for variables
    if no_cores == 1:
        # create empty bins dict
        bins = {}
        for i in np.arange(xs_len):
            x_i = xs[i]
            # print(x_i)
            # print xs
            if print_step>0 and bool((i+1)%print_step): 
                print(('{:'+str(len(str(xs_len)))+'.0f}/{} {}').format(i, xs_len, x_i), flush=True)
            # woebining on one variable
            bins[x_i] = woebin2(
              dtm = pd.DataFrame({'y':dt[y], 'variable':x_i, 'value':dt[x_i]}),
              breaks=breaks_list[x_i] if (breaks_list is not None) and (x_i in breaks_list.keys()) else None,
              spl_val=special_values[x_i] if (special_values is not None) and (x_i in special_values.keys()) else None,
              min_perc_fine_bin=min_perc_fine_bin,
              min_perc_coarse_bin=min_perc_coarse_bin,
              stop_limit=stop_limit, 
              max_num_bin=max_num_bin,
              method=method
            )
            # try catch:
            # "The variable '{}' caused the error: '{}'".format(x_i, error-info)
    else:
        pool = mp.Pool(processes=no_cores)
        # arguments
        args = zip(
          [pd.DataFrame({'y':dt[y], 'variable':x_i, 'value':dt[x_i]}) for x_i in xs], 
          [breaks_list[i] if (breaks_list is not None) and (i in list(breaks_list.keys())) else None for i in xs],
          [special_values[i] if (special_values is not None) and (i in list(special_values.keys())) else None for i in xs],
          [min_perc_fine_bin]*xs_len, [min_perc_coarse_bin]*xs_len, 
          [stop_limit]*xs_len, [max_num_bin]*xs_len, [method]*xs_len
        )
        # bins in dictionary
        bins = dict(zip(xs, pool.starmap(woebin2, args)))
        pool.close()
    
    # runingtime
    runingtime = time.time() - start_time
    if (runingtime >= 10):
        # print(time.strftime("%H:%M:%S", time.gmtime(runingtime)))
        print('Binning on {} rows and {} columns in {}'.format(dt.shape[0], dt.shape[1], time.strftime("%H:%M:%S", time.gmtime(runingtime))))
    # return
    return bins

def woebin_transform(data,bins,x,content,change_type='merge',*args,**kwargs):
    """
    content: 
            merge :[[0,1,2],[3,4]]
            change:[26,28,29,30] 
    """
    bins_new=deepcopy(bins)
    bin_df=bins[x].reset_index()
    if bin_df.iloc[0,2][0:1]=='[':
        if change_type=='merge':
            for i in content:
                if isinstance(i,list):
                    i_min,i_max=min(i),max(i)
                else:
                    i_min,i_max=min(content),max(content)
                break_new=bin_df.loc[i_max,'breaks']
                bin_df['breaks']=bin_df.apply(lambda x :break_new if (x['index']>=i_min) & (x['index']<=i_max) else x['breaks'],axis=1 )
                break_list={x:set(bin_df['breaks'].tolist()[:-1])}
        if change_type=='change':
            break_list={x:content}
        bin=woebin(data,x=x,breaks_list=break_list,*args,**kwargs)[x]
        bin['breaks']=bin['breaks'].astype('float')
        bin=bin.sort_values(by='breaks',ascending=True).reset_index(drop=True)
    else:
        if change_type=='change':
            break_list={x:content}
            bin=woebin(data,x=x,breaks_list=break_list,*args,**kwargs)[x]
    bins_new[x]=bin
    return bins_new  
        

def  woepoints_ply1(dtx, binx, x_i, woe_points):
    '''
    Transform original values into woe or porints for one variable.
    
    Params
    ------
    
    Returns
    ------
    
    '''
    # woe_points: "woe" "points"
    # binx = bins.loc[lambda x: x.variable == x_i] 
    binx = pd.merge(
      pd.DataFrame(binx['bin'].str.split('%,%').tolist(), index=binx['bin'])\
        .stack().reset_index().drop('level_1', axis=1),
      binx[['bin', woe_points]],
      how='left', on='bin'
    ).rename(columns={0:'V1',woe_points:'V2'})
    
    # dtx
    ## cut numeric variable
    if is_numeric_dtype(dtx[x_i]):
        is_sv = pd.Series(not bool(re.search(r'\[', str(i))) for i in binx.V1)
        binx_sv = binx.loc[is_sv]
        binx_other = binx.loc[~is_sv]
        # create bin column
        breaks_binx_other = np.unique(list(map(float, ['-inf']+[re.match(r'.*\[(.*),.+\).*', str(i)).group(1) for i in binx_other['bin']]+['inf'])))
        labels = ['[{},{})'.format(breaks_binx_other[i], breaks_binx_other[i+1]) for i in range(len(breaks_binx_other)-1)]
        
        dtx = dtx.assign(xi_bin = lambda x: pd.cut(x[x_i], breaks_binx_other, right=False, labels=labels))\
          .assign(xi_bin = lambda x: [i if (i != i) else str(i) for i in x['xi_bin']])
        # dtx.loc[:,'xi_bin'] = pd.cut(dtx[x_i], breaks_binx_other, right=False, labels=labels)
        # dtx.loc[:,'xi_bin'] = np.where(pd.isnull(dtx['xi_bin']), dtx['xi_bin'], dtx['xi_bin'].astype(str))
        #
        mask = dtx[x_i].isin(binx_sv['V1'])
        dtx.loc[mask,'xi_bin'] = dtx.loc[mask, x_i].astype(str)
        dtx = dtx[['xi_bin']].rename(columns={'xi_bin':x_i})
    ## to charcarter, na to missing
    if not is_string_dtype(dtx[x_i]):
        dtx.loc[:,x_i] = dtx.loc[:,x_i].astype(str).replace('nan', 'missing')
    # dtx.loc[:,x_i] = np.where(pd.isnull(dtx[x_i]), dtx[x_i], dtx[x_i].astype(str))
    # dtx = dtx.replace(np.nan, 'missing').assign(rowid = dtx.index)
    dtx = dtx.fillna('missing').assign(rowid = dtx.index)
    # rename binx
    binx.columns = ['bin', x_i, '_'.join([x_i,woe_points])]
    # merge
    dtx_suffix = pd.merge(dtx, binx, how='left', on=x_i).sort_values('rowid')\
      .set_index(dtx.index)[['_'.join([x_i,woe_points])]]
    return dtx_suffix
    
    
def woebin_ply(dt, bins, no_cores=None, print_step=0):
    '''
    WOE Transformation
    ------
    `woebin_ply` converts original input data into woe values 
    based on the binning information generated from `woebin`.
    
    Params
    ------
    dt: A data frame.
    bins: Binning information generated from `woebin`.
    no_cores: Number of CPU cores for parallel computation. 
      Defaults NULL. If no_cores is NULL, the no_cores will 
      set as 1 if length of x variables less than 10, and will 
      set as the number of all CPU cores if the length of x 
      variables greater than or equal to 10.
    print_step: A non-negative integer. Default is 1. If 
      print_step>0, print variable names by each print_step-th 
      iteration. If print_step=0 or no_cores>1, no message is print.
    
    Returns
    -------
    DataFrame
        a dataframe of woe values for each variables 
    
    Examples
    -------
    import scorecardpy as sc
    import pandas as pd
    
    # load data
    dat = sc.germancredit()
    
    # Example I
    dt = dat[["creditability", "credit.amount", "purpose"]]
    # binning for dt
    bins = sc.woebin(dt, y = "creditability")
    
    # converting original value to woe
    dt_woe = sc.woebin_ply(dt, bins=bins)
    
    # Example II
    # binning for germancredit dataset
    bins_germancredit = sc.woebin(dat, y="creditability")
    
    # converting the values in germancredit to woe
    ## bins is a dict
    germancredit_woe = sc.woebin_ply(dat, bins=bins_germancredit) 
    ## bins is a dataframe
    germancredit_woe = sc.woebin_ply(dat, bins=pd.concat(bins_germancredit))
    '''
    # start time
    start_time = time.time()
    # remove date/time col
    dt = rmcol_datetime_unique1(dt)
    # replace "" by NA
    dt = rep_blank_na(dt)
    # ncol of dt
    # if len(dt.index) <= 1: raise Exception("Incorrect inputs; dt should have at least two columns.")
    # print_step
    print_step = check_print_step(print_step)
    
    # bins # if (is.list(bins)) rbindlist(bins)
    if isinstance(bins, dict):
        bins = pd.concat(bins, ignore_index=True)
    # x variables
    xs_bin = bins['variable'].unique()
    xs_dt = list(dt.columns)
    xs = list(set(xs_bin).intersection(xs_dt))
    # length of x variables
    xs_len = len(xs)
    # initial data set
    dat = dt.loc[:,list(set(xs_dt) - set(xs))]
    
    # loop on xs
    if (no_cores is None) or (no_cores < 1):
        no_cores = 1 if xs_len<10 else mp.cpu_count()
    # 
    if no_cores == 1:
        for i in np.arange(xs_len):
            x_i = xs[i]
            # print xs
            # print(x_i)
            if print_step>0 and bool((i+1) % print_step): 
                print(('{:'+str(len(str(xs_len)))+'.0f}/{} {}').format(i, xs_len, x_i), flush=True)
            #
            binx = bins[bins['variable'] == x_i].reset_index()
                 # bins.loc[lambda x: x.variable == x_i] 
                 # bins.loc[bins['variable'] == x_i] # 
                 # bins.query('variable == \'{}\''.format(x_i))
            dtx = dt[[x_i]]
            dat = pd.concat([dat, woepoints_ply1(dtx, binx, x_i, woe_points="woe")], axis=1)
    else:
        pool = mp.Pool(processes=no_cores)
        # arguments
        args = zip(
          [dt[[i]] for i in xs], 
          [bins[bins['variable'] == i] for i in xs], 
          [i for i in xs], 
          ["woe"]*xs_len
        )
        # bins in dictionary
        dat_suffix = pool.starmap(woepoints_ply1, args)
        dat = pd.concat([dat]+dat_suffix, axis=1)
        pool.close()
    # runingtime
    runingtime = time.time() - start_time
    if (runingtime >= 10):
        # print(time.strftime("%H:%M:%S", time.gmtime(runingtime)))
        print('Woe transformating on {} rows and {} columns in {}'.format(dt.shape[0], xs_len, time.strftime("%H:%M:%S", time.gmtime(runingtime))))
    return dat

#===============================================================#
#-----------------------score_card&plt--------------------------#
#===============================================================#



# coefficients in scorecard
def ab(points0=600, odds0=1/60, pdo=50):
    # sigmoid function
    # library(ggplot2)
    # ggplot(data.frame(x = c(-5, 5)), aes(x)) + stat_function(fun = function(x) 1/(1+exp(-x)))
  
    # log_odds function
    # ggplot(data.frame(x = c(0, 1)), aes(x)) + stat_function(fun = function(x) log(x/(1-x)))
  
    # logistic function
    # p(y=1) = 1/(1+exp(-z)),
        # z = beta0+beta1*x1+...+betar*xr = beta*x
    ##==> z = log(p/(1-p)),
        # odds = p/(1-p) # bad/good <==>
        # p = odds/1+odds
    ##==> z = log(odds)
    ##==> score = a - b*log(odds)
  
    # two hypothesis
    # points0 = a - b*log(odds0)
    # points0 - PDO = a - b*log(2*odds0)
    if pdo > 0:
        b = pdo/np.log(2)
    else:
        b = -pdo/np.log(2)
    a = points0 + b*np.log(odds0) #log(odds0/(1+odds0))
    
    return {'a':a, 'b':b}


def scorecard(bins, model, xcolumns, points0=600, odds0=1/19, pdo=50, basepoints_eq0=False):
    '''
    Creating a Scorecard
    ------
    `scorecard` creates a scorecard based on the results from `woebin` 
    and LogisticRegression of sklearn.linear_model
    
    Params
    ------
    bins: Binning information generated from `woebin` function.
    model: A LogisticRegression model object.
    points0: Target points, default 600.
    odds0: Target odds, default 1/19. Odds = p/(1-p).
    pdo: Points to Double the Odds, default 50.
    basepoints_eq0: Logical, default is FALSE. If it is TRUE, the 
      basepoints will equally distribute to each variable.
    
    Returns
    ------
    DataFrame
        scorecard dataframe
    
    Examples
    ------
    import scorecardpy as sc
    
    # load data
    dat = sc.germancredit()
    
    # filter variable via missing rate, iv, identical value rate
    dt_sel = sc.var_filter(dat, "creditability")
    
    # woe binning ------
    bins = sc.woebin(dt_sel, "creditability")
    dt_woe = sc.woebin_ply(dt_sel, bins)
    
    y = dt_woe.loc[:,'creditability']
    X = dt_woe.loc[:,dt_woe.columns != 'creditability']
    
    # logistic regression ------
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(penalty='l1', C=0.9, solver='saga')
    lr.fit(X, y)
    
    # # predicted proability
    # dt_pred = lr.predict_proba(X)[:,1]
    # # performace
    # # ks & roc plot
    # sc.perf_eva(y, dt_pred)
    
    # scorecard
    # Example I # creat a scorecard
    card = sc.scorecard(bins, lr, X.columns)
    
    # credit score
    # Example I # only total score
    score1 = sc.scorecard_ply(dt_sel, card)
    # Example II # credit score for both total and each variable
    score2 = sc.scorecard_ply(dt_sel, card, only_total_score = False)
    '''
    
    # coefficients
    aabb = ab(points0, odds0, pdo)
    a = aabb['a'] 
    b = aabb['b']
    # odds = pred/(1-pred); score = a - b*log(odds)
    
    # bins # if (is.list(bins)) rbindlist(bins)
    if isinstance(bins, dict):
        bins = pd.concat(bins, ignore_index=True)
    xs = [re.sub('_woe$', '', i) for i in xcolumns]
    # coefficients
    coef_df = pd.Series(model.coef_[0], index=np.array(xs))\
      .loc[lambda x: x != 0]#.reset_index(drop=True)
    
    # scorecard
    len_x = len(coef_df)
    basepoints = a - b*model.intercept_[0]
    card = {}
    if basepoints_eq0:
        card['basepoints'] = pd.DataFrame({'variable':"basepoints", 'bin':np.nan, 'points':0}, index=np.arange(1))
        for i in coef_df.index:
            card[i] = bins.loc[bins['variable']==i,['variable', 'bin', 'woe']]\
              .assign(points = lambda x: round(-b*x['woe']*coef_df[i] + basepoints/len_x))\
              [["variable", "bin", "points"]]
    else:
        card['basepoints'] = pd.DataFrame({'variable':"basepoints", 'bin':np.nan, 'points':round(basepoints)}, index=np.arange(1))
        for i in coef_df.index:
            card[i] = bins.loc[bins['variable']==i,['variable', 'bin', 'woe']]\
              .assign(points = lambda x: round(-b*x['woe']*coef_df[i]))\
              [["variable", "bin", "points"]]
    return card



def scorecard_ply(dt, card,only_total_score=True, print_step=0,label=False):
    '''
    Score Transformation
    ------
    Params
    ------
    dt: Original data
    card: Scorecard generated from `scorecard`. dict 
    only_total_score: Logical, default is TRUE. If it is TRUE, then 
      the output includes only total credit score; Otherwise, if it 
      is FALSE, the output includes both total and each variable's 
      credit score.
    print_step: A non-negative integer. Default is 1. If print_step>0, 
      print variable names by each print_step-th iteration. If 
      print_step=0, no message is print.
    
    Return
    ------
    DataFrame
        Credit score
    '''
    assert isinstance(card,dict),"card param should be dict type  " 
    if label:
        xs = [x for x in card.keys() if x!= 'basepoints']
        for i in xs:
            dt[i]=dt[label].map(lambda x :x [i])
            dt=dt[xs]
    dt = dt.copy(deep=True)
    # replace "" by NA
    dt = rep_blank_na(dt)
    # print_step
    print_step = check_print_step(print_step)
    # card # if (is.list(card)) rbindlist(card)
    if isinstance(card, dict):
        card_df = pd.concat(card, ignore_index=True)

    xs = card_df.loc[card_df.variable != 'basepoints', 'variable'].unique()


    xs_len = len(xs)

    dat = dt.loc[:,list(set(dt.columns)-set(xs))]

    for i in np.arange(xs_len):
        x_i = xs[i]
        if print_step>0 : 
            print(('{:'+str(len(str(xs_len)))+'.0f}/{} {}').format(i, xs_len, x_i))
        
        cardx = card_df.loc[card_df['variable']==x_i]
        dtx = dt[[x_i]]
        # score transformation
        dtx_points = woepoints_ply1(dtx, cardx, x_i, woe_points="points")
        dat = pd.concat([dat, dtx_points], axis=1)
    
    # set basepoints
    card_basepoints = list(card_df.loc[card_df['variable']=='basepoints','points'])[0] if 'basepoints' in card_df['variable'].unique() else 0
    dat_score = dat[xs+'_points']
    dat_score = pd.concat([dat_score,pd.DataFrame(card_basepoints + dat_score.sum(axis=1),columns=['score'])],axis=1)
    if only_total_score: dat_score = dat_score[['score']]
    return dat_score



def eva_dfkslift(df, groupnum=None):
    if groupnum is None: groupnum=len(df.index)
    def n0(x): return sum(x==0)
    def n1(x): return sum(x==1)
    df_kslift = df.sort_values('pred', ascending=False).reset_index(drop=True)\
      .assign(group=lambda x: np.ceil((x.index+1)/(len(x.index)/groupnum)))\
      .groupby('group')['label'].agg([n0,n1])\
      .reset_index().rename(columns={'n0':'good','n1':'bad'})\
      .assign(
        group=lambda x: (x.index+1)/len(x.index),
        good_distri=lambda x: x.good/sum(x.good), 
        bad_distri=lambda x: x.bad/sum(x.bad), 
        badrate=lambda x: x.bad/(x.good+x.bad),
        cumbadrate=lambda x: np.cumsum(x.bad)/np.cumsum(x.good+x.bad),
        lift=lambda x: (np.cumsum(x.bad)/np.cumsum(x.good+x.bad))/(sum(x.bad)/sum(x.good+x.bad)),
        cumgood=lambda x: np.cumsum(x.good)/sum(x.good), 
        cumbad=lambda x: np.cumsum(x.bad)/sum(x.bad)
      ).assign(ks=lambda x:abs(x.cumbad-x.cumgood))
    # bind 0
    df_kslift=pd.concat([
      pd.DataFrame({'group':0, 'good':0, 'bad':0, 'good_distri':0, 'bad_distri':0, 'badrate':0, 'cumbadrate':np.nan, 'cumgood':0, 'cumbad':0, 'ks':0, 'lift':np.nan}, index=np.arange(1)),
      df_kslift
    ], ignore_index=True)
    # return
    return df_kslift
   
def eva_pks(dfkslift, title):
    dfks = dfkslift.loc[lambda x: x.ks==max(x.ks)].sort_values('group').iloc[0]
    ###### plot ###### 
    # fig, ax = plt.subplots()
    # ks, cumbad, cumgood
    plt.plot(dfkslift.group, dfkslift.ks, 'b-', 
      dfkslift.group, dfkslift.cumgood, 'k-', 
      dfkslift.group, dfkslift.cumbad, 'k-')
    # ks vline
    plt.plot([dfks['group'], dfks['group']], [0, dfks['ks']], 'r--')
    # set xylabel
    plt.gca().set(title=title+'K-S', 
      xlabel='% of population', ylabel='% of total Good/Bad', 
      xlim=[0,1], ylim=[0,1], aspect='equal')
    # text
    # plt.text(0.5,0.96,'K-S', fontsize=15,horizontalalignment='center')
    plt.text(0.2,0.8,'Bad',horizontalalignment='center')
    plt.text(0.8,0.55,'Good',horizontalalignment='center')
    plt.text(dfks['group'], dfks['ks'], 'KS:'+ str(round(dfks['ks'],4)), horizontalalignment='center',color='b')
    # plt.grid()
    # plt.show()
    # return fig

def eva_plift(dfkslift, title):
    badrate_avg = sum(dfkslift.bad)/sum(dfkslift.good+dfkslift.bad)
    ###### plot ###### 
    # fig, ax = plt.subplots()
    # ks, cumbad, cumgood
    plt.plot(dfkslift.group, dfkslift.cumbadrate, 'k-')
    # ks vline
    plt.plot([0, 1], [badrate_avg, badrate_avg], 'r--')
    # set xylabel
    plt.gca().set(title=title+'Lift', 
      xlabel='% of population', ylabel='% of Bad', 
      xlim=[0,1], ylim=[0,1], aspect='equal')
    # text
    # plt.text(0.5,0.96,'Lift', fontsize=15,horizontalalignment='center')
    plt.text(0.7,np.mean(dfkslift.cumbadrate),'cumulate badrate',horizontalalignment='center')
    plt.text(0.7,badrate_avg,'average badrate',horizontalalignment='center')
    # plt.grid()
    # plt.show()
    # return fig

def eva_dfrocpr(df):
    def n0(x): return sum(x==0)
    def n1(x): return sum(x==1)
    dfrocpr = df.sort_values('pred')\
      .groupby('pred')['label'].agg([n0,n1,len])\
      .reset_index().rename(columns={'n0':'countN','n1':'countP','len':'countpred'})\
      .assign(
        FN = lambda x: np.cumsum(x.countP), 
        TN = lambda x: np.cumsum(x.countN) 
      ).assign(
        TP = lambda x: sum(x.countP) - x.FN, 
        FP = lambda x: sum(x.countN) - x.TN
      ).assign(
        TPR = lambda x: x.TP/(x.TP+x.FN), 
        FPR = lambda x: x.FP/(x.TN+x.FP), 
        precision = lambda x: x.TP/(x.TP+x.FP), 
        recall = lambda x: x.TP/(x.TP+x.FN)
      ).assign(
        F1 = lambda x: 2*x.precision*x.recall/(x.precision+x.recall)
      )
    return dfrocpr

def eva_proc(dfrocpr, title):
    dfrocpr = pd.concat(
      [dfrocpr[['FPR','TPR']], pd.DataFrame({'FPR':[0,1], 'TPR':[0,1]})], 
      ignore_index=True).sort_values(['FPR','TPR'])
    auc = dfrocpr.sort_values(['FPR','TPR'])\
          .assign(
            TPR_lag=lambda x: x['TPR'].shift(1), FPR_lag=lambda x: x['FPR'].shift(1)
          ).assign(
            auc=lambda x: (x.TPR+x.TPR_lag)*(x.FPR-x.FPR_lag)/2
          )['auc'].sum()
    ###### plot ###### 
    # fig, ax = plt.subplots()
    # ks, cumbad, cumgood
    plt.plot(dfrocpr.FPR, dfrocpr.TPR, 'k-')
    # ks vline
    x=np.array(np.arange(0,1.1,0.1))
    plt.plot(x, x, 'r--')
    # fill 
    plt.fill_between(dfrocpr.FPR, 0, dfrocpr.TPR, color='blue', alpha=0.1)
    # set xylabel
    plt.gca().set(title=title+'ROC',
      xlabel='FPR', ylabel='TPR', 
      xlim=[0,1], ylim=[0,1], aspect='equal')
    # text
    # plt.text(0.5,0.96, 'ROC', fontsize=15, horizontalalignment='center')
    plt.text(0.55,0.45, 'AUC:'+str(round(auc,4)), horizontalalignment='center', color='b')
    # plt.grid()
    # plt.show()
    # return fig

def eva_ppr(dfrocpr, title):
    ###### plot ###### 
    # fig, ax = plt.subplots()
    # ks, cumbad, cumgood
    plt.plot(dfrocpr.recall, dfrocpr.precision, 'k-')
    # ks vline
    x=np.array(np.arange(0,1.1,0.1))
    plt.plot(x, x, 'r--')
    # set xylabel
    plt.gca().set(title=title+'P-R', 
      xlabel='Recall', ylabel='Precision', 
      xlim=[0,1], ylim=[0,1], aspect='equal')
    # text
    # plt.text(0.5,0.96, 'P-R', fontsize=15, horizontalalignment='center')
    # plt.grid()
    # plt.show()
    # return fig

def eva_pf1(dfrocpr, title):
    dfrocpr=dfrocpr.assign(pop=lambda x: np.cumsum(x.countpred)/sum(x.countpred))
    ###### plot ###### 
    # fig, ax = plt.subplots()
    # ks, cumbad, cumgood
    plt.plot(dfrocpr['pop'], dfrocpr['F1'], 'k-')
    # ks vline
    F1max_pop = dfrocpr.loc[dfrocpr['F1'].idxmax(),'pop']
    F1max_F1 = dfrocpr.loc[dfrocpr['F1'].idxmax(),'F1']
    plt.plot([F1max_pop,F1max_pop], [0,F1max_F1], 'r--')
    # set xylabel
    plt.gca().set(title=title+'F1', 
      xlabel='% of population', ylabel='F1', 
      xlim=[0,1], ylim=[0,1], aspect='equal')
    # pred text
    pred_0=dfrocpr.loc[dfrocpr['pred'].idxmin(),'pred']
    pred_F1max=dfrocpr.loc[dfrocpr['F1'].idxmax(),'pred']
    pred_1=dfrocpr.loc[dfrocpr['pred'].idxmax(),'pred']
    if np.mean(dfrocpr.pred) < 0 or np.mean(dfrocpr.pred) > 1: 
        pred_0 = -pred_0
        pred_F1max = -pred_F1max
        pred_1 = -pred_1
    plt.text(0, 0, 'pred \n'+str(round(pred_0,4)), horizontalalignment='left',color='b')
    plt.text(F1max_pop, 0, 'pred \n'+str(round(pred_F1max,4)), horizontalalignment='center',color='b')
    plt.text(1, 0, 'pred \n'+str(round(pred_1,4)), horizontalalignment='right',color='b')
    # title F1
    plt.text(F1max_pop, F1max_F1, 'F1 max: \n'+ str(round(F1max_F1,4)), horizontalalignment='center',color='b')
    # plt.grid()
    # plt.show()
    # return fig


def perf_eva(label, pred, title=None, groupnum=None, plot_type=["ks", "roc"], show_plot=True, positive="bad|1", seed=186):
    '''
    KS, ROC, Lift, PR
    ------
    perf_eva provides performance evaluations, such as 
    kolmogorov-smirnow(ks), ROC, lift and precision-recall curves, 
    based on provided label and predicted probability values.
    
    Params
    ------
    label: Label values, such as 0s and 1s, 0 represent for good 
      and 1 for bad.
    pred: Predicted probability or score.
    title: Title of plot, default is "performance".
    groupnum: The group number when calculating KS.  Default NULL, 
      which means the number of sample size.
    plot_type: Types of performance plot, such as "ks", "lift", "roc", "pr". 
      Default c("ks", "roc").
    show_plot: Logical value, default is TRUE. It means whether to show plot.
    positive: Value of positive class, default is "bad|1".
    seed: Integer, default is 186. The specify seed is used for random sorting data.
    
    Returns
    ------
    dict
        ks, auc, gini values, and figure objects
    
    Details
    ------
    Accuracy = 
        true positive and true negative/total cases
    Error rate = 
        false positive and false negative/total cases
    TPR, True Positive Rate(Recall or Sensitivity) = 
        true positive/total actual positive
    PPV, Positive Predicted Value(Precision) = 
        true positive/total predicted positive
    TNR, True Negative Rate(Specificity) = 
        true negative/total actual negative
    NPV, Negative Predicted Value = 
        true negative/total predicted negative
    '''  
    # inputs checking
    if len(label) != len(pred):
        warnings.warn('Incorrect inputs; label and pred should be list with the same length.')
    # if pred is score
    if np.mean(pred) < 0 or np.mean(pred) > 1:
        warnings.warn('Since the average of pred is not in [0,1], it is treated as predicted score but not probability.')
        pred = -pred
    # random sort datatable
    df = pd.DataFrame({'label':label, 'pred':pred}).sample(frac=1, random_state=seed)
    # remove NAs
    if any(np.unique(df.isnull())):
        warnings.warn('The NANs in \'label\' or \'pred\' were removed.')
        df = df.dropna()
    # check label
    df = check_y(df, 'label', positive)
    # title
    title='' if title is None else str(title)+': '
    
    ### data ###
    # dfkslift ------
    if any([i in plot_type for i in ['ks', 'lift']]):
        dfkslift = eva_dfkslift(df, groupnum)
        if 'ks' in plot_type: df_ks = dfkslift
        if 'lift' in plot_type: df_lift = dfkslift
    # dfrocpr ------
    if any([i in plot_type for i in ["roc","pr",'f1']]):
        dfrocpr = eva_dfrocpr(df)
        if 'roc' in plot_type: df_roc = dfrocpr
        if 'pr' in plot_type: df_pr = dfrocpr
        if 'f1' in plot_type: df_f1 = dfrocpr
    ### return list ### 
    rt = {}
    # plot, KS ------
    if 'ks' in plot_type:
        rt['KS'] = round(dfkslift.loc[lambda x: x.ks==max(x.ks),'ks'].iloc[0],4)
    # plot, ROC ------
    if 'roc' in plot_type:
        auc = pd.concat(
          [dfrocpr[['FPR','TPR']], pd.DataFrame({'FPR':[0,1], 'TPR':[0,1]})], 
          ignore_index=True).sort_values(['FPR','TPR'])\
          .assign(
            TPR_lag=lambda x: x['TPR'].shift(1), FPR_lag=lambda x: x['FPR'].shift(1)
          ).assign(
            auc=lambda x: (x.TPR+x.TPR_lag)*(x.FPR-x.FPR_lag)/2
          )['auc'].sum()
        ### 
        rt['AUC'] = round(auc, 4)
        rt['Gini'] = round(2*auc-1, 4)
    
    if show_plot:
        plist = ["eva_p"+i+'(df_'+i+',title)' for i in plot_type]
        subplot_nrows = np.ceil(len(plist)/2)
        subplot_ncols = np.ceil(len(plist)/subplot_nrows)
        
        fig = plt.figure()
        for i in np.arange(len(plist)):
            plt.subplot(subplot_nrows,subplot_ncols,i+1)
            eval(plist[i])
        plt.show()
        rt['pic'] = fig
    return rt
    

def perf_psi(score, label=None, title=None, x_limits=None, x_tick_break=50,y_max=0.2,show_plot=True, seed=186, return_distr_dat=False):
    '''
    PSI
    ------
    perf_psi calculates population stability index (PSI) and provides 
    credit score distribution based on credit score datasets.
    
    Params
    ------
    score: A list of credit score for actual and expected data samples. 
      For example, score = list(actual = score_A, expect = score_E), both 
      score_A and score_E are dataframes with the same column names.
    label: A list of label value for actual and expected data samples. 
      The default is NULL. For example, label = list(actual = label_A, 
      expect = label_E), both label_A and label_E are vectors or 
      dataframes. The label values should be 0s and 1s, 0 represent for 
      good and 1 for bad.
    title: Title of plot, default is NULL.
    y_max： max_y_lim default 0.2
    x_limits: x-axis limits, default is None.
    x_tick_break: x-axis ticker break, default is 50.
    show_plot: Logical, default is TRUE. It means whether to show plot.
    return_distr_dat: Logical, default is FALSE.
    seed: Integer, default is 186. The specify seed is used for random 
      sorting data.
    
    Returns
    ------
    The population stability index (PSI) formula is displayed below: 
    \deqn{PSI = \sum((Actual\% - Expected\%)*(\ln(\frac{Actual\%}{Expected\%}))).} 
    The rule of thumb for the PSI is as follows: Less than 0.1 inference 
    insignificant change, no action required; 0.1 - 0.25 inference some 
    minor change, check other scorecard monitoring metrics; Greater than 
    0.25 inference major shift in population, need to delve deeper.
    
    # Example I # psi
    psi1 = sc.perf_psi(
      score = {'train':train_score, 'test':test_score},
      label = {'train':y_train, 'test':y_test},
      x_limits = [250, 750],
      x_tick_break = 50
    )

    '''
    
    # inputs checking
    ## score
    if not isinstance(score, dict) and len(score) != 2:
        raise Exception("Incorrect inputs; score should be a dictionary with two elements.")
    else:
        if any([not isinstance(i, pd.DataFrame) for i in score.values()]):
            raise Exception("Incorrect inputs; score is a dictionary of two dataframes.")
        score_columns = [list(i.columns) for i in score.values()]
        if set(score_columns[0]) != set(score_columns[1]):
            raise Exception("Incorrect inputs; the column names of two dataframes in score should be the same.")
    ## label
    if label is not None:
        if not isinstance(label, dict) and len(label) != 2:
            raise Exception("Incorrect inputs; label should be a dictionary with two elements.")
        else:
            if set(score.keys()) != set(label.keys()):
                raise Exception("Incorrect inputs; the keys of score and label should be the same. ")
            for i in label.keys():
                if isinstance(label[i], pd.DataFrame):
                    if len(label[i].columns) == 1:
                        label[i] = label[i].iloc[:,0]
                    else:
                        raise Exception("Incorrect inputs; the number of columns in label should be 1.")
    # score dataframe column names
    score_names = score[list(score.keys())[0]].columns
    # merge label with score
    for i in score.keys():
        score[i] = score[i].copy(deep=True)
        if label is not None:
            score[i].loc[:,'y'] = label[i]
        else:
            score[i].copy(deep=True).loc[:,'y'] = np.nan
    # dateset of score and label
    dt_sl = pd.concat(score, names=['ae', 'rowid']).reset_index()\
      .sample(frac=1, random_state=seed)
      # ae refers to 'Actual & Expected'
    
    # PSI function
    def psi(dat):
        dt_bae = dat.groupby(['ae','bin']).size().reset_index(name='N')\
          .pivot_table(values='N', index='bin', columns='ae').fillna(0.9)\
          .agg(lambda x: x/sum(x))
        dt_bae.columns = ['A','E']
        psi_dt = dt_bae.assign(
          AE = lambda x: x.A-x.E,
          logAE = lambda x: np.log(x.A/x.E)
        ).assign(
          bin_PSI=lambda x: x.AE*x.logAE
        )['bin_PSI'].sum()
        return psi_dt
    
    # return psi and pic
    rt_psi = {}
    rt_pic = {}
    rt_dat = {}
    rt = {}
    for sn in score_names:
        # dataframe with columns of ae y sn
        dat = dt_sl[['ae', 'y', sn]]
        if len(dt_sl[sn].unique()) >=10:
            # breakpoints
            if x_limits is None:
                x_limits = dat[sn].quantile([0.02, 0.98])
                x_limits = round(x_limits/x_tick_break)*x_tick_break
                x_limits = list(x_limits)
        
            brkp = np.unique([np.floor(min(dt_sl[sn])/x_tick_break)*x_tick_break]+\
              list(np.arange(x_limits[0], x_limits[1], x_tick_break))+\
              [np.ceil(max(dt_sl[sn])/x_tick_break)*x_tick_break])
            # cut
            labels = ['[{},{})'.format(int(brkp[i]), int(brkp[i+1])) for i in range(len(brkp)-1)]
            dat.loc[:,'bin'] = pd.cut(dat[sn], brkp, right=False, labels=labels)
        else:
            dat.loc[:,'bin'] = dat[sn]
        # psi ------
        rt_psi[sn] = pd.DataFrame({'PSI':psi(dat)},index=np.arange(1)) 
    
        # distribution of scorecard probability
        def good(x): return sum(x==0)
        def bad(x): return sum(x==1)
        distr_prob = dat.groupby(['ae', 'bin'])\
          ['y'].agg([good, bad])\
          .assign(N=lambda x: x.good+x.bad,
            badprob=lambda x: x.bad/(x.good+x.bad)
          ).reset_index()
        distr_prob.loc[:,'distr'] = distr_prob.groupby('ae')['N'].transform(lambda x:x/sum(x))
        # pivot table
        distr_prob = distr_prob.pivot_table(values=['N','badprob', 'distr'], index='bin', columns='ae')
            
        # plot ------
        if show_plot:
            ###### param ######
            ind = np.arange(len(distr_prob.index))    # the x locations for the groups
            width = 0.35       # the width of the bars: can also be len(x) sequence
            ###### plot ###### 
            fig, ax1 = plt.subplots(figsize=(12,6))
            ax2 = ax1.twinx()
            title_string = sn+'_PSI: '+str(round(psi(dat),4))
            title_string = title_string if title is None else str(title)+' '+title_string
            # ax1
            p1 = ax1.bar(ind, distr_prob['distr'].iloc[:,0], width, color=(24/254, 192/254, 196/254), alpha=0.6)
            p2 = ax1.bar(ind+width, distr_prob['distr'].iloc[:,1], width, color=(246/254, 115/254, 109/254), alpha=0.6)
            # ax2
            p3 = ax2.plot(ind+width/2, distr_prob['badprob'].iloc[:,0], color=(24/254, 192/254, 196/254))
            ax2.scatter(ind+width/2, distr_prob['badprob'].iloc[:,0], facecolors='w', edgecolors=(24/254, 192/254, 196/254))
            p4 = ax2.plot(ind+width/2, distr_prob['badprob'].iloc[:,1], color=(246/254, 115/254, 109/254))
            ax2.scatter(ind+width/2, distr_prob['badprob'].iloc[:,1], facecolors='w', edgecolors=(246/254, 115/254, 109/254))
            # settings
            ax1.set_ylabel('Score distribution')
            ax2.set_ylabel('Bad probability')#, color='blue')
            # ax2.tick_params(axis='y', colors='blue')
            # ax1.set_yticks(np.arange(0, np.nanmax(distr_prob['distr'].values), 0.2))
            # ax2.set_yticks(np.arange(0, 1+0.2, 0.2))
            ax1.set_ylim([0,np.ceil(np.nanmax(distr_prob['distr'].values)*10)/10])
            ax2.set_ylim([0,y_max])
            plt.xticks(ind+width/2, distr_prob.index)
            plt.title(title_string, loc='left')
            ax1.legend((p1[0], p2[0]), list(distr_prob.columns.levels[1]), loc='upper left')
            ax2.legend((p3[0], p4[0]), list(distr_prob.columns.levels[1]), loc='upper right')
            # show plot 
            plt.show()
            
            # return of pic
            rt_pic[sn] = fig
        
        # return distr_dat ------
        if return_distr_dat:
            aa=distr_prob[['N','badprob']].reset_index()
            rt_dat[sn] = aa
            aa=pd.concat(  [aa,aa['N']*aa['badprob']],axis=1)
            aa.columns=['bin','cnt_test','cnt_train','bad_rate_test','bad_rate_train','bad_tset','bad_train']
            tmp1,tmp2=aa[['bin','cnt_train','bad_train','bad_rate_train']],aa[['bin','cnt_test','bad_tset','bad_rate_test']]
            tmp_list=[tmp1,tmp2]
            for i,tmp in enumerate( tmp_list):
                tmp.columns=['bin','cnt','bad','bad_rate']
                tmp['good']=tmp['cnt']-tmp['bad']
                tmp['good_rate']=tmp['good']/tmp['cnt']
                tmp['count_distr']=tmp['cnt']/tmp['cnt'].sum()
                tmp['cum_bad_rate']=tmp['bad'].cumsum(axis=0)/tmp['bad'].sum()
                tmp['cum_good_rate']=tmp['good'].cumsum(axis=0)/tmp['good'].sum()
                tmp['ks']=abs(tmp['cum_bad_rate']-tmp['cum_good_rate'])
                all_df=pd.DataFrame(['合计',tmp['cnt'].sum(),tmp['bad'].sum()
                                     ,1,tmp['good'].sum(),1,1,1,tmp['ks'].max()],index=tmp.columns.tolist()).T
                tmp=pd.concat([tmp,all_df],axis=0)
                tmp_list[i]=tmp
            rt_dat['df_c'] = pd.concat([tmp_list[0],tmp_list[1]],axis=1)
            rt_dat['df_r'] =  pd.concat([tmp_list[0],tmp_list[1]],axis=0)
    rt['psi'] = pd.concat(rt_psi).reset_index().rename(columns={'level_0':'variable'})[['variable', 'PSI']]
    rt['pic'] = rt_pic
    if return_distr_dat: rt['dat'] = rt_dat
    return rt
    
    
    
import numpy as np 
import pandas as pd 

def model_score_bins(dt,x,y='Y',pdo=20,xlim=[540,740],bin_type='cut',margins=True):
	"""
	dt,x,label：dataframe coontains socre(x) and label(y)
	xlim: min_score and max_score,if likes [a,b] means a,a+ceil((b-a)//pdo)  
		  if margins it contains inf and total summary
	bin_type： if 'cut' use cut , 'qcut' means qcut 


	"""
	df=dt.copy()
	if bin_type=='cut':
		if isinstance(xlim,list):
			x_min=xlim[0]
			x_max=xlim[0]+np.ceil((xlim[1]-xlim[0])//pdo+1)*pdo
		cut_list=[-np.inf]+list(np.arange(x_min,x_max,pdo))+[np.inf]
		df['cuts']=pd.cut(df[x],cut_list)
	elif bin_type=='qcut':
		q_nodes=10
		cut_list=np.linspace(0,1,q_nodes+1)
		df['cuts']=pd.qcut(df[x],cut_list)
	else:
		raise Exception("Incorrect params : bin_type ( cut or quct) ")
	df[y]=df[y].astype('float')
	df=df.groupby('cuts')[y].agg([('total','count')
	                                    ,('good',lambda y :(y==0).sum())
	                                    ,('bad',lambda y :(y==1).sum())
	                                  ])
	df['bad_rate']=df['bad']/df['total']
	df['count_rate']=df['total']/df['total'].sum()
	df['count_distr']=df['total'].cumsum()/df['total'].sum()
	df['cum_bad_rate']=df['bad'].cumsum()/df['bad'].sum()
	df['cum_good_rate']=df['good'].cumsum()/df['good'].sum()
	df['ks']=abs(df['cum_bad_rate']-df['cum_good_rate'])
	df['lift']=df['cum_bad_rate']/df['count_distr']
	all_x=lambda i :df[i].max()  if i  in ['ks','lift'] else 1 if i  not in ['total','good','bad'] else df[i].sum() 
	all_row=pd.DataFrame.from_dict({i:all_x(i) for i in df.columns  },orient='index').T
	all_row.index=['ALL']
	df=df.reset_index().rename({'cuts':'index'},axis=1)
	df=aa=pd.concat([df,all_row.reset_index()],axis=0,ignore_index=True).set_index('index')
	return df 
    
    
def auc_get(dt,x,y):
    y_true=dt[y].tolist()
    y_pred=dt[x].tolist()
    df=pd.DataFrame({'y_true':y_true,'y_pred':y_pred}).sort_values(by='y_pred',ascending=True)
    df=df.replace(np.nan,0)
    #正样本
    p_num=np.sum(df.y_true)
    n_num=len(df.y_true)-p_num
    wrong_count=0
    rest_p_count=p_num
    socre_labels=df.y_true
    for i in socre_labels:
        if i==1:
            rest_p_count-=1
        else:
            wrong_count+=rest_p_count
    auc=1-wrong_count/(p_num*n_num)
    return auc
