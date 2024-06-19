import pandas as pd
import numpy as np
import statsmodels.api as sm

# ----------------------------------- Utility functions -----------------------------------
# SEQUENCE
def sequence(n):
    return list(range(1, n+1))

# REGBETA
# get linear regression beta
def regbeta(y, X):
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model.params[1]

# Rolling OLS
def rolling_ols(x, window):
    if len(x) >= window:
        model = sm.OLS(sequence(window), x).fit()
        return model.params[0]
    else:
        return np.nan

# Rolling OLS std
def guotai_rolling_ols_std(X, y):
    try:
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        return np.std(model.resid)
    except:
        return np.nan, np.nan

# Rolling OLS for guotai
def guotai_rolling_ols(X, y):
    try:
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        return model.params[0], model.params[1]
    except:
        return np.nan, np.nan

# REGRESI
# get linear regression beta
def regresi(y, X):
    model = sm.OLS(y, X).fit()
    return model.resid.tolist()[-1]

# rolling with fix window
def rolling_window_fixedSize(a, window):
    '''
    Return a rolling window matrix
            Parameters:
                    a: a list or one dimentional array
                    window: fixed window size
            Returns:
                    two dimentional ndarray(length of a, window)
    '''
    a = a.to_numpy()
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

# TSRANK
def tsRank(df, window):
    data = df.copy()
    arr = rolling_window_fixedSize(data.T, window)
    order_arr = np.argsort(np.argsort(arr, axis=2), axis=2)
    for idx, pid in enumerate(data.columns):
        arr_b = arr[idx]
        order_b = (order_arr[idx] + 1).astype(float)
        for i in range(arr_b.shape[0]):
            n_idx = list(np.where(arr_b[i] == arr_b[i][-1])[0])
            if len(n_idx) > 1:
                mean_order = 0
                for j in n_idx:
                    mean_order += order_b[i][j]
                order_b[i][-1] = float(mean_order / len(n_idx))
        b = order_b[:, -1] / window
        b = [v if np.isnan(arr_b[i]).sum() == 0 else np.nan for i, v in enumerate(b)]
        b = [np.nan for i in range(window-1)] + b
        data[pid] = b
    return data

# DECAYLINEAR
def decayLinear(df, weight, window):
    data = df.copy()
    arr = rolling_window_fixedSize(data.T, window)
    for idx, pid in enumerate(data.columns):
        arr_b = arr[idx]
        b = []
        for i in arr_b:
            b.append(np.dot(i, weight))
        b = [np.nan for i in range(window-1)] + b
        data[pid] = b
    return data

# WMA
def wma(df, weight, window):
    data = df.copy()
    arr = rolling_window_fixedSize(data.T, window)
    for idx, pid in enumerate(data.columns):
        arr_b = arr[idx]
        b = []
        for i in arr_b:
            b.append(np.dot(i, weight) / np.sum(weight))
        b = [np.nan for i in range(window-1)] + b
        data[pid] = b
    return data

# ----------------------------------------- Alpha -----------------------------------------
# 基本面因子
def alphaPE(dataset):
    data = dataset.df.copy()
    alpha = data['PE'].rank(axis=1, pct=True)
    return alpha
#Rank_IC.mean, Rank_IC.std, IC_IR,   IC>0.%
#-0.0261,      0.0718,      0.3637,  0.349


def alphaPB(dataset):
    data = dataset.df.copy()
    alpha = data['PB'].rank(axis=1, pct=True)
    return alpha

# 價量因子
# Alpha001 : (-1*CORR(RANK(DELTA(LOG(VOLUME),1)),RANK(((CLOSE-OPEN)/OPEN)),6)
# RANK is the cross-sectional ranking of each stocks
def alpha001(dataset, window=6):
    data = dataset.df.copy()
    rank_sizenl = np.log(data['Volume']).diff(1).rank(axis=1, pct=True)
    rank_ret = ((data['Close'] - data['Open']) / data['Open']).rank(axis=1, pct=True)
    rel = rank_sizenl.rolling(window).corr(rank_ret)*(-1)
    return rel

# Alpha002 : -1*delta(((close-low)-(high-close))/(high-low),1)
def alpha002(dataset, window=1):
    data = dataset.df.copy()
    win_ratio = (2*data['Close']-data['Low']-data['High'])/(data['High']-data['Low'])
    return win_ratio.diff(window) * (-1)

# Alpha003 : -1*SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)
def alpha003(dataset, window=6):
    data = dataset.df.copy()
    alpha = data['Close'].diff(1)
    delay_close = data['Close'].shift(1)
    alpha[alpha > 0] = data['Close'][alpha > 0] - np.minimum(delay_close[alpha > 0], data['Low'][alpha > 0])
    alpha[alpha < 0] = data['Close'][alpha < 0] - np.minimum(delay_close[alpha < 0], data['High'][alpha < 0])
    return alpha.rolling(window).sum() * -1

# Alpha004 :
# (((SUM(CLOSE,8)/8)+STD(CLOSE,8))<(SUM(CLOSE,2)/2))
#     ?-1:(SUM(CLOSE,2)/2<(SUM(CLOSE,8)/8-STD(CLOSE,8))
#         ?1:(1<=(VOLUME/MEAN(VOLUME,20))
#             ?1:-1))
def alpha004(dataset):
    data = dataset.df.copy()
    alpha = data['Volume'].rolling(20).mean()
    condition1 = data['Close'].rolling(8).mean() + data['Close'].rolling(8).std() < data['Close'].rolling(2).mean()
    condition2 = data['Close'].rolling(2).mean() < data['Close'].rolling(8).mean() - data['Close'].rolling(8).std()
    condition3 = 1 <= data['Volume'] / data['Volume'].rolling(20).mean()
    alpha[condition1 & (alpha.isna() == False)] = -1
    alpha[(condition1 == False) & (condition2) & (alpha.isna() == False)] = 1
    alpha[(condition1 == False) & (condition2 == False) & condition3 & (alpha.isna() == False)] = 1
    alpha[(condition1 == False) & (condition2 == False) & (condition3 == False) & (alpha.isna() == False)] = -1
    return alpha
    
# Alpha005 : # -1*TSMAX(CORR(TSRANK(VOLUME,5),TSRANK(HIGH,5),5),3)
# Note: may have Nan, since rolling corr may has some series are constant over time.
# def alpha005(df):
#     data = df.copy()
#     alpha = data['Volume'].rolling(5).apply(lambda x: tsrank(x)/5).rolling(5).corr(data['High'].rolling(5).apply(lambda x: tsrank(x)/5)).rolling(3).max() * -1
#     return preprocess(alpha)

def alpha005(dataset, window1=5, window2=3):
    data = dataset.df.copy()
    ts_volume = tsRank(data['Volume'], window1)
    ts_high = tsRank(data['High'], window1)
    corr_ts = ts_volume.rolling(window1).corr(ts_high)
    return corr_ts.rolling(window2).max() * (-1)

# Alpha006 : -1*RANK(SIGN(DELTA(OPEN*0.85+HIGH*0.15,4)))
# Note: a little bit strange, why rank the sign ? (1, 0 ,-1)
def alpha006(dataset, window=4):
    data = dataset.df.copy()
    alpha = (data['Open'] * 0.85 + data['High'] *0.15).diff(window)
    alpha[alpha > 0] = 1
    alpha[alpha == 0] = 0
    alpha[alpha < 0] = -1
    alpha = alpha.rank(axis=1, pct=True)
    return -1 * alpha

# Alpha007 : (RANK(MAX(VWAP-CLOSE,3))+RANK(MIN(VWAP-CLOSE,3)))*RANK(DELTA(VOLUME,3))
def alpha007(dataset, window=3):
    data = dataset.df.copy()
    A = (data['VWAP'] - data['Close']).rolling(window).max().rank(axis=1, pct=True) 
    B =  (data['VWAP'] - data['Close']).rolling(window).min().rank(axis=1, pct=True) * data['Volume'].diff(window).rank(axis=1, pct=True)
    return A + B

# Alpha008 : RANK( DELTA((HIGH+LOW)/10+VWAP*0.8, 4) * -1 )
def alpha008(dataset, window=4):
    data = dataset.df.copy()
    alpha = ((data['High'] + data['Low']) / 10) + data['VWAP'] * 0.8
    return (alpha.diff(window) * -1).rank(axis=1, pct=True)

# Alpha009 : SMA( ((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2) * (HIGH-LOW)/VOLUME,7,2)
# Note: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html
def alpha009(dataset, window1=2, window2=7):
    data = dataset.df.copy()
    A = (data['High'] + data['Low']) / 2
    B = (data['High'].shift(1) + data['Low'].shift(1)) / 2
    C = (data['High'] - data['Low']) / data['Volume']
    return ((A - B) * C).ewm(adjust=False, alpha=window1/window2, min_periods=0, ignore_na=False).mean() # SMA(A, 7, 2)

# Alpha010 : RANK( MAX(((RET<0)?STD(RET,20):CLOSE)^2, 5) )
# Note: std^2 全部都比 5 小，因此被替換成 5，很奇怪
def alpha010(dataset, window=20):
    data = dataset.df.copy()
    ret = data['Close'].pct_change(1)
    part1 = ret.rolling(window).std()
    condition = (ret >= 0) & (part1.isna() == False)
    part1[condition] = data['Close'][condition]
    alpha = (part1 ** 2)
    alpha[alpha < 5] = 5
    return alpha.rank(axis=1, pct=True)

# Alpha011 : SUM(((CLOSE-LOW)-(HIGH-CLOSE))./(HIGH-LOW).*VOLUME,6)
def alpha011(dataset, window=6):
    data = dataset.df.copy()
    alpha = (2 * data['Close'] - data['Low'] - data['High']) / (data['High'] - data['Low']) * data['Volume']
    return alpha.rolling(window).sum()

# Alpha012 : (RANK((OPEN-(SUM(VWAP,10)/10)))) * (-1 * (RANK(ABS((CLOSE-VWAP)))))
def alpha012(dataset, window=10):
    data = dataset.df.copy()
    A = (data['Open'] - (data['VWAP'].rolling(window).sum() / window)).rank(axis=1, pct=True)
    B = (data['Close'] - data['VWAP']).abs().rank(axis=1, pct=True) * -1
    return A * B

# Alpha013 : (((HIGH*LOW)^0.5)-VWAP)
def alpha013(dataset):
    data = dataset.df.copy()
    alpha = (data['High'] * data['Low']) ** 0.5 - data['VWAP']
    return alpha

# Alpha014 : CLOSE-DELAY(CLOSE,5)
def alpha014(dataset, window=5):
    data = dataset.df.copy()
    alpha = data['Close'].diff(window)
    return alpha

# Alpha015 : OPEN/DELAY(CLOSE,1) - 1
def alpha015(dataset, window=1):
    data = dataset.df.copy()
    alpha = data['Open']/data['Close'].shift(window) - 1
    return alpha

# Alpha016 : -1 * TSMAX( RANK( CORR(RANK(VOLUME), RANK(VWAP), 5) ), 5)
# Note: may have Nan, since rolling corr may has some series are constant over time.
def alpha016(dataset, window1=5, window2=5):
    data = dataset.df.copy()
    alpha = data['Volume'].rank(axis=1, pct=True).rolling(window1).corr(data['VWAP'].rank(axis=1, pct=True))
    return -1 * alpha.rolling(window2).max()

# Alpha017 : RANK((VWAP-MAX(VWAP,15)))^DELTA(CLOSE,5) -> i think it should change to "TSMAX(VWAP,15)"
# too much inf value -> drop this alpha
def alpha017(dataset, window1=15, window2=5):
    data = dataset.df.copy()
    alpha = (data['VWAP'] - data['VWAP'].rolling(window1).max()).rank(axis=1, pct=True)
    delta = data['Close'].diff(window2)
    return alpha ** delta

# Alpha018 : CLOSE/DELAY(CLOSE,5)
def alpha018(dataset, window=5):
    data = dataset.df.copy()
    return data['Close'] / data['Close'].shift(window)

# Alpha019 : 
# (CLOSE<DELAY(CLOSE,5)
#     ?(CLOSE-DELAY(CLOSE,5))/DELAY(CLOSE,5): (CLOSE=DELAY(CLOSE,5)
#         ?0: (CLOSE-DELAY(CLOSE,5))/CLOSE))
def alpha019(dataset, window=5):
    data = dataset.df.copy()
    condition1 = (data['Close'] <= data['Close'].shift(window)) & (data['Close'].shift(window).isna() == False)
    alpha = data['Close'].shift(window)
    alpha[condition1] = (data['Close'].diff(window) / data['Close'].shift(window))[condition1]
    condition2 = (data['Close'] > data['Close'].shift(window)) & (data['Close'].shift(window).isna() == False) & (data['Close'] == data['Close'].shift(window))
    alpha[condition2] = 0
    condition3 = (data['Close'] > data['Close'].shift(window)) & (data['Close'].shift(window).isna() == False) & (data['Close'] != data['Close'].shift(window))
    alpha[condition3] = (data['Close'].diff(window) / data['Close'])[condition3]
    return alpha

# Alpha020 : (CLOSE/DELAY(CLOSE,6)-1) * 100
def alpha020(dataset, window=6):
    data = dataset.df.copy()
    return data['Close'].pct_change(window) * 100

# Alpha021 : REGBETA(MEAN(CLOSE,6),SEQUENCE(6))
# Note: adding intercept; using for loop, slow computing; 為何 y 是 1,2,3,4,5,6 ?
def alpha021(dataset, window=6):
    data = dataset.df.copy()
    df_X = data['Close'].rolling(window).mean()
    df_beta = df_X.copy()
    for col in df_X.columns:
        temp1 = df_X[col]
        df_beta[col] = temp1.rolling(window).apply(lambda x: rolling_ols(x, window))
    return df_beta

# Alpha022 : SMEAN( ((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6) - DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6), 3)) ,12, 1)
# Note: SMEAN may be SMA
def alpha022(dataset, window1=6, window2=3, window3=12):
    data = dataset.df.copy()
    A = (data['Close'] - data['Close'].rolling(window1).mean()) / data['Close'].rolling(window1).mean()
    alpha = A.diff(window2)
    return alpha.ewm(adjust=False, alpha=1/window3, min_periods=0, ignore_na=False).mean()

# Alpha023 : 
# SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1) / 
# ( SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1) + 
# SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1) )*100
def alpha023(dataset, window=20):
    data = dataset.df.copy()
    condition1 = (data['Close'] > data['Close'].shift(1)) & (data['Close'].shift(1).isna() == False)
    condition2 = (data['Close'] <= data['Close'].shift(1)) & (data['Close'].shift(1).isna() == False)
    A = data['Close'].shift(1)
    B = A.copy()
    A[condition1] = data['Close'].rolling(window).std()[condition1]
    A[condition2] = 0
    B[condition1] = 0
    A[condition2] = data['Close'].rolling(window).std()[condition2]
    A = A.ewm(adjust=False, alpha=1/window, ignore_na=False).mean()
    B = B.ewm(adjust=False, alpha=1/window, ignore_na=False).mean()
    return (A / (A + B)) * 100

# Alpha024 : SMA(CLOSE-DELAY(CLOSE,5),5,1)
def alpha024(dataset, window=5):
    data = dataset.df.copy()
    alpha = data['Close'].diff(window)
    return alpha.ewm(adjust=False, alpha=1/window, ignore_na=False).mean()

# Alpha025 : (-1 * RANK((DELTA(CLOSE, 7) * (1 - RANK(DECAYLINEAR((VOLUME/MEAN(VOLUME, 20)), 9)))))) * (1 + RANK(SUM(RET, 250)))
def alpha025(dataset, window1=9, window2=7, window3=20, window4=250):
    data = dataset.df.copy()
    w = np.array(range(1, window1+1)) / np.sum(range(1, window1+1))
    A = -1 * (data['Close'].diff(window2).rank(axis=1, pct=True))
    #B = 1 - (data['Volume'] / data['Volume'].rolling(window3).mean()).rolling(window1).apply(lambda x: np.dot(x, w)).rank(axis=1, pct=True)
    B = (data['Volume'] / data['Volume'].rolling(window3).mean())
    B = 1 - decayLinear(B, w, window1).rank(axis=1, pct=True)
    C = 1 + data['Close'].pct_change(1).rolling(window4).sum().rank(axis=1, pct=True)
    return A * B * C

# Alpha026 : (SUM(CLOSE,7)/7)-CLOSE + CORR(VWAP,DELAY(CLOSE,5),230)
def alpha026(dataset, window1=7, window2=230):
    data = dataset.df.copy()
    A = data['Close'].rolling(window1).mean() - data['Close']
    B = data['VWAP'].rolling(window2).corr(data['Close'].shift(5))
    return A + B

# Alpha027 : WMA( (CLOSE-DELAY(CLOSE,3))/DELAY(CLOSE,3)*100 + (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100, 12)
# Note: 不確定 WMA 的算法是否正確; * 100 是對誰乘
def alpha027(dataset, window=12):
    data = dataset.df.copy()
    A = (data['Close'].diff(3) / data['Close'].shift(3)) * 100
    B = (data['Close'].diff(6) / data['Close'].shift(6)) * 100
    w = np.array([0.9**i for i in range(window-1, -1, -1)])
    #return (A + B).rolling(window).apply(lambda x: np.dot(x, w) / np.sum(w))
    return wma(A + B, w, window)

# Alpha028 :
# 3 * SMA( (CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100, 3, 1)
# -2 * SMA( SMA( (CLOSE-TSMIN(LOW,9))/( TSMAX(HIGH,9)-TSMIN(LOW,9))*100, 3, 1), 3, 1)
# Note: 不確定 * 100 是對誰乘
def alpha028(dataset, window1=9, window2=3):
    data = dataset.df.copy()
    A = (data['Close'] - data['Low'].rolling(window1).min()) / (data['High'].rolling(window1).max() - data['Low'].rolling(window1).min()) * 100
    B = A.ewm(adjust=False, alpha=1/window2, ignore_na=False).mean()
    return 3 * B - 2 * B.ewm(adjust=False, alpha=1/window2, ignore_na=False).mean()

# Alpha029 : (CLOSE-DELAY(CLOSE,6)) / DELAY(CLOSE,6) * VOLUME
def alpha029(dataset, window=6):
    data = dataset.df.copy()
    return data['Close'].pct_change(window) * data['Volume']

# # Alpha030 : WMA( (REGRESI(RET,MKT,SMB,HML,60))^2, 20)
# # Note: mktvalue, PB 資料來源待確定 / rolling regression to get residual?
# def alpha030(df):
#     data = df.copy()
#     ret = data['Close'].pct_change()
#     df_res = ret.copy().dropna()
#     # generate SMB and HML
#     me30 = (data['MktValue'].T <= data['MktValue'].quantile(0.3, axis=1)).T
#     me70 = (data['MktValue'].T >= data['MktValue'].quantile(0.7, axis=1)).T
#     pb30 = (data['PB'].T <= data['PB'].quantile(0.3, axis=1)).T
#     pb70 = (data['PB'].T >= data['PB'].quantile(0.7, axis=1)).T
#     smb_ret = ret[me30].mean(axis=1, skipna=True) - ret[me70].mean(axis=1, skipna=True)
#     hml_ret = ret[pb70].mean(axis=1, skipna=True) - ret[pb30].mean(axis=1, skipna=True)
#     fama_french = pd.concat([mkt_ret, smb_ret, hml_ret], axis= 1)
#     fama_french.columns = ['MKT', 'SMB', 'HML']
#     fama_french['Constant'] = 1
#     # run fama french for each individual stock
#     for pid in ret.columns:
#         res_list = []
#         fama_french['R'] = ret[pid]
#         fama_french = fama_french.dropna()
#         for i in range(len(fama_french)):
#             if i >= 59:
#                 temp = fama_french.iloc[i - 59: i+1].reset_index(drop=True)
#                 res_list.append(regresi(temp['R'], temp[['MKT', 'SMB', 'HML', 'Constant']]) ** 2) # REGRESI will return t = T's residual
#             else:
#                 res_list.append(np.nan)
#         # save residual
#         df_res[pid] = res_list
#     # WMA(rsidual ** 2, 20)
#     w = np.array([0.9**i for i in range(19, -1, -1)])
#     #return df_res.rolling(20).apply(lambda x: np.dot(x, w) / np.sum(w))
#     return wma(df_res, w, 20)

# Alpha031 : (CLOSE-MEAN(CLOSE,12)) / MEAN(CLOSE,12) * 100
def alpha031(dataset, window=12):
    data = dataset.df.copy()
    return ((data['Close'] - data['Close'].rolling(window).mean()) / data['Close'].rolling(window).mean()) * 100

# Alpha032 : -1 * SUM( RANK( CORR( RANK(HIGH), RANK(VOLUME), 3) ), 3)
# Note: rolling corr may leads to multiple NaNs
def alpha032(dataset, window=3):
    data = dataset.df.copy()
    return data['High'].rank(axis=1, pct=True).rolling(window).corr(data['Volume'].rank(axis=1, pct=True)).rank(axis=1, pct=True).rolling(window).sum() * -1

# Alpha033 : (-1 * TSMIN(LOW,5)+DELAY(TSMIN(LOW,5),5)) * RANK( (SUM(RET,240)-SUM(RET,20)) / 220 ) * TSRANK(VOLUME,5)
def alpha033(dataset, window1=5, window2=240, window3=20):
    data = dataset.df.copy()
    A = data['Low'].rolling(window1).min().shift(window1) - data['Low'].rolling(window1).min()
    B = ((data['Close'].pct_change().rolling(window2).sum() - data['Close'].pct_change().rolling(window3).sum()) / (window2 - window3)).rank(axis=1, pct=True)
    # C = data['Volume'].rolling(window1).apply(lambda x: stats.rankdata(x)[-1] / window1)
    C = tsRank(data['Volume'], window1)
    return A * B * C

# Alpha034 : MEAN(CLOSE,12) / CLOSE
def alpha034(dataset, window=12):
    data = dataset.df.copy()
    return data['Close'].rolling(window).mean() / data['Close']

# Alpha035 : MIN( RANK( DECAYLINEAR( DELTA(OPEN, 1), 15) ), RANK(DECAYLINEAR( CORR(VOLUME, OPEN*0.65 + CLOSE*0.35, 17) , 7))) * -1
def alpha035(dataset, window1=15, window2=17, window3=7):
    data = dataset.df.copy()
    w1 = np.array(range(1, window1+1)) / np.sum(range(1, window1+1))
    #A = data['Open'].diff(1).rolling(window1).apply(lambda x: np.dot(x, w1)).rank(axis=1, pct=True)
    A = data['Open'].diff(1)
    A = decayLinear(A, w1, window1).rank(axis=1, pct=True)
    w2 = np.array(range(1, window3+1)) / np.sum(range(1, window3+1))
    #B = data['Volume'].rolling(window2).corr(data['Open'] * 0.65 + data['Close'] * 0.35).rolling(window3).apply(lambda x: np.dot(x, w2)).rank(axis=1, pct=True)
    B = data['Volume'].rolling(window2).corr(data['Open'] * 0.65 + data['Close'] * 0.35)
    B = decayLinear(B, w2, window3).rank(axis=1, pct=True)
    return pd.concat([A, B]).min(level=0) * -1

# Alpha036 : RANK( SUM( CORR(RANK(VOLUME), RANK(VWAP), 6), 2) )
# Note: rolling corr may leads to multiple NaNs
def alpha036(dataset, window1=6, window2=2):
    data = dataset.df.copy()
    return data['Volume'].rank(axis=1, pct=True).rolling(window1).corr(data['VWAP'].rank(axis=1, pct=True)).rolling(window2).sum().rank(axis=1, pct=True)

# Alpha037 : -1 * RANK( (SUM(OPEN,5) * SUM(RET,5)) - DELAY( (SUM(OPEN,5) * SUM(RET,5)), 10) )
def alpha037(dataset, window1=5, window2=10):
    data = dataset.df.copy()
    A = data['Open'].rolling(window1).sum() * data['Close'].pct_change().rolling(window1).sum()
    return -1 * (A - A.shift(window2)).rank(axis=1, pct=True)

#Alpha038 : 
# (SUM(HIGH,20)/20) < HIGH)
#     ?(-1 * DELTA(HIGH, 2)) :0)
def alpha038(dataset, window=20):
    data = dataset.df.copy()
    alpha = data['High'].rolling(window).mean()
    condition1 = data['High'].rolling(window).mean() < data['High']
    condition2 = (data['High'].rolling(window).mean() >= data['High']) & (data['High'].rolling(window).mean().isna() == False)
    alpha[condition1] = (data['High'].diff(2) * -1)[condition1]
    alpha[condition2] = 0
    return alpha

# Alpha039 : ( RANK( DECAYLINEAR( DELTA(CLOSE, 2), 8) ) - RANK( DECAYLINEAR( CORR( VWAP*0.3+OPEN*0.7, SUM( MEAN( VOLUME, 180), 37) , 14), 12)) ) * -1
def alpha039(dataset, window1=8, window2=12, window3=14, window4=180, window5=37):
    data = dataset.df.copy()
    w1 = np.array(range(1, window1+1)) / np.sum(range(1, window1+1))
    #A = data['Close'].diff(2).rolling(window1).apply(lambda x: np.dot(x, w1)).rank(axis=1, pct=True)
    A = data['Close'].diff(2)
    A = wma(A, w1, window1).rank(axis=1, pct=True)
    w2 = np.array(range(1, window2+1)) / np.sum(range(1, window2+1))
    #B = (data['VWAP'] * 0.3 + data['Open'] * 0.7).rolling(window3).corr(data['Volume'].rolling(window4).mean().rolling(window5).sum()).rolling(window2).apply(lambda x: np.dot(x, w2)).rank(axis=1, pct=True)
    B = (data['VWAP'] * 0.3 + data['Open'] * 0.7).rolling(window3).corr(data['Volume'].rolling(window4).mean().rolling(window5).sum())
    B = decayLinear(B, w2, window2).rank(axis=1, pct=True)
    return (A - B) * -1

# Alpha040 : 
# SUM( (CLOSE>DELAY(CLOSE,1)? VOLUME: 0), 26) / SUM( (CLOSE<=DELAY(CLOSE,1)? VOLUME: 0), 26) * 100
def alpha040(dataset, window=26):
    data = dataset.df.copy()
    A = data['Close'].diff(1).copy()
    B = data['Close'].diff(1).copy()
    condition1 = (data['Close'].diff(1) > 0) & (data['Close'].diff(1).isna() == False)
    condition2 = (data['Close'].diff(1) <= 0) & (data['Close'].diff(1).isna() == False)
    A[condition1] = data['Volume'][condition1]
    A[condition2] = 0
    B[condition2] = data['Volume'][condition2]
    B[condition1] = 0
    return (A.rolling(window).sum() / B.rolling(window).sum()) * 100

# Alpha041 : RANK( MAX( DELTA(VWAP, 3) , 5) ) * -1
# Note: MAX should be TSMAX ?
def alpha041(dataset, window=5):
    data = dataset.df.copy()
    return -1 * (data['VWAP'].diff(3).rolling(window).max().rank(axis=1, pct=True))

# Alpha042 : -1 *  RANK( STD(HIGH, 10)) *  CORR(HIGH, VOLUME, 10)
def alpha042(dataset, window=10):
    data = dataset.df.copy()
    A = data['High'].rolling(window).std().rank(axis=1, pct=True)
    B = data['High'].rolling(window).corr(data['Volume'])
    return  -1 * A * B

# Alpha043 : 
# SUM( 
# (CLOSE > DELAY(CLOSE,1)
# ?VOLUME:(CLOSE < DELAY(CLOSE,1)
# ?-VOLUME:0)), 
# 6)
def alpha043(dataset, window=6):
    data = dataset.df.copy()
    condition1 = data['Close'] > data['Close'].shift(1)
    condition2 = data['Close'] == data['Close'].shift(1)
    condition3 = data['Close'] < data['Close'].shift(1)
    alpha = data['Close'].shift(1).copy()
    alpha[condition1] = data['Volume'][condition1]
    alpha[condition2] = 0
    alpha[condition3] = - data['Volume'][condition3]
    return alpha.rolling(window).sum()

# Alpha044 : TSRANK( DECAYLINEAR( CORR(LOW, MEAN(VOLUME, 10), 7), 6), 4) + TSRANK( DECAYLINEAR( DELTA(VWAP, 3) , 10), 15)
def alpha044(dataset, window1=6, window2=7, window3=10, window4=4, window5=10, window6=3, window7=15):
    data = dataset.df.copy()
    w1 = np.array(range(1, window1+1)) / np.sum(range(1, window1+1))
    #A = data['Low'].rolling(window2).corr(data['Volume'].rolling(window3).mean()).rolling(window1).apply(lambda x: np.dot(x, w1)).rolling(window4).apply(lambda x: stats.rankdata(x)[-1] / window4)
    #A = data['Low'].rolling(window2).corr(data['Volume'].rolling(window3).mean()).rolling(window1).apply(lambda x: np.dot(x, w1))
    A = data['Low'].rolling(window2).corr(data['Volume'].rolling(window3).mean())
    A = decayLinear(A, w1, window1)
    A.replace([np.inf, -np.inf], np.nan, inplace=True)
    A = tsRank(A, window4)
    w2 = np.array(range(1, window5+1)) / np.sum(range(1, window5+1))
    #B = data['VWAP'].diff(window6).rolling(window5).apply(lambda x: np.dot(x, w2)).rolling(window7).apply(lambda x: stats.rankdata(x)[-1] / window7)
    #B = data['VWAP'].diff(window6).rolling(window5).apply(lambda x: np.dot(x, w2))
    B = decayLinear(data['VWAP'].diff(window6), w2, window5)
    B.replace([np.inf, -np.inf], np.nan, inplace=True)
    B = tsRank(B, window7)
    return A + B

# Alpha045 : RANK( DELTA( (CLOSE * 0.6)+(OPEN * 0.4) , 1) ) * RANK( CORR(VWAP, MEAN(VOLUME, 150), 15) )
def alpha045(dataset, window1=15, window2=150):
    data = dataset.df.copy()
    A = (data['Close'] * 0.6 + data['Open'] * 0.4).diff(1).rank(axis=1, pct=True)
    B = data['VWAP'].rolling(window1).corr(data['Volume'].rolling(window2).mean()).rank(axis=1, pct=True)
    return A * B

# Alpha046 : ( MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24) ) / (4*CLOSE)
def alpha046(dataset, window1=3, window2=6, window3=12, window4=24):
    data = dataset.df.copy()
    A = data['Close'].rolling(window1).mean() + data['Close'].rolling(window2).mean() + data['Close'].rolling(window3).mean() + data['Close'].rolling(window4).mean()
    B = 4 * data['Close']
    return A / B

# Alpha047 : SMA( (TSMAX(HIGH,6)-CLOSE) / ( TSMAX(HIGH,6) - TSMIN(LOW,6) ) * 100, 9, 1)
def alpha047(dataset, window1=6, window2=9):
    data = dataset.df.copy()
    A = data['High'].rolling(window1).max() - data['Close']
    B = data['High'].rolling(window1).max() - data['Low'].rolling(window1).min()
    alpha = (A / B) * 100
    return alpha.ewm(adjust=False, alpha=1/window2, min_periods=0, ignore_na=False).mean()

# Alpha048 : (-1 * (( RANK((( SIGN((CLOSE-DELAY(CLOSE,1))) + SIGN((DELAY(CLOSE,1)-DELAY(CLOSE,2)))) + SIGN((DELAY(CLOSE,2)-DELAY(CLOSE,3)))))) * SUM(VOLUME,5))/SUM(VOLUME,20))
def alpha048(dataset, window1=5, window2=20):
    data = dataset.df.copy()
    A = data['Close'].diff(1)
    B = A.shift(1)
    C = A.shift(2)
    A[A > 0] = 1
    A[A < 0 ] = -1
    B[B > 0] = 1
    B[B < 0 ] = -1
    C[C > 0] = 1
    C[C < 0 ] = -1
    part1 = (A + B + C).rank(axis=1, pct=True)
    part2 = data['Volume'].rolling(window1).sum() / data['Volume'].rolling(window2).sum()
    return -1 * (part1 * part2)

# Alpha049 : 
# SUM( (HIGH+LOW) >= (DELAY(HIGH,1)+DELAY(LOW,1)) ? 0: MAX( ABS(HIGH-DELAY(HIGH,1)), ABS(LOW-DELAY(LOW,1)) ), 12) /
# ( SUM( (HIGH+LOW) >= (DELAY(HIGH,1)+DELAY(LOW,1)) ? 0: MAX( ABS(HIGH-DELAY(HIGH,1)), ABS(LOW-DELAY(LOW,1)) ), 12) +
# SUM( (HIGH+LOW) <= (DELAY(HIGH,1)+DELAY(LOW,1)) ? 0: MAX( ABS(HIGH-DELAY(HIGH,1)), ABS(LOW-DELAY(LOW,1)) ), 12) )
def alpha049(dataset, window=12):
    data = dataset.df.copy()
    condition1 = (data['High'] + data['Low']) >= (data['High'].shift(1) + data['Low'].shift(1))
    condition2 = (data['High'] + data['Low']) < (data['High'].shift(1) + data['Low'].shift(1))
    A = data['High'].diff(1).abs()
    B = data['Low'].diff(1).abs()
    A[A < B] = B[B > A]
    part1 = data['High'].shift(1).copy()
    part1[condition1] = 0
    part1[condition2] = A[condition2]
    part1 = part1.rolling(window).sum()
    
    condition3 = (data['High'] + data['Low']) <= (data['High'].shift(1) + data['Low'].shift(1))
    condition4 = (data['High'] + data['Low']) > (data['High'].shift(1) + data['Low'].shift(1))
    part2 = data['High'].shift(1).copy()
    part2[condition3] = 0
    part2[condition4] = A[condition4]
    part2 = part2.rolling(window).sum()
    return part1 / (part1 + part2)

# Alpha050
# SUM( (HIGH+LOW) <= (DELAY(HIGH,1)+DELAY(LOW,1)) ? 0: MAX( ABS(HIGH-DELAY(HIGH,1)), ABS(LOW-DELAY(LOW,1))), 12)/
# ( SUM( (HIGH+LOW) <= (DELAY(HIGH,1)+DELAY(LOW,1)) ? 0: MAX( ABS(HIGH-DELAY(HIGH,1)), ABS(LOW-DELAY(LOW,1))), 12) +
# SUM( (HIGH+LOW) >= (DELAY(HIGH,1)+DELAY(LOW,1)) ? 0: MAX( ABS(HIGH-DELAY(HIGH,1)), ABS(LOW-DELAY(LOW,1))), 12) )

# - SUM( (HIGH+LOW) >= (DELAY(HIGH,1)+DELAY(LOW,1)) ? 0: MAX( ABS(HIGH-DELAY(HIGH,1)), ABS(LOW-DELAY(LOW,1))), 12)/
# ( SUM( (HIGH+LOW) >= (DELAY(HIGH,1)+DELAY(LOW,1)) ? 0: MAX( ABS(HIGH-DELAY(HIGH,1)), ABS(LOW-DELAY(LOW,1))), 12) +
# SUM( (HIGH+LOW) <= (DELAY(HIGH,1)+DELAY(LOW,1)) ? 0: MAX( ABS(HIGH-DELAY(HIGH,1)), ABS(LOW-DELAY(LOW,1))), 12) )
def alpha050(dataset, window=12):
    data = dataset.df.copy()
    condition1 = (data['High'] + data['Low']) >= (data['High'].shift(1) + data['Low'].shift(1))
    condition2 = (data['High'] + data['Low']) < (data['High'].shift(1) + data['Low'].shift(1))
    A = data['High'].diff(1).abs()
    B = data['Low'].diff(1).abs()
    A[A < B] = B[B > A]
    part1 = data['High'].shift(1).copy()
    part1[condition1] = 0
    part1[condition2] = A[condition2]
    part1 = part1.rolling(window).sum()
    
    condition3 = (data['High'] + data['Low']) <= (data['High'].shift(1) + data['Low'].shift(1))
    condition4 = (data['High'] + data['Low']) > (data['High'].shift(1) + data['Low'].shift(1))
    part2 = data['High'].shift(1).copy()
    part2[condition3] = 0
    part2[condition4] = A[condition4]
    part2 = part2.rolling(window).sum()

    return (part2 / (part1 + part2)) - (part1 / (part1 + part2))

# Alpha051
# SUM( (HIGH+LOW) <= (DELAY(HIGH,1)+DELAY(LOW,1)) ? 0: MAX( ABS(HIGH-DELAY(HIGH,1)), ABS(LOW-DELAY(LOW,1)) ), 12) /
# ( SUM( (HIGH+LOW) >= (DELAY(HIGH,1)+DELAY(LOW,1)) ? 0: MAX( ABS(HIGH-DELAY(HIGH,1)), ABS(LOW-DELAY(LOW,1)) ), 12) +
# SUM( (HIGH+LOW) <= (DELAY(HIGH,1)+DELAY(LOW,1)) ? 0: MAX( ABS(HIGH-DELAY(HIGH,1)), ABS(LOW-DELAY(LOW,1)) ), 12) )
def alpha051(dataset, window=12):
    data = dataset.df.copy()
    condition1 = (data['High'] + data['Low']) >= (data['High'].shift(1) + data['Low'].shift(1))
    condition2 = (data['High'] + data['Low']) < (data['High'].shift(1) + data['Low'].shift(1))
    A = data['High'].diff(1).abs()
    B = data['Low'].diff(1).abs()
    A[A < B] = B[B > A]
    part1 = data['High'].shift(1).copy()
    part1[condition1] = 0
    part1[condition2] = A[condition2]
    part1 = part1.rolling(window).sum()
    
    condition3 = (data['High'] + data['Low']) <= (data['High'].shift(1) + data['Low'].shift(1))
    condition4 = (data['High'] + data['Low']) > (data['High'].shift(1) + data['Low'].shift(1))
    part2 = data['High'].shift(1).copy()
    part2[condition3] = 0
    part2[condition4] = A[condition4]
    part2 = part2.rolling(window).sum()
    return part2 / (part1 + part2)

# Alpha052 : SUM( MAX(0, HIGH-DELAY((HIGH+LOW+CLOSE)/3,1)), 26) / SUM( MAX(0, DELAY((HIGH+LOW+CLOSE)/3,1)-LOW), 26) * 100
def alpha052(dataset, window=26):
    data = dataset.df.copy()
    A = data['High'] - ((data['High'] + data['Low'] + data['Close']) / 3).shift(1)
    A[A < 0] = 0
    A = A.rolling(window).sum()
    B = ((data['High'] + data['Low'] + data['Close']) / 3).shift(1) - data['Low']
    B[B < 0] = 0
    B = B.rolling(window).sum()
    return (A / B) * 100

# Alpha053 : COUNT( CLOSE>DELAY(CLOSE,1), 12) / 12 * 100
def alpha053(dataset, window=12):
    data = dataset.df.copy()
    alpha = data['Close'].diff(1)
    alpha[data['Close'].diff(1) > 0] = 1
    alpha[data['Close'].diff(1) < 0] = 0
    alpha = alpha.rolling(window).sum()
    return (alpha / window) * 100

# Alpha054 : -1 * RANK( STD(ABS(CLOSE-OPEN) + (CLOSE-OPEN) + CORR(CLOSE,OPEN,10) )
def alpha054(dataset, window=10):
    data = dataset.df.copy()
    A = (data['Close'] - data['Open']).abs().rolling(window).std()
    B = data['Close'] - data['Open']
    C = data['Close'].rolling(window).corr(data['Open'])
    return -1 * (A + B + C).rank(axis=1, pct=True)

# Alpha57: SMA((CLOSE-TSMIN (LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)
def alpha057(dataset, window=9):
    data = dataset.df.copy()
    alpha = (data['Close'] - data['Low'].rolling(window).min())/(data['High'].rolling(window).max() - data['Low'].rolling(window).min())*100
    alpha = alpha.ewm(adjust=False, alpha=1/3, ignore_na=False).mean()
    return alpha

# Alpha58: COUNT(CLOSE>DELAY(CLOSE,1),20)/20*100
def alpha058(dataset, window=20):                                  
    data = dataset.df.copy()
    alpha = (data['Close'].diff(1) > 0).rolling(window).mean()
    return alpha * 100

# Alpha 59: SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),20)
def alpha059(dataset, window=20):
    data = dataset.df.copy()
    alpha = data['Close'].diff(1)
    delay_close = data['Close'].shift(1)
    alpha[alpha>0] = data['Close'][alpha>0] - np.minimum(delay_close[alpha>0], data['Low'][alpha>0])
    alpha[alpha<0] = data['Close'][alpha<0] - np.minimum(delay_close[alpha<0], data['High'][alpha<0])
    return alpha.rolling(window).sum()

# Alpha60: SUM(((CLOSE-LOW)-(HIGH-CLOSE))./(HIGH-LOW).*VOLUME,20)
def alpha060(dataset, window=20):
    data = dataset.df.copy()
    alpha = (2*data['Close']-data['Low']-data['High'])/(data['High']-data['Low'])
    return alpha*data['Volume'].rolling(window).sum()

#Alpha61: (MAX(RANK(DECAYLINEAR(DELTA(VWAP, 1), 12)),
#           RANK(DECAYLINEAR(RANK(CORR((LOW),MEAN(VOLUME,80), 8)), 17))) * 1)
def alpha061(dataset, window1=12, window2=17, window3=80, window4=8):
    data = dataset.df.copy()
    w1 = np.array(range(1, window1+1)) / np.sum(range(1, window1+1))
    #a1 = data['VWAP'].diff(1).rolling(window1).apply(lambda x: np.dot(x, w1)).rank(axis=1, pct=True)
    a1 = data['VWAP'].diff(1)
    a1 = decayLinear(a1, w1, window1).rank(axis=1, pct=True)
    w2 = np.array(range(1, window2+1)) / np.sum(range(1, window2+1))
    mean = data['Volume'].rolling(window3).mean()
    corr = data['Low'].rolling(window4).corr(mean).rank(axis=1, pct=True)
    #a2 = corr.rolling(window2).apply(lambda x: np.dot(x, w2)).rank(axis=1, pct=True)
    a2 = decayLinear(corr, w2, window2).rank(axis=1, pct=True)
    a1[a1>=a2] = a1
    a1[a1<a2] = a2
    return a1

# Alpha62(-1 * CORR(HIGH, RANK(VOLUME), 5))
def alpha062(dataset, window=5):
    data = dataset.df.copy()
    vol = data['Volume'].rank(axis=1, pct=True)
    alpha = -1*data['High'].rolling(window).corr(vol)
    return alpha

# Alpha63: SMA(MAX(CLOSE-DELAY(CLOSE,1),0),6,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),6,1)*100
def alpha063(dataset, window=1):
    data = dataset.df.copy()
    part1 = data['Close'].diff(window)
    part2 = data['Close'].diff(window)
    part1[part1 >= 0] = part1
    part1[part1 < 0] = 0
    part2[part1 >= 0] = part2
    part2[part1 < 0] = part2*-1
    alpha1 = part1.ewm(adjust=False, alpha=1/6, ignore_na=False).mean()
    alpha2 = part2.ewm(adjust=False, alpha=1/6, ignore_na=False).mean()
    return alpha1/alpha2*100

# Alpha64: MAX(RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 4), 4)),
#RANK(DECAYLINEAR(MAX(CORR(RANK(CLOSE), RANK(MEAN(VOLUME,60)), 4), 13), 14))) * 1
def alpha064(dataset, window1=4, window2=60, window3=13, window4=14):
    data = dataset.df.copy()
    w1 = np.array(range(1, window1+1)) / np.sum(range(1, window1+1))
    rvwap = data['VWAP'].rank(axis=1, pct=True)
    rvol = data['Volume'].rank(axis=1, pct=True)
    a1 = rvwap.rolling(window1).corr(rvol)
    #a1 = a1.rolling(window1).apply(lambda x: np.dot(x, w1)).rank(axis=1, pct=True)
    a1.replace([np.inf, -np.inf], np.nan, inplace=True)
    a1 = decayLinear(a1, w1, window1).rank(axis=1, pct=True)
    a1.replace([np.inf, -np.inf], np.nan, inplace=True)
    muv = data['Volume'].rolling(window2).mean().rank(axis=1, pct=True)
    cor = data['Close'].rank(axis=1, pct=True).rolling(window1).corr(muv)
    cor = cor.rolling(window3).max()
    w2 = np.array(range(1, window4+1)) / np.sum(range(1, window4+1))
    #cor = cor.rolling(window4).apply(lambda x: np.dot(x, w2)).rank(axis=1, pct=True)
    cor.replace([np.inf, -np.inf], np.nan, inplace=True)
    cor = decayLinear(cor, w2, window4).rank(axis=1, pct=True)
    cor.replace([np.inf, -np.inf], np.nan, inplace=True)
    a1[a1>=cor] = a1
    a1[a1<cor] = cor
    return a1

# Alpha65 MEAN(CLOSE,6)/CLOSE
def alpha065(dataset, window=6):
    data = dataset.df.copy()
    alpha = data['Close'].rolling(window).mean() / data['Close']
    return alpha

# Alpha66: (CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)*100
def alpha066(dataset, window=6):
    data = dataset.df.copy()
    de = data['Close'].rolling(window).mean()
    nu = data['Close'] - de
    return nu/de*100

# Alpha67: SMA(MAX(CLOSE-DELAY(CLOSE,1),0),24,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),24,1)*100
def alpha067(dataset, window=1):
    data = dataset.df.copy()
    nu1 = data['Close'].diff(window)
    nu2 = data['Close'].diff(window)
    nu1[nu1>=0] = nu1
    nu1[nu1<0] = 0
    alpha1 = nu1.ewm(adjust=False, alpha=1/24, ignore_na=False).mean()
    nu2[nu2>=0] = nu2
    nu2[nu2<0] = nu2*-1
    alpha2 = nu2.ewm(adjust=False, alpha=1/24, ignore_na=False).mean()
    return alpha1/alpha2*100

# Alpha68: SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,15,2)
def alpha068(dataset, window=1):
    data = dataset.df.copy()
    A = (data['High'] + data['Low']) / 2
    B = (data['High'].shift(window) + data['Low'].shift(window)) / 2
    C = (data['High'] - data['Low']) / data['Volume']
    return ((A - B) * C).ewm(adjust=False, alpha=2/15, min_periods=0, ignore_na=False).mean() 

# Alpha69: (SUM(DTM,20)>SUM(DBM,20)？(SUM(DTM,20)-SUM(DBM,20))/SUM(DTM,20)
#           ：(SUM(DTM,20)=SUM(DBM,20)? 0 :(SUM(DTM,20)-SUM(DBM,20))/SUM(DBM,20)))
# DTM: (OPEN<=DELAY(OPEN,1) ? 0 : MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1))))
# DBM: (OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1))))
def alpha069(dataset, window=20):
    data = dataset.df.copy()
    DTM = data['Open'].diff(1).copy()
    DBM = data['Open'].diff(1).copy()
    conditionT = (data['Open'].diff(1)<=0) 
    conditionB = (data['Open'].diff(1)>=0) 
    condition1 = data['High']-data['Open'] - data['Open'].diff(1)
    condition2 = data['Open']-data['Low'] - data['Open'].diff(1)
    
    DTM[conditionT] = 0
    DTM[conditionB & condition1>=0] = data['High']-data['Open']
    DTM[conditionB & condition1<0] = data['Open'].diff(1)
    
    DBM[conditionB] = 0
    DBM[conditionT & condition2>=0] = data['Open']-data['Low']
    DBM[conditionT & condition2<0] = data['Open'].diff(1)
    
    alpha = DTM.rolling(window).sum() - DBM.rolling(window).sum()
    
    alpha[alpha>0] = alpha/DTM.rolling(window).sum()
    alpha[alpha==0] = 0
    alpha[alpha<0] = alpha/DBM.rolling(window).sum()
    return alpha

# Alpha70: STD(AMOUNT,6)
def alpha070(dataset, window=6):
    data = dataset.df.copy()
    alpha = data.Amount.rolling(window).std()
    return alpha

# Alpha71: (CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)*100
def alpha071(dataset, window=24):
    data = dataset.df.copy()
    alpha = data.Close.rolling(window).mean()
    return (data.Close - alpha) / alpha * 100

# Alpha72: SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX (HIGH,6)-TSMIN (LOW,6))*100,15,1)
def alpha072(dataset, window=6):
    data = dataset.df.copy()
    nu = data.High.rolling(window).max() - data.Close
    de = data.High.rolling(window).max() - data.Low.rolling(window).min()
    alpha = nu/de*100
    return alpha.ewm(adjust=False, alpha=1/15, ignore_na=False).mean()

# Alpha73: (TSRANK(DECAYLINEAR(DECAYLINEAR(CORR((CLOSE), VOLUME, 10), 16), 4), 5)
#            - RANK(DECAYLINEAR(CORR(VWAP, MEAN(VOLUME,30), 4),3))) *-1
def alpha073(dataset, window1=16, window2=4, window3=3, window4=10, window5=5, window6=4, window7=30):
    data = dataset.df.copy()
    w1 = np.array(range(1, window1+1)) / np.sum(range(1, window1+1))
    w2 = np.array(range(1, window2+1)) / np.sum(range(1, window2+1))
    w3 = np.array(range(1, window3+1)) / np.sum(range(1, window3+1))
    alpha1 = data.Close.rolling(window4).corr(data.Volume)
    # alpha1 = alpha1.rolling(window1).apply(lambda x: np.dot(x, w1))
    # alpha1 = alpha1.rolling(window2).apply(lambda x: np.dot(x, w2))
    alpha1.replace([np.inf, -np.inf], np.nan, inplace=True)
    alpha1 = decayLinear(alpha1, w1, window1)
    alpha1.replace([np.inf, -np.inf], np.nan, inplace=True)
    alpha1 = decayLinear(alpha1, w2, window2)
    alpha1.replace([np.inf, -np.inf], np.nan, inplace=True)
    alpha1 = alpha1.rolling(window5).max()
    alpha2 = data.VWAP.rolling(window6).corr(data.Volume.rolling(window7).mean())
    #alpha2 = alpha2.rolling(window3).apply(lambda x: np.dot(x, w3)).rank(axis=1, pct=True)
    alpha2.replace([np.inf, -np.inf], np.nan, inplace=True)
    alpha2 = decayLinear(alpha2, w3, window3).rank(axis=1, pct=True)
    alpha2.replace([np.inf, -np.inf], np.nan, inplace=True)
    return (alpha1 - alpha2)*-1

# Alpha74: (RANK(CORR(SUM(((LOW * 0.35) + (VWAP * 0.65)), 20), SUM(MEAN(VOLUME,40), 20), 7)) +
#          RANK( CORR(RANK(VWAP), RANK(VOLUME), 6)))
def alpha074(dataset, window1=20, window2=40, window3=7, window4=6):
    data = dataset.df.copy()
    sum1 = (data.Low*0.35 + data.VWAP*0.65).rolling(window1).sum()
    sum2 = data.Volume.rolling(window2).mean().rolling(window1).sum()
    cor1 = sum1.rolling(window3).corr(sum2).rank(axis=1, pct=True)
    r1 = data.VWAP.rank(axis=1, pct=True)
    r2 = data.Volume.rank(axis=1, pct=True)
    cor2 =r1.rolling(window4).corr(r2).rank(axis=1, pct=True)
    return cor1+cor2

# Alpha76: STD(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20) / 
#           MEAN(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)
def alpha076(dataset, window=20):
    data = dataset.df.copy()
    alpha = (data.Close/data.Close.shift(1) - 1).abs() / data.Volume
    return alpha.rolling(window).std() / alpha.rolling(window).mean()

# Alpha77: MIN(RANK(DECAYLINEAR(((((HIGH + LOW) / 2) + HIGH)-(VWAP + HIGH)), 20)),
#           RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME,40), 3), 6)))
def alpha077(dataset, window1=20, window2=6, window3=3, window4=40):
    data = dataset.df.copy()
    w1 = np.array(range(1, window1+1)) / np.sum(range(1, window1+1))
    d1 = (data.High + data.Low)/2 - data.VWAP 
    #d1 = d1.rolling(window1).apply(lambda x: np.dot(x, w1)).rank(axis=1, pct=True)
    d1.replace([np.inf, -np.inf], np.nan, inplace=True)
    d1 = decayLinear(d1, w1, window1)
    d1.replace([np.inf, -np.inf], np.nan, inplace=True)
    d1 = d1.rank(axis=1, pct=True)
    w2 = np.array(range(1, window2+1)) / np.sum(range(1, window2+1))
    d2 = ((data.High + data.Low)/2).rolling(window3).corr(data.Volume.rolling(window4).mean())
    #d2 = d2.rolling(window2).apply(lambda x: np.dot(x, w2)).rank(axis=1, pct=True)
    d2.replace([np.inf, -np.inf], np.nan, inplace=True)
    d2 = decayLinear(d2, w2, window2)
    d2.replace([np.inf, -np.inf], np.nan, inplace=True)
    d2 = d2.rank(axis=1, pct=True)
    d1[d1>d2] = d2
    d1[d1<d2] = d1
    return d1

# Alpha78: ((HIGH+LOW+CLOSE)/3-MA((HIGH+LOW+CLOSE)/3,12)) / 
# (0.015*MEAN( ABS(CLOSE-MEAN( (HIGH+LOW+CLOSE)/3, 12), 12))
# MA may be moving avg, 0.015 ignore
def alpha078(dataset, window=12):
    data = dataset.df.copy()
    alpha = (data['High'] + data['Low'] + data['Close']) / 3
    part1 = alpha - alpha.rolling(window).mean()
    part2 = (data['Close'] - alpha.rolling(window).mean()).abs().rolling(window).mean()
    return part1 / part2

# Alpha79: SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1) /
#           SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100
def alpha079(dataset, window=1):
    data = dataset.df.copy()
    alpha1 = data.Close.diff(window)
    alpha1[alpha1>0] = alpha1
    alpha1[alpha1<=0] = 0
    alpha1 = alpha1.ewm(adjust=False, alpha=1/12, ignore_na=False).mean()
    alpha2 = data.Close.diff(window).abs()
    alpha2 = alpha2.ewm(adjust=False, alpha=1/12, ignore_na=False).mean()
    return alpha1/alpha2*100

#Alpha80: (VOLUME-DELAY(VOLUME,5))/DELAY(VOLUME,5)*100
def alpha080(dataset, window=5):
    data = dataset.df.copy()
    return data.Volume.diff(window) / data.Volume.shift(window) *100

# Alpha81: SMA(VOLUME,21,2)
def alpha081(dataset):
    data = dataset.df.copy()
    return data.Volume.ewm(adjust=False, alpha=2/21, ignore_na=False).mean()

# ALpha82: SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,20,1)
def alpha082(dataset, window=6):
    data = dataset.df.copy()
    nu = data.High.rolling(window).max() - data.Close
    de = data.High.rolling(window).max() - data.Low.rolling(window).min()
    alpha = nu/de*100
    return alpha.ewm(adjust=False, alpha=1/20, ignore_na=False).mean()

# Alpha83:-1 * RANK(CORR(RANK(HIGH), RANK(VOLUME), 5)))
def alpha083_1(dataset, window=5):
    data = dataset.df.copy()
    alpha1 = (data.High.rank(axis=1, pct=True))
    alpha2 = data.Volume.rank(axis=1, pct=True)
    alpha = alpha1.rolling(window).corr(alpha2)
    return alpha.rank(axis=1, pct=True) * -1

# -1 * RANK(COV(RANK(HIGH), RANK(VOLUME), 5)))
def alpha083_2(dataset, window=5):
    data = dataset.df.copy()
    alpha1 = (data.High.rank(axis=1, pct=True))
    alpha2 = data.Volume.rank(axis=1, pct=True)
    alpha = alpha1.rolling(window).corr(alpha2)
    alpha = alpha * alpha1.rolling(window).std() * alpha2.rolling(window).std()
    return alpha.rank(axis=1, pct=True) * -1

# ALpha084: SUM((CLOSE>DELAY(CLOSE,1)?
#                   VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),20)
def alpha084(dataset, window=20):
    data = dataset.df.copy()
    alpha = data.Volume
    condition = data.Close.diff(1)
    alpha[condition>0] = alpha
    alpha[condition<0] = -1*alpha
    alpha[condition == 0] = 0
    return alpha.rolling(window).sum()

# Alpha85: (TSRANK((VOLUME/MEAN(VOLUME,20)),20) * 
#           TSRANK((-1 * DELTA(CLOSE, 7)), 8))
def alpha085(dataset, window1=20, window2=7, window3=8):
    data = dataset.df.copy()
    alpha1 = data.Volume / data.Volume.rolling(window1).mean()
    alpha1 = tsRank(alpha1, window1)
    alpha2 = data.Close.diff(window2) * -1
    alpha2 = tsRank(alpha2, window3)
    return alpha1*alpha2

# Alpha86: ((0.25 < (((DELAY(CLOSE, 20)-DELAY(CLOSE, 10)) / 10)-((DELAY(CLOSE, 10)-CLOSE) / 10)))
# ? (-1 * 1) : 
#   (((((DELAY(CLOSE, 20)-DELAY(CLOSE, 10)) / 10)-((DELAY(CLOSE, 10) CLOSE) / 10)) < 0) ? 
#   1 : ((-1 * 1) *(CLOSE-DELAY(CLOSE, 1)))))
def alpha086(dataset, window1=20, window2=10):
    data = dataset.df.copy()
    condition = (data.Close.shift(window1) - data.Close.shift(window2))/(window1-window2) - (data.Close.shift(window2) - data.Close)/window2
    condition[condition>0.25] = -1
    condition[condition<0] = 1
    other = (condition>=0) & (condition<=0.25)
    condition[other] = data.Close.diff(1)[other] * (-1)
    return condition

# Alpha87: ((RANK(DECAYLINEAR(DELTA(VWAP, 4), 7)) + 
#           TSRANK(DECAYLINEAR(((((LOW * 0.9) + (LOW * 0.1))-VWAP)/(OPEN-((HIGH + LOW) / 2))), 11), 7)) *-1)
def alpha087(dataset, window1=7, window2=11, window3=4, window4=7):
    data = dataset.df.copy()
    w1 = np.array(range(1, window1+1)) / np.sum(range(1, window1+1))
    w2 = np.array(range(1, window2+1)) / np.sum(range(1, window2+1))
    #part1 = data.VWAP.diff(window3).rolling(window1).apply(lambda x: np.dot(x, w1)).rank(axis=1, pct=True)
    part1 = data.VWAP.diff(window3)
    part1 = decayLinear(part1, w1, window1)
    part1.replace([np.inf, -np.inf], np.nan, inplace=True)
    part1 = part1.rank(axis=1, pct=True)
    part2 = (data.Low - data.VWAP) / (data.Open - ((data.High+data.Low)/2))
    #part2 = part2.rolling(window2).apply(lambda x: np.dot(x, w2)).rank(axis=1, pct=True)
    # part2 = part2.rolling(window4).apply(lambda x: stats.rankdata(x)[-1] / window4)
    part2.replace([np.inf, -np.inf], np.nan, inplace=True)
    part2 = decayLinear(part2, w2, window2)
    part2.replace([np.inf, -np.inf], np.nan, inplace=True)
    part2 = part2.rank(axis=1, pct=True)
    part2 = tsRank(part2, window4)
    return (part1+part2)*-1

# Alpha88: (CLOSE-DELAY(CLOSE,20))/DELAY(CLOSE,20)*100
def alpha088(dataset, window=20):
    data = dataset.df.copy()
    return data.Close.diff(window)/data.Close.shift(window)*100

# Alpha89: 2*(SMA(CLOSE,13,2)-SMA(CLOSE,27,2)-SMA(SMA(CLOSE,13,2)-SMA(CLOSE,27,2),10,2))
def alpha089(dataset):
    data = dataset.df.copy()
    sma1 = data.Close.ewm(adjust=False, alpha=2/13, ignore_na=False).mean()
    sma2 = data.Close.ewm(adjust=False, alpha=2/27, ignore_na=False).mean()
    sma3  = (sma1-sma2).ewm(adjust=False, alpha=2/10, ignore_na=False).mean()
    return 2*(sma1-sma2-sma3)

# Alpha90: (RANK(CORR(RANK(VWAP), RANK(VOLUME), 5)) *-1)
def alpha090(dataset, window=5):
    data = dataset.df.copy()
    alpha = (data.VWAP.rank(axis=1, pct=True)).rolling(window).corr(data.Volume.rank(axis=1, pct=True))
    return alpha.rank(axis=1, pct=True) * -1

# Alpha91: ((RANK((CLOSE-MAX(CLOSE, 5)))*RANK(CORR((MEAN(VOLUME,40)), LOW, 5))) *-1)
# MAX should be TSMAX ??
def alpha091(dataset, window1=5, window2=40, window3=5):
    data = dataset.df.copy()
    r1 = (data.Close - data.Close.rolling(window1).max()).rank(axis=1, pct=True)
    r2 = (data.Volume.rolling(window2).mean()).rolling(window3).corr(data.Low).rank(axis=1, pct=True)
    return r1*r2*-1

# Alpha92: (MAX(RANK(DECAYLINEAR(DELTA(((CLOSE * 0.35) + (VWAP *0.65)), 2), 3)),
#           TSRANK(DECAYLINEAR(ABS(CORR((MEAN(VOLUME,180)), CLOSE, 13)), 5), 15)) *-1)
def alpha092(dataset, window1=3, window2=5, window3=180, window4=13, window5=15):
    data = dataset.df.copy()
    w1 = np.array(range(1, window1+1)) / np.sum(range(1, window1+1))
    part1 = (data.Close*0.35 + data.VWAP*0.65).diff(2)
    #part1 = part1.rolling(window1).apply(lambda x: np.dot(x, w1)).rank(axis=1, pct=True)
    part1 = decayLinear(part1, w1, window1)
    part1.replace([np.inf, -np.inf], np.nan, inplace=True)
    part1 = part1.rank(axis=1, pct=True)
    w2 = np.array(range(1, window2+1)) / np.sum(range(1, window2+1))
    part2 = ((data.Volume.rolling(window3).mean()).rolling(window4).corr(data.Close)).abs()
    #part2 = part2.rolling(window2).apply(lambda x: np.dot(x, w2)).rank(axis=1, pct=True)
    part2 = decayLinear(part2, w2, window2)
    part2.replace([np.inf, -np.inf], np.nan, inplace=True)
    part2 = part2.rank(axis=1, pct=True)
    # part2 = part2.rolling(window5).apply(lambda x: stats.rankdata(x)[-1] / window5)
    part2 = tsRank(part2, window5)
    part1[part1>=part2] = part1
    part1[part1<=part2] = part2
    return part1*-1

# Alpha93: SUM((OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1)))),20)
def alpha093(dataset, window=20):
    data = dataset.df.copy()
    condition = data.Open.diff(1)
    condition[condition>=0] = 0
    c1 = data.Open - data.Low
    c2 = data.Open.diff(1)
    c1[c1>c2] = c1
    c1[c1<c2] = c2
    condition[condition<0] = c1
    return condition.rolling(window).sum()

# Alpha94: SUM((CLOSE>DELAY(CLOSE,1) ? 
#               VOLUME: (CLOSE<DELAY(CLOSE,1)? -VOLUME:0)),30)
def alpha094(dataset, window=30):
    data = dataset.df.copy()
    condition = data.Close.diff(1)
    condition[condition>0] = data.Volume
    condition[condition<0] = data.Volume*-1    
    condition[condition==0] = 0
    return condition.rolling(window).sum()

# Alpha95: STD(AMOUNT, 20)
def alpha095(dataset, window=20):
    data = dataset.df.copy()
    alpha = data.Amount.rolling(window).std()
    return alpha

# Alpha96: SMA(SMA((CLOSE-TSMIN(LOW,9)) / (TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1)
def alpha096(dataset, window=9):
    data = dataset.df.copy()
    nu = data.Close - data.Low.rolling(window).min()
    de = data.High.rolling(window).max() - data.Low.rolling(window).min()
    alpha = nu / de *100
    alpha = alpha.ewm(adjust=False, alpha=1/3, ignore_na=False).mean()
    return alpha.ewm(adjust=False, alpha=1/3, ignore_na=False).mean()

# Alpha97: STD(VOLUME,10)
def alpha097(dataset, window=10):
    data = dataset.df.copy()
    alpha = data.Volume.rolling(window).std()
    return alpha

# Alpha98: ((((DELTA((SUM(CLOSE, 100) / 100), 100) / DELAY(CLOSE, 100)) < 0.05) || 
#           ((DELTA((SUM(CLOSE, 100) / 100), 100) / DELAY(CLOSE, 100)) == 0.05)) ? 
#               (-1 * (CLOSE-TSMIN(CLOSE, 100))) :-1 * DELTA(CLOSE, 3))
def alpha098(dataset, window1=100, window2=3):
    data = dataset.df.copy()
    condition = (data.Close.rolling(window1).sum()/window1).diff(window1) / data.Close.shift(window1)
    condition[condition <= 0.05] =  (data.Close - data.Close.rolling(window1).min()) * -1
    condition[condition > 0.05] = data.Close.diff(window2) * -1
    return condition

# Alpha99_1: -1 * RANK(CORR(RANK(CLOSE), RANK(VOLUME), 5))
def alpha099_1(dataset, window=5):
    data = dataset.df.copy()
    alpha1 = (data.Close.rank(axis=1, pct=True))
    alpha2 = data.Volume.rank(axis=1, pct=True)
    alpha = alpha1.rolling(window).corr(alpha2)
    return alpha.rank(axis=1, pct=True) * -1

# Alpha99_2: -1 * RANK(COVIANCE(RANK(CLOSE), RANK(VOLUME), 5))
def alpha099_2(dataset, window=5):
    data = dataset.df.copy()
    alpha1 = (data.Close.rank(axis=1, pct=True))
    alpha2 = data.Volume.rank(axis=1, pct=True)
    alpha = alpha1.rolling(window).corr(alpha2)
    alpha = alpha * alpha1.rolling(window).std() * alpha2.rolling(window).std()
    return alpha.rank(axis=1, pct=True) * -1

# Alpha100: STD(VOLUME,20)
def alpha100(dataset, window=20):
    data = dataset.df.copy()
    alpha = data.Volume.rolling(window).std()
    return alpha

# Alpha102: SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)/SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1)*100
def alpha102(dataset, window=1): 
    data = dataset.df.copy()
    alpha1 = data.Volume - data.Volume.shift(window)
    alpha1[alpha1>=0] = alpha1
    alpha1[alpha1<0] = 0
    alpha2 = data.Volume - data.Volume.shift(window)
    alpha2[alpha2>=0] = alpha2
    alpha2[alpha2<0] = -1*alpha2
    sma1 = alpha1.ewm(adjust=False, alpha=1/6, ignore_na=False).mean()
    sma2 = alpha2.ewm(adjust=False, alpha=1/6, ignore_na=False).mean()
    return  sma1/sma2 * 100

# Alpha103: ((20-LOWDAY(LOW,20 ))/20)*100 
def alpha103(dataset, window=20):
    data = dataset.df.copy()
    lowday = (window-1) - data.Low.rolling(window).apply(np.argmin)
    return (window-lowday)/window*100

# Alpha104: (-1 * (DELTA(CORR(HIGH, VOLUME, 5), 5) * RANK(STD(CLOSE, 20))))
def alpha104(dataset, window1=5, window2=20):
    data = dataset.df.copy()
    part1 = data.High.rolling(window1).corr(data.Volume).diff(window1)
    part2 = data.Close.rolling(window2).std().rank(axis=1, pct=True)
    return -1 * part1 * part2

# Alpha105: (-1 * CORR(RANK(OPEN), RANK(VOLUME), 10))
def alpha105(dataset, window=10):
    data = dataset.df.copy()
    return -1 * (data.Open.rank(axis=1, pct=True)).rolling(window).corr(data.Volume.rank(axis=1, pct=True))

def alpha106(dataset, window=20):
    data = dataset.df.copy()
    return data.Close.diff(window)

# Alpha107: (((-1 * RANK((OPEN-DELAY(HIGH, 1)))) * RANK((OPEN-DELAY(CLOSE, 1)))) * 
#                   RANK((OPEN-DELAY(LOW, 1))))
def alpha107(dataset, window=1):
    data = dataset.df.copy()
    r1 = -1 * (data.Open - data.High.shift(window)).rank(axis=1, pct=True)
    r2 = (data.Open - data.Close.shift(window)).rank(axis=1, pct=True)
    r3 = (data.Open - data.Low.shift(window)).rank(axis=1, pct=True)
    return r1 * r2 * r3

# Alpha108: ((RANK((HIGH-MIN(HIGH, 2)))^RANK(CORR((VWAP), (MEAN(VOLUME,120)), 6))) *-1)
# MIN should be TSMIN ???
def alpha108(dataset, window1=2, window2=6, window3=120):
    data = dataset.df.copy()
    r1 = (data.High - data.High.rolling(window1).min()).rank(axis=1, pct=True)
    r2 = (data.VWAP.rolling(window2).corr(data.Volume.rolling(window3).mean())).rank(axis=1, pct=True)
    return (r1 ** r2) * -1

# Alpha109: SMA(HIGH-LOW,10,2)/SMA(SMA(HIGH-LOW,10,2),10,2)
def alpha109(dataset):
    data = dataset.df.copy()
    sma1 = (data.High - data.Low).ewm(adjust=False, alpha=2/10, ignore_na=False).mean()
    sma2 = sma1.ewm(adjust=False, alpha=2/10, ignore_na=False).mean()
    return sma1 / sma2

# Alpha110: SUM(MAX(0,HIGH-DELAY(CLOSE,1)),20)/SUM(MAX(0,DELAY(CLOSE,1)-LOW),20)*100
def alpha110(dataset, window=20):
    data = dataset.df.copy()
    alpha1 = data.High - data.Close.shift(1)
    alpha1[alpha1>0] = alpha1
    alpha1[alpha1<=0] = 0
    sum1 = alpha1.rolling(window).sum()
    alpha2 = data.Close.shift(1) - data.Low
    alpha2[alpha2>0] = alpha2
    alpha2[alpha2<=0] = 0
    sum2 = alpha2.rolling(window).sum()
    return sum1 / sum2 * 100

# Alpha111: SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),11,2) - 
#           SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),4,2)
# VOL should be VOLUME ???
def alpha111(dataset):
    data = dataset.df.copy()
    alpha = data.Volume * ((data.Close-data.Low)-(data.High-data.Close)) / (data.High-data.Low)
    sma1 = alpha.ewm(adjust=False, alpha=2/11, ignore_na=False).mean()
    sma2 = alpha.ewm(adjust=False, alpha=2/4, ignore_na=False).mean()
    return sma1 - sma2

def alpha112(dataset, window=12):
    data = dataset.df.copy()
    alpha = data.Close.diff(1)
    alpha1 = data.Close.diff(1)
    alpha[alpha > 0] = alpha
    alpha[alpha <= 0] = 0
    alpha1[alpha1 >= 0] = 0
    alpha1[alpha1 < 0] = alpha1.abs()
    alpha = alpha.rolling(window).sum()
    alpha1 = alpha1.rolling(window).sum()
    de = (alpha+alpha1).replace(0, np.nan) #　how to let 0 to non
    return (alpha-alpha1) / de *100

# Alpha113: -1 * ((RANK((SUM(DELAY(CLOSE, 5), 20) / 20)) * CORR(CLOSE, VOLUME, 2)) * 
#                   RANK(CORR(SUM(CLOSE, 5),SUM(CLOSE, 20), 2)))
def alpha113(dataset, window1=5, window2=20, window3=2):
    data = dataset.df.copy()
    p1 = (data.Close.shift(window1).rolling(window2).sum() / window2).rank(axis=1, pct=True)
    p2 = data.Close.rolling(window3).corr(data.Volume)    
    p3 = (data.Close.rolling(window1).sum()).rolling(window3).corr(data.Close.rolling(window2).sum()).rank(axis=1, pct=True)
    return -1 * p1 * p2 * p3

# Alpha114: RANK(DELAY(((HIGH-LOW) / (SUM(CLOSE, 5) / 5)), 2)) * 
#           RANK(RANK(VOLUME) /   (((HIGH-LOW) /(SUM(CLOSE, 5) / 5)) / (VWAP-CLOSE)))
def alpha114(dataset, window1=5, window2=2):
    data = dataset.df.copy()
    r1 = ((data.High - data.Low) / data.Close.rolling(window1).mean()).shift(window2).rank(axis=1, pct=True)
    r2 = (data.Volume.rank(axis=1, pct=True)) / (((data.High - data.Low) / data.Close.rolling(window1).mean()) / (data.VWAP-data.Close))
    return r1 * (r2.rank(axis=1, pct=True))

# Alpha115: (RANK(CORR(((HIGH * 0.9) + (CLOSE * 0.1)), MEAN(VOLUME,30), 10)) ^ 
#               RANK(CORR(TSRANK(((HIGH + LOW) /2), 4), TSRANK(VOLUME, 10), 7)))
def alpha115(dataset, window1=10, window2=30, window3=4, window4=10, window5=7):
    data = dataset.df.copy()
    r1 = (data.High*0.9 + data.Close*0.1).rolling(window1).corr(data.Volume.rolling(window2).mean()).rank(axis=1, pct=True)
    #r2_1 = ((data.High+data.Low)/2).rolling(window3).apply(lambda x: stats.rankdata(x)[-1] / window3)
    r2_1 = tsRank((data.High+data.Low)/2, window3)
    #r2_2 = data.Volume.rolling(window4).apply(lambda x: stats.rankdata(x)[-1] / window4)
    r2_2 = tsRank(data.Volume, window4)
    r2 = r2_1.rolling(window5).corr(r2_2)
    return r1 ** r2

# Alpha117: ((TSRANK(VOLUME, 32) * (1-TSRANK(((CLOSE + HIGH)-LOW), 16))) * (1-TSRANK(RET, 32)))
# RET: (close/close-1)
def alpha117(dataset, window1=32, window2=16):
    data = dataset.df.copy()
    # p1 = data.Volume.rolling(window1).apply(lambda x: stats.rankdata(x)[-1] / window1)
    p1 = tsRank(data.Volume, window1)
    # p2 = 1 - ((data.Close + data.High - data.Low).rolling(window2).apply(lambda x: stats.rankdata(x)[-1] / window2))
    p2 = 1 - tsRank(data.Close + data.High - data.Low, window2)
    # p3 = 1 - ((data.Close/data.Close.shift(1) - 1).rolling(window1).apply(lambda x: stats.rankdata(x)[-1] / window1))
    p3 = 1 - tsRank(data.Close/data.Close.shift(1) - 1, window1)
    return p1 * p2 * p3

# Alpha118: SUM(HIGH-OPEN,20)/SUM(OPEN-LOW,20)*100
def alpha118(dataset, window=20):
    data = dataset.df.copy()
    s1 = (data.High-data.Open).rolling(window).sum()
    s2 = (data.Open-data.Low).rolling(window).sum()
    return s1 / s2 * 100

# Alpha119: (RANK(DECAYLINEAR(CORR(VWAP, SUM(MEAN(VOLUME,5), 26), 5), 7)) - 
#            RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(OPEN), RANK(MEAN(VOLUME,15)), 21), 9), 7), 8)))
# MIN should be TSMIN ???
def alpha119(dataset, window1=7, window2=5, window3=26, window4=8, window5=21, window6=15, window7=9, window8=7):
    data = dataset.df.copy()
    w1 = np.array(range(1, window1+1)) / np.sum(range(1, window1+1))
    p1 = data.VWAP.rolling(window2).corr((data.Volume.rolling(window2).mean()).rolling(window3).sum())
    #p1 = p1.rolling(window1).apply(lambda x: np.dot(x, w1)).rank(axis=1, pct=True)
    p1.replace([np.inf, -np.inf], np.nan, inplace=True)
    p1 = decayLinear(p1, w1, window1)
    p1.replace([np.inf, -np.inf], np.nan, inplace=True)
    p1 = p1.rank(axis=1, pct=True)
    w2 = np.array(range(1, window4+1)) / np.sum(range(1, window4+1))
    cor = (data.Open.rank(axis=1, pct=True)).rolling(window5).corr(data.Volume.rolling(window6).mean().rank(axis=1, pct=True))
    #p2 = cor.rolling(window7).min().rolling(window8).apply(lambda x: stats.rankdata(x)[-1] / window8)
    p2 = cor.rolling(window7).min()
    p2.replace([np.inf, -np.inf], np.nan, inplace=True)
    p2 = tsRank(p2, window8)
    #p2 = p2.rolling(window4).apply(lambda x: np.dot(x, w2)).rank(axis=1, pct=True)
    p2.replace([np.inf, -np.inf], np.nan, inplace=True)
    p2 = decayLinear(p2, w2, window4)
    p2.replace([np.inf, -np.inf], np.nan, inplace=True)
    p2 = p2.rank(axis=1, pct=True)
    return p1 - p2

# Alpha120: (RANK((VWAP-CLOSE)) / RANK(( VWAP + CLOSE)))
def alpha120(dataset):
    data = dataset.df.copy()
    nu = (data.VWAP - data.Close).rank(axis=1, pct=True)
    de = (data.VWAP + data.Close).rank(axis=1, pct=True)
    return nu / de

# Alpha121: ((RANK((VWAP-MIN(VWAP, 12))) ^ 
#             TSRANK(CORR(TSRANK(VWAP, 20), TSRANK(MEAN(VOLUME,60), 2), 18), 3)) *-1)
# MIN should be TSMIN ???
def alpha121(dataset, window1=12, window2=20, window3=60, window4=2, window5=18, window6=3):
    data = dataset.df.copy()
    p1 = (data.VWAP - data.VWAP.rolling(window1).min()).rank(axis=1, pct=True)
    # tr1 = data.VWAP.rolling(window2).apply(lambda x: stats.rankdata(x)[-1] / window2)
    # tr2 = (data.Volume.rolling(window3).mean()).rolling(window4).apply(lambda x: stats.rankdata(x)[-1] / window4)
    # p2 = (tr1.rolling(window5).corr(tr2)).rolling(window6).apply(lambda x: stats.rankdata(x)[-1] / window6)
    tr1 = tsRank(data.VWAP, window2)
    tr2 = tsRank(data.Volume.rolling(window3).mean(), window4)
    p2 = tr1.rolling(window5).corr(tr2)
    p2.replace([np.inf, -np.inf], np.nan, inplace=True)
    p2 = tsRank(p2, window6)
    return (p1 ** p2) * -1

# Alpha122: (SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2) - 
#                  DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1)) / 
#           DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1)
def alpha122(dataset, window=1):
    data = dataset.df.copy()
    alpha = np.log(data.Close)
    
    for _ in range(3):
        alpha = alpha.ewm(adjust=False, alpha=2/13, ignore_na=False).mean()
    
    alpha1 = alpha.shift(window)
    
    return (alpha - alpha1) / alpha1

# Alpha124: (CLOSE-VWAP) / DECAYLINEAR(RANK(TSMAX(CLOSE, 30)),2)
def alpha124(dataset, window1=30, window2=2):
    data = dataset.df.copy()
    nu = data.Close - data.VWAP
    de = data.Close.rolling(window1).max().rank(axis=1, pct=True)
    w = np.array(range(1, window2+1)) / np.sum(range(1, window2+1))
    #de = de.rolling(window2).apply(lambda x: np.dot(x, w)).rank(axis=1, pct=True)
    de.replace([np.inf, -np.inf], np.nan, inplace=True)
    de = decayLinear(de, w, window2)
    de.replace([np.inf, -np.inf], np.nan, inplace=True)
    de = de.rank(axis=1, pct=True)
    return nu / de

# Alpha125: (RANK(DECAYLINEAR(CORR((VWAP), MEAN(VOLUME,80),17), 20)) / 
#           RANK(DECAYLINEAR(DELTA(((CLOSE * 0.5)+ (VWAP * 0.5)), 3), 16)))
def alpha125(dataset, window1=20, window2=16, window3=17, window4=80, window5=3):
    data = dataset.df.copy()
    w1 = np.array(range(1, window1+1)) / np.sum(range(1, window1+1))
    r1 = data.VWAP.rolling(window3).corr(data.Volume.rolling(window4).mean())
    #r1 = r1.rolling(window1).apply(lambda x: np.dot(x, w1)).rank(axis=1, pct=True)
    r1.replace([np.inf, -np.inf], np.nan, inplace=True)
    r1 = decayLinear(r1, w1, window1)
    r1.replace([np.inf, -np.inf], np.nan, inplace=True)
    r1 = r1.rank(axis=1, pct=True)
    w2 = np.array(range(1, window2+1)) / np.sum(range(1, window2+1))
    r2 = (data.Close*0.5  + data.VWAP*0.5).diff(window5)
    #r2 = r2.rolling(window2).apply(lambda x: np.dot(x, w2)).rank(axis=1, pct=True)
    r2.replace([np.inf, -np.inf], np.nan, inplace=True)
    r2 = decayLinear(r2, w2, window2)
    r2.replace([np.inf, -np.inf], np.nan, inplace=True)
    r2 = r2.rank(axis=1, pct=True)
    return r1 / r2

# Alpha126: (CLOSE+HIGH+LOW)/3
def alpha126(dataset):
    data = dataset.df.copy()
    return (data.Close + data.High + data.Low) / 3

# Alpha127: (MEAN(100*(CLOSE-MAX(CLOSE,12))/(MAX(CLOSE,12)))^2)^(1/2)
# MAX should be TSMAX ???
# ^2 and ^1/2 position not sure
def alpha127(dataset, window=12):
    data = dataset.df.copy()
    maxc = data.Close.rolling(window).max()
    alpha = 100*(data.Close - maxc) / maxc
    return ((alpha**2).rolling(window).mean()) ** 1/2

# Alpha128: 100-(100/(1+SUM(((HIGH+LOW+CLOSE)/3>DELAY((HIGH+LOW+CLOSE)/3 ,1) ? 
#           (HIGH+LOW+CLOSE)/3*VOLUME:0),14) / 
#           SUM(((HIGH+LOW+CLOSE)/3<DELAY((HIGH+LOW+CLOSE)/3,1) ? 
#           (HIGH+LOW+CLOSE)/3*VOLUME:0),14)))
def alpha128(dataset, window1=1, window2=14): 
    data = dataset.df.copy()
    condition = (data.Close + data.High + data.Low) / 3
    condition[condition>condition.shift(window1)] = condition*data.Volume
    condition[condition<=condition.shift(window1)] = 0
    condition = condition.rolling(window2).sum()
    cond = (data.Close + data.High + data.Low) / 3
    cond[cond<cond.shift(window1)] = cond*data.Volume
    cond[cond>=cond.shift(window1)] = 0
    cond = cond.rolling(window2).sum()
    return 100-(100/(1+condition/cond))

# Alpha129: SUM((CLOSE-DELAY(CLOSE,1)<0 ? ABS(CLOSE-DELAY(CLOSE,1)):0),12)
def alpha129(dataset, window1=1, window2=12):
    data = dataset.df.copy()
    alpha = data.Close.diff(window1)
    alpha[alpha>=0] = 0
    alpha[alpha<0] = alpha.abs()
    return alpha.rolling(window2).sum()

# Alpha130: (RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME,40), 9), 10)) /
#           RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 7),3)))
def alpha130(dataset, window1=10, window2=3, window3=9, window4=7, window5=40):
    data = dataset.df.copy()
    w1 = np.array(range(1, window1+1)) / np.sum(range(1, window1+1))
    r1 = ((data.High+data.Low)/2).rolling(window3).corr(data.Volume.rolling(window5).mean())
    #r1 = r1.rolling(window1).apply(lambda x: np.dot(x, w1)).rank(axis=1, pct=True)
    r1.replace([np.inf, -np.inf], np.nan, inplace=True)
    r1 = decayLinear(r1, w1, window1)
    r1.replace([np.inf, -np.inf], np.nan, inplace=True)
    r1 = r1.rank(axis=1, pct=True)
    w2 = np.array(range(1, window2+1)) / np.sum(range(1, window2+1))
    r2 = data.VWAP.rank(axis=1, pct=True).rolling(window4).corr(data.Volume.rank(axis=1, pct=True))
    #r2 = r2.rolling(window2).apply(lambda x: np.dot(x, w2)).rank(axis=1, pct=True)
    r2.replace([np.inf, -np.inf], np.nan, inplace=True)
    r2 = decayLinear(r2, w2, window2)
    r2.replace([np.inf, -np.inf], np.nan, inplace=True)
    r2 = r2.rank(axis=1, pct=True)
    return r1 / r2

# Alpha131: (RANK(DELAT(VWAP, 1))^TSRANK(CORR(CLOSE,MEAN(VOLUME,50), 18), 18))
# DELAT should be DELTA ???
def alpha131(dataset, window1=18, window2=50, window3=1):
    data = dataset.df.copy()
    part1 = data.VWAP.diff(window3).rank(axis=1, pct=True)
    part2 = data.Close.rolling(window1).corr(data.Volume.rolling(window2).mean())
    # part2 = part2.rolling(window1).apply(lambda x: stats.rankdata(x)[-1] / window1)
    part2.replace([np.inf, -np.inf], np.nan, inplace=True)
    part2 = tsRank(part2, window1)
    return part1 ** part2

# Alpha132: MEAN(AMOUNT, 20)
def alpha132(dataset, window=20):
    data = dataset.df.copy()
    return data.Amount.rolling(window).mean()

# Alpha133: ((20-HIGHDAY(HIGH,20))/20)*100 - 
#           ((20-LOWDAY(LOW,20)))/20)*100
def alpha133(dataset, window=20):
    data = dataset.df.copy()
    highday = (window-1) - data.High.rolling(window).apply(np.argmax)
    lowday = (window-1) - data.Low.rolling(window).apply(np.argmax)
    part1 = (window-highday)/window * 100
    part2 = (window-lowday)/window * 100
    return part1 - part2

# Alpha134: (CLOSE-DELAY(CLOSE,12))/DELAY(CLOSE,12)*VOLUME
def alpha134(dataset, window=12):
    data = dataset.df.copy()
    return data.Close.diff(window) / data.Close.shift(window) * data.Volume

# Alpha135: SMA(DELAY(CLOSE/DELAY(CLOSE,20),1),20,1)
def alpha135(dataset, window1=1, window2=20):
    data = dataset.df.copy()
    alpha = (data.Close / data.Close.shift(window2)).shift(window1)
    return alpha.ewm(adjust=False, alpha=1/20, ignore_na=False).mean()

# Alpha136: ((-1 * RANK(DELTA(RET, 3))) * CORR(OPEN, VOLUME, 10))
def alpha136(dataset, window1=3, window2=10):
    data = dataset.df.copy()
    part1 = data.Close/data.Close.shift(1) - 1
    part1 = part1.diff(window1).rank(axis=1, pct=True) * -1
    part2 = data.Open.rolling(window2).corr(data.Volume)
    return part1 * part2

# ( 
# RANK( DECAYLINEAR( DELTA((((LOW*0.7)+(VWAP*0.3))), 3), 20) ) -
# TSRANK( DECAYLINEAR( TSRANK( CORR(TSRANK(LOW, 8), TSRANK(MEAN(VOLUME, 60), 17), 5), 19), 16), 7)
# ) * -1
def alpha138(dataset, window1=20, window2=16, window3=8, window4=60, window5=17, window6=5, window7=19, window8=7):
    data = dataset.df.copy()
    # part1
    w1 = np.array(range(1, window1+1)) / np.sum(range(1, window1+1))
    #part1 = (data['Low'] * 0.7 + data['VWAP'] * 0.3).diff(3).rolling(window1).apply(lambda x: np.dot(x, w1)).rank(axis=1, pct=True)
    part1 = (data['Low'] * 0.7 + data['VWAP'] * 0.3).diff(3)
    part1 = decayLinear(part1, w1, window1)
    part1.replace([np.inf, -np.inf], np.nan, inplace=True)
    part1 = part1.rank(axis=1, pct=True)
    # part2
    w2 = np.array(range(1, window2+1)) / np.sum(range(1, window2+1))
    # a1 = data['Low'].rolling(window3).apply(lambda x: stats.rankdata(x)[-1] / window3)
    # a2 = data['Volume'].rolling(window4).mean().rolling(window5).apply(lambda x: stats.rankdata(x)[-1] / window5)
    # part2 = a1.rolling(window6).corr(a2).rolling(window7).apply(lambda x: stats.rankdata(x)[-1] / window7).rolling(window2).apply(lambda x: np.dot(x, w2)).rolling(window8).apply(lambda x: stats.rankdata(x)[-1] / window8)
    a1 = tsRank(data['Low'], window3)
    a2 = tsRank(data['Volume'].rolling(window4).mean(), window5)
    part2 = a1.rolling(window6).corr(a2)
    part2.replace([np.inf, -np.inf], np.nan, inplace=True)
    #part2 = tsRank(part2, window7).rolling(window2).apply(lambda x: np.dot(x, w2))
    part2 = tsRank(part2, window7)
    part2.replace([np.inf, -np.inf], np.nan, inplace=True)
    part2 = decayLinear(part2, w2, window2)
    part2.replace([np.inf, -np.inf], np.nan, inplace=True)
    part2 = tsRank(part2, window8)
    return (part1 - part2) * -1

# -1 * CORR(OPEN, VOLUME, 10)
def alpha139(dataset, window=10):
    data = dataset.df.copy()
    return data['Open'].rolling(window).corr(data['Volume']) * -1

# MIN( RANK( DECAYLINEAR(( (RANK(OPEN)+RANK(LOW))-(RANK(HIGH)+RANK(CLOSE)) ), 8) ) , TSRANK( DECAYLINEAR( CORR( TSRANK(CLOSE,8), TSRANK(MEAN(VOLUME,60), 20) , 8), 7) , 3) )
def alpha140(dataset, window1=8, window2=8, window3=60, window4=20, window5=8, window6=7, window7=3):
    data = dataset.df.copy()
    # part1
    w1 = np.array(range(1, window1+1)) / np.sum(range(1, window1+1))
    part1 = data['Open'].rank(axis=1, pct=True) + data['Low'].rank(axis=1, pct=True) - data['High'].rank(axis=1, pct=True) - data['Close'].rank(axis=1, pct=True)
    part1 = part1.rolling(window1).apply(lambda x: np.dot(x, w1)).rank(axis=1, pct=True)
    # part2
    w2 = np.array(range(1, window6+1)) / np.sum(range(1, window6+1))
    # a1 = data['Close'].rolling(window2).apply(lambda x: stats.rankdata(x)[-1] / window2)
    # a2 = data['Volume'].rolling(window3).mean().rolling(window4).apply(lambda x: stats.rankdata(x)[-1] / window4)
    # part2 = a1.rolling(window5).corr(a2).rolling(window6).apply(lambda x: np.dot(x, w2)).rolling(window7).apply(lambda x: stats.rankdata(x)[-1] / window7)
    a1 = tsRank(data['Close'], window2)
    a2 = tsRank(data['Volume'].rolling(window3).mean(), window4)
    part2 = a1.rolling(window5).corr(a2).rolling(window6).apply(lambda x: np.dot(x, w2))
    part2.replace([np.inf, -np.inf], np.nan, inplace=True)
    part2 = tsRank(part2, window7)
    alpha = part1.copy()
    condition = part1 > part2
    alpha[condition] = part2
    return alpha

# RANK( CORR( RANK(HIGH), RANK(MEAN(VOLUME,15)), 9) ) * -1
def alpha141(dataset, window1=15, window2=9):
    data = dataset.df.copy()
    a1 = data['High'].rank(axis=1, pct=True)
    a2 = data['Volume'].rolling(window1).mean().rank(axis=1, pct=True)
    return a1.rolling(window2).corr(a2).rank(axis=1, pct=True) * -1

# (((-1 * RANK( TSRANK(CLOSE, 10) )) * 
# RANK(DELTA(DELTA(CLOSE,1),1))) * 
# RANK(TSRANK((VOLUME/MEAN(VOLUME,20)),5)))
def alpha142(dataset, window1=10, window2=2, window3=20, window4=5):
    data = dataset.df.copy()
    #a1 = data['Close'].rolling(window1).apply(lambda x: stats.rankdata(x)[-1] / window1).rank(axis=1, pct=True)
    a1 = tsRank(data['Close'], window1).rank(axis=1, pct=True)
    a2 = data['Close'].diff(window2).rank(axis=1, pct=True)
    #a3 = ( data['Volume'] / data['Volume'].rolling(window3).mean() ).rolling(window4).apply(lambda x: stats.rankdata(x)[-1] / window4).rank(axis=1, pct=True)
    a3 = tsRank(data['Volume'] / data['Volume'].rolling(window3).mean(), window4).rank(axis=1, pct=True)
    return -1*a1*a2*a3

# SELF: 特殊變量，出現在Alpha143，表示t-1 日的 Alpha143 因子計算結果
# CLOSE > DELAY(CLOSE,1)?
# (CLOSE-DELAY(CLOSE,1)) / DELAY(CLOSE,1) * SELF:
# SELF
def alpha143(dataset, window=1):
    data = dataset.df.copy()
    a1 = data['Close'].pct_change(window)
    a2 = data['Close'].pct_change(window).shift(1)
    condition = a1 > 0
    a2[condition] = a1 * a2
    return a2

# SUMIF( ABS(CLOSE/DELAY(CLOSE,1)-1)/AMOUNT, 20 , CLOSE < DELAY(CLOSE,1) ) 
# / 
# COUNT( CLOSE < DELAY(CLOSE,1), 20 )
def alpha144(dataset, window1=1, window2=20):
    data = dataset.df.copy()
    # part1
    a1 = data['Close'].pct_change(window1).abs() / data['Amount'] 
    part1 = a1.rolling(window2).sum()
    condition = data['Close'].pct_change(window1) >= 0
    part1[condition] = a1
    # part2
    part2 = (data['Close'].diff(window1) < 0).rolling(window2).sum()
    return part1 / part2

# (MEAN(VOLUME,9)-MEAN(VOLUME,26))/MEAN(VOLUME,12) * 100
def alpha145(dataset, window1=9, window2=26, window3=12):
    data = dataset.df.copy()
    return (data['Volume'].rolling(window1).mean() - data['Volume'].rolling(window2).mean()) / data['Volume'].rolling(window3).mean()

# A = (CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)
# MEAN( A - SMA(A, 61, 2), 20) *
# ( A - SMA(A, 61, 2)) / 
# SMA( (A - (A - SMA( A, 61, 2)))^2, 60 )
# Note: last SMA 缺一個值，改成 40, 2
def alpha146(dataset, window1=1, window2=2, window3=40, window4=20):
    data = dataset.df.copy()
    a = data['Close'].pct_change(window1)
    # part1
    part1 = (a - a.ewm(adjust=False, alpha=window2/window3, min_periods=0, ignore_na=False).mean()).rolling(window4).mean() * (a - a.ewm(adjust=False, alpha=window2/window3, min_periods=0, ignore_na=False).mean())
    # part2
    part2 = ((a.ewm(adjust=False, alpha=window2/window3, min_periods=0, ignore_na=False).mean())**2).ewm(adjust=False, alpha=window2/window3, min_periods=0, ignore_na=False).mean()
    return part1/part2

# (CLOSE+HIGH+LOW)/3*VOLUME
def alpha150(dataset):
    data = dataset.df.copy()
    return ((data['Close']+data['High']+data['Low'])/3) * data['Volume']

# SMA( CLOSE-DELAY(CLOSE,20) , 20, 1)
def alpha151(dataset, window=20):
    data = dataset.df.copy()
    alpha = data['Close'].diff(window)
    return alpha.ewm(adjust=False, alpha=1/20, min_periods=0, ignore_na=False).mean()

# SMA(
# MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1), 12) - 
# MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1), 26), 
# 9, 1)
def alpha152(dataset, window1=9, window2=1, window3=12, window4=26):
    data = dataset.df.copy()
    a = (data['Close'] / data['Close'].shift(window1)).diff(window2).ewm(adjust=False, alpha=1/9, min_periods=0, ignore_na=False).mean().diff(window2)
    return (a.rolling(window3).mean() - a.rolling(window4).mean()).ewm(adjust=False, alpha=1/9, min_periods=0, ignore_na=False).mean()

# (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/4
def alpha153(dataset, window1=3, window2=6, window3=12, window4=24,):
    data = dataset.df.copy()
    return (data['Close'].rolling(window1).mean() + data['Close'].rolling(window2).mean() + data['Close'].rolling(window3).mean() + data['Close'].rolling(window4).mean()) / 4

# A = SMA(VOLUME,13,2) - SMA(VOLUME,27,2)
# A - SMA( A, 10, 2)
def alpha155(dataset):
    data = dataset.df.copy()
    a = data['Volume'].ewm(adjust=False, alpha=2/13, min_periods=0, ignore_na=False).mean() - data['Volume'].ewm(adjust=False, alpha=2/27, min_periods=0, ignore_na=False).mean()
    return a - a.ewm(adjust=False, alpha=2/10, min_periods=0, ignore_na=False).mean()

# MAX( RANK(DECAYLINEAR(DELTA(VWAP,5),3)), RANK( DECAYLINEAR(((DELTA(((OPEN * 0.15)+(LOW * 0.85)),2)/((OPEN * 0.15)+(LOW * 0.85))) * -1), 3))) * -1
def alpha156(dataset, window1=5, window2=3, window3=2):
    data = dataset.df.copy()
    # part1
    w = np.array(range(1, window2+1)) / np.sum(range(1, window2+1))
    #part1 = data['VWAP'].diff(window1).rolling(window2).apply(lambda x: np.dot(x, w)).rank(axis=1, pct=True)
    part1 = data['VWAP'].diff(window1)
    part1 = decayLinear(part1, w, window2)
    part1.replace([np.inf, -np.inf], np.nan, inplace=True)
    part1 = part1.rank(axis=1, pct=True)
    # part2
    #part2 = (((data['Open']*0.15 + data['Low']*0.85).diff(window3) / (data['Open']*0.15 + data['Low']*0.85)) * -1).rolling(window2).apply(lambda x: np.dot(x, w)).rank(axis=1, pct=True)
    part2 = (((data['Open']*0.15 + data['Low']*0.85).diff(window3) / (data['Open']*0.15 + data['Low']*0.85)) * -1)
    part2 = decayLinear(part2, w, window2)
    part2.replace([np.inf, -np.inf], np.nan, inplace=True)
    part2 = part2.rank(axis=1, pct=True)
    condition = part1 < part2
    alpha = part1.copy()
    alpha[condition] = part2
    return alpha * -1

# ( (HIGH-SMA(CLOSE,15,2)) - ( LOW-SMA(CLOSE,15,2) ) ) / CLOSE
def alpha158(dataset):
    data = dataset.df.copy()
    return (data['High'] - data['Low']) / data['Close']

# ((CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),6))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),6)*12*24 + 
# (CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),12))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),12)*6*24 +
# (CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),24))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),24)*6*24)*100 /
# (6*12+6*24+12*24)
def alpha159(dataset, window1=6, window2=12, window3=24):
    data = dataset.df.copy()
    # part1
    part1 = data['Low'].copy()
    a1 = data['Close'].shift(1)
    condition1 = part1 > a1
    part1[condition1] = a1
    # part2
    part2 = data['High'].copy()
    condition2 = part2 < a1
    part2[condition2] = a1
    # combine
    alpha1 = (data['Close'] - part1.rolling(window1).sum()) / ((part2 - part1).rolling(window1).sum()*window2*window3)
    alpha2 = (data['Close'] - part1.rolling(window2).sum()) / ((part2 - part1).rolling(window2).sum()*window1*window3)
    alpha3 = (data['Close'] - part1.rolling(window3).sum()) / ((part2 - part1).rolling(window3).sum()*window1*window3)
    return ((alpha1+alpha2+alpha3)*100) / (window1*window2+window1*window3+window2*window3)

# SMA(
# (CLOSE<=DELAY(CLOSE,1)? STD(CLOSE,20): 0)
# , 20, 1)
def alpha160(dataset, window1=1, window2=20):
    data = dataset.df.copy()
    condition = data['Close'].diff(window1) > 0
    alpha = data['Close'].rolling(window2).std()
    alpha[condition] = 0
    return alpha.ewm(adjust=False, alpha=1/20, min_periods=0, ignore_na=False).mean()

# MEAN(
# MAX( MAX( (HIGH-LOW), ABS(DELAY(CLOSE,1)-HIGH) ), ABS(DELAY(CLOSE,1)-LOW))
# , 12)
def alpha161(dataset, window1=12):
    data = dataset.df.copy()
    # part1
    part1 = data['High'] - data['Low']
    a1 = (data['Close'].shift(1) - data['High']).abs()
    condition1 = part1 < a1
    part1[condition1] = a1
    # part2
    part2 = (data['Close'].shift(1) - data['Low']).abs()
    alpha = part1.copy()
    condition2 = part1 < part2
    alpha[condition2] = part2
    return alpha.rolling(window1).mean()

# A = MAX(CLOSE-DELAY(CLOSE,1), 0)
# B = ABS(CLOSE-DELAY(CLOSE,1))
# ( SMA(A,12,1) / SMA(B,12,1) * 100 - 
# MIN( SMA(A,12,1) / SMA(B,12,1) * 100, 12) ) /
# ( MAX(SMA(A,12,1) / SMA(B,12,1) * 100, 12) - 
# MIN( SMA(A,12, 1)/SMA(B,12,1) * 100, 12) )
def alpha162(dataset, window1=1, window2=12):
    data = dataset.df.copy()
    A = data['Close'].diff(window1)
    A[A < 0] = 0
    B = data['Close'].diff(window1).abs()
    C = (A.ewm(adjust=False, alpha=window1/window2, min_periods=0, ignore_na=False).mean() / B.ewm(adjust=False, alpha=window1/window2, min_periods=0, ignore_na=False).mean()) * 100
    # part1
    part1 = C.copy()
    part1[C > 12] = 12
    part1 = C - part1
    # part2
    part2 = C.copy()
    part2[C < 12] = 12
    part3 = C.copy()
    part3[C > 12] = 12
    return part1 / (part2 - part3)

# RANK(((((-1 * RET) * MEAN(VOLUME, 20))*VWAP)*(HIGH-CLOSE)))
def alpha163(dataset, window=20):
    data = dataset.df.copy()
    return ((data['Close'].pct_change(1)*data['Volume'].rolling(window).mean()*data['VWAP']*(data['High'] - data['Close'])) * -1).rank(axis=1, pct=True)

# SMA(
# (((CLOSE>DELAY(CLOSE,1))? 1/(CLOSE-DELAY(CLOSE,1)): 1) -
# MIN(
# ((CLOSE>DELAY(CLOSE,1))? 1/(CLOSE-DELAY(CLOSE,1)): 1)
# , 12)) / (HIGH-LOW) * 100,
#  13, 2)
# MIN -> TSMIN ?
def alpha164(dataset, window1=1, window2=12):
    data = dataset.df.copy()
    a = data['Close'].diff(window1)
    condition = a < 0
    part1 = 1/a
    part1[condition] = 1
    part2 = part1.rolling(window2).min()
    return (((part1 - part2) / (data['High'] - data['Low']))*100).ewm(adjust=False, alpha=2/13, min_periods=0, ignore_na=False).mean()

# MAX(SUMAC(CLOSE-MEAN(CLOSE,48))) - MIN(SUMAC(CLOSE-MEAN(CLOSE,48))) / STD(CLOSE,48)
def alpha165(dataset, window=48):
    data = dataset.df.copy()
    alpha = (data['Close'] - data['Close'].rolling(window).mean()).rolling(window).sum()
    return (alpha.rolling(window).max() - alpha.rolling(window).min()) / data['Close'].rolling(window).std()

# -20*(20-1)^1.5*SUM(CLOSE/DELAY(CLOSE,1)-1-MEAN(CLOSE/DELAY(CLOSE,1)-1,20),20)/((20-1)*(20-2)*(SUM((CLOSE/DELAY(CLOSE,1))^2,20))^1.5)
def alpha166(dataset, window=20):
    data = dataset.df.copy()
    part1 = data['Close'].pct_change(1)-(data['Close'].pct_change(1).rolling(window).mean())
    part1 = part1.rolling(window).sum() * ((-window) * (window-1) ** 1.5)
    part2 = (((data['Close']/data['Close'].shift(1)) ** 2).rolling(window).sum() ** 1.5) * (window-1) * (window-2)
    return part1 / part2

# SUM(
# (CLOSE-DELAY(CLOSE,1)>0? CLOSE-DELAY(CLOSE,1): 0)
# , 12)
def alpha167(dataset, window=12):
    data = dataset.df.copy()
    alpha = data['Close'].diff(1)
    alpha[alpha < 0] = 0
    return alpha.rolling(window).sum()

# -1 * (VOLUME/MEAN(VOLUME,20))
def alpha168(dataset, window=20):
    data = dataset.df.copy()
    return (data['Volume'] / data['Volume'].rolling(window).mean()) * -1

# A = DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1)
# SMA(
# MEAN(A, 12) - MEAN(A, 26)
# , 10, 1)
def alpha169(dataset, window1=12, window2=26):
    data = dataset.df.copy()
    a = data['Close'].diff(1).ewm(adjust=False, alpha=1/9, min_periods=0, ignore_na=False).mean().shift(1)
    part1 = a.rolling(window1).mean()
    part2 = a.rolling(window2).mean()
    return (part1 - part2).ewm(adjust=False, alpha=1/10, min_periods=0, ignore_na=False).mean().shift(1)

# ((RANK(1/CLOSE)*VOLUME)/MEAN(VOLUME,20)) * (HIGH*RANK(HIGH-CLOSE)/(SUM(HIGH,5)/5)) - RANK(VWAP-DELAY(VWAP,5))
def alpha170(dataset, window1=20, window2=5):
    data = dataset.df.copy()
    part1 = ((1 / data['Close']).rank(axis=1, pct=True) * data['Volume']) / data['Volume'].rolling(window1).mean()
    part2 = (data['High'] * (data['High'] - data['Close']).rank(axis=1, pct=True)) / data['High'].rolling(window2).mean()
    part3 = data['VWAP'].diff(window2).rank(axis=1, pct=True)
    return part1 * part2 - part3

# (-1*(LOW-CLOSE)*(OPEN^5))/((CLOSE-HIGH)*(CLOSE^5))
def alpha171(dataset, window=5):
    data  = dataset.df.copy()
    part1 = (data['Low']-data['Close']) * (data['Open'] ** window) * (-1)
    part2 = (data['Close']-data['High']) * (data['Close'] ** window)
    return part1 / part2

# 就是DMI-ADX
# HD  HIGH-DELAY(HIGH,1)
# LD  DELAY(LOW,1)-LOW
# TR  MAX(MAX(HIGH-LOW,ABS(HIGH-DELAY(CLOSE,1))),ABS(LOW-DELAY(CLOSE,1)))
# A = SUM((LD>0&LD>HD)?LD:0,14)*100
# B = SUM((HD>0&HD>LD)?HD:0,14)*100
# C = SUM(TR,14)
# MEAN(
# ABS(A/C - B/C) / (A/C + B/C)*100
# , 6)
def alpha172(dataset, window1=14, window2=6):
    data = dataset.df.copy()
    hd = data['High'].diff(1)
    ld = -data['Low'].diff(1)
    # tr
    a1 = data['High'] - data['Low']
    a2 = (data['High'] - data['Close'].shift(1)).abs()
    a3 = (data['Low'] - data['Close'].shift(1)).abs()
    tr = a1.copy()
    condition1 = a2 > a1
    tr[condition1] = a2
    condition2 = a3 > tr
    tr[condition2] = a3
    # A, B, C
    condition1 = (ld > 0) & (ld > hd)
    part1 = ld.copy()
    part1[~condition1] = 0
    part1 = part1.rolling(window1).sum() * 100
    condition2 = (hd > 0) & (hd > ld)
    part2 = hd.copy()
    part2[~condition2] = 0
    part2 = part2.rolling(window1).sum() * 100
    part3 = tr.rolling(window1).sum()
    return (((part1/part3 - part2/part3).abs() / (part1/part3 - part2/part3)) * 100).rolling(window2).mean()

# 3*SMA(CLOSE,13,2) - 2*SMA(SMA(CLOSE,13,2),13,2) + SMA( SMA( SMA(LOG(CLOSE) ,13,2) ,13,2) ,13,2)
def alpha173(dataset):
    data = dataset.df.copy()
    part1 = 3 * data['Close'].ewm(adjust=False, alpha=2/13, min_periods=0, ignore_na=False).mean()
    part2 = 2 * data['Close'].ewm(adjust=False, alpha=2/13, min_periods=0, ignore_na=False).mean().ewm(adjust=False, alpha=2/13, min_periods=0, ignore_na=False).mean()
    part3 = np.log(data['Close']).ewm(adjust=False, alpha=2/13, min_periods=0, ignore_na=False).mean().ewm(adjust=False, alpha=2/13, min_periods=0, ignore_na=False).mean().ewm(adjust=False, alpha=2/13, min_periods=0, ignore_na=False).mean()
    return part1 - part2 + part3

# SMA(
# (CLOSE>DELAY(CLOSE,1)? STD(CLOSE,20): 0)
# ,20,1)
def alpha174(dataset, winodw=20):
    data = dataset.df.copy()
    alpha = data['Close'].rolling(winodw).std()
    condition = data['Close'].diff(1) < 0
    alpha[condition] = 0
    return alpha.ewm(adjust=False, alpha=1/20, min_periods=0, ignore_na=False).mean()

# MEAN(
# MAX(MAX((HIGH-LOW), ABS(DELAY(CLOSE,1)-HIGH)), ABS(DELAY(CLOSE,1)-LOW))
# , 6)
def alpha175(dataset, window=6):
    data = dataset.df.copy()
    a1 = data['High'] - data['Low']
    a2 = (data['Close'].shift(1) - data['High']).abs()
    a3 = (data['Close'].shift(1) - data['Low']).abs()
    a1[a1 < a2] = a2
    a1[a1 < a3] = a3
    return a1.rolling(window).mean()

# CORR(
# RANK( ((CLOSE-TSMIN(LOW, 12)) / (TSMAX(HIGH, 12)-TSMIN(LOW, 12))) ), RANK(VOLUME)
# , 6)
def alpha176(dataset, window1=12, window2=6):
    data = dataset.df.copy()
    part1 = ((data['Close'] - data['Low'].rolling(window1).min()) / (data['High'].rolling(window1).max() - data['Low'].rolling(window1).min())).rank(axis=1, pct=True)
    part2 = data['Volume'].rank(axis=1, pct=True)
    return part1.rolling(window2).corr(part2)

# ((20-HIGHDAY(HIGH,20))/20)*100
def alpha177(dataset, window=20):
    data = dataset.df.copy()
    highday = (window-1) - data.High.rolling(window).apply(np.argmax)
    return (window-highday)/window * 100

# (CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*VOLUME
def alpha178(dataset, window=1):
    data = dataset.df.copy()
    alpha = data['Close'].pct_change(window)
    return alpha * data['Volume']

# RANK( CORR(VWAP, VOLUME, 4) ) * RANK( CORR( RANK(LOW), RANK(MEAN(VOLUME, 50)), 12) )
def alpha179(dataset, window1=4, window2=50, window3=12):
    data = dataset.df.copy()
    # part1
    part1 = data['VWAP'].rolling(window1).corr(data['Volume']).rank(axis=1, pct=True)
    # part2
    part2 = data['Low'].rank(axis=1, pct=True).rolling(window3).corr(data['Volume'].rolling(window2).mean().rank(axis=1, pct=True)).rank(axis=1, pct=True)
    return part1 * part2

# ((MEAN(VOLUME,20)<VOLUME)? ((-1 * TSRANK(ABS(DELTA(CLOSE,7)),60)) * SIGN(DELTA(CLOSE,7)): (-1 * VOLUME)))
def alpha180(dataset, window1=7, window2=60, window3=20):
    data = dataset.df.copy()
    # part1
    #part1 = data['Close'].diff(window1).abs().rolling(window2).apply(lambda x: stats.rankdata(x)[-1] / window2) * -1
    part1 = data['Close'].diff(window1).abs()
    part1 = tsRank(part1, window2) * -1
    a1 = data['Close'].diff(window1)
    a1[a1 > 0] = 1
    a1[a1 < 0] = -1
    part1 = part1 * a1
    # part2
    part2 = data['Volume'] * -1
    condition = (data['Volume'].rolling(window3).mean() - data['Volume']) > 0
    alpha = part1.copy()
    alpha[condition] = part2
    return alpha

# MAX(SUMAC(CLOSE-MEAN(CLOSE,24))) - MIN(SUMAC(CLOSE-MEAN(CLOSE,24))) / STD(CLOSE,24)
def alpha183(dataset, window=24):
    data = dataset.df.copy()
    a = (data['Close'] - data['Close'].rolling(window).mean()).rolling(window).sum()
    return (a.rolling(window).max() - a.rolling(window).min()) / data['Close'].rolling(window).std()

# RANK( CORR(DELAY((OPEN-CLOSE),1), CLOSE, 200) ) + RANK((OPEN-CLOSE))
def alpha184(dataset, window=200):
    data = dataset.df.copy()
    part1 = (data['Open'] - data['Close']).shift(1).rolling(window).corr(data['Close']).rank(axis=1, pct=True)
    part2 = (data['Open'] - data['Close']).rank(axis=1, pct=True)
    return part1 + part2

# RANK((-1*((1-(OPEN/CLOSE))^2)))
def alpha185(dataset):
    data = dataset.df.copy()
    return (((1 - (data['Open'] / data['Close']))**2) * -1).rank(axis=1, pct=True)

# A = SUM((LD>0 & LD>HD)?LD:0,14)*100
# B = SUM((HD>0 & HD>LD)?HD:0,14)*100
# C = SUM(TR,14)
# ( MEAN(ABS(A/C-B/C) / (A/C+B/C)*100, 6) + DELAY( MEAN(ABS(A/C-B/C) / (A/C+B/C)*100, 6), 6) 
# ) / 2
def alpha186(dataset, window1=14, window2=6):
    data = dataset.df.copy()
    hd = data['High'].diff(1)
    ld = -data['Low'].diff(1)
    # tr
    a1 = data['High'] - data['Low']
    a2 = (data['High'] - data['Close'].shift(1)).abs()
    a3 = (data['Low'] - data['Close'].shift(1)).abs()
    tr = a1.copy()
    condition1 = a2 > a1
    tr[condition1] = a2
    condition2 = a3 > tr
    tr[condition2] = a3
    # A, B, C
    condition1 = (ld > 0) & (ld > hd)
    part1 = ld.copy()
    part1[~condition1] = 0
    part1 = part1.rolling(window1).sum() * 100
    condition2 = (hd > 0) & (hd > ld)
    part2 = hd.copy()
    part2[~condition2] = 0
    part2 = part2.rolling(window1).sum() * 100
    part3 = tr.rolling(window1).sum()
    alpha = (((part1/part3 - part2/part3).abs() / (part1/part3 - part2/part3)) * 100).rolling(window2).mean()
    return (alpha + alpha.shift(window2)) / 2

# SUM(
# (OPEN<=DELAY(OPEN,1)? 0: MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1))))
# , 20)
def alpha187(dataset, window=20):
    data = dataset.df.copy()
    a1 = data['High'] - data['Open']
    a2 = data['Open'].diff(1)
    condition1 = a1 < a2
    a1[condition1] = a2
    condition2 = data['Open'].diff(1) <= 0
    alpha = a1.copy()
    alpha[condition2] = 0
    return alpha.rolling(window).sum()

# ( (HIGH-LOW–SMA(HIGH-LOW,11,2)) / SMA(HIGH-LOW,11,2) ) * 100
def alpha188(dataset):
    data = dataset.df.copy()
    part1 = data['High'] - data['Low'] - (data['High'] - data['Low']).ewm(adjust=False, alpha=2/11, min_periods=0, ignore_na=False).mean()
    part2 = (data['High'] - data['Low']).ewm(adjust=False, alpha=2/11, min_periods=0, ignore_na=False).mean()
    return part1/part2 * 100

# MEAN(
# ABS(CLOSE-MEAN(CLOSE,6))
# , 6)
def alpha189(dataset, window=6):
    data = dataset.df.copy()
    alpha = (data['Close'] - data['Close'].rolling(window).mean()).abs()
    return alpha.rolling(window).mean()

# LOG(
# (COUNT( RET>((CLOSE/DELAY(CLOSE,19))^(1/20)-1), 20)-1)
# *SUMIF( (RET-(CLOSE/DELAY(CLOSE,19))^(1/20)-1)^2, 20, RET<(CLOSE/DELAY(CLOSE,19))^(1/20)-1)
# /
# (COUNT(RET<(CLOSE/DELAY(CLOSE,19))^(1/20)-1,20)
# *SUMIF((RET-((CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,RET>(CLOSE/DELAY(CLOSE,19))^(1/20)-1))
# )
def alpha190(dataset, window=19):
    data = dataset.df.copy()
    # part1
    condition1 = data['Close'].pct_change(1) > (data['Close'] / data['Close'].shift(window))**(1/(window+1)) - 1
    a1 = condition1.rolling(window+1).sum() - 1
    a2 = (data['Close'].pct_change(1) - (data['Close'] / data['Close'].shift(window))**(1/(window+1)) - 1)**2
    a2[~condition1] = a2.rolling(window+1).sum()
    part1 = a1*a2
    # part2
    condition2 = data['Close'].pct_change(1) < (data['Close'] / data['Close'].shift(window))**(1/(window+1)) - 1
    a3 = condition2.rolling(window+1).sum()
    a4 = (data['Close'].pct_change(1) - (data['Close'] / data['Close'].shift(window))**(1/(window+1)) - 1)**2
    a4[condition1] = a4.rolling(window+1).sum()
    part2 = a3*a4
    return np.log(part1/part2)

# CORR( MEAN(VOLUME,20), LOW , 5) + (HIGH+LOW)/2) - CLOSE
def alpha191(dataset, window1=20, window2=5):
    data = dataset.df.copy()
    return data['Volume'].rolling(window1).mean().rolling(window2).corr(data['Low']) + data['Close'] + (data['High'] + data['Low']) / 2

# -------------------------------- HuaTai Technical alphas --------------------------------
# HAlpha
def alpha_HAlpha(dataset, window1=20, window2=60):
    data = dataset.df['Close'].pct_change(window1)
    data_rm = dataset.df_rm.pct_change(window1)
    df_alpha = data.copy()
    for col in data.columns:
        temp = pd.DataFrame(
            {
                'y': data[col].values,
                'X': data_rm['收盤價'].values
            }
        )
        # convert to npy
        npy_y = rolling_window_fixedSize(temp['y'].T, window2)
        npy_x = rolling_window_fixedSize(temp['X'].T, window2)
        # rolling OLS
        alphas_= [np.nan for _ in range(window2-1)]
        for i in range(len(npy_x)):
            alpha_, beta_ = guotai_rolling_ols(npy_x[i], npy_y[i])
            alphas_.append(alpha_)
        # save results
        df_alpha[col] = alphas_
    return df_alpha

# beta
def alpha_beta(dataset, window1=20, window2=60):
    data = dataset.df['Close'].pct_change(window1)
    data_rm = dataset.df_rm.pct_change(window1)
    df_beta  = data.copy()
    for col in data.columns:
        temp = pd.DataFrame(
            {
                'y': data[col].values,
                'X': data_rm['收盤價'].values
            }
        )
        # convert to npy
        npy_y = rolling_window_fixedSize(temp['y'].T, window2)
        npy_x = rolling_window_fixedSize(temp['X'].T, window2)
        # rolling OLS
        betas_ = [np.nan for _ in range(window2-1)]
        for i in range(len(npy_x)):
            alpha_, beta_ = guotai_rolling_ols(npy_x[i], npy_y[i])
            betas_.append(beta_)
        # save results
        df_beta[col]  = betas_
    return df_beta

# return_Nm
def alpha_return_Nm(dataset, window=20):
    data = dataset.df.copy()
    return data['Close'].pct_change(window)

# wgt_return_Nm
def alpha_wgt_return_Nm(dataset, window=20):
    data  = dataset.df.copy()
    alpha = data['Close'].pct_change(1) * data['TO']
    return alpha.rolling(window).mean()

# exp_wgt_return_Nm
def alpha_exp_wgt_return_Nm(dataset, window):
    data  = dataset.df.copy()
    alpha = data['Close'].pct_change(1) * data['TO']
    return alpha.ewm(span=window, adjust=False, min_periods=window).mean()

# std_FF3factor_Nm
def alpha_std_FF3factor_Nm(dataset, window):
    data = dataset.df.copy()
    ret = data['Close'].pct_change()
    data_rm = dataset.df_rm.pct_change()
    df_res_std = ret.copy()
    # generate SMB and HML
    me30 = (data['MktValue'].T <= data['MktValue'].quantile(0.3, axis=1)).T
    me70 = (data['MktValue'].T >= data['MktValue'].quantile(0.7, axis=1)).T
    pb30 = (data['PB'].T <= data['PB'].quantile(0.3, axis=1)).T
    pb70 = (data['PB'].T >= data['PB'].quantile(0.7, axis=1)).T
    smb_ret = ret[me30].mean(axis=1, skipna=True) - ret[me70].mean(axis=1, skipna=True)
    hml_ret = ret[pb70].mean(axis=1, skipna=True) - ret[pb30].mean(axis=1, skipna=True)
    # rolling OLS
    for col in ret.columns:
        temp = pd.DataFrame(
            {
                'y': ret[col].values,
                'X1': data_rm['收盤價'].values,
                'X2': smb_ret.values,
                'X3': hml_ret.values
            }
        )
        # convert to npy
        npy_y = rolling_window_fixedSize(temp['y'].T, window)
        npy_x1 = rolling_window_fixedSize(temp['X1'].T, window)
        npy_x2 = rolling_window_fixedSize(temp['X2'].T, window)
        npy_x3 = rolling_window_fixedSize(temp['X3'].T, window)
        # rolling OLS
        res_std_ = [np.nan for _ in range(window-1)]
        for i in range(len(npy_y)):
            X = np.concatenate((npy_x1[i].reshape(window,1),npy_x2[i].reshape(window,1),npy_x3[i].reshape(window,1)), axis=1)
            res_std = guotai_rolling_ols_std(X, npy_y[i])
            res_std_.append(res_std)
        # save results
        df_res_std[col] = res_std_
    return df_res_std

# std_Nm
def alpha_std_Nm(dataset, window):
    data = dataset.df.copy()
    ret  = data['Close'].pct_change()
    return ret.rolling(window).std()

# ln_capital
def alpha_ln_capital(dataset):
    data = dataset.df.copy()
    return np.log(data['MktValue'])

# turn_Nm
def alpha_turn_Nm(dataset, window):
    data = dataset.df.copy()
    return data['TO'].rolling(window).mean()

# bias_turn_Nm 
def alpha_bias_turn_Nm(dataset, window1, window2=500):
    data = dataset.df.copy()
    return (data['TO'].rolling(window1).mean() / data['TO'].rolling(window2).mean()) - 1

# MACD
def alpha_MACD(dataset, window1=10, window2=30, window3=15):
    data = dataset.df.copy()
    exp1 = data['Close'].ewm(span=window1, adjust=False, min_periods=window1).mean()
    exp2 = data['Close'].ewm(span=window2, adjust=False, min_periods=window2).mean()
    dif  = exp1 - exp2
    dea  = dif.ewm(span=window3, adjust=False, min_periods=window3).mean()
    return dif - dea

# DEA
def alpha_DEA(dataset, window1=10, window2=30, window3=15):
    data = dataset.df.copy()
    exp1 = data['Close'].ewm(span=window1, adjust=False, min_periods=window1).mean()
    exp2 = data['Close'].ewm(span=window2, adjust=False, min_periods=window2).mean()
    dif  = exp1 - exp2
    dea  = dif.ewm(span=window3, adjust=False, min_periods=window3).mean()
    return dea

# DIF
def alpha_DIF(dataset, window1=10, window2=30, window3=15):
    data = dataset.df.copy()
    exp1 = data['Close'].ewm(span=window1, adjust=False, min_periods=window1).mean()
    exp2 = data['Close'].ewm(span=window2, adjust=False, min_periods=window2).mean()
    dif  = exp1 - exp2
    dea  = dif.ewm(span=window3, adjust=False, min_periods=window3).mean()
    return dif

# RSI
def alpha_RSI(dataset, window=20):
    data = dataset.df.copy()
    up   = data['Close'].diff(1)
    up[up < 0] = 0
    down = data['Close'].diff(1) 
    down[down > 0] = 0
    down = -down
    return (100*up.rolling(window).mean()) / (up.rolling(window).mean() + down.rolling(window).mean())

# PSY
def alpha_PSY(dataset, window=20):
    data  = dataset.df.copy()
    alpha = data['Close'].diff(1)
    alpha = alpha > 0
    return alpha.rolling(window).mean()

# BIAS
def alpha_BIAS(dataset, window=20):
    data  = dataset.df.copy()
    return (data['Close'] - data['Close'].rolling(window).mean()) / data['Close'].rolling(window).mean()

# ---------------------------- GuoTai Technical alphas 2021Ver -----------------------------
# Alpha Sto
def alphaSto(dataset, window=1):
    data = dataset.df.copy()
    return data['TO'].rolling(window).mean()

# Alpha Corr_cp_turn
def alphaCorrCPTO(dataset, window=5):
    data = dataset.df.copy()
    return data['Close'].rolling(window).corr(data['TO'])

# Alpha Vol
def alphaVol(dataset, window=5):
    data = dataset.df.copy()
    return data['Close'].pct_change(1).rolling(window).std()

# Alpha Vstd
def alphaVstd(dataset, window=5):
    data = dataset.df.copy()
    return data['Volume'].rolling(window).std() / data['Volume'].rolling(window).mean()

# AlphaAD
def alphaAD(dataset):
    data = dataset.df.copy()
    clv = ((data['Close']*2 - data['Low'] - data['High']) / (data['High'] - data['Low'])) * data['Volume']
    return clv.cumsum()

# -------------------------- Founder Securities Technical alphas ---------------------------
def alphaOverMOM(dataset, window=15):
    data = dataset.df.copy()
    m1 = (data['Open'] / data['Close'].shift(1)) - 1
    return m1.rolling(window).sum()

def alphaIntraOverMOM(dataset, window=15):
    data = dataset.df.copy()
    m1 = ((data['Open'] / data['Close'].shift(1)) - 1).rolling(window).sum()
    m0 = ((data['Close'] / data['Open']) - 1).rolling(window).sum()
    return (m0.rank(axis=1, pct=True) + m1.rank(axis=1, pct=True, ascending=False)) / 2

def alphaFR(dataset, window=20):
    data = dataset.df.copy()
    to = data['TO'] - data['TO'].rolling(window).mean()
    ret = data['Close'].pct_change() - data['Close'].pct_change().rolling(window).mean()
    return to.rolling(window).corr(ret)