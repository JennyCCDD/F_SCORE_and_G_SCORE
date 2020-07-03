from datareader import loadData
import pandas as pd
from getTradingDate import getTradingDateFromJY
dataPathPrefix = 'D:\caitong_security'  ######################修改储存数据的路径

'''
20200606: 重大修订：新增stock_dif函数，所有函数的数据框长度全部经过调整

'''

################################ 定义一个函数使得股票无法对齐时，df1的股票column与df2保持一致
def stock_dif(df1,df2):
    column_num = len(df2.index)
    df_merge = pd.DataFrame(columns=df2.index, index=df2.columns)
    dff = pd.merge(df1.T, df_merge, how='inner', left_index=True, right_index=True)
    dff = pd.merge(dff, df_merge, how='outer', left_index=True, right_index=True)
    dff = dff.iloc[:, :-column_num]
    dff = dff.dropna(how='all', axis=1)
    dff = dff.T
    return dff

def data_initial_F(para):
    tradingDateList = getTradingDateFromJY(20091231, para.endDate, ifTrade=True, Period='M')

    ################################# 涨跌停数据：1表示是涨停，-1表示跌停，0表示非涨跌停
    UpDownLimitStatus = pd.read_hdf(
        dataPathPrefix + '\DataBase\Data_AShareEODPrices\BasicDailyFactor_UpDownLimitStatus.h5')
    LimitStatus = UpDownLimitStatus.loc[para.startDate:para.endDate, :]

    ################################# 盈利类因子
    #################################### 提取ROA因子
    ROA = loadData(para='ROA')
    ROA_ = ROA.BasicDailyFactorAlpha.loc[para.startDate:para.endDate, :]
    ROA_ = stock_dif(ROA_, LimitStatus)
    ROA_[ROA_ > 0] = 1
    ROA_[ROA_ <= 0] = 0
    ROA_.index = ROA.BasicDailyFactorAlpha.loc[para.startDate:para.endDate, :].index

    #################################### CFOA因子
    CFO = loadData(para='CFO')
    CFO_ = CFO.BasicDailyFactorAlpha.loc[para.startDate:para.endDate, :]
    CFO_ = stock_dif(CFO_, LimitStatus)
    CFO_[CFO_ > 0] = 1
    CFO_[CFO_ <= 0] = 0
    CFO_.index = CFO.BasicDailyFactorAlpha.loc[para.startDate:para.endDate, :].index

    #################################### delta_ROA因子
    GROAQ = loadData(para='GROAQ')
    GROAQ_ = GROAQ.BasicDailyFactorAlpha.loc[para.startDate:para.endDate, :]
    GROAQ_ = stock_dif(GROAQ_, LimitStatus)
    GROAQ_[GROAQ_ > 0] = 1
    GROAQ_[GROAQ_ <= 0] = 0
    GROAQ_.index = GROAQ.BasicDailyFactorAlpha.loc[para.startDate:para.endDate, :].index

    # #################################### Accrual 应计量
    Accrual = CFO.BasicDailyFactorAlpha.loc[para.startDate:para.endDate, :] \
              - ROA.BasicDailyFactorAlpha.loc[para.startDate:para.endDate, :]
    Accrual_ = Accrual.copy()
    Accrual_ = stock_dif(Accrual_, LimitStatus)
    Accrual_[Accrual_ < 0] = 1
    Accrual_[Accrual_ >= 0] = 0
    Accrual_.index = Accrual.loc[para.startDate:para.endDate, :].index

    # #################################### delta_leverage因子
    BLEV = loadData(para='BLEV')
    delta_BLEV = BLEV.BasicDailyFactorAlpha.diff()
    delta_BLEV_ = delta_BLEV.loc[para.startDate:para.endDate, :]
    delta_BLEV_ = stock_dif(delta_BLEV_, LimitStatus)
    delta_BLEV_[delta_BLEV_ <= 0] = 1
    delta_BLEV_[delta_BLEV_ > 0] = 0
    delta_BLEV_.index = delta_BLEV.loc[para.startDate:para.endDate, :].index

    # ################################### delta_CurrentRatio因子
    CurrentRatio = loadData(para='CurrentRatio')
    delta_CurrentRatio = CurrentRatio.BasicDailyFactorAlpha.diff()
    delta_CurrentRatio_ = delta_CurrentRatio.loc[para.startDate:para.endDate, :]
    delta_CurrentRatio_ = stock_dif(delta_CurrentRatio_, LimitStatus)
    delta_CurrentRatio_[delta_CurrentRatio_ <= 0] = 0
    delta_CurrentRatio_[delta_CurrentRatio_ > 0] = 1
    delta_CurrentRatio_.index = delta_CurrentRatio.loc[para.startDate:para.endDate, :].index

    logreturn = pd.read_csv(para.data_path + 'logreturn.csv', index_col=0)
    logreturn = logreturn.loc[para.startDate:para.endDate, :]
    # ################################### EG_OFFER 过去一年是否增发或配股
    EG_OFFER = pd.read_csv(para.data_path + 'EG_OFFER.csv', index_col=0)
    column_num = len(logreturn.index)
    df_merge = pd.DataFrame(columns=logreturn.index, index=logreturn.columns)
    EG_OFFER_row = pd.merge(EG_OFFER, df_merge, how='inner', left_index=True, right_index=True)
    EG_OFFER = pd.merge(EG_OFFER_row, df_merge, how='outer', left_index=True, right_index=True)
    EG_OFFER = EG_OFFER.iloc[:, :-column_num]
    EG_OFFER = EG_OFFER.dropna(how='all', axis=1)
    EG_OFFER.columns = tradingDateList[:-1]
    EG_OFFER = EG_OFFER.T

    # ################################### delta_GrossProfitMargin
    GrossProfitMargin = loadData(para='GrossProfitMargin')
    delta_GrossProfitMargin = GrossProfitMargin.BasicDailyFactorAlpha.diff()
    delta_GrossProfitMargin_ = delta_GrossProfitMargin.loc[para.startDate:para.endDate, :]
    delta_GrossProfitMargin_ = stock_dif(delta_GrossProfitMargin_, LimitStatus)
    delta_GrossProfitMargin_[delta_GrossProfitMargin_ <= 0] = 0
    delta_GrossProfitMargin_[delta_GrossProfitMargin_ > 0] = 1
    delta_GrossProfitMargin_.index = delta_GrossProfitMargin.loc[para.startDate:para.endDate, :].index

    # ################################### delta_AssetsTurn
    AssetsTurn = loadData(para='AssetsTurn')
    delta_AssetsTurn = AssetsTurn.BasicDailyFactorAlpha.diff()
    delta_AssetsTurn_ = delta_AssetsTurn.loc[para.startDate:para.endDate, :]
    delta_AssetsTurn_ = stock_dif(delta_AssetsTurn_, LimitStatus)
    delta_AssetsTurn_[delta_AssetsTurn_ <= 0] = 0
    delta_AssetsTurn_[delta_AssetsTurn_ > 0] = 1
    delta_AssetsTurn_.index = delta_AssetsTurn.loc[para.startDate:para.endDate, :].index

    return ROA_,CFO_,GROAQ_,Accrual_,\
           delta_BLEV_,delta_CurrentRatio_,EG_OFFER,\
           delta_GrossProfitMargin_,delta_AssetsTurn_


def data_initial_G(para):
    ################################# 涨跌停数据：1表示是涨停，-1表示跌停，0表示非涨跌停
    UpDownLimitStatus = pd.read_hdf(
        dataPathPrefix + '\DataBase\Data_AShareEODPrices\BasicDailyFactor_UpDownLimitStatus.h5')
    LimitStatus = UpDownLimitStatus.loc[para.startDate:para.endDate, :]

    ################################# ST/ST*数据： 1表示正常
    StockTradeStatus = pd.read_hdf(
        dataPathPrefix + '\DataBase\Data_AShareTradeStatus\BasicDailyFactor_StockTradeStatus.h5')
    Status = StockTradeStatus.loc[para.startDate:para.endDate, :]

    ############################### 过去交易的天数
    StockListDateNum = pd.read_hdf(
        dataPathPrefix + '\DataBase\Data_AShareDescription\BasicDailyFactor_StockListDateNum.h5')
    listDateNum = StockListDateNum.loc[para.startDate:para.endDate, :]

    ################################# 行业分类数据
    Data_AShareIndustryClass = pd.read_hdf(
        dataPathPrefix + '\DataBase\Data_AShareIndustryClass\AShareIndustriesClassCITICSNew_FirstIndustries.h5')
    Industry = Data_AShareIndustryClass.loc[para.startDate:para.endDate, :]
    Industry = stock_dif(Industry, LimitStatus)
    Industry.index = Data_AShareIndustryClass.loc[para.startDate:para.endDate, :].index

    ################################# A股总市值
    StockTotalMV = pd.read_hdf(
        dataPathPrefix + '\DataBase\Data_AShareEODDerivativeIndicator\BasicDailyFactor_StockTotalMV.h5')
    Size = StockTotalMV.loc[para.startDate:para.endDate, :]
    Size = stock_dif(Size, LimitStatus)
    Size.index = StockTotalMV.loc[para.startDate:para.endDate, :].index

    logreturn = pd.read_csv(para.data_path + 'logreturn.csv', index_col=0)
    logreturn = logreturn.loc[para.startDate:para.endDate, :]

    tradingDateList = getTradingDateFromJY(para.startDate, para.endDate, ifTrade=True, Period='M')

    ################################# 盈利类因子
    #################################### 提取ROA因子
    ROA = loadData(para='ROA')
    ROA_ = ROA.BasicDailyFactorAlpha.loc[para.startDate:para.endDate, :]
    ROA_ = stock_dif(ROA_, LimitStatus)
    ROA_.index =  ROA.BasicDailyFactorAlpha.loc[para.startDate:para.endDate, :].index

    #################################### CFOA因子
    CFO = loadData(para='CFO')
    CFO_ = CFO.BasicDailyFactorAlpha.loc[para.startDate:para.endDate, :]
    CFO_ = stock_dif(CFO_, LimitStatus)
    CFO_.index = CFO.BasicDailyFactorAlpha.loc[para.startDate:para.endDate, :].index

    # #################################### Accrual 应计量
    Accrual = CFO.BasicDailyFactorAlpha.loc[para.startDate:para.endDate, :] \
              - ROA.BasicDailyFactorAlpha.loc[para.startDate:para.endDate, :]
    Accrual_ = Accrual.copy()
    Accrual_ = stock_dif(Accrual_, LimitStatus)
    Accrual_.index = Accrual.loc[para.startDate:para.endDate, :].index

    # #################################### ROA的方差
    ROA_row = ROA_
    ROA_VAR_list = []
    for i, Date in enumerate(tradingDateList):
        if i < para.backtestwindow: continue
        ROA_row = ROA_row.iloc[i - para.backtestwindow:i, :].dropna(axis=1, how='all')
        ROA_VAR = ROA_row.var()
        ROA_VAR_list.append(ROA_VAR)
    ROA_VAR = pd.DataFrame(ROA_VAR_list, index=tradingDateList[para.backtestwindow:])

    ###################################### 营业收入同比增量率的方差
    Sales_G_TTM = loadData(para='OperatingRevenueQYOY').BasicDailyFactorAlpha
    Sales_G_TTM = stock_dif(Sales_G_TTM, LimitStatus)
    Sales_G_TTM_list = []
    for i, Date in enumerate(tradingDateList):
        if i < para.backtestwindow: continue
        Sales_G_TTM_row = Sales_G_TTM.iloc[i - para.backtestwindow:i, :].dropna(axis=1, how='all')
        Sales_G_TTM_VAR = Sales_G_TTM_row.var()
        Sales_G_TTM_list.append(Sales_G_TTM_VAR)
    Sales_G_TTM_VAR = pd.DataFrame(Sales_G_TTM_list, index=tradingDateList[para.backtestwindow:])
    Sales_G_TTM_VAR = stock_dif(Sales_G_TTM_VAR, LimitStatus)
    Sales_G_TTM_VAR.index = tradingDateList[para.backtestwindow:]

    ##################################### R&D /总资产
    # R&D数据是用中报和年报的数据填充，4月底开始填充去年年报，9月底开始填充当前中报
    R_and_D = pd.read_csv(para.data_path + 'RandD.csv', index_col=0)
    column_num = len(logreturn.index)
    df_merge = pd.DataFrame(columns=logreturn.index, index=logreturn.columns)
    R_and_D_row = pd.merge(R_and_D.T, df_merge, how='inner', left_index=True, right_index=True)
    R_and_D_row_ = pd.merge(R_and_D_row, df_merge, how='outer', left_index=True, right_index=True)
    R_and_D_row_ = R_and_D_row_.iloc[:, :-column_num]
    R_and_D_row_ = R_and_D_row_.dropna(how='all', axis=1)
    R_and_D_row_.columns = tradingDateList[:-1]
    R_and_D_ = R_and_D_row_.T

    Total_MV = loadData(para='TotalMV').BasicDailyFactorAlpha
    RD_MV_list = []
    for i, Date in enumerate(list(R_and_D.index)):
        RD_MV = R_and_D_.iloc[i, :] / Total_MV.loc[Date, :]
        RD_MV_list.append(RD_MV)
    RD_MV_ = pd.DataFrame(RD_MV_list, index=tradingDateList[:-1])

    ###################################### 销售费用 /总资产
    # 销售费用用季报数据填充，分段时间点为4月底（去年年报），8月底用今年中报，10月底用今年三季报
    Sales = pd.read_csv(para.data_path + 'sales.csv', index_col=0)
    column_num = len(logreturn.index)
    df_merge = pd.DataFrame(columns=logreturn.index, index=logreturn.columns)
    Sales_row = pd.merge(Sales.T, df_merge, how='inner', left_index=True, right_index=True)
    Sales_row_ = pd.merge(Sales_row, df_merge, how='outer', left_index=True, right_index=True)
    Sales_row_ = Sales_row_.iloc[:, :-column_num]
    Sales_row_ = Sales_row_.dropna(how='all', axis=1)
    Sales_row_.columns = tradingDateList[:-1]
    Sales_ = Sales_row_.T

    Total_MV = loadData(para='TotalMV').BasicDailyFactorAlpha
    Sales_MV_list = []
    for i, Date in enumerate(list(Sales_.index)):
        Sales__MV = Sales_.iloc[i, :] / Total_MV.loc[Date, :]
        Sales_MV_list.append(Sales__MV)
    Sales_MV_ = pd.DataFrame(Sales_MV_list, index=tradingDateList[:-1])

    ###################################### 资本性支出 /总资产
    # 资本性支出用季报数据填充，分段时间点为4月底（去年年报），8月底用今年中报，10月底用今年三季报
    Expenditure = pd.read_csv(para.data_path + 'expenditure.csv', index_col=0)
    column_num = len(logreturn.index)
    df_merge = pd.DataFrame(columns=logreturn.index, index=logreturn.columns)
    Expenditure_row = pd.merge(Expenditure.T, df_merge, how='inner', left_index=True, right_index=True)
    Expenditure_row_ = pd.merge(Expenditure_row, df_merge, how='outer', left_index=True, right_index=True)
    Expenditure_row_ = Expenditure_row_.iloc[:, :-column_num]
    Expenditure_row_ = Expenditure_row_.dropna(how='all', axis=1)
    Expenditure_row_.columns = tradingDateList[:-1]
    Expenditure_ = Expenditure_row_.T


    Total_MV = loadData(para='TotalMV').BasicDailyFactorAlpha
    Expenditure_MV_list = []
    for i, Date in enumerate(list(Expenditure_.index)):
        Expenditure_MV = Sales_.iloc[i, :] / Total_MV.loc[Date, :]
        Expenditure_MV_list.append(Expenditure_MV)
    Expenditure_MV_ = pd.DataFrame(Expenditure_MV_list, index=tradingDateList[:-1])

    return ROA_,CFO_,Accrual_,\
           ROA_VAR,Sales_G_TTM_VAR,\
           RD_MV_,Sales_MV_,Expenditure_MV_

# class Para():
#     startDate = 20091231
#     endDate = 20200508
#     groupnum = 10
#     weightMethod = '简单加权'
#     data_path = '.\\data\\'
#     factor = 'BP'
#     result_path = '.\\result\\'
#     listnum = 121
#     backtestwindow = 60
#     pass
# para = Para()
#
# ROA_,CFO_,GROAQ_,Accrual_,\
#            delta_BLEV_,delta_CurrentRatio_,EG_OFFER,\
#            delta_GrossProfitMargin_,delta_AssetsTurn_,logreturn,\
#             LimitStatus,Status,listDateNum,Industry,Size = data_initial_F(para)
# print('factor',len(ROA_.index),len(ROA_.columns),
#       'return',len(delta_BLEV_.index),len(delta_BLEV_.columns),
#       # 'market',len(mrkDF.index),len(mrkDF.columns),
#       # 'LimitStatus',len(LimitStatus.index),len(LimitStatus.columns),
#       # 'Status',len(Status.index),len(Status.columns),
#       #  'listDateNum',len(listDateNum.index),len(listDateNum.columns),
#       # 'Industry',len(Industry.index),len(Industry.columns),
#       # 'Size',len(Size.index),len(Size.columns))
#       )
