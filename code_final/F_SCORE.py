# -*- coding: utf-8 -*-
__author__ = "Mengxuan Chen"
__email__  = "chenmx19@mails.tsinghua.edu.cn"
__date__   = "20200701"


import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr
from getTradingDate import getTradingDateFromJY
from utils import weightmeanFun, basic_data, stock_dif, performance, performance_anl
from MAC_RP import withoutboundary
from data_initial import data_initial_F
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class Para():
    startDate = 20091231
    endDate = 20200508
    groupnum = 4
    weightMethod = '简单加权'
    ret_calMethod = '简单' # 对数
    factor = 'F'
    sample = 'in_sample' # in_sample out_of_sample
    data_path = '.\\data\\'
    result_path = '.\\result\\'
    listnum = 121
    backtestwindow = 60
    fin_stock = 'no'
    dataPathPrefix = 'D:\caitong_security'
    pass

class main():
    def __init__(self):
        self.tradingDateList = getTradingDateFromJY(para.startDate, para.endDate, ifTrade=True, Period='M')
        pass
    def data_get(self):
        self.ROA_, self.CFO_, self.GROAQ_, self.Accrual_, \
        self.delta_BLEV_, self.delta_CurrentRatio_, self.EG_OFFER, self.delta_GrossProfitMargin_, \
        self.delta_AssetsTurn_\
        = data_initial_F(para)
        self.Price, self.LimitStatus, self.Status, self.listDateNum, self.Industry, self.Size = basic_data(para)
        return

    def every_month(self):
        self.Price = self.Price.loc[para.startDate:para.endDate, :]
        F_list = []
        meanlist = []
        for i,Date in enumerate(tqdm(self.tradingDateList[:-1])):
            lastDate = self.tradingDateList[self.tradingDateList.index(Date) - 1]
            nextDate = self.tradingDateList[self.tradingDateList.index(Date) + 1]

            if para.sample == 'in_sample':
                # use different method to calculate the return
                # logreturn for short time period and simple return calculation for long time period
                if para.ret_calMethod == '对数':
                    self.ret = np.log(self.Price.loc[Date, :] / self.Price.loc[lastDate, :])
                elif para.ret_calMethod == '简单':
                    self.ret = self.Price.loc[Date, :] / self.Price.loc[lastDate, :] - 1
            elif para.sample == 'out_of_sample':
                if para.ret_calMethod == '对数':
                    self.ret = np.log(self.Price.loc[nextDate, :] / self.Price.loc[Date, :])
                elif para.ret_calMethod == '简单':
                    self.ret = self.Price.loc[nextDate, :] / self.Price.loc[Date, :] - 1

            dataFrame = pd.concat([
                # 盈利类因子
                self.ROA_.loc[Date, :], self.CFO_.loc[Date, :], self.GROAQ_.loc[Date, :], self.Accrual_.loc[Date, :],
                # 杠杆和融资
                self.delta_BLEV_.loc[Date, :], self.EG_OFFER.loc[Date,:], self.delta_CurrentRatio_.loc[Date, :],
                # 经营效率
                self.delta_GrossProfitMargin_.loc[Date, :], self.delta_AssetsTurn_.loc[Date, :],
                self.ret,
                self.LimitStatus.loc[Date, :], self.Status.loc[Date, :], self.listDateNum.loc[Date, :],
                self.Industry.loc[Date, :], self.Size.loc[Date, :]], axis=1, sort=True)

            dataFrame = dataFrame.reset_index()
            dataFrame.columns = ['stockid',
                                 'ROA',
                                 'CFOA',
                                 'delta_ROA',
                                 'Accrual',
                                 'delta_leverage',
                                 'EQ_offer',
                                 'delta_liquidity',
                                 'delta_margin',
                                 'delta_turn',
                                 'RET',
                                 'LimitStatus',
                                 'Status',
                                 'listDateNum',
                                 'Industry',
                                 'Size']
            dataFrame['F_SCORE'] = dataFrame.loc[:, 'ROA':'delta_turn'].apply(lambda x: x.sum(), axis=1)
            F = dataFrame['F_SCORE']
            F_list.append(F)

            dataFrame = dataFrame.dropna()
            dataFrame = dataFrame.loc[dataFrame['LimitStatus'] == 0]  # 提取非涨跌停的正常交易的数据
            dataFrame = dataFrame.loc[dataFrame['Status'] == 1]  # 提取非ST/ST*/退市的正常交易的数据
            dataFrame = dataFrame.loc[dataFrame['listDateNum'] >= para.listnum]  # 提取上市天数超过listnum的股票
            if para.fin_stock == 'no': # 非银行金融代号41
                dataFrame = dataFrame.loc[dataFrame['Industry'] != 41]


            dataFrame.drop(['ROA',
                            'CFOA',
                            'delta_ROA',
                            'Accrual',
                            'delta_leverage',
                            'EQ_offer',
                            'delta_liquidity',
                            'delta_margin',
                            'delta_turn',
                            'LimitStatus',
                            'Status',
                            'listDateNum'
                            ],
                            axis=1, inplace=True)

            # 对单因子进行排序打分
            dataFrame = dataFrame.sort_values(by='F_SCORE', ascending=False)  # 降序排列
            Des = dataFrame['F_SCORE'].describe()
            dataFrame['Score'] = ''
            eachgroup = int(Des['count'] / para.groupnum)
            for groupi in range(0, para.groupnum - 1):
                dataFrame.iloc[groupi * eachgroup:(groupi + 1) * eachgroup, -1] = groupi + 1
            dataFrame.iloc[(para.groupnum - 1) * eachgroup:, -1] = para.groupnum

            dataFrame['Score'].type = np.str
            # simple average weights
            if para.weightMethod == '简单加权':
                meanlist.append(np.array(dataFrame.groupby('Score')['RET'].mean()))
            # weights according to the size( market value of the stocks)
            elif para.weightMethod == '市值加权':
                meanlist_group = []
                for groupi in range(0, para.groupnum):
                    dataFrame_ = dataFrame.iloc[groupi * eachgroup:(groupi + 1) * eachgroup, :]
                    meanlist_g = weightmeanFun(dataFrame_)
                    meanlist_group.append(meanlist_g)
                meanlist.append(meanlist_group)

            # weights according to risk control method
            # MAC：Markovitz Mean-Variance Model
            # RP: risk parity
            # https://blog.csdn.net/weixin_42294255/article/details/103836548
            elif para.weightMethod == 'MAC' or 'RP':

                NAV = self.Price.loc[:nextDate, :]

                meanlist_group = []
                for groupi in range(1, para.groupnum + 1):
                    data_ = dataFrame.loc[dataFrame['Score'] == groupi]
                    data_ = data_.set_index('stockid')
                    data_nav = pd.merge(data_, NAV.T, how='inner', left_index=True, right_index=True)
                    ret = data_nav['RET']
                    data_nav = data_nav.drop(columns=['factor',
                                                      'RET',
                                                      'Status',
                                                      'listDateNum',
                                                      'LimitStatus',
                                                      'Industry',
                                                      'Size',
                                                      'Score'])
                    data_nav = data_nav.T
                    data_nav['date'] = data_nav.index.copy()
                    data_nav['date'] = data_nav['date'].apply(lambda x: str(x))
                    data_nav['date'] = data_nav['date'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d'))
                    data_nav = data_nav.set_index('date')
                    weights = withoutboundary(data_nav, period=1, rollingtime=21,
                                              method=para.weightMethod)
                    menlist_g = np.sum(np.array(weights) * np.array(ret))
                    # append all of the groups
                    meanlist_group.append(menlist_g)

                # append all of the days
                meanlist.append(meanlist_group)

            # self.meanDf is the portfolio monthly return timeseries list for each group
        self.meanDf = pd.DataFrame(meanlist, index=self.tradingDateList[:-1])
        self.FDf = pd.DataFrame(F_list,index = self.tradingDateList[:-1])

        return self.meanDf, self.FDf

    def portfolio_test(self):
        # portfolio_test function is to calculate the index of the porfolio
        # https://blog.csdn.net/weixin_42294255/article/details/103836548
        sharp_list = []
        ret_list = []
        std_list = []
        mdd_list = []
        r2var_list = []
        cr2var_list = []
        compare = pd.DataFrame()
        for oneleg in tqdm(range(len(self.meanDf.columns))):
            portfolioDF = pd.DataFrame()
            portfolioDF['ret'] = self.meanDf.iloc[:, oneleg]
            portfolioDF['nav'] = (portfolioDF['ret'] + 1).cumprod()
            performance_df = performance(portfolioDF, para)
            # performance_df_anl = performance_anl(portfolioDF,para)
            sharp_list.append(np.array(performance_df.iloc[:, 0].T)[0])
            ret_list.append(np.array(performance_df.iloc[:, 1].T)[0])
            std_list.append(np.array(performance_df.iloc[:, 2].T)[0])
            mdd_list.append(np.array(performance_df.iloc[:, 3].T)[0])
            r2var_list.append(np.array(performance_df.iloc[:, 4].T)[0])
            cr2var_list.append(np.array(performance_df.iloc[:, 5].T)[0])
            compare[str(oneleg)] = portfolioDF['nav']
        performanceDf = pd.concat([pd.Series(sharp_list),
                                   pd.Series(ret_list),
                                   pd.Series(std_list),
                                   pd.Series(mdd_list),
                                   pd.Series(r2var_list),
                                   pd.Series(cr2var_list)],
                                  axis=1, sort=True)
        performanceDf.columns = ['Sharp',
                                 'RetYearly',
                                 'STD',
                                 'MDD',
                                 'R2VaR',
                                 'R2CVaR']
        compare.index = self.meanDf.index
        plt.plot(range(len(compare.iloc[1:, 1])),
                 compare.iloc[1:, :])
        plt.title(para.factor)
        plt.xticks([0, 25, 50, 75, 100, 125],
                   ['2009/12/31', '2011/01/31', '2013/02/28', '2015/03/31', '2017/04/30', '2020/04/30'])
        plt.grid(True)
        plt.xlim((0, 125))
        plt.legend()
        plt.savefig(para.result_path + para.factor + '_' + para.weightMethod + '_performance_nav.png')
        plt.show()
        return performanceDf, compare

    def F_test(self):
        F_Des = self.FDf.describe()
        print(F_Des)
        return F_Des

if __name__ == "__main__":
    para = Para()
    main_fun = main()
    data_ = main_fun.data_get()
    mean, F = main_fun.every_month()
    print(mean, F)
    mean.to_csv(para.result_path+'mean_'+para.factor+'.csv')
    F.to_csv(para.result_path +'_'+ para.factor + '.csv')
    F_ = main_fun.F_test()
    test, test_nav = main_fun.portfolio_test()
    print(test)
    test.to_csv(para.result_path +'_'+ para.factor + '_' +para.weightMethod
                +'_performance.csv')
    test_nav.to_csv(para.result_path + '_'+ para.factor + '_' +para.weightMethod
                +'_result.csv')