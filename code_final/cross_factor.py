# -*- coding: utf-8 -*-
__author__ = "Mengxuan Chen"
__email__  = "chenmx19@mails.tsinghua.edu.cn"
__date__   = "20200601"

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr
from getTradingDate import getTradingDateFromJY
from utils import weightmeanFun, basic_data, stock_dif, performance, performance_anl
from MAC_RP import withoutboundary
from datareader import loadData
from G_SCORE import Gmain
import matplotlib.pyplot as plt
import warnings
from performance_output import performance
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class Para():
    startDate = 20141231 # 20091231
    endDate = 20200508
    groupnum = 3
    groupnum2 = 3
    factor = 'BP'
    factor2 = 'G'
    weightMethod = '市值加权'
    ret_calMethod = '简单' # 对数
    sample = 'out_of_sample'  # in_sample out_of_sample
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
        Factor = loadData(para = para.factor).BasicDailyFactorAlpha.loc[para.startDate:para.endDate, :]
        self.Factor2  = pd.read_csv(para.result_path+'_'+para.factor2+'.csv',index_col=0)
        self.Price, self.LimitStatus, self.Status, self.listDateNum, self.Industry, self.Size = basic_data(para)
        self.Factor = stock_dif(Factor, self.LimitStatus)
        self.Factor.index = self.LimitStatus.index.copy()
        # self.Factor2 = stock_dif(Factor2, self.LimitStatus)
        # self.Factor2.index = self.LimitStatus.index.copy()
        # _, self.Factor2 = Gmain().every_month()
        # self.Factor2.columns = self.LimitStatus.columns.copy()
        pass

    def every_month(self):
        self.dfList = []
        mktlist = []
        for i, Date in enumerate(tqdm(self.tradingDateList[:-1])):
            lastDate = self.tradingDateList[self.tradingDateList.index(Date) - 1]
            nextDate = self.tradingDateList[self.tradingDateList.index(Date) + 1]

            if para.sample == 'in_sample':
                if para.ret_calMethod == '对数':
                    self.ret = np.log(self.Price.loc[Date, :] / self.Price.loc[lastDate, :])
                elif para.ret_calMethod == '简单':
                    self.ret = self.Price.loc[Date, :] / self.Price.loc[lastDate, :] - 1
            elif para.sample == 'out_of_sample':
                if para.ret_calMethod == '对数':
                    self.ret = np.log(self.Price.loc[nextDate, :] / self.Price.loc[Date, :])
                elif para.ret_calMethod == '简单':
                    self.ret = self.Price.loc[nextDate, :] / self.Price.loc[Date, :] - 1

            self.dataFrame = pd.concat([self.Factor2.loc[Date, :],
                                        self.Factor.loc[Date, :],
                                        self.ret,
                                        self.LimitStatus.loc[Date, :],
                                        self.Status.loc[Date, :],
                                        self.listDateNum.loc[Date,:],
                                        self.Industry.loc[Date, :],
                                        self.Size.loc[Date, :]],
                                        axis=1, sort=True)
            self.dataFrame.columns = ['%s' % para.factor2,
                                        '%s' % para.factor,
                                        'RET',
                                        'LimitStatus',
                                        'Status',
                                        'listDateNum',
                                        'Industry',
                                        'Size']
            # self.dataFrame = self.dataFrame.dropna()
            self.dataFrame = self.dataFrame.loc[self.dataFrame['LimitStatus'] == 0]  # 提取非涨跌停的正常交易的数据
            self.dataFrame = self.dataFrame.loc[self.dataFrame['Status'] == 1]  # 提取非ST/ST*/退市的正常交易的数据
            self.dataFrame = self.dataFrame.loc[self.dataFrame['listDateNum'] >= para.listnum]  # 提取上市天数超过listnum的股票
            if para.fin_stock == 'no':  # 非银行金融代号41
                self.dataFrame = self.dataFrame.loc[self.dataFrame['Industry'] != 41]

            # self.dataFrame['%s' % para.factor2] = self.Factor2.iloc[i, :]
            # self.dataFrame.dropna(subset=['%s' % para.factor2],inplace=True)

            ############################################ 计算市场投资组合
            # mktlist = dataFrame.apply(weightmeanFun)
            # self.dataFrame['weight'] = self.dataFrame['Size'] / self.dataFrame['Size'].sum()
            # self.dataFrame.loc[:, 'RETmWeight'] = self.dataFrame.loc[:, 'RET'] * self.dataFrame.loc[:, 'weight']
            # self.dataFrame.loc[:, 'RETmWeight_'] = self.dataFrame['RETmWeight'].sum()
            # df = self.dataFrame.drop_duplicates(subset='RETmWeight_', keep='last', inplace=False)
            # df = df.loc[:, 'RETmWeight_']
            # mktlist.append(np.array(df)[0])

            ############################################ 计算分域投资组合
            # 对单因子进行排序打分
            self.dataFrame = self.dataFrame.sort_values(by='%s' % para.factor, ascending=False)
            # factor描述性统计分析：count, std, min, 25%, 50%, 75%, max
            self.Des = self.dataFrame['%s' % para.factor].describe()

            self.dataFrame['Score'] = ''
            eachgroup = int(self.Des['count']/ para.groupnum)
            for groupi in range(0,para.groupnum-1):
                self.dataFrame.iloc[groupi*eachgroup:(groupi+1)*eachgroup,-1] = groupi+1
            self.dataFrame.iloc[(para.groupnum-1) * eachgroup:, -1] = int(para.groupnum)

            self.meanlist = []
            for k in range(1, int(para.groupnum) + 1):
                self.dataFrame2 = self.dataFrame.loc[self.dataFrame['Score'] == k, :]
                self.dataFrame2.sort_values(by='%s' % para.factor2, inplace=True, ascending=False)
                Des2 = self.dataFrame2['%s' % para.factor2].describe()

                self.dataFrame2['Score2'] = ''
                eachgroup2 = int(Des2['count'] / para.groupnum2)
                for groupj in range(0, int(para.groupnum2)-1):
                    self.dataFrame2.iloc[groupj * eachgroup2:(groupj + 1) * eachgroup2, -1] = groupj+1
                self.dataFrame2.iloc[(para.groupnum2 -1) * eachgroup2:, -1] = int(para.groupnum2)

                if para.weightMethod == '简单加权':
                    self.meanlist.append(np.array(self.dataFrame2.groupby('Score2')['RET'].mean()))
                elif para.weightMethod == '市值加权':
                    meanlist_group = []
                    for kk in range(1, para.groupnum2 + 1):
                        self.dataFrame2_ = self.dataFrame2.loc[self.dataFrame2['Score2'] == kk, :]
                        meanlist_g = weightmeanFun(self.dataFrame2_)
                        meanlist_group.append(meanlist_g)
                    self.meanlist.append(meanlist_group)
            self.meanDf = pd.DataFrame(self.meanlist).unstack()
            self.dfList.append(self.meanDf)
        self.df = pd.DataFrame(self.dfList, index=self.tradingDateList[:-1])
        return self.df

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
        for oneleg in tqdm(range(len(self.df.columns))):
            portfolioDF = pd.DataFrame()
            portfolioDF['ret'] = self.df.iloc[:, oneleg]
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
        compare.index = self.df.index
        plt.plot(range(len(compare.iloc[1:, 1])),
                 compare.iloc[1:, :]
                 )
        plt.title(para.factor +'_'+para.factor2)
        # plt.xticks([0, 25, 50, 75, 100, 125],
        #            ['2009/12/31', '2011/01/31', '2013/02/28', '2015/03/31', '2017/04/30', '2020/04/30'])
        # plt.grid(True)
        # plt.xlim((0, 125))
        plt.xticks([0, 25, 50, 65],
                   ['2014/12/31', '2016/12/30', '2018/12/31', '2020/04/30'])
        plt.grid(True)
        plt.xlim((0, 65))
        plt.legend()
        plt.savefig(para.result_path + para.factor +'_' + para.factor2 + '_' + para.weightMethod + '_performance_nav.png')
        plt.show()
        return performanceDf, compare


if __name__ == "__main__":
    para = Para()
    main_fun = main()
    mean = main_fun.every_month()
    mean.to_csv(para.result_path + 'mean_' + para.factor +'_' + para.factor2 + '_.csv')
    test, test_nav = main_fun.portfolio_test()
    print(test)
    test.to_csv(para.result_path +'_'+ para.factor  +'_' + para.factor2 +  '_' +para.weightMethod
                +'_performance.csv')
    test_nav.to_csv(para.result_path + '_'+ para.factor  +'_' + para.factor2 +  '_' +para.weightMethod
                +'_result.csv')