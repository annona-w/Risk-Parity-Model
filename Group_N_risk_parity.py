# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from scipy import stats
import math
from tabulate import tabulate
from scipy.optimize import minimize
import empyrical


# -------------------------------------------------------------Data pre-processing-----------------------------------------------------------------
# load the data in and drop the stocks which contains "NaN"
data = pd.read_csv("F:\\新加坡国立大学\\FE5107 Risk Analyses And Management\\Final project\\closePrice2006to2020.csv",engine="python",index_col = 0)

data.plot(legend = False,figsize=(16,8))
plt.show()
# ax = data.plot(legend = False,figsize=(16,8))
# ax.figsize=(12,16)
# fig = ax.get_figure()
#fig.savefig('SZ50.jpg')
df = data.dropna(axis=1,how="any")
print(df.shape)

# show the first five data
print(df.iloc[:5])
# show the left stocks
print(df.columns)

# calculate the log return and drop the first row which is empty
df1=np.log(df).diff(axis=0)
df2=df1.dropna(axis=0,how="any")

# generate a time_slot list for each month
time_period=[]
for y in range(6,21):
    for m in range(1,13):
        if y<10:
            if m < 10:
                period="200"+str(y)+"-"+"0"+str(m)
            else:
                period="200"+str(y)+"-"+str(m)
        else:
            if m < 10:
                period="20"+str(y)+"-"+"0"+str(m)
            else:
                period="20"+str(y)+"-"+str(m)
        time_period.append(period)
# print(time_period)

# generate two empty dict for storing each month's data
return_dict={}
cov_dict = {}
for i in range(len(time_period)):
    if i == len(time_period)-1:
        temp=df2[df2.index>time_period[i]]
    else:
        temp=df2[np.logical_and(df2.index>time_period[i],df2.index<time_period[i+1])]
    cov_dict[time_period[i]]=np.cov(temp,rowvar=False)
    return_dict[time_period[i]]=temp.apply(lambda x:x.sum(),axis=0)

# -------------------------------------------------------------Apply risk-parity model-----------------------------------------------------------------
# annualized the return and volatility
V = np.matrix(cov_dict[time_period[1]])*144
R = (np.matrix(return_dict[time_period[1]]).T)*12

def calculate_portfolio_var(w,V):
    # 计算组合风险的函数
    w = np.matrix(w)
    return (w*V*w.T)[0,0]

def calculate_risk_contribution(w,V):
    # 计算单个资产对总体风险贡献度的函数
    w = np.matrix(w)
    sigma = np.sqrt(calculate_portfolio_var(w,V))
    # 边际风险贡献
    MRC = V*w.T
    # 风险贡献
    RC = np.multiply(MRC,w.T)/sigma
    return RC

def risk_budget_objective(x,pars):
    # 计算组合风险
    V = pars[0]# 协方差矩阵
    x_t = pars[1] # 组合中资产预期风险贡献度的目标向量
    sig_p =  np.sqrt(calculate_portfolio_var(x,V)) # portfolio sigma
    risk_target = np.asmatrix(np.multiply(sig_p,x_t))
    asset_RC = calculate_risk_contribution(x,V)
    J = sum(np.square(asset_RC-risk_target.T))[0,0] # sum of squared error
    return J

def total_weight_constraint(x):
    return np.sum(x)-1.0

def long_only_constraint(x):
    return x

def calcu_w(x):
    w0 = np.ones([1,R.shape[0]])/R.shape[0]
    #x_t = [1/22,1/22,...,1/22] # 目标是让22个资产风险贡献度相等，即都为1/22
    x_t = x
    cons = ({'type': 'eq', 'fun': total_weight_constraint},
    {'type': 'ineq', 'fun': long_only_constraint})
    res = minimize(risk_budget_objective, w0, args=[V,x_t], method='SLSQP',constraints=cons, options={'disp': True})
    #w_rb = np.asmatrix(res.x)
    w_rb = res.x
    return w_rb

weight_array=[]
for i in range(len(time_period)):
    V = np.matrix(cov_dict[time_period[i]])*144
    R = (np.matrix(return_dict[time_period[i]]).T)*12
    weight_array.append(calcu_w(np.ones([1,R.shape[0]])/R.shape[0]))
weight_matrix=np.mat(weight_array)

print(weight_matrix)

# -------------------------------------------------------------Benchmark calculation-----------------------------------------------------------------
mon_rt = pd.DataFrame.from_dict(return_dict, orient = "index")
# print(mon_rt)
mon_rt.to_csv("mon_rt.csv", index = True, header = True)
mon_rt['price_weighted_return'] = mon_rt.mean(axis = 1)
print(mon_rt)

plt.clf()
plt.hist(mon_rt['price_weighted_return'], bins = 40, density = True)
plt.show()

fig1 = qqplot(mon_rt['price_weighted_return'], line='q', dist = stats.norm, fit = True)
plt.show()

mon_rt_sort_values = mon_rt.sort_values('price_weighted_return', ascending = True)
print(mon_rt_sort_values)

# -------------------------------------------------------------Calculate VaR and ES-----------------------------------------------------------------
print(len(mon_rt_sort_values['price_weighted_return']))

ValueLocForPercentile_90 = math.ceil(len(mon_rt_sort_values['price_weighted_return']) * (1 - 0.9))
VaR_90 = mon_rt_sort_values['price_weighted_return'][ValueLocForPercentile_90-1]
ES_90 = mon_rt_sort_values['price_weighted_return'][0: (ValueLocForPercentile_90-1)].mean()
print("VaR_90",VaR_90)
print("ES_90:",ES_90)

ValueLocForPercentile_95 = math.ceil(len(mon_rt_sort_values['price_weighted_return']) * (1 - 0.95))
VaR_95 = mon_rt_sort_values['price_weighted_return'][ValueLocForPercentile_95-1]
ES_95 = mon_rt_sort_values['price_weighted_return'][0: (ValueLocForPercentile_95-1)].mean()
print("VaR_95",VaR_95)
print("ES_95:",ES_95)

ValueLocForPercentile_99 = math.ceil(len(mon_rt_sort_values['price_weighted_return']) * (1 - 0.99))
VaR_99 = mon_rt_sort_values['price_weighted_return'][ValueLocForPercentile_99-1]
ES_99 = mon_rt_sort_values['price_weighted_return'][0: (ValueLocForPercentile_99-1)].mean()
print("VaR_99",VaR_99)
print("ES_99:",ES_99)

print(tabulate([["90%", VaR_90, ES_90],["95%", VaR_95, ES_95], ["99%", VaR_99, ES_99]], headers = ['Confidence Level', 'Value at Risk', 'Expected Shortfall']))

# -------------------------------------------------------------Metrics-----------------------------------------------------------------
'''
weight_matrix: 权重矩阵  格式：pd.Dataframe
return_matrix：收益率矩阵 格式：pd.Dataframe
significant_level：显著性水平 格式: float (0~1)
'''
def calIndicator(weight_matrix, return_matrix, significant_level):

    tmp_return_matrix = return_matrix[return_matrix.index.isin(weight_matrix.index)]
    tmp_return_matrix = tmp_return_matrix * weight_matrix
    myReturn = tmp_return_matrix.apply(lambda x: x.sum(), 1)
    net_value, annual_ret, annual_vol, ret_to_risk, win_rate, max_drawdown = BKtest_indicators(myReturn)
    VaR = np.percentile(myReturn.dropna(), significant_level*100)
    ES = myReturn[myReturn <= VaR].mean()
    print('回测结果：')
    print('年化收益率', annual_ret)
    print('年化波动率', annual_vol)
    print('收益风险比', ret_to_risk)
    print('胜率', win_rate)
    print('最大回撤', max_drawdown)
    print('VaR', VaR)
    print('ES', ES)
    print('=======================')
    return

'''
my_return: 组合月收益率序列
'''
def BKtest_indicators(my_return):
    annual_ret = empyrical.annual_return(my_return, period='monthly')
    annual_vol = np.std(my_return) * np.sqrt(12)
    ret_to_risk = empyrical.annual_return(my_return, period='monthly') / (np.std(my_return) * np.sqrt(12))
    win_rate = np.sum(my_return > 0) / len(my_return)  # 胜率
    net_value = (1 + my_return).cumprod()
    max_drawdown = np.min(net_value / net_value.cummax() - 1)  # 最大回撤
    return net_value, annual_ret, annual_vol, ret_to_risk, win_rate, max_drawdown

mon_rt = mon_rt.drop(columns = ['price_weighted_return'])
weight_matrix_benchmark = np.ones(mon_rt.shape)*1/mon_rt.shape[1]
print(mon_rt.columns)
weight_matrix = pd.DataFrame(weight_matrix, columns = mon_rt.columns, index = mon_rt.index )
weight_matrix_benchmark = pd.DataFrame(weight_matrix_benchmark, columns = mon_rt.columns, index = mon_rt.index )

# confidence level here is 99%
print("The confidence level here is 99%:")
calIndicator(weight_matrix, mon_rt, 0.01)
calIndicator(weight_matrix_benchmark, mon_rt, 0.01)
print("\n")

# confidence level here is 95%
print("The confidence level here is 95%:")
calIndicator(weight_matrix, mon_rt, 0.05)
calIndicator(weight_matrix_benchmark, mon_rt, 0.05)
print("\n")

# confidence level here is 90%
print("The confidence level here is 90%:")
calIndicator(weight_matrix, mon_rt, 0.1)
calIndicator(weight_matrix_benchmark, mon_rt, 0.1)
print("\n")

# -------------------------------------------------------------Risk parity in stressed market-----------------------------------------------------------------
print("The result below is from 2007-10 to 2008-:")
start_index = list(weight_matrix.index).index('2007-10')
end_index = list(weight_matrix.index).index('2008-12')
calIndicator(weight_matrix.iloc[start_index : end_index,:], mon_rt, 0.05)
calIndicator(weight_matrix_benchmark.iloc[start_index : end_index,:], mon_rt, 0.05)
print("\n")

print("The result below is from 2015-05 to 2016-04:")
start_index = list(weight_matrix.index).index('2015-05')
end_index = list(weight_matrix.index).index('2016-04')
calIndicator(weight_matrix.iloc[start_index : end_index,:], mon_rt, 0.05)
calIndicator(weight_matrix_benchmark.iloc[start_index : end_index,:], mon_rt, 0.05)
print("\n")