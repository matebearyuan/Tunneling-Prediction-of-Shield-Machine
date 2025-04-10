import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
Data = pd.read_csv('leftline_time.csv')
print(Data.head())
print(Data.shape)
Data_new = Data.drop(Data[Data['s'] >=50].index)
Data_new = Data_new.drop(Data_new[Data_new['a'] >=50].index)
print(Data_new.head())
fig = plt.figure(figsize=(8, 6))
plt.plot(Data['ring'],Data['s'], 'o',color='Blue')
plt.xlabel('ring')
plt.title('stop_time')
plt.show()
fig = plt.figure(figsize=(8, 6))
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 确保中文能显示
bins=range(0,51,1)
values, bins, patches=plt.hist(Data_new['s'], bins=bins, color='Blue', ec='black')
plt.xlabel('time (hour)')
plt.title('stop_time')
plt.tight_layout()  # 自动调整
plt.show()

fig = plt.figure(figsize=(8, 6))
plt.plot(Data['ring'],Data['a'], 'o',color='Blue')
plt.xlabel('ring')
plt.title('assembly_time')
plt.show()
fig = plt.figure(figsize=(8, 6))
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 确保中文能显示
bins=range(0,51,1)
plt.hist(Data_new['a'], bins=bins, color='Blue', ec='black')
plt.xlabel('time (hour)')
#axs[0].set_ylabel('assembly_time')
plt.title('assembly_time')
plt.tight_layout()  # 自动调整
plt.show()

# 定义高斯函数模型
def gaussian(x, a, b, c):
    return a * np.exp(-((x - b) ** 2) / (2 * c**2))
# 调用curve_fit()函数进行拟合
x= range(0,50,1)
print(x)
print(values)
y=values
df1 = pd.DataFrame({'hour': x})
df2 = pd.DataFrame({'value': values})
result = pd.concat([df1, df2], axis=1)
print(result)

result.to_csv("output.txt", index=False, sep='\t')

params, _ = curve_fit(gaussian, x, y)
a_fit, b_fit, c_fit = params

# 输出拟合结果
print("拟合参数：")
print("amplitude:", a_fit)
print("mean:", b_fit)
print("stddev:", c_fit)
# 绘制原始数据点及拟合曲线
plt.figure(5,figsize=(8, 6))
plt.scatter(x, y, label='Data')
plt.plot(x, gaussian(x, a_fit, b_fit, c_fit), 'r', label='Fit')
plt.legend()
plt.show()
