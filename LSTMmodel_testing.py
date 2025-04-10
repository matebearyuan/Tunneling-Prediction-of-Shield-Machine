from keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load

# 数据加载和预处理
Data = pd.read_csv('D:\\pythonProject\\PyxLSTM-main\\TBM\\c_510_726.csv')
Data1 = Data[['geology_1', 'geology_2', 'geology_3', 'geology_4', 'geology_5', 'geology_6',
              'cutter speed', 'total thrust', 'grouting pressure', 'sync grouting volume',
              'stock solution ratio', 'foam pressure', 'expansion rate', 'dosage',
              'cutter torque', 'tunneling speed']]
Data1_clean = Data1.dropna()
filtered_data = Data1_clean[Data1_clean['geology_4']>0.6]

X = filtered_data.iloc[:, :-2]  # 特征列
Y = filtered_data[['tunneling speed','cutter torque']]  # 标签列
x = X.values.astype('float32')
y = Y.values.astype('float32')
print(x.shape[1])
print(y.shape)
# 数据标准化
scaler = load('xlstm.joblib')
scaled_features = scaler.fit_transform(x)
scaled_target = scaler.fit_transform(y.reshape(-1, 2))
reshaped_scaled_features = scaled_features.reshape(scaled_features.shape[0], 1, scaled_features.shape[1])
model = load_model("STM_LSTMs.h5")
y_pred = model.predict(reshaped_scaled_features)

orignal_y_pred = scaler.inverse_transform(y_pred)
orignal_y_test = y              #scaler.inverse_transform(y_test)

means = np.mean(np.abs((orignal_y_test - orignal_y_pred)/orignal_y_test),axis=0)
maxi = np.max(np.abs((orignal_y_test - orignal_y_pred)),axis=0)
mini = np.min(np.abs((orignal_y_test - orignal_y_pred)),axis=0)
print(f'mean  Error: {means}')
print(f'max  Error: {maxi}')
print(f'mini  Error: {mini}')

plt.figure(3, dpi=120, figsize=(8, 6))
plt.rc('font', family='Times New Roman')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(linestyle='--')
#plt.xlabel("ring", fontsize=18)
plt.ylabel("tunneling speed", fontsize=18)
plt.plot(orignal_y_test[:,0], '-',color='gray', linewidth=2.5, label='ground truth')
plt.plot(orignal_y_pred[:,0], '-', color='orange',linewidth=2.5, label='prediction')

plt.legend(fontsize=16, loc='upper left')
# plt.savefig('multivar_training.jpg', dpi=300)
plt.show()

plt.figure(4, dpi=120, figsize=(8, 6))
plt.rc('font', family='Times New Roman')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(linestyle='--')
#plt.xlabel("ring", fontsize=18)
plt.ylabel("cutter torque", fontsize=18)
plt.plot(orignal_y_test[:,1], '-', color='gray', linewidth=2.5, label='ground truth')
plt.plot(orignal_y_pred[:,1], '-', color='green', linewidth=2.5, label='prediction')

plt.legend(fontsize=16, loc='upper left')
plt.show()