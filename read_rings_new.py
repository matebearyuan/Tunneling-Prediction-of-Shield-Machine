import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime,timedelta
import matplotlib.pyplot  as mqt
import seaborn as sea
import scipy.signal as signal
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from keras import layers
from keras.models import Sequential
from joblib import dump, load



Data = pd.read_excel('left_new.xls')
Data1 = pd.read_excel('right_new.xls')
Data=Data.append(Data1,ignore_index=True)
Data['tunnelling hours'] = Data['tunnelling time'].apply(lambda x: x.hour+x.minute/60+x.minute/3600 if pd.notna(x) else None)
Data['assembly hours'] = Data['assembly time'].apply(lambda x: x.hour+x.minute/60+x.minute/3600 if pd.notna(x) else None)
Data1 = Data[['geology_1','geology_2','geology_3','geology_4','geology_5','geology_6', 'grouting pressure', 'sync grouting volume', 'stock solution ratio', 'foam pressure', 'expansion rate', 'dosage', 'excavated', 'top pressure', 'total thrust_u', 'cutter speed_u', 'tunneling speed_u', 'cutter torque_u']]
Data1_clean=Data1.dropna()
X = Data1_clean[['geology_1','geology_2','geology_3','geology_4','geology_5','geology_6', 'grouting pressure', 'sync grouting volume',
                 'stock solution ratio', 'foam pressure', 'expansion rate', 'dosage',
                 'total thrust_u', 'cutter speed_u']]
    # the othe remaining columns are selected in X
Y = Data1_clean[['tunneling speed_u']]

x = X.values
y = Y.values

# 3. 数据标准化
scaler = StandardScaler()
x = scaler.fit_transform(x)
dump(scaler, 'std_scaler.bin', compress=True)
# 4. 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# 5. 构建 CNN 模型
model = Sequential([
    layers.Reshape((x_train.shape[1], 1), input_shape=(x_train.shape[1],)),
    layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    #layers.Dropout(0.1),
    layers.MaxPooling1D(pool_size=2),
    layers.LSTM(32, activation='relu'),
    #layers.Dropout(0.50),
    #layers.MaxPooling1D(pool_size=2),
    layers.Flatten(),
    layers.Dense(units=64, activation='relu'),
    layers.Dropout(0.1),
    layers.LayerNormalization(epsilon=1e-6),
    layers.Dense(units=1)  # 输出层，因为是回归任务，所以不使用激活函数
])
# 6. 编译模型
model.compile(optimizer='adam', loss='mean_squared_error')

# 7. 训练模型
history = model.fit(x_train, y_train, epochs=180, batch_size=32, validation_data=(x_test, y_test))

print("Saving model to disk \n")
model.save("STM.h5")
plt.figure(2, dpi=120, figsize=(8, 6))
plt.rc('font', family='Times New Roman')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(linestyle='--')
plt.xlabel("epoch", fontsize=18)
plt.ylabel("loss", fontsize=18)
plt.plot(history.history['loss'], 'g-', linewidth=3, label='training_loss')
plt.plot(history.history['val_loss'], 'b-', linewidth=3, label='test_loss')

plt.legend(fontsize=16, loc='upper left')
# plt.savefig('multivar_training.jpg', dpi=300)
plt.show()
# 8. 使用模型进行预测
y_pred = model.predict(x_test)
print("actual:",y_test.shape)
print("predcit:",y_pred.shape)
# 9. 评估模型性能
mse = np.mean((y_test - y_pred)**2)
print(f'Mean Squared Error: {mse}')

error = np.abs(y_test - y_pred)
print("error:",error.shape)
percentage_error = np.divide(error,np.abs(y_test))*100
print("误差百分比：",percentage_error.shape)

plt.figure(3, dpi=120, figsize=(8, 6))
plt.rc('font', family='Times New Roman')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(linestyle='--')
plt.xlabel("ring", fontsize=18)
plt.ylabel("tunneling speed", fontsize=18)
plt.plot(y_test, 'k-', linewidth=3, label='ground truth')
plt.plot(y_pred.squeeze(), 'b-', linewidth=3, label='prediction')

plt.legend(fontsize=16, loc='upper left')
# plt.savefig('multivar_training.jpg', dpi=300)
plt.show()

plt.figure(4, dpi=120, figsize=(8, 6))
plt.rc('font', family='Times New Roman')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(linestyle='--')
plt.xlabel("ring", fontsize=18)
plt.ylabel("percentage_error_prediction", fontsize=18)
plt.plot(percentage_error, 'b-', linewidth=3, label='prediction')

plt.legend(fontsize=16, loc='upper left')
plt.show()

'''

plt.figure(1, dpi=120, figsize=(8, 6))
plt.rc('font', family='Times New Roman')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(linestyle='--')
plt.xlabel("ring", fontsize=18)
plt.ylabel("geology", fontsize=18)
plt.plot(Data['ring number'], Data['geology'],'b',linewidth=2)
plt.show()

plt.figure(2, dpi=120, figsize=(8, 6))
plt.rc('font', family='Times New Roman')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(linestyle='--')
plt.xlabel("ring", fontsize=18)
plt.ylabel("grouting pressure", fontsize=18)
plt.plot(Data['ring number'], Data['grouting pressure'],'g',linewidth=2)
plt.show()
plt.figure(3, dpi=120, figsize=(8, 6))
plt.rc('font', family='Times New Roman')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(linestyle='--')
plt.xlabel("ring", fontsize=18)
plt.ylabel("sync grouting volume", fontsize=18)
plt.plot(Data['ring number'], Data['sync grouting volume'],'g',linewidth=2)
plt.show()
plt.figure(4, dpi=120, figsize=(8, 6))
plt.rc('font', family='Times New Roman')

# Calculate the correlation matrix
pearson =Data1.corr(method="pearson")
sea.heatmap(pearson,annot=True, cmap='coolwarm', fmt=".2f")
mqt.show()

kendall_correlation = Data1.corr(method='kendall')
sea.heatmap(kendall_correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()

spearman_correlation  = Data1.corr(method='spearman')
sea.heatmap(spearman_correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()





plt.figure(11, dpi=120, figsize=(8, 6))
plt.rc('font', family='Times New Roman')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(linestyle='--')
plt.xlabel("ring", fontsize=18)
plt.ylabel("hour", fontsize=18)
plt.plot(Data['ring number'], Data['tunnelling hours'],'dodgerblue',linewidth=2,label='tunnelling time')
plt.plot(Data['ring number'], Data['assembly hours'],'hotpink',linewidth=2,label='assembly time')
plt.legend(fontsize=16)
plt.show()

plt.figure(10, dpi=120, figsize=(8, 6))
plt.rc('font', family='Times New Roman')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(linestyle='--')
plt.xlabel("ring", fontsize=18)
plt.ylabel("cutter torque", fontsize=18)
plt.plot(Data['ring number'], Data['cutter torque_d'],'lightcoral',linewidth=2,label='cutter torque down')
plt.plot(Data['ring number'], Data['cutter torque_u'],'darkred',linewidth=2,label='cutter torque up')
plt.legend()
plt.show()
plt.figure(9, dpi=120, figsize=(8, 6))
plt.rc('font', family='Times New Roman')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(linestyle='--')
plt.xlabel("ring", fontsize=18)
plt.ylabel("tunneling speed", fontsize=18)
plt.plot(Data['ring number'], Data['tunneling speed_d'],'limegreen',linewidth=2,label='tunneling speed down')
plt.plot(Data['ring number'], Data['tunneling speed_u'],'darkgreen',linewidth=2,label='tunneling speed up')
plt.legend()
plt.show()
plt.figure(7, dpi=120, figsize=(8, 6))
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', family='Times New Roman')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(linestyle='--')
plt.xlabel("ring", fontsize=18)
plt.ylabel("cutter speed", fontsize=18)
#plt.plot(Data['ring number'], Data['cutter speed_d'],color=(0,0,0),linewidth=2,label='cutter speed down')
plt.plot(Data['ring number'], Data['cutter speed_u'],color='purple',linewidth=2,label='cutter speed up')
#plt.legend()
plt.show()
plt.figure(8, dpi=120, figsize=(8, 6))
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', family='Times New Roman')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(linestyle='--')
plt.xlabel("ring", fontsize=18)
plt.ylabel("total thrust", fontsize=18)
plt.plot(Data['ring number'], Data['total thrust_d'],'plum',linewidth=2,label='total thrust down')
plt.plot(Data['ring number'], Data['total thrust_u'],'purple',linewidth=2,label='total thrust up')
plt.legend()
plt.show()
plt.figure(5, dpi=120, figsize=(8, 6))
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', family='Times New Roman')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(linestyle='--')
plt.xlabel("ring", fontsize=18)
plt.ylabel("top pressure", fontsize=18)
plt.plot(Data['ring number'], Data['top pressure'],'g',linewidth=2)
plt.show()
plt.figure(6, dpi=120, figsize=(8, 6))
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', family='Times New Roman')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(linestyle='--')
plt.xlabel("ring", fontsize=18)
plt.ylabel("excavated", fontsize=18)
plt.plot(Data['ring number'], Data['excavated'],'g',linewidth=2)
plt.show()




plt.grid(linestyle='--')
plt.subplot(2, 2, 1)
plt.plot(Data['ring number'], Data['stock solution ratio'],linewidth=2)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.subplot(2, 2, 2)
plt.plot(Data['ring number'], Data['foam pressure'],linewidth=2)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.subplot(2, 2, 3)
plt.plot(Data['ring number'], Data['expansion rate'],linewidth=2)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.subplot(2, 2, 4)
plt.plot(Data['ring number'], Data['dosage'],linewidth=2)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()
'''