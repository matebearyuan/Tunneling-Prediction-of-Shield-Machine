import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Dense, LSTM
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from joblib import dump
from sklearn.model_selection import train_test_split
timesteps= 1

# 数据加载和预处理
Data = pd.read_csv('D:\\pythonProject\\PyxLSTM-main\\TBM\\c_510_726.csv')
Data1 = Data[['geology_1', 'geology_2', 'geology_3', 'geology_4', 'geology_5', 'geology_6',
              'cutter speed', 'total thrust', 'grouting pressure', 'sync grouting volume',
              'stock solution ratio', 'foam pressure', 'expansion rate', 'dosage',
              'cutter torque', 'tunneling speed']]
Data1_clean = Data1.dropna()

X = Data1_clean.iloc[:, :-2]  # 特征列
Y = Data1_clean[['tunneling speed','cutter torque']]  # 标签列
x = X.values.astype('float32')
y = Y.values.astype('float32')
print(x.shape[1])
print(y.shape)
# 数据标准化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(x)
scaled_target = scaler.fit_transform(y.reshape(-1, 2))
dataset = np.hstack((scaled_features, scaled_target))

print(dataset)

dump(scaler, 'xlstm.joblib', compress=True)
reshaped_scaled_features = scaled_features.reshape(scaled_features.shape[0], 1, scaled_features.shape[1])
# 定义模型
model = Sequential()
# 输入层
model.add(Input(shape=(1,x.shape[1])))  # 5个输入变量
# 第一层 XLSTM（假设 XLSTM 为标准 LSTM）
model.add(LSTM(64, activation='tanh', return_sequences=True))
# 第二层 XLSTM
model.add(LSTM(32, activation='tanh', return_sequences=False))
# 全连接层
model.add(Dense(16, activation='relu'))
# 输出层
model.add(Dense(2, activation='linear'))  # 2 个输出变量
# 编译模型
model.compile(optimizer='adam', loss='mse', metrics=['mae','accuracy'])
# 打印模型摘要
model.summary()
# 可视化模型架构
#tf.keras.utils.plot_model(model, to_file='xlstm_model.png', show_shapes=True, show_layer_names=True)


# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(reshaped_scaled_features, scaled_target, test_size=0.2, random_state=40)



history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test))
print("Saving model to disk \n")
model.save("STM_LSTMs.h5")
# 查看训练集上的损失值和评价指标
train_loss = history.history['loss']
train_mae = history.history['mae']
train_accuracy = history.history['accuracy']

# 查看验证集上的损失值和评价指标
val_loss = history.history['val_loss']
val_mae = history.history['val_mae']
val_accuracy = history.history['val_accuracy']
# 打印最后一个 epoch 的结果
print(f"Final Training Loss: {val_loss[-1]}")
print(f"Final Validation MAE: {val_mae[-1]}")
print(f"Final Validation accuracy: {val_accuracy[-1]}")
plt.figure(2, dpi=120, figsize=(8, 6))
plt.rc('font', family='Times New Roman')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(linestyle='--')
plt.xlabel("epoch", fontsize=18)
plt.ylabel("loss", fontsize=18)
plt.plot(history.history['loss'], 'g-', linewidth=3, label='training_loss')
plt.plot(history.history['val_loss'], 'b-', linewidth=3, label='testing_loss')
plt.legend(fontsize=16, loc='upper left')
plt.show()

y_pred = model.predict(x_test)
print(y_pred.shape)

orignal_y_pred = scaler.inverse_transform(y_pred)
orignal_y_test = scaler.inverse_transform(y_test)
# 13. 评估模型性能
mse = np.mean((y_test - y_pred) ** 2)
print(f'Mean Squared Error: {mse}')
means = np.mean(np.abs((orignal_y_test - orignal_y_pred)),axis=0)
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
plt.plot(orignal_y_test[:,0], 'k-', linewidth=3, label='ground truth')
plt.plot(orignal_y_pred[:,0], 'b-', linewidth=3, label='prediction')

plt.legend(fontsize=16, loc='upper left')
# plt.savefig('multivar_training.jpg', dpi=300)
plt.show()

plt.figure(4, dpi=120, figsize=(8, 6))
plt.rc('font', family='Times New Roman')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(linestyle='--')
#plt.xlabel("ring", fontsize=18)
plt.ylabel("cutter torgue", fontsize=18)
plt.plot(orignal_y_test[:,1], 'k-', linewidth=3, label='ground truth')
plt.plot(orignal_y_pred[:,1], 'b-', linewidth=3, label='prediction')

plt.legend(fontsize=16, loc='upper left')
plt.show()