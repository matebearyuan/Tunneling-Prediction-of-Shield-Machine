from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from joblib import dump
from xLSTM.model import xLSTM

# 1. 创建时间序列数据函数
def create_sequences(data, target, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(data) - seq_length):
        X_seq.append(data[i:i + seq_length])  # 取连续 seq_length 个时间步的数据
        y_seq.append(target[i + seq_length])  # 目标值为下一时间步的 tunneling speed
    return np.array(X_seq), np.array(y_seq)

# 2. 加载并处理数据
Data = pd.read_csv('D:\\pythonProject\\PyxLSTM-main\\TBM\\c_510_726.csv', index_col='ring number')
Data1 = Data[['geology_1', 'geology_2', 'geology_3', 'geology_4', 'geology_5', 'geology_6',
              'cutter speed', 'total thrust', 'grouting pressure', 'sync grouting volume',
              'stock solution ratio', 'foam pressure', 'expansion rate', 'dosage', 'cutter torque', 'tunneling speed']]

Data1_clean = Data1.dropna().astype(float).sort_index()

X = Data1_clean.iloc[:, :-2].values  # 所有输入特征
Y = Data1_clean[['tunneling speed']].values.ravel()  # 目标值
print(X.shape)
print(Y.shape)
# 3. 标准化所有输入特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Y_scaled = scaler.fit_transform(Y.reshape(-1, 1))
dump(scaler, 'slstm_ring_scaler.joblib', compress=True)

# 4. 创建时间序列数据
seq_length = 10
X_seq, Y_seq = create_sequences(X_scaled, Y_scaled, seq_length)
print(X_seq.shape)
print(Y_seq.shape)
# 5. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_seq, Y_seq, test_size=0.2, random_state=0, shuffle=False)

# 6. 设置设备为 CPU
device = torch.device('cpu')

# 7. 构建 xLSTM 模型
input_size = X_train.shape[2]
model = xLSTM(
    vocab_size=input_size,
    embedding_size=128,
    hidden_size=128,
    num_layers=2,
    num_blocks=1,
    dropout=0.3,
    lstm_type='slstm'
)
model.input_proj = nn.Linear(input_size, 128)
model.output_layer = nn.Linear(128, 1)
model.to(device)

# 8. 初始化优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)  # 1e-5为L2正则化系数
criterion = torch.nn.MSELoss()

# 9. 数据转换为 Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# 打印每一步的数据形状
print(f"X_train_tensor shape before model: {X_train_tensor.shape}")

# 10. 训练模型
n_steps = 300
losses = []
val_losses = []
# 调整学习率策略
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)


for epoch in range(n_steps):
    model.train()
    optimizer.zero_grad()
    outputs, _ = model(X_train_tensor)  # 直接传入三维输入
    loss = criterion(outputs.squeeze(), y_train_tensor)
    loss.backward()
    optimizer.step()

    # 验证集评估
    model.eval()
    with torch.no_grad():
        val_outputs, _ = model(X_test_tensor)  # 直接传入三维输入
        val_loss = criterion(val_outputs.squeeze(), y_test_tensor)

    losses.append(loss.item())
    val_losses.append(val_loss.item())
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{n_steps}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
    # 在每个epoch后进行学习率调整
    scheduler.step(val_loss)

# 11. 保存模型
torch.save(model, 'sLSTM_model_ring.pt')
print("Model saved to disk as 'sLSTM_model_ring.pt'.")

# 12. 加载模型并预测
model = torch.load('sLSTM_model_ring.pt', map_location=device)
model.eval()

with torch.no_grad():
    y_pred, _ = model(X_test_tensor)
    y_pred = y_pred.cpu().numpy().squeeze()

# 13. 评估模型性能
mse = np.mean((y_test - y_pred) ** 2)
print(f'Mean Squared Error: {mse}')

# 14. 绘制 Loss 变化图
plt.figure(1, figsize=(8, 6), dpi=120)
plt.rc('font', family='Times New Roman')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(linestyle='--')
plt.plot(range(1, n_steps + 1), losses, 'r-', linewidth=2, label='Training Loss')
plt.plot(range(1, n_steps + 1), val_losses, 'b-', linewidth=2, label='Validation Loss')
plt.xlabel("Epoch", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.title("Training and Validation Loss")
plt.legend(fontsize=16, loc='upper right')
plt.grid()
plt.show()

# 15. 绘制真实值与预测值对比图
plt.figure(2, figsize=(8, 6), dpi=120)
plt.rc('font', family='Times New Roman')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(linestyle='--')
plt.plot(y_test, 'k-', linewidth=2, label='Ground Truth')
plt.plot(y_pred, 'b-', linewidth=2, label='Prediction')
plt.xlabel("Sample Index", fontsize=16)
plt.ylabel("Tunneling Speed", fontsize=16)
plt.title("Ground Truth vs Prediction")
plt.legend(fontsize=16, loc='upper right')
plt.grid()
plt.show()

# 16. 绘制误差百分比图
error = np.abs(y_test - y_pred)
percentage_error = (error / np.abs(y_test)) * 100

plt.figure(3, figsize=(8, 6), dpi=120)
plt.rc('font', family='Times New Roman')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(linestyle='--')
plt.plot(percentage_error, 'r-', linewidth=2, label='Percentage Error')
plt.xlabel("Sample Index", fontsize=16)
plt.ylabel("Percentage Error (%)", fontsize=16)
plt.title("Prediction Error Percentage")
plt.legend(fontsize=16, loc='upper right')
plt.grid()
plt.show()

# 输出误差统计
print(f"Average Percentage Error: {np.mean(percentage_error):.2f}%")
