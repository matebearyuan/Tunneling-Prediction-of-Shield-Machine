from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from joblib import dump, load
from xLSTM.model import xLSTM

# 1. 加载数据
Data = pd.read_csv('D:\\pythonProject\\PyxLSTM-main\\TBM\\c_510_726.csv', index_col='ring number')
Data1 = Data[['geology_1', 'geology_2', 'geology_3', 'geology_4', 'geology_5', 'geology_6',
              'cutter speed', 'total thrust', 'grouting pressure', 'sync grouting volume',
              'stock solution ratio', 'foam pressure', 'expansion rate', 'dosage', 'cutter torque', 'tunneling speed']]
Data1_clean = Data1.dropna().astype(float)

X = Data1_clean.iloc[:, :-1]
Y = Data1_clean[['tunneling speed']]
y = Y.values.ravel()  # 转换为一维数组，因为MSELoss期望一维标签

# 3. 对数值特征进行标准化
scaler = StandardScaler()
x_encoded = X.copy()  # 复制 X 原数据
x_encoded[:] = scaler.fit_transform(X)  # 对所有列进行标准化
dump(scaler, 'slstm_scaler.joblib', compress=True)

# 将数据转为 numpy 数组
x = x_encoded.values


# 4. 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# 5. 设置设备为 CPU
device = torch.device('cpu')  # 明确指定使用 CPU

# 6. 构建 xLSTM 模型
input_size = x_train.shape[1]  # 输入特征数为 15

model = xLSTM(
    vocab_size=input_size,
    embedding_size=256,
    hidden_size=256,
    num_layers=3,
    num_blocks=2,
    dropout=0.3,
    lstm_type='slstm'
)
model.input_proj = nn.Linear(input_size, 256)  # 输入映射层: 15 -> 256
model.output_layer = nn.Linear(256, 1)  # 修改输出层为标量输出
model.to(device)


# 7. 初始化优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.MSELoss()

# 8. 训练模型
n_steps = 300
losses = []

# 数据转换：添加时间步维度
x_train_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

x_test_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

print("x_train_tensor shape:", x_train_tensor.shape)  # 应为 (batch_size, 1, input_size)
print("y_train_tensor shape:", y_train_tensor.shape)

# 训练循环
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.7)  # 每50个epoch学习率乘以0.5
for epoch in range(n_steps):
    def closure():
        optimizer.zero_grad()
        outputs, _ = model(x_train_tensor)  # 直接传入 x_train_tensor，不用重复映射
        loss = criterion(outputs.squeeze(), y_train_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    loss_value = closure().item()
    losses.append(loss_value)
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}/{n_steps}, Loss: {loss_value}')

    # 调整学习率
    scheduler.step()
    print(f"Current Learning Rate: {scheduler.get_last_lr()[0]}")

    # 验证集评估
    with torch.no_grad():
        val_outputs, _ = model(x_test_tensor)
        val_loss = criterion(val_outputs.squeeze(), y_test_tensor)
        print(f"Validation Loss at Epoch {epoch + 1}: {val_loss.item():.4f}")

# 9. 保存整个模型
print("Saving the entire model to disk \n")
torch.save(model, 'sLSTM_model_complete.pt')

# 10. 加载模型并进行预测
model = torch.load('sLSTM_model_complete.pt', map_location=device)
# 预测部分
model.eval()  # 设置为评估模式

# 预测部分
with torch.no_grad():
    y_pred, _ = model(x_test_tensor)  # 直接传入 x_test_tensor
    y_pred = y_pred.cpu().numpy().squeeze()

# 11. 评估模型性能
mse = np.mean((y_test - y_pred) ** 2)
print(f'Mean Squared Error: {mse}')

# 12. 绘制 Loss 变化图
plt.figure(1, dpi=120, figsize=(8, 6))
plt.rc('font', family='Times New Roman')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(linestyle='--')
plt.xlabel("Epoch", fontsize=18)
plt.ylabel("Loss", fontsize=18)
plt.plot(range(1, n_steps + 1), losses, 'r-', linewidth=2, label='Training Loss')
plt.legend(fontsize=16, loc='upper right')
plt.show()

# 11. 计算误差百分比
error = np.abs(y_test - y_pred)
percentage_error = (error / np.abs(y_test)) * 100
print("误差百分比：", percentage_error.shape)

# 13. 绘制真实值与预测值对比图
plt.figure(3, dpi=120, figsize=(8, 6))
plt.rc('font', family='Times New Roman')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(linestyle='--')
plt.xlabel("Ring", fontsize=18)
plt.ylabel("Tunneling Speed", fontsize=18)
plt.plot(y_test, 'k-', linewidth=3, label='Ground Truth')
plt.plot(y_pred.squeeze(), 'b-', linewidth=3, label='Prediction')
plt.legend(fontsize=16, loc='upper left')
plt.show()

# 14. 绘制误差百分比图
plt.figure(4, dpi=120, figsize=(8, 6))
plt.rc('font', family='Times New Roman')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(linestyle='--')
plt.xlabel("Ring", fontsize=18)
plt.ylabel("Percentage Error", fontsize=18)
plt.plot(percentage_error, 'b-', linewidth=3, label='Prediction')
plt.legend(fontsize=16, loc='upper left')
plt.show()


