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

# 2. 对类别特征重新编码为整数索引
x_encoded = X.copy()
for col in ['geology_1', 'geology_2', 'geology_3', 'geology_4', 'geology_5', 'geology_6']:
    le = LabelEncoder()
    x_encoded[col] = le.fit_transform(X[col])

# 3. 对数值特征进行标准化
scaler = StandardScaler()
numerical_cols = ['cutter speed', 'total thrust', 'grouting pressure', 'sync grouting volume',
                  'stock solution ratio', 'foam pressure', 'expansion rate', 'dosage', 'cutter torque']
x_encoded[numerical_cols] = scaler.fit_transform(X[numerical_cols])
dump(scaler, 'mlstm_scaler.joblib', compress=True)

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
    lstm_type='mlstm'
)
model.input_proj = nn.Linear(input_size, 256)  # 输入映射层: 15 -> 256
model.output_layer = nn.Linear(256, 1)  # 修改输出层为标量输出
model.to(device)

# 8. 训练模型
n_steps = 2
losses = []

# 数据转换：添加时间步维度
x_train_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

x_test_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

print("x_train_tensor shape:", x_train_tensor.shape)  # 应为 (batch_size, 1, input_size)
print("y_train_tensor shape:", y_train_tensor.shape)

# 训练循环
from torch.utils.data import DataLoader, TensorDataset

# 定义批量大小
batch_size = 128

# 创建 DataLoader
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 设置小学习率
optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)
criterion = torch.nn.MSELoss()

# 训练循环
# 初始化一个默认值（假设初始损失为 0.0）
last_valid_loss = 0.0

for epoch in range(n_steps):
    total_loss = 0.0
    valid_batches = 0

    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()

        # 前向传播
        batch_outputs, _ = model(batch_x)

        # 检查输出 NaN
        if torch.isnan(batch_outputs).any() or torch.isnan(batch_y).any():
            print("NaN detected in inputs or outputs!")
            continue

        # 计算损失
        loss = criterion(batch_outputs.squeeze(), batch_y)
        if torch.isnan(loss):
            print("NaN detected in loss calculation, skipping this batch.")
            continue

        # 反向传播 + 梯度裁剪
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        valid_batches += 1
        last_valid_loss = loss.item()  # 更新有效的 loss

    # 记录损失
    if valid_batches > 0:
        average_loss = total_loss / valid_batches
        losses.append(average_loss)
    else:
        losses.append(last_valid_loss)  # 如果所有批次都失败，使用上一个有效 loss

    print(f"Epoch {epoch + 1}/{n_steps}, Loss: {losses[-1]:.4f}")

# 9. 保存整个模型
print("Saving the entire model to disk \n")
torch.save(model, 'mLSTM_model_complete.pt')

# 10. 加载模型并进行预测
model = torch.load('mLSTM_model_complete.pt', map_location=device)
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
if len(losses) > 0:
    plt.figure(1, dpi=120, figsize=(8, 6))
    plt.rc('font', family='Times New Roman')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(linestyle='--')
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("Loss", fontsize=18)
    plt.plot(range(1, len(losses) + 1), losses, 'r-', linewidth=2, label='Training Loss')
    plt.legend(fontsize=16, loc='upper right')
    plt.show()
else:
    print("No valid losses to plot.")


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


