import torch
import torch.nn as nn
from xLSTM.model import xLSTM
from xLSTM.data import LanguageModelingDataset, Tokenizer
from xLSTM.utils import load_config, set_seed, get_device
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from joblib import dump, load
from sklearn.preprocessing import StandardScaler

# 读取数据
Data = pd.read_csv('D:\\pythonProject\\PyxLSTM-main\\TBM\\c_510_726.csv')

# 选择特征列
Data1 = Data[['geology_1', 'geology_2', 'geology_3', 'geology_4', 'geology_5', 'geology_6', 'cutter speed', 'total thrust', 'grouting pressure', 'sync grouting volume',
              'stock solution ratio', 'foam pressure', 'expansion rate', 'dosage', 'cutter torque', 'tunneling speed']]
Data1_clean = Data1.dropna()

X = Data1_clean.iloc[:, :-1]  # 选取特征列
Y = Data1_clean[['tunneling speed']]
x = X.values
y = Y.values.ravel()  # 转换为一维数组，因为MSELoss期望一维标签

# 数据标准化
scaler = StandardScaler()
x = scaler.fit_transform(x)
dump(scaler, 'kan.joblib', compress=True)

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# 加载配置文件
config = load_config('path_to_your_config_file.yaml')
device = get_device()

# 初始化tokenizer和数据集
tokenizer = Tokenizer(config.vocab_file)
train_dataset = LanguageModelingDataset(config.train_data, tokenizer, config.max_length)

# 创建xLSTM模型
model = xLSTM(len(tokenizer), config.embedding_size, config.hidden_size,
              config.num_layers, config.num_blocks, config.dropout,
              config.bidirectional, config.lstm_type)
model.to(device)

# 定义优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
criterion = nn.MSELoss()

# 定义正则化项计算函数（如果模型内部未集成）
def compute_l1_l2_regularizer(model, lamb):
    l1_reg = sum(p.abs().sum() for p in model.parameters())
    l2_reg = sum(p.pow(2.0).sum() for p in model.parameters())
    return lamb * l1_reg + lamb * l2_reg

n_steps = 500  # 设置训练步数
losses = []  # 用于存储每个 epoch 的 loss

for epoch in range(n_steps):
    def closure():
        optimizer.zero_grad()
        inputs = torch.tensor(x_train, dtype=torch.float32).to(device)
        targets = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # 添加L1和L2正则化项到损失中
        loss += compute_l1_l2_regularizer(model, lamb=0.1)
        loss.backward()
        return loss

    optimizer.step(closure)
    loss_value = closure().item()
    losses.append(loss_value)  # 记录 loss
    print(f'Step {epoch + 1}/{n_steps}, Loss: {loss_value}')

# 保存模型
print("Saving model to disk \n")
torch.save(model.state_dict(), 'xLSTM_model.pt')  # 保存模型状态字典

# 加载模型进行预测（如果需要的话，这部分通常在另一个独立的脚本或模型部署阶段）
model.load_state_dict(torch.load('xLSTM_model.pt'))
model.eval()  # 设置模型为评估模式

# 使用模型进行预测
with torch.no_grad():
    inputs = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_pred = model(inputs).cpu().numpy()

print("actual:", y_test.shape)
print("predict:", y_pred.shape)

# 评估模型性能
mse = np.mean((y_test - y_pred.squeeze()) ** 2)
print(f'Mean Squared Error: {mse}')

error = np.abs(y_test - y_pred.squeeze())
percentage_error = np.divide(error, np.abs(y_test)) * 100
print("误差百分比：", percentage_error.shape)

# 绘制 Loss 变化图
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