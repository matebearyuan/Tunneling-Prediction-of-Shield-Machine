from keras.models import Sequential
import pandas as pd
import scipy.signal as signal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
def med_filter(Data,filter_size):
    Data_filtered = Data
    for col in range(Data.shape[1]):
        y=Data.iloc[:, col]

        y_med = signal.medfilt(y, kernel_size=filter_size)
        Data_filtered.iloc[:,col]=y_med
    return Data_filtered
def normalization(data,label):

    mm_x=MinMaxScaler() # 导入sklearn的预处理容器
    mm_y=MinMaxScaler()
    data=data.values    # 将pd的系列格式转换为np的数组格式
    label=label.values
    data=mm_x.fit_transform(data) # 对数据和标签进行归一化等处理
    label=mm_y.fit_transform(label)
    return data,label,mm_x,mm_y
def build_model():
    model = Sequential()
    #model.add(BatchNormalization(input_dim=12))
    #model.add(Conv1D(input_shape=(x_train_np.shape[1], 1), filters=8, kernel_size=1, strides=3, padding='same', activation='relu'))
    #model.add(MaxPooling1D(pool_size=1, strides=2, padding='same'))
    #model.add(Conv1D(filters=16, kernel_size=1, activation='relu'))
    #model.add(MaxPooling1D(pool_size=1, strides=2, padding='same'))
    #model.add(Conv1D(filters=64, kernel_size=1, activation='relu'))
    #model.add(MaxPooling1D(pool_size=1, strides=2, padding='same'))
    #model.add(Flatten())

    model.add(Dense(12,  activation='relu', use_bias=True))

    model.add(Dense(64, activation='relu',use_bias=True))
    model.add(Dropout(0.4))
    model.add(Dense(1, use_bias=True,activation='linear'))

    #model.summary()
    #plot_model(model, to_file='./cnn_model.png', show_shapes=True)
    model.compile(loss='mse', optimizer='Adam')
    return model
if __name__ == "__main__":
    Data = pd.read_excel('1right0124.xlsx')
    print(Data.shape)
    print(Data.head())
    Data_filtered = med_filter(Data, 5)
    Y = Data_filtered.iloc[:, [4]]
    # the othe remaining columns are selected in X
    X = Data_filtered.iloc[:, [1,3,5,6,7,8,9,10,11,12,13,14]]
    Xn, Yn, mmx, mmy = normalization(X, Y)
    x_train, x_test, y_train, y_test = train_test_split(Xn, Yn, test_size=0.3, random_state=42, shuffle=False)
    model = build_model()
    history = model.fit(x_train, y_train, epochs=200, batch_size=30, validation_data=(x_test, y_test), verbose=2,
                        shuffle=True)

    # 输出最终的权值
    # weights = model.get_weights()  # 获取整个网络模型的全部参数
    print(history)
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