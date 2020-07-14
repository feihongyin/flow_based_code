#-*- coding:utf-8 -*-

import time
import os
import psutil as psutil
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, Bidirectional
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.utils import np_utils # 生成one-hot编码
from keras import backend as K
from numpy import interp
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from keras.layers import Reshape
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle


# from tensorflow.contrib.layers import xavier_initializer


# 显示当前 python 程序占用的内存大小
def show_memory_info(hint):
    pid = os.getpid()
    p = psutil.Process(pid)

    info = p.memory_full_info()
    memory = info.uss / 1024. / 1024
    print('{} memory used: {} MB'.format(hint, memory))

def train_test_generation(Data_all):
    """生成训练测和测试集"""
    data_train = []
    data_test = []
    label_train = []
    label_test = []

    for device_data in Data_all:
        x_data = np.array(device_data[:, :-1], dtype="float32")
        y_data = np.array(device_data[:, -1], dtype="int32")
        device_data_train, device_data_test, device_label_train, device_label_test = train_test_split(x_data, y_data, test_size=0.2, random_state=0)  # random_state:填0或不填，数据集每次都会不一样。
        data_train.append(device_data_train)
        data_test.append(device_data_test)
        label_train.append(device_label_train)
        label_test.append(device_label_test)

    data_train = np.concatenate(np.array(data_train), axis=0)
    data_test = np.concatenate(np.array(data_test), axis=0)
    label_train = np.concatenate(np.array(label_train), axis=0)
    label_test = np.concatenate(np.array(label_test), axis=0)

    return data_train, data_test, label_train, label_test





def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens,  # 这个地方设置混淆矩阵的颜色主题，这个主题看着就干净~
                          normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(15, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=60)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()

    ticks = np.array(range(len(device_name)))
    plt.xticks(ticks, device_name, rotation=90)
    plt.yticks(ticks, device_name, rotation=360)

    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    # 这里这个savefig是保存图片，如果想把图存在什么地方就改一下下面的路径，然后dpi设一下分辨率即可。
    # plt.savefig('/content/drive/My Drive/Colab Notebooks/confusionmatrix32.png',dpi=350)
    plt.show()



# 显示混淆矩阵
def plot_confuse(model, x_val, y_val):
    predictions = model.predict_classes(x_val, batch_size=2000)
    truelabel = y_val.argmax(axis=-1)  # 将one-hot转化为label
    conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
    plt.figure()
    plot_confusion_matrix(conf_mat, normalize=False, target_names=device_name, title='Confusion Matrix')



def ROC_curve(n_classes, y_test, y_score):
    """绘制ROC曲线"""
    # Plot linewidth.
    lw = 2

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(1)
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = itertools.cycle(['firebrick', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'black', 'green', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

    # Zoom in view of the upper left corner.
    plt.figure(2)
    plt.xlim(0, 0.2)
    plt.ylim(0.8, 1)
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()





# step 1: 读取数据
read_data_time = time.time()

device_name = ["Amazon_Echo_numpy", "Belkin_wemo_motion_sensor_numpy", "Belkin_Wemo_switch_numpy",
               "HP_Printer_numpy", "Insteon_Camera_numpy", "Light_Bulbs_LiFX_Smart_Bulb_numpy",
               "Netatmo_weather_station_numpy", "Netatmo_Welcome_numpy", "PIX_STAR_Photo_frame_numpy",
               "Samsung_SmartCam_numpy", "Smart_Things_numpy", "TP_Link_Day_Night_Cloud_camera_numpy",
               "TP_Link_Smart_plug_numpy", "Withings_Aura_smart_sleep_sensor_numpy",
               "Withings_Smart_Baby_Monitor_numpy"]

show_memory_info('initial')
print("##################开始读取数据#################")
# 由于数据量较大，在使用pd.read_csv读取数据时，可以设置数据的格式为uint8, 可以将数据量减小为原来的四分之一
Amazon_Echo = pd.read_csv("/home/dell/datasets/packet_based_classification/3_packet_label_discard_csv/Amazon_Echo_clear.csv", dtype=np.uint8) # 867817
Belkin_wemo_motion_sensor = pd.read_csv("/home/dell/datasets/packet_based_classification/3_packet_label_discard_csv/Belkin_wemo_motion_sensor_clear.csv", dtype=np.uint8) # 277485
Belkin_Wemo_switch = pd.read_csv("/home/dell/datasets/packet_based_classification/3_packet_label_discard_csv/Belkin_Wemo_switch_clear.csv", dtype=np.uint8) # 192415
HP_Printer = pd.read_csv("/home/dell/datasets/packet_based_classification/3_packet_label_discard_csv/HP_Printer_clear.csv", dtype=np.uint8) # 71863
Insteon_Camera = pd.read_csv("/home/dell/datasets/packet_based_classification/3_packet_label_discard_csv/Insteon_Camera_clear.csv", dtype=np.uint8) # 413205
Light_Bulbs_LiFX_Smart_Bulb = pd.read_csv("/home/dell/datasets/packet_based_classification/3_packet_label_discard_csv/Light_Bulbs_LiFX_Smart_Bulb_clear.csv", dtype=np.uint8) # 49527
Netatmo_weather_station = pd.read_csv("/home/dell/datasets/packet_based_classification/3_packet_label_discard_csv/Netatmo_weather_station_clear.csv", dtype=np.uint8) # 48642
Netatmo_Welcome = pd.read_csv("/home/dell/datasets/packet_based_classification/3_packet_label_discard_csv/Netatmo_Welcome_clear.csv", dtype=np.uint8) # 343951
PIX_STAR_Photo_frame = pd.read_csv("/home/dell/datasets/packet_based_classification/3_packet_label_discard_csv/PIX_STAR_Photo_frame_clear.csv", dtype=np.uint8) # 42305
Samsung_SmartCam = pd.read_csv("/home/dell/datasets/packet_based_classification/3_packet_label_discard_csv/Samsung_SmartCam_clear.csv", dtype=np.uint8) # 338607
Smart_Things = pd.read_csv("/home/dell/datasets/packet_based_classification/3_packet_label_discard_csv/Smart_Things_clear.csv", dtype=np.uint8) # 83473
TP_Link_Day_Night_Cloud_camera = pd.read_csv("/home/dell/datasets/packet_based_classification/3_packet_label_discard_csv/TP_Link_Day_Night_Cloud_camera_clear.csv", dtype=np.uint8) # 217395
TP_Link_Smart_plug = pd.read_csv("/home/dell/datasets/packet_based_classification/3_packet_label_discard_csv/TP_Link_Smart_plug_clear.csv", dtype=np.uint8) # 25271
Withings_Aura_smart_sleep_sensor = pd.read_csv("/home/dell/datasets/packet_based_classification/3_packet_label_discard_csv/Withings_Aura_smart_sleep_sensor_clear.csv", dtype=np.uint8) # 92542
Withings_Smart_Baby_Monitor = pd.read_csv("/home/dell/datasets/packet_based_classification/3_packet_label_discard_csv/Withings_Smart_Baby_Monitor_clear.csv", dtype=np.uint8) # 143578
show_memory_info('after read_csv')
print(f"读取数据花费时间为：{time.time() - read_data_time}")

print(f"Amazon_Echo.shape[0]:{Amazon_Echo.shape[0]}")
print(f"Belkin_wemo_motion_sensor.shape[0]:{Belkin_wemo_motion_sensor.shape[0]}")
print(f"Belkin_Wemo_switch.shape[0]:{Belkin_Wemo_switch.shape[0]}")
print(f"HP_Printer.shape[0]:{HP_Printer.shape[0]}")
print(f"Insteon_Camera.shape[0]:{Insteon_Camera.shape[0]}")
print(f"Light_Bulbs_LiFX_Smart_Bulb.shape[0]:{Light_Bulbs_LiFX_Smart_Bulb.shape[0]}")
print(f"Netatmo_weather_station.shape[0]:{Netatmo_weather_station.shape[0]}")
print(f"Netatmo_Welcome.shape[0]:{Netatmo_Welcome.shape[0]}")
print(f"PIX_STAR_Photo_frame.shape[0]:{PIX_STAR_Photo_frame.shape[0]}")
print(f"Samsung_SmartCam.shape[0]:{Samsung_SmartCam.shape[0]}")
print(f"Smart_Things.shape[0]:{Smart_Things.shape[0]}")
print(f"TP_Link_Day_Night_Cloud_camera.shape[0]:{TP_Link_Day_Night_Cloud_camera.shape[0]}")
print(f"TP_Link_Smart_plug.shape[0]:{TP_Link_Smart_plug.shape[0]}")
print(f"Withings_Aura_smart_sleep_sensor.shape[0]:{Withings_Aura_smart_sleep_sensor.shape[0]}")
print(f"Withings_Smart_Baby_Monitor.shape[0]:{Withings_Smart_Baby_Monitor.shape[0]}")


# step 2:将 dataframe 格式数据转为 numpy
tran_data_time = time.time()
print("##################开始转换数据#################")
# # del操作没用
# del Amazon_Echo
# gc.collect()
# show_memory_info('after1')
Amazon_Echo_numpy = Amazon_Echo.values[:, 1:][:200000]
Belkin_wemo_motion_sensor_numpy = Belkin_wemo_motion_sensor.values[:, 1:][:150000] # 150000
Belkin_Wemo_switch_numpy = Belkin_Wemo_switch.values[:, 1:][:100000] # 100000
HP_Printer_numpy = HP_Printer.values[:, 1:][:80000] # 71863
Insteon_Camera_numpy = Insteon_Camera.values[:, 1:][:150000] # 150000
Light_Bulbs_LiFX_Smart_Bulb_numpy = Light_Bulbs_LiFX_Smart_Bulb.values[:, 1:][:50000] # 49527
Netatmo_weather_station_numpy = Netatmo_weather_station.values[:, 1:][:50000] # 48642
Netatmo_Welcome_numpy = Netatmo_Welcome.values[:, 1:][:200000] # 200000
PIX_STAR_Photo_frame_numpy = PIX_STAR_Photo_frame.values[:, 1:][:50000] # 42305
Samsung_SmartCam_numpy = Samsung_SmartCam.values[:, 1:][:200000] # 200000
Smart_Things_numpy = Smart_Things.values[:, 1:][:90000] # 83473
TP_Link_Day_Night_Cloud_camera_numpy = TP_Link_Day_Night_Cloud_camera.values[:, 1:][:100000] # 100000
TP_Link_Smart_plug_numpy = TP_Link_Smart_plug.values[:, 1:][:30000] # 25271
Withings_Aura_smart_sleep_sensor_numpy = Withings_Aura_smart_sleep_sensor.values[:, 1:][:100000] # 100000
Withings_Smart_Baby_Monitor_numpy = Withings_Smart_Baby_Monitor.values[:, 1:][:80000] # 80000


print(f"转换数据花费时间为：{time.time() - tran_data_time}")

Data_all = (Amazon_Echo_numpy, Belkin_wemo_motion_sensor_numpy, Belkin_Wemo_switch_numpy, HP_Printer_numpy,
            Insteon_Camera_numpy, Light_Bulbs_LiFX_Smart_Bulb_numpy, Netatmo_weather_station_numpy,
            Netatmo_Welcome_numpy, PIX_STAR_Photo_frame_numpy, Samsung_SmartCam_numpy, Smart_Things_numpy,
            TP_Link_Day_Night_Cloud_camera_numpy, TP_Link_Smart_plug_numpy, Withings_Aura_smart_sleep_sensor_numpy,
            Withings_Smart_Baby_Monitor_numpy)



# Data_all = (Belkin_Wemo_switch_numpy, HP_Printer_numpy)

data_train, data_test, label_train, label_test = train_test_generation(Data_all)

print(f"total samples:{len(Data_all)}")
print(f"train samples:{len(data_train)}")
print(f"test samples:{len(data_test)}")


# ==========================================================================
def labels_transform(mlist, classes):
    batch_label = np.zeros((len(mlist), classes), dtype="i4")
    for i in range(len(mlist)):
        batch_label[i][mlist[i]] = 1
    return batch_label


# ============================================================================

# parameter
learning_rate = 0.001
img_rows, img_cols = 40, 40
class_num = 15
batch_size = 128
lstm_neuron_num = 128
lstm_input_size = 160
lstm_timestep_size = 10
lstm_hidden_layers = 2
train_iter = 15000


# 根据不同的backend定下不同的格式
if K.image_data_format() == 'th':
    X_train = data_train.reshape(data_train.shape[0], 1, img_rows, img_cols)
    X_test = data_test.reshape(data_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = data_train.reshape(data_train.shape[0], img_rows, img_cols, 1) # (None, 40, 40, 1)
    X_test = data_test.reshape(data_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


print(X_train.shape)

# 转换为one_hot类型
Y_train = np_utils.to_categorical(label_train, class_num)
Y_Test = np_utils.to_categorical(label_test, class_num)

# 构建模型

model = Sequential()
# 输入 train_x: 40*40*1
model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid', input_shape=input_shape))
# 36*36*32
model.add(BatchNormalization()) # 正则化
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')) #池化层
# 18*18*32
model.add(Dropout(0.5)) # dropout


model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='valid'))
# 16*16*64
model.add(BatchNormalization()) # 正则化
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')) #池化层
# 8*8*64
model.add(Dropout(0.5)) # dropout
# 好像需要改一下数据的维度
model.add(Flatten()) # 8*8*64=4096
model.add(Dense(1600, activation='relu'))
model.add(Dropout(0.5)) #随机失活

# 在使用LSTM网络之前，需要改一维度，从[None, 1600]改为[None, 10, 160]
model.add(Reshape((lstm_timestep_size, lstm_input_size)))
# LSTM网络
lstm1 = LSTM(units=256, input_shape=(lstm_timestep_size, lstm_input_size), return_sequences=True) #返回所有节点的输出
model.add(Bidirectional(lstm1)) # 双向LSTM
model.add(Dropout(0.25))
lstm2 = LSTM(units=256, return_sequences=False) #返回最后一个节点的输出
model.add(Bidirectional(lstm2))
model.add(Dropout(0.25))
model.add(Dense(class_num, activation='softmax'))

#查看网络结构
model.summary()
#编译模型
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#训练模型
model.fit(X_train, Y_train, batch_size=512, epochs=36, verbose=1) # verbose = 2 为每个epoch输出一行记录
#评估模型
pre = model.evaluate(X_test, Y_Test, batch_size=2000, verbose=1)


y_pred_rf = model.predict_proba(X_test)[:, 1]
ROC_curve(class_num, Y_Test, y_pred_rf)

# =========================================================================================
# 最后调用这个函数即可。 test_x是测试数据，test_y是测试标签（这里用的是One——hot向量）
# labels是一个列表，存储了你的各个类别的名字，最后会显示在横纵轴上。
# 比如这里我的labels列表

plot_confuse(model, X_test, Y_Test)

