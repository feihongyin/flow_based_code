#-*- coding:utf-8 -*-

# 网格搜索优化


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
from sklearn.metrics import confusion_matrix, auc
from sklearn.model_selection import train_test_split
from keras.layers import Reshape
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import seaborn as sns
from itertools import cycle
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier # 将 Keras 模型包装起来在 scikit-learn 中使用
from hyperopt import Trials, STATUS_OK, rand
from hyperas import optim
from hyperas.distributions import uniform, choice


# from tensorflow.contrib.layers import xavier_initializer

# import tensorflow as tf

# from inpudata import DataSet






# from sklearn import metrics
# from sklearn.metrics import roc_curve, auc
# import matplotlib as mpl
# from itertools import cycle


# 显示当前 python 程序占用的内存大小
from sklearn.pipeline import Pipeline


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

    print(data_train.shape)
    print(label_train.shape)

    train_samples = np.concatenate((data_train, label_train.reshape(1, -1).T), axis=1)
    print(train_samples.shape)
    np.random.shuffle(train_samples)
    print(train_samples.shape)
    data_train = train_samples[:, :1600]
    label_train = train_samples[:, 1600].astype(np.int32)
    return data_train, data_test, label_train, label_test






def plot_confusion_matrix(cm, target_names,  title='Confusion matrix', cmap=plt.cm.Greens, normalize=True):
    """绘制 confusion matrix 曲线"""
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


def ROC_curve(n_classes, y_test, y_score):
    """绘制ROC曲线"""
    print(f"y_test.shape:{y_test.shape}")
    print(f"y_score.shape:{y_score.shape}")
    # Plot linewidth.
    lw = 1
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Plot all ROC curves
    plt.figure()

    colors = cycle(['firebrick', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'black', 'green', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat'])
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




def plot_confuse(model, x_val, y_val):
    """ 绘制混淆矩阵 """
    predictions = model.predict_classes(x_val, batch_size=2000)
    truelabel = y_val.argmax(axis=-1)  # 将one-hot转化为label
    conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
    plt.figure()
    plot_confusion_matrix(conf_mat, normalize=False, target_names=device_name, title='Confusion Matrix')






# step 1: 读取数据
read_data_time = time.time()

device_name = ["Amazon_Echo_numpy", "Belkin_wemo_motion_sensor_numpy", "Belkin_Wemo_switch_numpy",
               "HP_Printer_numpy", "Insteon_Camera_numpy", "Light_Bulbs_LiFX_Smart_Bulb_numpy",
               "Netatmo_weather_station_numpy", "Netatmo_Welcome_numpy", "PIX_STAR_Photo_frame_numpy",
               "Samsung_SmartCam_numpy", "Smart_Things_numpy", "TP_Link_Day_Night_Cloud_camera_numpy",
               "TP_Link_Smart_plug_numpy", "Withings_Aura_smart_sleep_sensor_numpy",
               "Withings_Smart_Baby_Monitor_numpy", "Blipcare_Blood_Pressure_meter", "Dropcam", "iHome",
               "Nest_Dropcam", "NEST_Protect_smoke_alarm", "Triby_Speaker", "Withings_Smart_scale"]

show_memory_info('initial')
print("##################开始读取数据#################")
# 由于数据量较大，在使用pd.read_csv读取数据时，可以设置数据的格式为uint8, 可以将数据量减小为原来的四分之一
Amazon_Echo = pd.read_csv("/home/dell/datasets/flow_based_classification/3_flow_label_discard_csv/Amazon_Echo_clear_dec.csv", dtype=np.uint8)
Belkin_wemo_motion_sensor = pd.read_csv("/home/dell/datasets/flow_based_classification/3_flow_label_discard_csv/Belkin_wemo_motion_sensor_clear_dec.csv", dtype=np.uint8)
Belkin_Wemo_switch = pd.read_csv("/home/dell/datasets/flow_based_classification/3_flow_label_discard_csv/Belkin_Wemo_switch_clear_dec.csv", dtype=np.uint8)
HP_Printer = pd.read_csv("/home/dell/datasets/flow_based_classification/3_flow_label_discard_csv/HP_Printer_clear_dec.csv", dtype=np.uint8)
Insteon_Camera = pd.read_csv("/home/dell/datasets/flow_based_classification/3_flow_label_discard_csv/Insteon_Camera_clear_dec.csv", dtype=np.uint8)
Light_Bulbs_LiFX_Smart_Bulb = pd.read_csv("/home/dell/datasets/flow_based_classification/3_flow_label_discard_csv/Light_Bulbs_LiFX_Smart_Bulb_clear_dec.csv", dtype=np.uint8)
Netatmo_weather_station = pd.read_csv("/home/dell/datasets/flow_based_classification/3_flow_label_discard_csv/Netatmo_weather_station_clear_dec.csv", dtype=np.uint8)
Netatmo_Welcome = pd.read_csv("/home/dell/datasets/flow_based_classification/3_flow_label_discard_csv/Netatmo_Welcome_clear_dec.csv", dtype=np.uint8)
PIX_STAR_Photo_frame = pd.read_csv("/home/dell/datasets/flow_based_classification/3_flow_label_discard_csv/PIX_STAR_Photo_frame_clear_dec.csv", dtype=np.uint8)
Samsung_SmartCam = pd.read_csv("/home/dell/datasets/flow_based_classification/3_flow_label_discard_csv/Samsung_SmartCam_clear_dec.csv", dtype=np.uint8)
Smart_Things = pd.read_csv("/home/dell/datasets/flow_based_classification/3_flow_label_discard_csv/Smart_Things_clear_dec.csv", dtype=np.uint8)
TP_Link_Day_Night_Cloud_camera = pd.read_csv("/home/dell/datasets/flow_based_classification/3_flow_label_discard_csv/TP_Link_Day_Night_Cloud_camera_clear_dec.csv", dtype=np.uint8)
TP_Link_Smart_plug = pd.read_csv("/home/dell/datasets/flow_based_classification/3_flow_label_discard_csv/TP_Link_Smart_plug_clear_dec.csv", dtype=np.uint8)
Withings_Aura_smart_sleep_sensor = pd.read_csv("/home/dell/datasets/flow_based_classification/3_flow_label_discard_csv/Withings_Aura_smart_sleep_sensor_clear_dec.csv", dtype=np.uint8)
Withings_Smart_Baby_Monitor = pd.read_csv("/home/dell/datasets/flow_based_classification/3_flow_label_discard_csv/Withings_Smart_Baby_Monitor_clear_dec.csv", dtype=np.uint8)
Blipcare_Blood_Pressure_meter = pd.read_csv("/home/dell/datasets/flow_based_classification/3_flow_label_discard_csv/Blipcare_Blood_Pressure_meter_clear_dec.csv", dtype=np.uint8)
Dropcam = pd.read_csv("/home/dell/datasets/flow_based_classification/3_flow_label_discard_csv/Dropcam_clear_dec.csv", dtype=np.uint8)
iHome = pd.read_csv("/home/dell/datasets/flow_based_classification/3_flow_label_discard_csv/iHome_clear_dec.csv", dtype=np.uint8)
Nest_Dropcam = pd.read_csv("/home/dell/datasets/flow_based_classification/3_flow_label_discard_csv/Nest_Dropcam_clear_dec.csv", dtype=np.uint8)
NEST_Protect_smoke_alarm = pd.read_csv("/home/dell/datasets/flow_based_classification/3_flow_label_discard_csv/NEST_Protect_smoke_alarm_clear_dec.csv", dtype=np.uint8)
Triby_Speaker = pd.read_csv("/home/dell/datasets/flow_based_classification/3_flow_label_discard_csv/Triby_Speaker_clear_dec.csv", dtype=np.uint8)
Withings_Smart_scale = pd.read_csv("/home/dell/datasets/flow_based_classification/3_flow_label_discard_csv/Withings_Smart_scale_clear_dec.csv", dtype=np.uint8)


# show_memory_info('after read_csv')
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
print(f"Blipcare_Blood_Pressure_meter.shape[0]:{Blipcare_Blood_Pressure_meter.shape[0]}")
print(f"Dropcam.shape[0]:{Dropcam.shape[0]}")
print(f"iHome.shape[0]:{iHome.shape[0]}")
print(f"Nest_Dropcam.shape[0]:{Nest_Dropcam.shape[0]}")
print(f"NEST_Protect_smoke_alarm.shape[0]:{NEST_Protect_smoke_alarm.shape[0]}")
print(f"Triby_Speaker.shape[0]:{Triby_Speaker.shape[0]}")
print(f"Withings_Smart_scale.shape[0]:{Withings_Smart_scale.shape[0]}")

# step 2:将 dataframe 格式数据转为 numpy
tran_data_time = time.time()
print("##################开始转换数据#################")
Amazon_Echo_numpy = Amazon_Echo.values[:, 1:]
Belkin_wemo_motion_sensor_numpy = Belkin_wemo_motion_sensor.values[:, 1:]
Belkin_Wemo_switch_numpy = Belkin_Wemo_switch.values[:, 1:]
HP_Printer_numpy = HP_Printer.values[:, 1:]
Insteon_Camera_numpy = Insteon_Camera.values[:, 1:]
Light_Bulbs_LiFX_Smart_Bulb_numpy = Light_Bulbs_LiFX_Smart_Bulb.values[:, 1:]
Netatmo_weather_station_numpy = Netatmo_weather_station.values[:, 1:]
Netatmo_Welcome_numpy = Netatmo_Welcome.values[:, 1:]
PIX_STAR_Photo_frame_numpy = PIX_STAR_Photo_frame.values[:, 1:]
Samsung_SmartCam_numpy = Samsung_SmartCam.values[:, 1:]
Smart_Things_numpy = Smart_Things.values[:, 1:]
TP_Link_Day_Night_Cloud_camera_numpy = TP_Link_Day_Night_Cloud_camera.values[:, 1:]
TP_Link_Smart_plug_numpy = TP_Link_Smart_plug.values[:, 1:]
Withings_Aura_smart_sleep_sensor_numpy = Withings_Aura_smart_sleep_sensor.values[:, 1:]
Withings_Smart_Baby_Monitor_numpy = Withings_Smart_Baby_Monitor.values[:, 1:]
Blipcare_Blood_Pressure_meter_numpy = Blipcare_Blood_Pressure_meter.values[:, 1:]
Dropcam_numpy = Dropcam.values[:, 1:]
iHome_numpy = iHome.values[:, 1:]
Nest_Dropcam_numpy = Nest_Dropcam.values[:, 1:]
NEST_Protect_smoke_alarm_numpy = NEST_Protect_smoke_alarm.values[:, 1:]
Triby_Speaker_numpy = Triby_Speaker.values[:, 1:]
Withings_Smart_scale_numpy = Withings_Smart_scale.values[:, 1:]


print(f"转换数据花费时间为：{time.time() - tran_data_time}")

Data_all = (Amazon_Echo_numpy, Belkin_wemo_motion_sensor_numpy, Belkin_Wemo_switch_numpy, HP_Printer_numpy,
            Insteon_Camera_numpy, Light_Bulbs_LiFX_Smart_Bulb_numpy, Netatmo_weather_station_numpy,
            Netatmo_Welcome_numpy, PIX_STAR_Photo_frame_numpy, Samsung_SmartCam_numpy, Smart_Things_numpy,
            TP_Link_Day_Night_Cloud_camera_numpy, TP_Link_Smart_plug_numpy, Withings_Aura_smart_sleep_sensor_numpy,
            Withings_Smart_Baby_Monitor_numpy, Blipcare_Blood_Pressure_meter_numpy, Dropcam_numpy, iHome_numpy, Nest_Dropcam_numpy,
            Triby_Speaker_numpy, Withings_Smart_scale_numpy)

# Data_all = (Belkin_Wemo_switch_numpy, HP_Printer_numpy)

data_train, data_test, label_train, label_test = train_test_generation(Data_all)




# ============================================================================

# parameter
img_rows, img_cols = 40, 40
class_num = 22
lstm_neuron_num = 128
lstm_input_size = 160
lstm_timestep_size = 10
dropout1 = 0.13749544245386258
dropout2 = 0.587606728324542
dropout3 = 0.3746350041674067
dropout4 = 0.5838315653286106
dropout5 = 0.8713141896816126



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


# 构建模型的函数
def create_model(optimizer='RMSprop'):
    # cleanup
    K.clear_session() # 不这样做可能会导致内存爆掉
    # 构建模型
    model = Sequential()
    # 输入 train_x: 40*40*1
    model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid',
                     input_shape=input_shape))
    # 36*36*32
    model.add(BatchNormalization())  # 正则化
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))  # 池化层
    # 18*18*32
    model.add(Dropout(dropout1))  # dropout

    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='valid'))
    # 16*16*64
    model.add(BatchNormalization())  # 正则化
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))  # 池化层
    # 8*8*64
    model.add(Dropout(dropout2))  # dropout
    # 好像需要改一下数据的维度
    model.add(Flatten())  # 8*8*64=4096
    model.add(Dense(1600, activation='relu'))
    model.add(Dropout(dropout3))  # 随机失活

    # 在使用LSTM网络之前，需要改一维度，从[None, 1600]改为[None, 10, 160]
    model.add(Reshape((lstm_timestep_size, lstm_input_size)))
    # LSTM网络
    lstm1 = LSTM(units=256, input_shape=(lstm_timestep_size, lstm_input_size), return_sequences=True)  # 返回所有节点的输出
    model.add(Bidirectional(lstm1))  # 双向LSTM
    model.add(Dropout(dropout4))
    lstm2 = LSTM(units=256, return_sequences=False)  # 返回最后一个节点的输出
    model.add(Bidirectional(lstm2))
    model.add(Dropout(dropout5))
    model.add(Dense(class_num, activation='softmax'))

    # 查看网络结构
    model.summary()
    # 编译模型
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
# 训练模型
# model = create_model()
# model.fit(X_train, Y_train, batch_size=512, epochs=1, verbose=1) # verbose = 2 为每个epoch输出一行记录
# #评估模型
# pre = model.evaluate(X_test, Y_Test, batch_size=2000, verbose=1)
#
# # =========================================================================================
# # 绘制ROC曲线
# y_pred_rf = model.predict(X_test)
# ROC_curve(class_num, Y_Test, y_pred_rf) # 调用 ROC_curve 函数

# # =========================================================================================
# # X_test 是测试数据，Y_Test 是测试标签（这里用的是One——hot向量）
# plot_confuse(model, X_test, Y_Test) # 调用 plot_confuse 函数





# 定义网格搜索参数

model = KerasClassifier(build_fn=create_model, epochs=33, verbose=1)
# 定义网格搜索参数
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
batch_size = [64, 80, 100, 128, 256, 512]
param_grid = dict(optimizer=optimizer, batch_size=batch_size)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=2, n_jobs=1) # n_jobs：并行数，int：个数,-1：跟CPU核数一致, 1:默认值。
grid_result = grid.fit(X_train, Y_train)
# 总结结果
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))



