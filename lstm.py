import os
import datetime
import sys

import keras
import tensorflow as tf
import IPython

import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


#用7天预测3天
class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df, test_df,label_columns=None):
    # Store the raw data.
    self.row_df = df
    self.train_df = train_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns#确定标签列在数据集中的索引位置
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width#输入宽度
    self.label_width = label_width#输出宽度
    self.shift = shift#

    self.total_window_size = input_width + shift#总宽度

    self.input_slice = slice(0, input_width)#输入切片
    self.input_indices = np.arange(self.total_window_size)[self.input_slice] #输入的索引切片

    self.label_start = self.total_window_size - self.label_width#输出开始索引
    self.labels_slice = slice(self.label_start, None)#输出切片
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]#输出索引切片

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])

  #拆分
def split_window(self, features):
    inputs = features[:, self.input_slice, :]#将表示对 features 张量进行切片操作，保持批次维度不变，选择 self.input_slice 索引范围内的窗口维度，保持特征维度不变。
    labels = features[:, self.labels_slice, :]#此时的labels包含全部特征
    if self.label_columns is not None:
        labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1)#将labels指定到某一特征上

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  #labels 张量的静态形状被设置为 [None, self.label_width, None]。其中：
  #第一个维度被设置为 None，表示批次大小可以是任意值。
  #第二个维度被设置为 self.label_width，表示标签窗口的宽度是固定的。
  #第三个维度被设置为 None，表示标签列的数量可以是任意值。
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])
    return inputs, labels

WindowGenerator.split_window = split_window


#此 make_dataset 方法将获取时间序列 DataFrame
#并使用 tf.keras.utils.timeseries_dataset_from_array 函数将其转换为 (input_window, label_window) 对的 tf.data.Dataset。
def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)#将输入的 data 转换为 NumPy 数组，并指定数据类型为 np.float32。
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=True,
        batch_size=32,)#将输入的数据数组 data 转换为时间序列数据集对象 ds。batch_size: 每个批次的样本数。

    ds = ds.map(self.split_window)#使用 map 方法将 split_window 函数应用于时间序列数据集 ds 的每个样本。

    return ds

WindowGenerator.make_dataset = make_dataset

#使用之前定义的 make_dataset 方法添加属性以作为 tf.data.Dataset 访问它们
@property
def train(self):
  return self.make_dataset(self.train_df)
@property
def test(self):
  return self.make_dataset(self.test_df)

WindowGenerator.train = train
WindowGenerator.test = test




def compile_and_fit(model, window, MAX_EPOCHS=200,patience=20):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',#mean_absolute_error
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      callbacks=[early_stopping])
  return history

#读取与拆分数据
if __name__ == "__main__":
    df = pd.read_excel(sys.argv[1],sheet_name='sheet1')
    # df = pd.read_excel("C:/Users/13183/PycharmProjects/pythonProject/program/daorushuju.xlsx",sheet_name='Sheet2')
    # Slice [start:stop:step], starting from index 5 take every 6th record.
    column_indices = {name: i for i, name in enumerate(df.columns)}
    n = len(df)
    num_features = df.shape[1]
    column_names = df.columns.tolist()#全部列名
    print(num_features)
    performances={}
    lin1 = [0,0,0]
    lin2 = [0,0,0]
    lin3 = [0,0,0]
    new_line=[]
    for i in range(num_features-35):
        column_name=column_names[i+3] #列名
        data=df[['总去','总回','温差',column_name]]
        train_df = data[0:-3]
        test_df = data[-11:]
        print(data)
        OUT_STEPS = 3
        multi_window = WindowGenerator(input_width=7,
                                       label_width=OUT_STEPS,
                                       shift=OUT_STEPS,
                                       train_df=train_df,
                                       test_df=test_df,
                                       label_columns=[column_name]
                                       )
        multi_lstm_model = tf.keras.Sequential([
            # Shape [batch, time, features] => [batch, lstm_units].
            # Adding more `lstm_units` just overfits more quickly.
            tf.keras.layers.LSTM(32, return_sequences=True),
            tf.keras.layers.LSTM(32, return_sequences=False),
            tf.keras.layers.Dense(32, activation='relu'),
            # Shape => [batch, out_steps*features].
            tf.keras.layers.Dense(OUT_STEPS * num_features,
                                  kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features].
            tf.keras.layers.Reshape([OUT_STEPS, num_features])
        ])
        history = compile_and_fit(multi_lstm_model, multi_window)
        IPython.display.clear_output()
        performances[column_name] = multi_lstm_model.evaluate(multi_window.test, verbose=0)  # MAE  温差 温度
        column_indices = {name: i for i, name in enumerate(data.columns)}
        plot_col_index = column_indices[column_name]
        # 画图
        # inputs = []
        # for j in range(len(data) - 7 ):  # 这个地方的-7+1不确定
        #     inputs.append(np.array(data[j:j + 7]))
        # inputs = tf.convert_to_tensor(inputs)
        # # inputs, labels = wide_window.example
        # predictions = multi_lstm_model(inputs)
        # print(predictions)
        # plt.plot(range(len(data)), data[column_name], label='Inputs', marker='.', zorder=-10)
        # plt.plot(range(7, len(df)), predictions[:, 0, plot_col_index])
        # plt.plot(range(8, len(df)+1), predictions[:, 1, plot_col_index])
        # plt.plot(range(9, len(df)+2), predictions[:, 2, plot_col_index])
        # plt.savefig("C:/Users/13183/PycharmProjects/pythonProject/pic-mult-{}".format(column_name))
        # plt.clf()
        inputs = []
        inputs.append(np.array(data[-7:]))
        inputs = tf.convert_to_tensor(inputs)
        predictions = multi_lstm_model(inputs)

        y = np.squeeze(predictions)
        y_unique = y[:,0]

        print(data)
        print(y_unique)
        lin1.append(y_unique[0])
        lin2.append(y_unique[1])
        lin3.append(y_unique[2])
        print(performances)
        # plt.plot(range(len(df)),df['wendu'],label='Inputs', marker='.', zorder=-10)
        # plt.plot(range(len(df)-7,len(df)),y,marker='X', label='Predictions',c='#ff7f0e')
        # plt.show()
    new_line.append(lin1)
    new_line.append(lin2)
    new_line.append(lin3)
    print(new_line)
# # 创建一个新的 DataFrame 来存储预测的 y 数据
# y_df = pd.DataFrame(new_line, columns=df.columns)
# # 将 y_df 添加到原始的 df 中
# df.append(y_df)
# # 将更新后的 df 保存到 Excel 文件中
# df.to_excel("your_file.xlsx", sheet_name='Sheet1', index=False)
#




