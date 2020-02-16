# """
# LeNet详细介绍
# https://blog.csdn.net/daydayup_668819/article/details/79932548
# """
#
# """
#
# 数据集的内容：包含0-9的手写数字
# 数据集的数量：60000个训练集/10000个次测试集
# 数据集的格式：28*28
# 数据集通道数：灰度图（一个通道）
#
# """

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2

# MNIST数据集的加载
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# 随机选一个图片并查看label
image_index = 123 # [0,59999]
# plt.imshow(x_train[image_index]) #彩色显示
# # plt.imshow(x_train[image_index], cmap="Greys")
# plt.show()
print(y_train[image_index])

# 将图片从28*28扩充为32*32
x_train = np.pad(x_train,((0,0),(2,2),(2,2)),'constant',constant_values=0) # pad详解见 https://blog.csdn.net/hustqb/article/details/77726660
x_test = np.pad(x_test,((0,0),(2,2),(2,2)),'constant',constant_values=0)
print(x_train.shape)

# 数据类型转换
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


# 数据正则化
x_train /= 255
x_test /= 255

# 数据维度转换
x_train = x_train.reshape(x_train.shape[0], 32, 32, 1)
x_test = x_test.reshape(x_test.shape[0], 32, 32, 1)
print(x_train.shape)

# LeNet模型的建立
# 定义模型
# 模型的构建:  tf.keras.Model 和tf.keras.layers
# 模型的损失函数：tf.keras.losses
# 模型的优化器：   tf.keras.optimizer
# 模型的评估：   tf.keras.metrics
class LeNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv_layer_1 = tf.keras.layers.Conv2D(
            filters = 6,
            kernel_size = (5, 5),
            padding = 'valid',
            activation = tf.nn.relu
        )

        self.pool_layer_1 = tf.keras.layers.AveragePooling2D(pool_size = (2, 2), padding = 'same')

        self.conv_layer_2 = tf.keras.layers.Conv2D(
            filters = 16,
            kernel_size = (5, 5),
            padding = 'valid',
            activation = tf.nn.relu
        )

        self.pool_layer_2 = tf.keras.layers.AveragePooling2D(pool_size = (2, 2), padding = 'same')

        self.flatten = tf.keras.layers.Flatten()

        self.fc_layer_1 = tf.keras.layers.Dense(
            units = 120,
            activation = tf.nn.relu
        )

        self.fc_layer_2 = tf.keras.layers.Dense(
            units=84,
            activation=tf.nn.relu
        )

        self.output_layer = tf.keras.layers.Dense(
            units=10,
            activation=tf.nn.softmax
        )

    def call(self, inputs):
        x = self.conv_layer_1(inputs)
        x = self.pool_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.pool_layer_2(x)
        x = self.flatten(x)
        x = self.fc_layer_1(x)
        x = self.fc_layer_2(x)
        output = self.output_layer(x)

        return output

# 另一种构建方法:  tf.keras.models.Sequential 和tf.keras.layers
model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=6,kernel_size=(5,5),padding='valid',activation=tf.nn.relu,input_shape=(32,32,1)),
    tf.keras.layers.AveragePooling2D(pool_size=(2,2),strides=(2,2),padding='same'),
    tf.keras.layers.Conv2D(filters=16,kernel_size=(5,5),padding='valid',activation=tf.nn.relu),
    tf.keras.layers.AveragePooling2D(pool_size=(2,2),strides=(2,2),padding='same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=120,activation=tf.nn.relu),
    # tf.keras.layers.Conv2D(filters=120,kernel_size=(5,5),strides=(1,1),activation='tanh',padding='valid'),
    # tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=84,activation=tf.nn.relu),
    tf.keras.layers.Dense(units=10,activation=tf.nn.softmax)
])

#模型展示
model.summary()

# 超参数设置
epoch_num = 10
batch_size = 64
learning_rate = 0.001

# 优化器
optimizer = tf.keras.optimizers.Adam(learning_rate)

# 编译模型
model.compile(optimizer = optimizer, loss = tf.keras.losses.sparse_categorical_crossentropy, metrics = ['accuracy'])

import datetime
start_time = datetime.datetime.now()

model.fit(
    x = x_train,
    y = y_train,
    batch_size = batch_size,
    epochs = epoch_num
)

end_time = datetime.datetime.now()
cost_time = end_time - start_time
print('cost_time:',cost_time)

model.save('./LeNet_Model.h5')

# 输出评估指标
print(model.evaluate(x_test, y_test))

# 预测
image_index = 8888
print(x_test[image_index].shape)
plt.imshow(x_test[image_index].reshape(32, 32), cmap='Greys')

pred = model.predict((x_test[image_index].reshape(1, 32, 32, 1)))
print(pred.argmax())


# 自己测试手写图片
model = tf.keras.models.load_model('./LeNet_Model.h5')
img = cv2.imread('E:/python-again/4.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 转灰
plt.imshow(img, cmap = 'Greys')
plt.show()

img = cv2.bitwise_not(img) # 将图片的底色和字的颜色取反

# 将底变成纯白色，将字变成纯黑色
img[img<200] = 0
img[img>200] = 255
plt.imshow(img, cmap = 'Greys')
plt.show()

img = cv2.resize(img, (32, 32)) # 将图片转换为规定尺寸

img = img.astype('float32') # 改数据类型

img /= 255 # 数据归一化

img = img.reshape(1, 32, 32, 1) #转换成规定格式

# 开始预测
pred = model.predict(img)

print(pred.argmax()) # 输出结果

