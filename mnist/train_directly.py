#
# Verify Reading Dataset via MnistDataloader class
#
# %matplotlib inline
# import random
import torch
import numpy as np
from os.path import join
import matplotlib.pyplot as plt

import dataset

#
# Set file paths based on added MNIST Datasets
#
input_path = "../dataset/mnist/"
training_images_filepath = join(
    input_path, "train-images-idx3-ubyte/train-images-idx3-ubyte"
)
training_labels_filepath = join(
    input_path, "train-labels-idx1-ubyte/train-labels-idx1-ubyte"
)
test_images_filepath = join(input_path, "t10k-images-idx3-ubyte/t10k-images-idx3-ubyte")
test_labels_filepath = join(input_path, "t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte")

#
# Load MINST dataset
#
mnist_dataloader = dataset.MnistDataloader(
    training_images_filepath,
    training_labels_filepath,
    test_images_filepath,
    test_labels_filepath,
)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

## 准备数据集

train_x = np.array(x_train, dtype=np.float32)  # x 是 float 类型
test_x = np.array(x_test, dtype=np.float32)
train_y = np.array(y_train, dtype=np.int64)  # y 是 int 类型
test_y = np.array(y_test, dtype=np.int64)

# print(train_x.shape, train_y.shape)
# print(test_x.shape, test_y.shape)

## 处理数据为一维
train_xx = train_x.reshape(train_x.shape[0], train_x.shape[1] * train_x.shape[2])
test_xx = test_x.reshape(test_x.shape[0], test_x.shape[1] * test_x.shape[2])

# print(train_xx.shape, test_xx.shape)

## 将数据转换为 tensor 格式
train_xxx, test_xxx, train_yyy, test_yyy = map(
    torch.tensor, (train_xx, test_xx, train_y, test_y)
)

# print(train_xxx.shape, train_yyy.shape)
# print(test_xxx.shape, test_yyy.shape)

## 准备参数

# 定义网络模型深度和参数
b1 = torch.zeros(28, requires_grad=True, dtype=torch.float32)
w1 = torch.randn(784, 28, requires_grad=True, dtype=torch.float32)
b2 = torch.zeros(10, requires_grad=True, dtype=torch.float32)
w2 = torch.randn(28, 10, requires_grad=True, dtype=torch.float32)
# print(b1.shape, b2.shape)
# print(w1.shape, w2.shape)

# Learning rate
lr1 = 0.01
lr2 = 0.001


## 模型训练


def train(x, y, w1, b1, w2, b2, lr):
    ### 1. 前向传播
    # 第一层网络运算
    t1 = x @ w1 + b1
    # 激活函数
    t1r = torch.nn.functional.relu(t1)

    # 第二层网络运算
    t2 = t1r @ w2 + b2

    ### 2. 计算损失
    loss = torch.nn.functional.cross_entropy(t2, y)
    # print(f"{i+1}: {loss}")

    # softmax 将结果转换为多分类问题的概率(各项概率和为 1)

    # one-hot 向量: 多分类任务中, 将指定分类位置设置为 1 其余位置都为 0 的向量
    # 例如 [0,1,2] 3 分类, 代表 `0` 位置的 one-hot 向量为 [1, 0, 0]

    ### 3. 反向传播
    loss.backward()

    ### 4. 求梯度更新参数
    w1.data.add_(-lr * w1.grad)
    b1.data.add_(-lr * b1.grad)

    w1.grad.zero_()
    b1.grad.zero_()

    w2.data.add_(-lr * w2.grad)
    b2.data.add_(-lr * b2.grad)

    w2.grad.zero_()
    b2.grad.zero_()

    return (lr, loss)


epochs = 5
batch_size = 64

for epoch in range(epochs):
    lr = lr1 if epoch < 3 else lr2

    for i in range(train_xxx.shape[0] // batch_size + 1):
        st = i * batch_size
        ed = st + batch_size

        x = train_xxx[st:ed]
        y = train_yyy[st:ed]

        lr, loss = train(x, y, w1, b1, w2, b2, lr)

        if i % 100 == 0:
            print(f"  {epoch+1}-{i} lr {lr} loss: {loss}")

    print(f"{epoch+1} lr {lr} loss: {loss}")
