import numpy as np
import h5py
import matplotlib.pyplot as plt
from PIL import Image
import pickle

def sigmoid(z):
    """
    输出函数
    :param z: 输入的数据
    :return: 激活后的值
    """
    return 1 / (1 + np.exp(-z))

class SingleNeural:

    def __init__(self, in_feature, bias=True):
        """
        无训练好的的参数的话就初始化为0的参数，w和b 一样
        因为只有一个神经元，所以可以初始化为0
        但多个神经元就一定不能为0了，必须随机给数字，不然就跟一个神经元一样了

        :param in_feature: 接收的特征点的数目
        :param bias: 是否加偏置项b
        """
        self.weight = np.zeros((in_feature, 1))
        if bias:
            self.bias = 0

    def __call__(self, x):

        return self.forward(x)

    def forward(self, x):
        """
        前向传播
        :param x: 输入的数据 必须为一个特征向量的形式传进来
        :return: 输出的结果
        """
        assert(x.ndim == 2)  # x的形状为 N V结构

        x = x.T  # 转置为 可以跟w相乘的形式

        z = np.dot(self.weight.T, x)

        if hasattr(self, "bias"):
            z += self.bias

        A = sigmoid(z)
        
        return A

class BCELoss:

    def __call__(self, A, Y, input_data):
        """
        默认求均值损失
        :param A: 输出的结果
        :param Y: 数据的标签
        :param input_data: 输入的原始数据
        :return: self 对象本身
        """
        self.A = A
        self.Y = Y
        self.X = input_data
        self.m = Y.size  # 样本的个数
        self.loss = -np.sum((Y * np.log(A) + (1-Y) * np.log(1-A))) / self.m  #所有样本的平均损失
        return self

    def backward(self):
        """
        反向传播求导
        :return: None
        """

        dz = self.A - self.Y

        dw = np.dot(self.X.T, dz.T) / self.m

        db = np.sum(dz) / self.m

        return {"dw": dw, "db": db}

    def float(self):
        return self.loss

class optimizer:
    """
    # 定义优化器， 更新梯度
    # :param net: 需要更新的网络参数
    # :return:None
    """
    def __init__(self, net, lr=0.01):
        self.net = net
        self.lr = lr

    def step(self, grads):

        self.net.weight = self.net.weight - self.lr * grads['dw']
        if hasattr(self.net, "bias"):
            self.net.bias = self.net.bias - self.lr * grads['db']

class Train:

    def __init__(self):
        """
        初始化神经元网络
        初始化训练集和测试集
        """
        self.net = SingleNeural(64*64*3)
        self.train_set, self.test_set = self.get_dataset()
        self.loss_func = BCELoss()
        self.optimizer = optimizer(self.net)

    def get_dataset(self):
        """
        读取数据集
        :return: 训练集， 测试集
        """
        train_set = h5py.File("datasets/train_catvnoncat.h5", mode="r")
        test_set = h5py.File("datasets/test_catvnoncat.h5", mode="r")

        return train_set, test_set

    def load_train_data(self, train_set):
        """
        加载训练集的数据、标签、类别
        classes：non-cat 和 cat 两类
        :param train_set: 训练集
        :return: 训练集数据， 标签和类别
        """

        train_target = train_set["train_set_y"][:]
        train_data = train_set["train_set_x"][:] / 255.  # 归一化
        classes = train_set["list_classes"][:]

        return train_data, train_target, classes

    def load_test_data(self, test_set):
        """
        同上（测试集）
        :param test_set:
        :return:
        """
        test_target = test_set["test_set_y"][:]

        test_data = test_set["test_set_x"][:] / 255.  # 归一化

        classes = test_set["list_classes"][:]

        return test_data, test_target, classes

    def train(self):
        """
        开始训练网络
        :return:
        """
        cost = []  # 累计所有的损失，并画出历史损失走势图
        for i in range(10000):
            input, target, classes = self.load_train_data(self.train_set)
            # 输入数据做形状变换
            input = input.reshape(input.shape[0], -1)
            # 前向传播
            output = self.net(input)
            # 把结果和标签代入损失函数求损失
            loss = self.loss_func(output, target, input)
            # 损失再对w和b求导
            grads = loss.backward()
            # 通过优化器来更新梯度
            self.optimizer.step(grads)

            if i % 100 == 0:
                cost.append(loss.float())
                print("第{}次优化后，损失是:{}".format(i, loss.float()))
                plt.clf()
                plt.plot(cost)
                plt.pause(0.1)
        self.save(self.net, "my_net.m")


    def save(self, net, path):
        """
        使用pickle 序列化到本地文件中，保存模型
        :param net:
        :param path:
        :return:
        """
        with open(path, "wb") as f:
            pickle.dump(net, f)

    def load(self, path):
        """
        使用pickle 反序列化到内存中，读取模型
        :param path:
        :return: 保存的网络模型
        """
        with open(path, "rb") as f:
            net = pickle.load(f)
            return net


    def predict(self):
        """
        拿测试集做预测
        :return: 输出预测的正确率
        """
        input, target, classes = self.load_test_data(self.test_set)
        net = self.load("my_net.m")

        # # 传入训练好的参数，实例化网络
        x = input
        input = input.reshape(input.shape[0], -1)  # reshape to N V  struct
        output = net(input)
        # 把结果拿来做预测，1就是有猫，0就是没猫
        # print(output)
        prediction = np.where(output >= 0.5, 1, 0)
        # for i in range(x.shape[0]):
        #     plt.clf()
        #     plt.axis("off")
        #     plt.imshow(x[i])
        #     plt.text(0, -2, "cat" if prediction[0, i] > 0 else "non-cat", fontsize=20, color="red")
        #     plt.pause(1)

        # print(prediction)
        # 当然用 output.round() 也可以，直接四舍五入了返回0，1的结果
        result = (prediction.flatten() == target).mean()
        # print("正确率:", str(result * 100) + "%")

        img = Image.open(r"/Users/Mical/Documents/dataset_image/bg_pic/pic99.jpg")
        img = img.resize((64, 64), Image.ANTIALIAS)
        img = np.array(img) / 255.

        # img = img[np.newaxis,:] # 等价于 np.expand_dims() 添加一个维度
        # print(img.shape)
        img = np.expand_dims(img, axis=0)  # 等价于 torch.unsqueeze(0)
        output = net(img.reshape(1, -1))
        if output.round().item() == 1:
            print("猫，置信度为:{}".format(str(output.item() * 100) + "%"))
        else:
            print("不是猫,置信度为:{}".format(str(output.item() * 100) + "%"))



if __name__ == '__main__':

    t = Train()
    # t.train()
    t.predict()


