"""
注意力汇聚：Nadaraya-Watson 核回归
    上节介绍了框架下的注意力机制的主要成分：
        查询（自主提示）和键（非自主提示）之间的交互形成了注意力汇聚； 
        注意力汇聚有选择地聚合了值（感官输入）以生成最终的输出。 

    本节将介绍注意力汇聚的更多细节， 以便从宏观上了解注意力机制在实践中的运作方式。 
        具体来说，1964年提出的Nadaraya-Watson核回归模型 是一个简单但完整的例子，可以用于演示具有注意力机制的机器学习。
"""
import torch
from torch import nn
from d2l import torch as d2l
"""
        [生成数据集]
        简单起见，考虑下面这个回归问题： 给定的成对的“输入－输出”数据集  {(𝑥1,𝑦1),…,(𝑥𝑛,𝑦𝑛)}， 如何学习 𝑓 来预测任意新输入 𝑥 的输出 𝑦̂ =𝑓(𝑥) ？

        根据下面的非线性函数生成一个人工数据集， 其中加入的噪声项为 𝜖 ：
                                                        𝑦𝑖=2sin(𝑥𝑖)+𝑥𝑖^0.8+𝜖,
"""
n_train = 50  # 训练样本数
x_train, _ = torch.sort(torch.rand(n_train) * 5)   # 排序后的训练样本

def f(x):
    return 2 * torch.sin(x) + x**0.8

y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))  # 训练样本的输出
x_test = torch.arange(0, 5, 0.1)  # 测试样本
y_truth = f(x_test)  # 测试样本的真实输出
n_test = len(x_test)  # 测试样本数
print(n_test)

"""
        下面的函数将绘制所有的训练样本（样本由圆圈表示）， 不带噪声项的真实数据生成函数 𝑓 （标记为“Truth”）， 以及学习得到的预测函数（标记为“Pred”）。
"""
def plot_kernel_reg(y_hat):
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'], xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5);

"""
        平均汇聚
            先使用最简单的估计器来解决回归问题。 基于平均汇聚来计算所有训练样本输出值的平均值：

                               n
                    𝑓(𝑥)=1/𝑛 * ∑𝑦𝑖, (1)
                              i=1

        如下图所示，这个估计器确实不够聪明。 真实函数 𝑓 （“Truth”）和预测函数（“Pred”）相差很大。
"""
y_hat = torch.repeat_interleave(y_train.mean(), n_test)
plot_kernel_reg(y_hat)

"""
        [非参数注意力汇聚]

        显然，平均汇聚忽略了输入 𝑥𝑖。 于是Nadaraya和 Watson提出了一个更好的想法， 根据输入的位置对输出 𝑦𝑖 进行加权：

                 n         n
            𝑓(𝑥)=∑(𝐾(𝑥−𝑥𝑖)/∑𝐾(𝑥−𝑥𝑗))𝑦𝑖, (2)
                i=1       j=1

            其中 𝐾 是核（kernel）。 
        公式所描述的估计器被称为 Nadaraya-Watson核回归（Nadaraya-Watson kernel regression）。 
        这里不会深入讨论核函数的细节，但受此启发，我们可以从./attention_cues.py中的注意力机制框架的角度重写nadaraya-watson， 
        成为一个更加通用的注意力汇聚（attention pooling）公式：

                 n
            𝑓(𝑥)=∑𝛼(𝑥,𝑥𝑖)𝑦𝑖, (3)
                i=1

            其中 𝑥 是查询， (𝑥𝑖,𝑦𝑖) 是键值对。 
        比较 (1) 和 (3)， 
            前者注意力汇聚是 𝑦𝑖 的加权平均。 
            后者将查询 𝑥 和键 𝑥𝑖 之间的关系建模为 注意力权重（attention weight） 𝛼(𝑥,𝑥𝑖)，这个权重将被分配给每一个对应值 𝑦𝑖。 
            对于任何查询，模型在所有键值对注意力权重都是一个有效的概率分布： 它们是非负的，并且总和为1。

        为了更好地理解注意力汇聚， 下面考虑一个高斯核（Gaussian kernel），其定义为：

            𝐾(𝑢)=1/√2𝜋 * exp(−𝑢^2/2).
         

        将高斯核代入 (2) 和 (3) 可以得到：

                 n
            𝑓(𝑥)=∑𝛼(𝑥,𝑥𝑖)𝑦𝑖
                i=1
                 n                     n
                =∑(exp(−1/2 * (𝑥−𝑥𝑖)^2)/∑exp(−1/2 * (𝑥−𝑥𝑗)^2)) * 𝑦𝑖
                i=1                   j=1
                 n
                =∑softmax(−1/2 * (𝑥−𝑥𝑖)^2)𝑦𝑖. (4)
                i=1
         

            在 (4) 中， 如果一个键 𝑥𝑖 越是接近给定的查询 𝑥 ，那么分配给这个键对应值 𝑦𝑖的注意力权重就会越大，也就“获得了更多的注意力”。

        值得注意的是，Nadaraya-Watson核回归是一个非参数模型。 
        因此，(4) 是 非参数的注意力汇聚（nonparametric attention pooling）模型。 
        接下来，我们将基于这个非参数的注意力汇聚模型来绘制预测结果。 
        从绘制的结果会发现新的模型预测线是平滑的，并且比平均汇聚的预测更接近真实。
"""
# X_repeat的形状:(n_test,n_train),
# 每一行都包含着相同的测试输入（例如：同样的查询）
X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
# x_train包含着键。attention_weights的形状：(n_test,n_train),
# 每一行都包含着要在给定的每个查询的值（y_train）之间分配的注意力权重
attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2 / 2, dim=1)
# y_hat的每个元素都是值的加权平均值，其中的权重是注意力权重
y_hat = torch.matmul(attention_weights, y_train)
plot_kernel_reg(y_hat)

"""
        现在来观察注意力的权重。 
        这里测试数据的输入相当于查询，而训练数据的输入相当于键。 
        因为两个输入都是经过排序的，因此由观察可知“查询-键”对越接近， 注意力汇聚的[注意力权重]就越高。
"""
d2l.show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')

"""
        [带参数注意力汇聚]

        非参数的Nadaraya-Watson核回归具有一致性（consistency）的优点： 如果有足够的数据，此模型会收敛到最优结果。 
        尽管如此，我们还是可以轻松地将可学习的参数集成到注意力汇聚中。

            例如，与 (4) 略有不同， 在下面的查询 𝑥 和键 𝑥𝑖 之间的距离乘以可学习参数 𝑤 ：

                     n
                𝑓(𝑥)=∑𝛼(𝑥,𝑥𝑖)𝑦𝑖
                    i=1
                     n                          n
                    =∑(exp(−1/2 * ((𝑥−𝑥𝑖)𝑤 )^2)/∑exp(−1/2 * ((𝑥−𝑥𝑗)𝑤 )^2)) * 𝑦𝑖
                    i=1                        j=1
                     n
                    =∑softmax(−1/2 * ((𝑥−𝑥𝑖)𝑤 )^2)𝑦𝑖. (5)
                    i=1

        本节的余下部分将通过训练这个模型 (5) 来学习注意力汇聚的参数。

        批量矩阵乘法

        为了更有效地计算小批量数据的注意力， 我们可以利用深度学习开发框架中提供的批量矩阵乘法。

        假设第一个小批量数据包含 𝑛 个矩阵 𝐗1,…,𝐗𝑛， 形状为 𝑎×𝑏， 第二个小批量包含 𝑛 个矩阵 𝐘1,…,𝐘𝑛， 形状为 𝑏×𝑐。 
        它们的批量矩阵乘法得到 𝑛 个矩阵  𝐗1𝐘1,…,𝐗𝑛𝐘𝑛， 形状为 𝑎×𝑐。 
        因此，[假定两个张量的形状分别是 (𝑛,𝑎,𝑏) 和 (𝑛,𝑏,𝑐)， 它们的批量矩阵乘法输出的形状为 (𝑛,𝑎,𝑐)]。
"""
X = torch.ones((2, 1, 4))
Y = torch.ones((2, 4, 6))
torch.bmm(X, Y).shape

"""
        在注意力机制的背景中，我们可以[使用小批量矩阵乘法来计算小批量数据中的加权平均值]。
"""
weights = torch.ones((2, 10)) * 0.1
values = torch.arange(20.0).reshape((2, 10))
torch.bmm(weights.unsqueeze(1), values.unsqueeze(-1))

"""
        定义模型
        基于 (4) 中的 [带参数的注意力汇聚]，使用小批量矩阵乘法， 定义Nadaraya-Watson核回归的带参数版本为：
"""
class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries, keys, values):
        # queries和attention_weights的形状为(查询个数，“键－值”对个数)
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        self.attention_weights = nn.functional.softmax(-((queries - keys) * self.w)**2 / 2, dim=1)
        # values的形状为(查询个数，“键－值”对个数)
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)

"""
    训练
    接下来，[将训练数据集变换为键和值]用于训练注意力模型。 
    在带参数的注意力汇聚模型中， 任何一个训练样本的输入都会和除自己以外的所有训练样本的“键－值”对进行计算， 从而得到其对应的预测输出。
"""
# X_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输入
X_tile = x_train.repeat((n_train, 1))
# Y_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输出
Y_tile = y_train.repeat((n_train, 1))
# keys的形状:('n_train'，'n_train'-1)
keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
# values的形状:('n_train'，'n_train'-1)
values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))

"""
    [训练带参数的注意力汇聚模型]时，使用平方损失函数和随机梯度下降。
"""
net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

for epoch in range(5):
    trainer.zero_grad()
    l = loss(net(x_train, keys, values), y_train)
    l.sum().backward()
    trainer.step()
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))

"""
    如下所示，训练完带参数的注意力汇聚模型后可以发现： 
    在尝试拟合带噪声的训练数据时， [预测结果绘制]的线不如之前非参数模型的平滑。
"""

# keys的形状:(n_test，n_train)，每一行包含着相同的训练输入（例如，相同的键）
keys = x_train.repeat((n_test, 1))
# value的形状:(n_test，n_train)
values = y_train.repeat((n_test, 1))
y_hat = net(x_test, keys, values).unsqueeze(1).detach()
plot_kernel_reg(y_hat)

"""
    为什么新的模型更不平滑了呢？ 下面看一下输出结果的绘制图： 
    与非参数的注意力汇聚模型相比， 带参数的模型加入可学习的参数后， [曲线在注意力权重较大的区域变得更不平滑]。
"""

d2l.show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')

"""
    小结
    Nadaraya-Watson核回归是具有注意力机制的机器学习范例。
    Nadaraya-Watson核回归的注意力汇聚是对训练数据中输出的加权平均。
    从注意力的角度来看，分配给每个值的注意力权重取决于将值所对应的键和查询作为输入的函数。
    注意力汇聚可以分为非参数型和带参数型。
"""
