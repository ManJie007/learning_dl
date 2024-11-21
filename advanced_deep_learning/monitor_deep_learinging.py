"""
使用 Keras 回调函数和 TensorBoard 来检查并监控深度学习模型
    
    本节将介绍在训练过程中如何更好地访问并控制模型内部过程的方法。使用 model.fit() 或 model.fit_generator() 在一个大型数据集上启动数十轮的训练，有点类似于扔一架纸飞 机，一开始给它一点推力，之后你便再也无法控制其飞行轨迹或着陆点。
    如果想要避免不好的 结果（并避免浪费纸飞机），更聪明的做法是不用纸飞机，而是用一架无人机，它可以感知其环境，将数据发回给操纵者，并且能够基于当前状态自主航行。
    我们下面要介绍的技术，可以让 model.fit() 的调用从纸飞机变为智能的自主无人机，可以自我反省并动态地采取行动。

训练过程中将回调函数作用于模型

    训练模型时，很多事情一开始都无法预测。
    尤其是你不知道需要多少轮才能得到最佳验证损失。
    前面所有例子都采用这样一种策略：训练足够多的轮次，这时模型已经开始过拟合，根据这第一次运行来确定训练所需要的正确轮数，然后使用这个最佳轮数从头开始再启动一次新的训练。
        当然，这种方法很浪费。
    处理这个问题的更好方法是，当观测到验证损失不再改善时就停止训练。
        这可以使用 Keras 回调函数来实现。
        回调函数（callback）是在调用 fit 时传入模型的一个对象（即实现特定方法的类实例），它在训练过程中的不同时间点都会被模型调用。
        它可以访问关于模型状态与性能的所有可用数据，还可以采取行动：中断训练、保存模型、加载一组不同的权重或改变模型的状态。
        
        回调函数的一些用法示例如下所示。
            模型检查点（model checkpointing）：在训练过程中的不同时间点保存模型的当前权重。
            提前终止（early stopping）：如果验证损失不再改善，则中断训练（当然，同时保存在训练过程中得到的最佳模型）。
            在训练过程中动态调节某些参数值：比如优化器的学习率。
            在训练过程中记录训练指标和验证指标，或将模型学到的表示可视化（这些表示也在不断更新）：你熟悉的 Keras 进度条就是一个回调函数！

        keras.callbacks 模块包含许多内置的回调函数，下面列出了其中一些，但还有很多没有列出来。
            keras.callbacks.ModelCheckpoint
            keras.callbacks.EarlyStopping
            keras.callbacks.LearningRateScheduler
            keras.callbacks.ReduceLROnPlateau
            keras.callbacks.CSVLogger

    下面介绍其中几个回调函数，让你了解如何使用它们：ModelCheckpoint、EarlyStopping 和 ReduceLROnPlateau。

        1. ModelCheckpoint 与 EarlyStopping 回调函数
            如果监控的目标指标在设定的轮数内不再改善，可以用 EarlyStopping 回调函数来中断训练。
                比如，这个回调函数可以在刚开始过拟合的时候就中断训练，从而避免用更少的轮次重新训练模型。
                这个回调函数通常与 ModelCheckpoint 结合使用，后者可以在训练过程中持续不断地保存模型（你也可以选择只保存目前的最佳模型，即一轮结束后具有最佳性能的模型）。
"""
import keras

#通过 fit 的 callbacks 参数将回调函数传入模型中，这个参数 接收一个回调函数的列表。你可以传入任意个数的回调函数
callbacks_list = [ 
 """
 如果不再改善，就中断训练
 monitor:监控模型的验证精度
 patience:如果精度在多于一轮的时间（即两轮）内不再改善，中断训练
 """
 keras.callbacks.EarlyStopping(monitor='acc', patience=1,),
 """
 在每轮过后保存当前权重
 filepath:目标模型文件的保存路径
 monitor:
 save_best_only:
 这两个参数的含义是，如果 val_loss 没有改善，那么不需要覆盖模型文件。
 这就可以始终保存在训练过程中见到的最佳模型
 """
 keras.callbacks.ModelCheckpoint(filepath='my_model.h5', monitor='val_loss', save_best_only=True,)
]

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc']) #你监控精度，所以它应该是模型指标的一部分

"""
注意，由于回调函数要监控验证损失和验证精度，所以在调用 fit 时需要传入 validation_data（验证数据）
"""
model.fit(x, y, epochs=10, batch_size=32, callbacks=callbacks_list, validation_data=(x_val, y_val)) 

"""
2. ReduceLROnPlateau 回调函数

如果验证损失不再改善，你可以使用这个回调函数来降低学习率。
    在训练过程中如果出现了损失平台（loss plateau），那么增大或减小学习率都是跳出局部最小值的有效策略。
下面这个示例使用了 ReduceLROnPlateau 回调函数。
"""

"""
monitor:监控模型的验证损失
factor:触发时将学习率除以 10
patience:如果验证损失在 10 轮内都没有改善，那么就触发这个回调函数
"""
callbacks_list = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss' ,factor=0.1, patience=10,)]

"""
注意，由于回调函数要监控验证损失和验证精度，所以在调用 fit 时需要传入 validation_data（验证数据）
"""
model.fit(x, y, epochs=10, batch_size=32, callbacks=callbacks_list, validation_data=(x_val, y_val))

"""
3. 编写你自己的回调函数
如果你需要在训练过程中采取特定行动，而这项行动又没有包含在内置回调函数中，那么可以编写你自己的回调函数。
    回调函数的实现方式是创建 keras.callbacks.Callback 类的子类。
    然后你可以实现下面这些方法（从名称中即可看出这些方法的作用），它们分别在训练过程中的不同时间点被调用。
        on_epoch_begin      在每轮开始时被调用           
        on_epoch_end        在每轮结束时被调用
        on_batch_begin      在处理每个批量之前调用 
        on_batch_end        在处理每个批量之后调用
        on_train_begin      在训练开始时被调用
        on_train_end        在训练结束时被调用

    这些方法被调用时都有一个 logs 参数，这个参数是一个字典，里面包含前一个批量、前一个轮次或前一次训练的信息，即训练指标和验证指标等。
    此外，回调函数还可以访问下列属性。
        self.model：调用回调函数的模型实例。
        self.validation_data：传入 fit 作为验证数据的值。

下面是一个自定义回调函数的简单示例，它可以在每轮结束后将模型每层的激活保存到硬盘（格式为 Numpy 数组），这个激活是对验证集的第一个样本计算得到的。
"""

import keras
import numpy as np

class ActivationLogger(keras.callbacks.Callback):
    def set_model(self, model):
        self.model = model #在训练之前由父模型调用，告诉回调函数是哪个模型在调用它
        layer_outputs = [layer.output for layer in model.layers]
        self.activations_model = keras.models.Model(model.input, layer_outputs) #模型实例，返回每层的激活

    def on_epoch_end(self, epoch, logs=None):
        if self.validation_data is None:
            raise RuntimeError('Requires validation_data.')
        #获取验证数据的第一个输入样本
        validation_sample = self.validation_data[0][0:1] 
        activations = self.activations_model.predict(validation_sample)
        #将数组保存 到硬盘
        f = open('activations_at_epoch_' + str(epoch) + '.npz', 'w') 
        np.savez(f, activations)
        f.close()

#关于回调函数你只需要知道这么多，其他的都是技术细节，很容易就能查到。
#现在，你已经可以在训练过程中对一个 Keras 模型执行任何类型的日志记录或预定程序的干预。

"""
TensorBoard 简介：TensorFlow 的可视化框架

想要做好研究或开发出好的模型，在实验过程中你需要丰富频繁的反馈，从而知道模型内部正在发生什么。
这正是运行实验的目的：获取关于模型表现好坏的信息，越多越好。
取得进展是一个反复迭代的过程（或循环）：
    首先你有一个想法，并将其表述为一个实验，用于验证你的想法是否正确。
    你运行这个实验，并处理其生成的信息。
    这又激发了你的下一个想法。

    在这个循环中实验的迭代次数越多，你的想法也就变得越来越精确、越来越强大。
    Keras 可以帮你在最短的时间内将想法转化成实验，而高速 GPU 可以帮你尽快得到实验结果。
    但如何处理实验结果呢？这就需要 TensorBoard 发挥作用了。

    本节将介绍 TensorBoard，一个内置于 TensorFlow 中的基于浏览器的可视化工具。
        注意，只有当 Keras 使用 TensorFlow 后端时，这一方法才能用于 Keras 模型。
        TensorBoard 的主要用途是，在训练过程中帮助你以可视化的方法监控模型内部发生的一切。
        如果你监控了除模型最终损失之外的更多信息，那么可以更清楚地了解模型做了什么、没做什么，并且能够更快地取得进展。
        TensorBoard 具有下列巧妙的功能，都在浏览器中实现。
            在训练过程中以可视化的方式监控指标
            将模型架构可视化
            将激活和梯度的直方图可视化
            以三维的形式研究嵌入

我们用一个简单的例子来演示这些功能：在 IMDB 情感分析任务上训练一个一维卷积神经网络。
    这个模型类似于 ../words_sequence/convnet_sequence.py 节的模型。我们将只考虑 IMDB 词表中的前 2000 个单词，这样更易于将词嵌入可视化。
"""
from keras import layers
from keras.datasets import imdb
from keras.preprocessing import sequence

max_features = 2000 #作为特征的单词个数
max_len = 500 #在这么多单词之后截断文本（这些单词都属于前 max_features 个最常见的单词）

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

model = keras.models.Sequential()
model.add(layers.Embedding(max_features, 128, input_length=max_len, name='embed'))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))

model.summary()
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

"""
在开始使用 TensorBoard 之前，我们需要创建一个目录，用于保存它生成的日志文件。
    $mkdir my_log_dir
我们用一个 TensorBoard 回调函数实例来启动训练。
    这个回调函数会将日志事件写入硬盘的指定位置。
        log_dir:日志文件将被写入这个位置
        histogram_freq:每一轮之后记录激活直方图
        embeddings_freq:每一轮之后记录嵌入数据
"""
callbacks = [keras.callbacks.TensorBoard(log_dir='my_log_dir', histogram_freq=1, embeddings_freq=1,)]

history = model.fit(x_train, y_train, epochs=20, batch_size=128, validation_split=0.2, callbacks=callbacks)

"""
现在，你可以在命令行启动 TensorBoard 服务器，指示它读取回调函数当前正在写入的日志。
在安装 TensorFlow 时（比如通过 pip），tensorboard 程序应该已经自动安装到计算机里了。
    $ tensorboard --logdir=my_log_dir
然后可以用浏览器打开 http://localhost:6006，并查看模型的训练过程。
除了训练指标和验证指标的实时图表之外，你还可以访问 HISTOGRAMS（直方图）标签页，并查看美观的直方图可视化，直方图中是每层的激活值。

    EMBEDDINGS（嵌入）标签页让你可以查看输入词表中 2000 个单词的嵌入位置和空间关系，它们都是由第一个 Embedding 层学到的。
    因为嵌入空间是 128 维的，所以 TensorBoard 会使用你选择的降维算法自动将其降至二维或三维，可选的降维算法有主成分分析（PCA）和 t-分布 随机近邻嵌入（t-SNE）。
    在图所示的点状云中，可以清楚地看到两个簇：
        正面含义的词和负面含义的词。
    从可视化图中可以立刻明显地看出，将嵌入与特定目标联合训练得到的模型是完全针对这个特定任务的，这也是为什么使用预训练的通用词嵌入通常不是一个好主意。

    GRAPHS（图）标签页显示的是 Keras 模型背后的底层 TensorFlow 运算图的交互式可视化。
    可见，图中的内容比之前想象的要多很多。
        对于你刚刚构建的模型，在 Keras 中定义模型时可能看起来很简单，只是几个基本层的堆叠；
        但在底层，你需要构建相当复杂的图结构来使其生效。
        其中许多内容都与梯度下降过程有关。
            你所见到的内容与你所操作的内容之间存在这种复杂度差异，这正是你选择使用 Keras 来构建模型、而不是使用原始 TensorFlow 从头开始定义所有内容的主要动机。
        Keras 让工作流程变得非常简单。

注意，Keras 还提供了另一种更简洁的方法——keras.utils.plot_model 函数，它可以将模型绘制为层组成的图，而不是 TensorFlow 运算组成的图。
使用这个函数需要安装 Python 的 pydot 库和 pydot-ng 库，还需要安装 graphviz 库。
"""

from keras.utils import plot_model

plot_model(model, to_file='model.png')

"""
你还可以选择在层组成的图中显示形状信息。

下面这个例子使用 plot_model 函数及show_shapes 选项将模型拓扑结构可视化
"""

from keras.utils import plot_model

plot_model(model, show_shapes=True, to_file='model.png')

"""
　小结
    Keras 回调函数提供了一种简单方法，可以在训练过程中监控模型并根据模型状态自动采取行动。
    使用 TensorFlow 时，TensorBoard 是一种在浏览器中将模型活动可视化的好方法。
        在Keras 模型中你可以通过 TensorBoard 回调函数来使用这种方法。
"""
