"""
本章将介绍几种强大的工具，可以让你朝着针对困难问题来开发最先进模型这一目标更近一步。
|----利用 Keras 函数式 API，你可以构建类图（graph-like）模型、在不同的输入之间共享某一层，并且还可以像使用 Python 函数一样使用 Keras 模型。
|----Keras 回调函数和 TensorBoard 基于浏览器的可视化工具，让你可以在训练过程中监控模型。
|----我们还会讨论其他几种最佳实践，包括批标准化、残差连接、超参数优化和模型集成。

不用 Sequential 模型的解决方案：Keras 函数式 API
    到目前为止，本书介绍的所有神经网络都是用 Sequential 模型实现的。
    Sequential 模型假设，网络只有一个输入和一个输出，而且网络是层的线性堆叠。

             output
               ^
               | 
        |-------------|
        |  ---------  |
        |  | layer |  |   
        |  ---------  |
        |  ---------  |
        |  | layer |  |   
        |  ---------  |
        |  ---------  |
        |  | layer |  |   
        |  ---------  |
        |-------------|
    Sequential
               ^
               | 
             input
    这是一个经过普遍验证的假设。
        这种网络配置非常常见，以至于本书前面只用 Sequential模型类就能够涵盖许多主题和实际应用。
        但有些情况下这种假设过于死板。
        有些网络需要多个独立的输入，有些网络则需要多个输出，而有些网络在层与层之间具有内部分支，这使得网络 看起来像是层构成的图（graph），而不是层的线性堆叠。

            例如，有些任务需要多模态（multimodal）输入。
                这些任务合并来自不同输入源的数据，并使用不同类型的神经层处理不同类型的数据。

            假设有一个深度学习模型，试图利用下列输入来预测一件二手衣服最可能的市场价格：
                用户提供的元数据（比如商品品牌、已使用年限等）、用户提供的文本描述与商品照片。
                    如果你只有元数据，那么可以使用 one-hot 编码，然后用密集连接网络来预测价格。
                    如果你只有文本描述，那么可以使用循环神经网络或一维卷积神经网络。 
                    如果你只有图像，那么可以使用二维卷积神经网络。
                但怎么才能同时使用这三种数据呢？
                    一种朴素的方法是训练三个独立的模型，然后对三者的预测做加权平均。
                        但这种方法可能不是最优的，因为模型提取的信息可能存在冗余。
                    更好的方法是使用一个可以同时查看所有可用的输入模态的模型，从而联合学习一个更加精确的数据模型——这个模型具有三个输入分支

                                                   价格预测
                                                      ^
                                                      |
                                                 ------------
                                                 | 合并模块 |
                                                 ------------
                                    /                  |                 \
                        --------------          --------------          ------------------
                        | Dense 模块 |          |  RNN  模块 |          |卷积神经网络模块|
                        --------------          --------------          ------------------
                               ^                       ^                         ^      
                               |                       |                         |      
                             元数据                 文本描述                    图片  

        同样，有些任务需要预测输入数据的多个目标属性。
            给定一部小说的文本，你可能希望将它按类别自动分类（比如爱情小说或惊悚小说），同时还希望预测其大致的写作日期。
            当然，你可以训练两个独立的模型：一个用于划分类别，一个用于预测日期。
            但由于这些属性并不是统计无关的，你可以构建一个更好的模型，用这个模型来学习同时预测类别和日期。
            这种联合模型将有两个输出，或者说两个头（head）。
            因为类别和日期之间具有相关性，所以知道小说的写作日期有助于模型在小说类别的空间中学到丰富而又准确的表示，反之亦然。

                             类别                    日期
                              ^                       ^                         ^      
                              |                       |                         |      
                        --------------          --------------
                        | 类别分类器 |          | 日期回归器 |
                        --------------          --------------
                                  \                 /
                                    ----------------
                                    | 文本处理模块 |
                                    ----------------
                                           ^
                                           |
                                        小说文本
                                一个多输出（或多头）模型

        此外，许多最新开发的神经架构要求非线性的网络拓扑结构，即网络结构为有向无环图。
            比如，Inception 系列网络（由 Google 的 Szegedy 等人开发） 依赖于 Inception 模块，其输入被多个并行的卷积分支所处理，然后将这些分支的输出合并为单个张量。

                                                                    
                                                                  output
                                                                    |
                                                             --------------
                                                             |    连接    |
                                                             --------------
                                                                    |
                              |-------------------------------------------------------------------------|
                              |                       |                       |                  --------------
                              |                       |                       |                  |   Conv2D   |
                              |                       |                       |                  | 3x3 step=2 |
                              |                       |                       |                  --------------
                              |                       |                       |                         |
                              |                 --------------           --------------          --------------
                              |                 |   Conv2D   |           | AvgRool2D  |          |   Conv2D   |
                              |                 | 3x3 step=2 |           |    3x3     |          |     3x3    |
                              |                 --------------           --------------          --------------
                              |                       |                         |                       |
                        --------------          --------------           --------------          --------------
                        |   Conv2D   |          |   Conv2D   |           | AvgRool2D  |          |   Conv2D   |
                        | 1x1 step=2 |          |     1x1    |           | 3x3 step=2 |          |     1x1    |
                        --------------          --------------           --------------          --------------
                              |                        |                        |                       |
                              |-------------------------------------------------------------------------|
                                                                    |
                                                                    |
                                                                  input
                                                Inception 模块：层组成的子图，具有多个并行卷积分支

            最近还有一种趋势是向模型中添加残差连接（residual connection），它最早出现于 ResNet 系列网络（由微软的何恺明等人开发）。
                残差连接是将前面的输出张量与后面的输出张量相加，从而将前面的表示重新注入下游数据流中，这有助于防止信息处理流程中的信息损失。
                这种类图 网络还有许多其他示例。   

                                                 -----------
                                                 |  layer  |
                                                 -----------
                                                      |
                                                      o <---------
                                                      |          |
                                                 -----------     |
                                                 |  layer  |     |
                                                 -----------     |  残查连接
                                                      |          |
                                                 -----------     |
                                                 |  layer  |     |
                                                 -----------     |
                                                      |-----------
                                                 -----------
                                                 |  layer  |
                                                 -----------

                                残差连接：通过特征图相加将前面的信息重新注入下游数据

这三个重要的使用案例（多输入模型、多输出模型和类图模型），只用 Keras 中的 Sequential 模型类是无法实现的。
但是还有另一种更加通用、更加灵活的使用 Keras 的方式，就是函数式 API（functional API）。
    本节将会详细介绍函数式 API 是什么、能做什么以及如何使用它。
"""

"""
函数式 API 简介
    使用函数式 API，你可以直接操作张量，也可以把层当作函数来使用，接收张量并返回张量（因此得名函数式 API）。
"""

from keras import Input, layers

input_tensor = Input(shape=(32,)) #一个张量

dense = layers.Dense(32, activation='relu') #一个层是一个函数 

output_tensor = dense(input_tensor) #可以在一个张量上调用一个层，它会返回一个张量

#我们首先来看一个最简单的示例，并列展示一个简单的 Sequential 模型以及对应的函数式 API 实现。

from keras.api.models import Sequential, Model
from keras.api import layers
from keras.api import Input

#前面学过的 Sequential 模型
seq_model = Sequential() 
seq_model.add(layers.Dense(32, activation='relu', input_shape=(64,)))
seq_model.add(layers.Dense(32, activation='relu'))
seq_model.add(layers.Dense(10, activation='softmax'))

"""
对应的函数式 API 实现
"""
input_tensor = Input(shape=(64,)) 
x = layers.Dense(32, activation='relu')(input_tensor)
x = layers.Dense(32, activation='relu')(x)
output_tensor = layers.Dense(10, activation='softmax')(x)

"""
Model 类将输入张量和输出张量转换为一个模型
"""
model = Model(input_tensor, output_tensor) 
model.summary() #查看模型

"""
>>> model.summary() #查看模型
Model: "functional_3"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Layer (type)                        ┃ Output Shape               ┃        Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ input_layer_1 (InputLayer)          │ (None, 64)                 │              0 │
├─────────────────────────────────────┼────────────────────────────┼────────────────┤
│ dense_3 (Dense)                     │ (None, 32)                 │          2,080 │
├─────────────────────────────────────┼────────────────────────────┼────────────────┤
│ dense_4 (Dense)                     │ (None, 32)                 │          1,056 │
├─────────────────────────────────────┼────────────────────────────┼────────────────┤
│ dense_5 (Dense)                     │ (None, 10)                 │            330 │
└─────────────────────────────────────┴────────────────────────────┴────────────────┘
 Total params: 3,466 (13.54 KB)
 Trainable params: 3,466 (13.54 KB)
 Non-trainable params: 0 (0.00 B)

这里只有一点可能看起来有点神奇，就是将 Model 对象实例化只用了一个输入张量和一个输出张量。
Keras 会在后台检索从 input_tensor 到 output_tensor 所包含的每一层，并将这些层组合成一个类图的数据结构，即一个 Model。
当然，这种方法有效的原因在于，output_tensor 是通过对 input_tensor 进行多次变换得到的。
如果你试图利用不相关的输 入和输出来构建一个模型，那么会得到 RuntimeError。

>>> unrelated_input = Input(shape=(32,))
>>> bad_model = model = Model(unrelated_input, output_tensor)
RuntimeError: Graph disconnected: cannot
obtain value for tensor Tensor("input_1:0", shape=(?, 64), dtype=float32) at layer "input_1".

这个报错告诉我们，Keras 无法从给定的输出张量到达 input_1。
对这种 Model 实例进行编译、训练或评估时，其 API 与 Sequential 模型相同。
"""
model.compile(optimizer='rmsprop', loss='categorical_crossentropy') #编译模型

#生成用于训练的虚构 Numpy 数据
import numpy as np 
x_train = np.random.random((1000, 64))
y_train = np.random.random((1000, 10))

model.fit(x_train, y_train, epochs=10, batch_size=128) #训练 10 轮模型

score = model.evaluate(x_train, y_train) #评估模型

"""
多输入模型
    函数式 API 可用于构建具有多个输入的模型。
    通常情况下，这种模型会在某一时刻用一个可以组合多个张量的层将不同的输入分支合并，张量组合方式可能是相加、连接等。
    这通常利用 Keras 的合并运算来实现，比如 keras.layers.add、keras.layers.concatenate 等。

    我们来看一个非常简单的多输入模型示例——一个问答模型。
        典型的问答模型有两个输入：一个自然语言描述的问题和一个文本片段（比如新闻文章），后者提供用于回答问题的信息。
        然后模型要生成一个回答，在最简单的情况下，这个回答只包含一个词，可以通过对某个预定义的词表做 softmax 得到。

                             回答
                              ^
                              |
                       ---------------
                       |    Dense    |
                       ---------------
                              ^
                              |
                       ---------------
                       | Concatenate |
                       ---------------
                        /            \                  
            --------------          -------------                            
            |    LSTM    |          |   LSTM    |                            
            --------------          -------------                            
                  |                       |
            --------------          -------------                            
            | Embedding  |          | Embedding |                            
            --------------          -------------                            
                  |                       |
               参考文本                  问题

下面这个示例展示了如何用函数式 API 构建这样的模型。
    我们设置了两个独立分支，首先将文本输入和问题输入分别编码为表示向量，然后连接这些向量，最后，在连接好的表示上添加一个 softmax 分类器。
"""
#用函数式 API 实现双输入问答模型
from keras.models import Model
from keras import layers
from keras import Input

text_vocabulary_size = 10000
question_vocabulary_size = 10000
answer_vocabulary_size = 500

#文本输入是一个长度可变的 整数序列。
#注意，你可以选 择对输入进行命名
text_input = Input(shape=(None,), dtype='int32', name='text') 
#将输入嵌入长度为 64 的向量
embedded_text = layers.Embedding(text_vocabulary_size, 64)(text_input) 
#利用 LSTM 将向量编码为单个向量
encoded_text = layers.LSTM(32)(embedded_text) 

#对问题进行相同的处理（使用不同的层实例）
question_input = Input(shape=(None,), dtype='int32', name='question') 
embedded_question = layers.Embedding(question_vocabulary_size, 32)(question_input)
encoded_question = layers.LSTM(16)(embedded_question)

#将编码后的问题和文本连接起来
concatenated = layers.concatenate([encoded_text, encoded_question], axis=-1) 

#在上面添加一个 softmax 分类器
answer = layers.Dense(answer_vocabulary_size, activation='softmax')(concatenated)

#在模型实例化时，指定 两个输入和输出
model = Model([text_input, question_input], answer) 
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

"""
接下来要如何训练这个双输入模型呢？
    有两个可用的 API：
        我们可以向模型输入一个由Numpy 数组组成的列表，
        或者也可以输入一个将输入名称映射为 Numpy 数组的字典。
            当然，只有输入具有名称时才能使用后一种方法。
"""

#将数据输入到多输入模型中

import numpy as np

num_samples = 1000
max_length = 100

#生成虚构的 Numpy 数据
text = np.random.randint(1, text_vocabulary_size, size=(num_samples, max_length)) 
question = np.random.randint(1, question_vocabulary_size, size=(num_samples, max_length))
answers = np.random.randint(answer_vocabulary_size, size=(num_samples))
answers = keras.utils.to_categorical(answers, answer_vocabulary_size) #回答是 one-hot 编 码的，不是整数

#使用输入组成的 列表来拟合
model.fit([text, question], answers, epochs=10, batch_size=128) 
#使用输入组成的字典来拟合 （只有对输入进行命名之后才 能用这种方法）
model.fit({'text': text, 'question': question}, answers, epochs=10, batch_size=128)

"""
多输出模型

利用相同的方法，我们还可以使用函数式 API 来构建具有多个输出（或多头）的模型。
    一个简单的例子就是一个网络试图同时预测数据的不同性质，比如一个网络，输入某个匿名人士 的一系列社交媒体发帖，然后尝试预测那个人的属性，比如年龄、性别和收入水平。

                              年龄                    收入                  性别  
                               ^                       ^                      ^      
                               |                       |                      |      
                          -------------          -------------          -------------
                          |   Dense   |          |   Dense   |          |   Dense   |
                          -------------          -------------          -------------
                                    \                  |                 /
                                              ------------------
                                              |一纬卷积神经网络|
                                              ------------------
                                                      ^
                                                      |
                                                 社交媒体发帖

                                            具有三个头的社交媒体模型
"""

#用函数式 API 实现一个三输出模型
from keras import layers
from keras import Input
from keras.models import Model

vocabulary_size = 50000
num_income_groups = 10

posts_input = Input(shape=(None,), dtype='int32', name='posts')
embedded_posts = layers.Embedding(256, vocabulary_size)(posts_input)
x = layers.Conv1D(128, 5, activation='relu')(embedded_posts)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation='relu')(x)

age_prediction = layers.Dense(1, name='age')(x) 
income_prediction = layers.Dense(num_income_groups, activation='softmax', name='income')(x)
gender_prediction = layers.Dense(1, activation='sigmoid', name='gender')(x) 

model = Model(posts_input, [age_prediction, income_prediction, gender_prediction])

"""
重要的是，训练这种模型需要能够对网络的各个头指定不同的损失函数，例如，年龄预测是标量回归任务，而性别预测是二分类任务，二者需要不同的训练过程。
但是，梯度下降要求将一个标量最小化，所以为了能够训练模型，我们必须将这些损失合并为单个标量。
    合并不同 损失最简单的方法就是对所有损失求和。
        在 Keras 中，你可以在编译时使用损失组成的列表或字典来为不同输出指定不同损失，然后将得到的损失值相加得到一个全局损失，并在训练过程中将这个损失最小化。
"""

#多输出模型的编译选项：多重损失
model.compile(optimizer='rmsprop', loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'])

model.compile(optimizer='rmsprop', loss={'age': 'mse', 'income': 'categorical_crossentropy', 'gender': 'binary_crossentropy'})


"""
注意，严重不平衡的损失贡献会导致模型表示针对单个损失值最大的任务优先进行优化，而不考虑其他任务的优化。

为了解决这个问题，我们可以为每个损失值对最终损失的贡献分配不同大小的重要性。

如果不同的损失值具有不同的取值范围，那么这一方法尤其有用。比如，用于年龄回归任务的均方误差（MSE）损失值通常在 3~5 左右，而用于性别分类任务的交叉熵损失值可能低至 0.1。
    在这种情况下，为了平衡不同损失的贡献，我们可以让交叉熵损失的权重取 10，而 MSE 损失的权重取 0.5。
"""

#多输出模型的编译选项：损失加权
loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'], loss_weights=[0.25, 1., 10.])

#与上述写法等效（只有输出层具有名称时才能采用这种写法）
model.compile(optimizer='rmsprop', loss={'age': 'mse', 'income': 'categorical_crossentropy', 'gender': 'binary_crossentropy'}, loss_weights={'age': 0.25, 'income': 1., 'gender': 10.})

#与多输入模型相同，多输出模型的训练输入数据可以是 Numpy 数组组成的列表或字典。
#将数据输入到多输出模型中

#假设 age_targets、 income_targets 和 gender_targets 都 是 Numpy 数组
model.fit(posts, [age_targets, income_targets, gender_targets], epochs=10, batch_size=64) 

#与上述写法等效（只 有输出层具有名称时 才能采用这种写法）
model.fit(posts, {'age': age_targets, 'income': income_targets, 'gender': gender_targets}, epochs=10, batch_size=64)

"""
层组成的有向无环图

    利用函数式 API，我们不仅可以构建多输入和多输出的模型，而且还可以实现具有复杂的内部拓扑结构的网络。

    Keras 中的神经网络可以是层组成的任意有向无环图（directed acyclic graph）。
        无环（acyclic）这个限定词很重要，即这些图不能有循环。
            张量 x 不能成为生成 x 的 某一层的输入。
        唯一允许的处理循环（即循环连接）是循环层的内部循环。
        
一些常见的神经网络组件都以图的形式实现。
    两个著名的组件是 Inception 模块和残差连接。
    为了更好地理解如何使用函数式 API 来构建层组成的图，我们来看一下如何用 Keras 实现这二者。


1. Inception 模块
    Inception 是一种流行的卷积神经网络的架构类型，它由 Google 的 Christian Szegedy 及其同事在 2013—2014 年开发，其灵感来源于早期的 network-in-network 架构。 
    
        它是模块的堆叠，这些模块本身看起来像是小型的独立网络，被分为多个并行分支。
        Inception 模块最基本的形式包含 3~4 个分支，首先是一个 1×1 的卷积，然后是一个 3×3 的卷积，最后将所得到的特征连接在一起。
        这种设置有助于网络分别学习空间特征和逐通道的特征，这比联合学习这两种特征更加有效。

    Inception 模块也可能具有更复杂的形式，通常会包含池化运算、不同尺寸的空间卷积（比如在某些分支上使用 5×5 的卷积代替 3×3 的卷积）和不包含空间卷积的分支（只有一个1×1 卷积）。

    下图给出了这种模块的一个示例，它来自于 Inception V3。

                                                                  output
                                                                    |
                                                             --------------
                                                             |    连接    |
                                                             --------------
                                                                    |
                              |-------------------------------------------------------------------------|
                              |                       |                       |                  --------------
                              |                       |                       |                  |   Conv2D   |
                              |                       |                       |                  | 3x3 step=2 |
                              |                       |                       |                  --------------
                              |                       |                       |                         |
                              |                 --------------           --------------          --------------
                              |                 |   Conv2D   |           | AvgRool2D  |          |   Conv2D   |
                              |                 | 3x3 step=2 |           |    3x3     |          |     3x3    |
                              |                 --------------           --------------          --------------
                              |                       |                         |                       |
                        --------------          --------------           --------------          --------------
                        |   Conv2D   |          |   Conv2D   |           | AvgRool2D  |          |   Conv2D   |
                        | 1x1 step=2 |          |     1x1    |           | 3x3 step=2 |          |     1x1    |
                        --------------          --------------           --------------          --------------
                              |                        |                        |                       |
                              |-------------------------------------------------------------------------|
                                                                    |
                                                                    |
                                                                  input
                                                Inception 模块：层组成的子图，具有多个并行卷积分支

        1×1 卷积的作用
            我们已经知道，卷积能够在输入张量的每一个方块周围提取空间图块，并对所有图块应用相同的变换。
            极端情况是提取的图块只包含一个方块。
                这时卷积运算等价于让每个方块向量经过一个 Dense 层：它计算得到的特征能够将输入张量通道中的信息混合在一起，但不会将跨空间的信息混合在一起（因为它一次只查看一个方块）。
                这种 1×1 卷积［也叫作逐点卷积（pointwise convolution）］是 Inception 模块的特色，它有助于区分开通道特征学习和空间特征学习。
                如果你假设每个通道在跨越空间时是高度自相关的，但不同的通道之间可能并不高度相关，那么这种做法是很合理的。
"""

#使用函数式 API 可以实现上图中的模块，其代码如下所示。
#这个例子假设我们有一个四 维输入张量 x。

from keras import layers

"""
每个分支都有相同的步幅值（2），这对于保持所有分支输出具有相同的尺寸是很有必要的，这样你才能将它们连接在一起
"""
branch_a = layers.Conv2D(128, 1, activation='relu', strides=2)(x) 
"""
在这个分支中，空间卷积层用到了步幅
"""
branch_b = layers.Conv2D(128, 1, activation='relu')(x) 
branch_b = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_b) 

"""
在这个分支中，在这个分支中，平均池化层用到了步幅
"""
branch_c = layers.AveragePooling2D(3, strides=2)(x) 
branch_c = layers.Conv2D(128, 3, activation='relu')(branch_c)

branch_d = layers.Conv2D(128, 1, activation='relu')(x)
branch_d = layers.Conv2D(128, 3, activation='relu')(branch_d)
branch_d = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_d)

"""
将分支输出连接在一起，得到模块输出
"""
output = layers.concatenate([branch_a, branch_b, branch_c, branch_d], axis=-1)

"""
注意，完整的Inception V3架构内置于Keras中，位置在keras.applications.inception_v3.InceptionV3，其中包括在 ImageNet 数据集上预训练得到的权重。

与其密切相关的另一个模型是 Xception，它也是 Keras 的 applications 模块的一部分。
    Xception 代表极端 Inception（extreme inception），它是一种卷积神经网络架构，其灵感可能来自于 Inception。
    Xception 将分别进行通道特征学习与空间特征学习的想法推向逻辑上的极端，并将 Inception 模块替换为深度可分离卷积，其中包括一个逐深度卷积（即一个空间卷积，分别对每个输入通道进行处理）和后面的一个逐点卷积（即一个 1×1 卷积）。
        这个深度可分离卷积实际上是 Inception 模块的一种极端形式，其空间特征和通道特征被完全分离。
    Xception 的参数个数与 Inception V3 大致相同，但因为它对模型参数的使用更加高效，所以在 ImageNet 以及其他大规模数据集上的运行性能更好，精度也更高。
"""

"""
2. 残差连接

残差连接（residual connection）是一种常见的类图网络组件，在 2015 年之后的许多网络架构（包括 Xception）中都可以见到。
2015 年末，来自微软的何恺明等人在 ILSVRC ImageNet 挑战赛中获胜 ，其中引入了这一方法。
    残差连接解决了困扰所有大规模深度学习模型的两个共性问题： 梯度消失和表示瓶颈。
    
        |--------------------------------------------------------------------------------------------------------------------|
        |深度学习中的表示瓶颈                                                                                                |
        |   在 Sequential 模型中，每个连续的表示层都构建于前一层之上，这意味着它只能访问前一层激活中包含的信息。             |
        |   如果某一层太小（比如特征维度太低），那么模型将会受限于该层激活中能够塞入多少信息。                               |
        |   你可以通过类比信号处理来理解这个概念：                                                                           |
        |       假设你有一条包含一系列操作的音频处理流水线，                                                                 |
        |       每个操作的输入都是前一个操作的输出，                                                                         |
        |       如果某个操作将信号裁剪到低频范围（比如0~15 kHz），                                                           |
        |       那么下游操作将永远无法恢复那些被丢弃的频段。                                                                 |
        |                                                                                                                    |
        |   任何信息的丢失都是永久性的。残差连接可以将较早的信息重新注入到下游数据中，从而部分解决了深度学习模型的这一问题。 |
        |--------------------------------------------------------------------------------------------------------------------|

        |----------------------------------------------------------------------------------------------------------------------------------------------------------------|
        |深度学习中的梯度消失                                                                                                                                            |
        |   反向传播是用于训练深度神经网络的主要算法，其工作原理是将来自输出损失的反馈信号向下传播到更底部的层。                                                         |
        |   如果这个反馈信号的传播需要经过很多层，那么信号可能会变得非常微弱，甚至完全丢失，导致网络无法训练。                                                           |
        |   这个问题被称为梯度消失（vanishing gradient）。                                                                                                               |
        |   深度网络中存在这个问题，在很长序列上的循环网络也存在这个问题。                                                                                               |
        |   在这两种情况下，反馈信号的传播都必须通过一长串操作。                                                                                                         |
        |       我们已经知道 LSTM 层是如何在循环网络中解决这个问题的：它引入了一个携带轨道（carry track），可以在与主处理轨道平行的轨道上传播信息。                      |
        |       残差连接在前馈深度网络中的工作原理与此类似，但它更加简单：它引入了一个纯线性的信息携带轨道，与主要的层堆叠方向平行，从而有助于跨越任意深度的层来传播梯度 |
        |-----------------------------------------------------------------------------------------------------------------------------------------------------------------

    通常来说，向任何多于 10 层的模型中添加残差连接，都可能会有所帮助。 
        残差连接是让前面某层的输出作为后面某层的输入，从而在序列网络中有效地创造了一条捷径。
        前面层的输出没有与后面层的激活连接在一起，而是与后面层的激活相加（这里假设两个激活的形状相同）。
        如果它们的形状不同，我们可以用一个线性变换将前面层的激活改变成目标形状（例如，这个线性变换可以是不带激活的 Dense 层；对于卷积特征图，可以是不带激活 1×1 卷积）。
        如果特征图的尺寸相同，在 Keras 中实现残差连接的方法如下，用的是恒等残差连接（identity residual connection）。

这个例子假设我们有一个四维输入张量 x。
"""

from keras import layers

x = ...
y = layers.Conv2D(128, 3, activation='relu', padding='same')(x) #对 x 进行变换

y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)
y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)

y = layers.add([y, x]) #将原始 x 与输出特征相加

"""
如果特征图的尺寸不同，实现残差连接的方法如下，用的是线性残差连接（linear residual connection）。
同样，假设我们有一个四维输入张量 x。
"""
from keras import layers

x = ...
y = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)
y = layers.MaxPooling2D(2, strides=2)(y)

"""
使用 1×1 卷积，将原始 x 张量线性下采样为与 y 具有相同的形状
"""
residual = layers.Conv2D(128, 1, strides=2, padding='same')(x) 

y = layers.add([y, residual]) #将残差张量与输出特征相加

"""
共享层权重
    函数式 API 还有一个重要特性，那就是能够多次重复使用一个层实例。
    如果你对一个层实例调用两次，而不是每次调用都实例化一个新层，那么每次调用可以重复使用相同的权重。
    这样你可以构建具有共享分支的模型，即几个分支全都共享相同的知识并执行相同的运算。也就是说，这些分支共享相同的表示，并同时对不同的输入集合学习这些表示。

    举个例子，假设一个模型想要评估两个句子之间的语义相似度。
        这个模型有两个输入（需要比较的两个句子），并输出一个范围在 0~1 的分数，0 表示两个句子毫不相关，1 表示两个句子完全相同或只是换一种表述。
        这种模型在许多应用中都很有用，其中包括在对话系统中删除重复的自然语言查询。

        在这种设置下，两个输入句子是可以互换的，因为语义相似度是一种对称关系，A 相对于 B 的相似度等于 B 相对于 A 的相似度。
        因此，学习两个单独的模型来分别处理两个输入句子是没有道理的。
        相反，你需要用一个 LSTM 层来处理两个句子。
        这个 LSTM 层的表示（即它的权重）是同时基于两个输入来学习的。
            我们将其称为连体 LSTM（Siamese LSTM）或共享 LSTM（shared LSTM）模型。 

    使用 Keras 函数式 API 中的层共享（层重复使用）可以实现这样的模型，其代码如下所示。
"""

from keras import layers
from keras import Input
from keras.models import Model

lstm = layers.LSTM(32) #将一个 LSTM 层实例化一次

"""
构建模型的左分支：输入是长度128 的向量组成的变长序列
"""
left_input = Input(shape=(None, 128)) 
left_output = lstm(left_input)

"""
构建模型的右分支：如果调用已有的层实例，那么就会重复使用它的权重
"""
right_input = Input(shape=(None, 128)) 
right_output = lstm(right_input)

"""
在上面构建一个分类器
"""
merged = layers.concatenate([left_output, right_output], axis=-1) 
predictions = layers.Dense(1, activation='sigmoid')(merged)

"""
将模型实例化并训练：训练这种模型时，基于两个输入对 LSTM 层的权重进行更新
"""
model = Model([left_input, right_input], predictions) 
model.fit([left_data, right_data], targets)

"""
自然地，一个层实例可能被多次重复使用，它可以被调用任意多次，每次都重复使用一组相同的权重。

将模型作为层
重要的是，在函数式 API 中，可以像使用层一样使用模型。   
    实际上，你可以将模型看作“更大的层”。
    Sequential 类和 Model 类都是如此。
    这意味着你可以在一个输入张量上调用模型，并得到一个输出张量。
        y = model(x)

    如果模型具有多个输入张量和多个输出张量，那么应该用张量列表来调用模型。
        y1, y2 = model([x1, x2])

在调用模型实例时，就是在重复使用模型的权重，正如在调用层实例时，就是在重复使用层的权重。
    调用一个实例，无论是层实例还是模型实例，都会重复使用这个实例已经学到的表示，这很直观。

通过重复使用模型实例可以构建一个简单的例子，就是一个使用双摄像头作为输入的视觉模型：两个平行的摄像头，相距几厘米（一英寸）。
    这样的模型可以感知深度，这在很多应用中都很有用。
    你不需要两个单独的模型从左右两个摄像头中分别提取视觉特征，然后再将二者合并。
    这样的底层处理可以在两个输入之间共享，即通过共享层（使用相同的权重，从而共享相同的表示）来实现。
    在 Keras 中实现连体视觉模型（共享卷积基）的代码如下所示。
"""

from keras import layers
from keras import applications
from keras import Input

xception_base = applications.Xception(weights=None, include_top=False) #图像处理基础模型是 Xception 网络（只包 括卷积基）

#输入是 250×250 的 RGB 图像
left_input = Input(shape=(250, 250, 3)) 
right_input = Input(shape=(250, 250, 3))

#对相同的视觉模型调用两次
left_features = xception_base(left_input) 
right_input = xception_base(right_input)

#合并后的特征包含来自左右两个视觉输入中的信息
merged_features = layers.concatenate([left_features, right_input], axis=-1)

"""
小结
以上就是对 Keras 函数式 API 的介绍，它是构建高级深度神经网络架构的必备工具。本节我们学习了以下内容。

    如果你需要实现的架构不仅仅是层的线性堆叠，那么不要局限于 Sequential API。
    如何使用 Keras 函数式 API 来构建多输入模型、多输出模型和具有复杂的内部网络拓扑结构的模型。
    如何通过多次调用相同的层实例或模型实例，在不同的处理分支之间重复使用层或模型的权重。
"""
