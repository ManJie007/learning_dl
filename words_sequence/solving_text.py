"""
文本是最常用的序列数据之一，可以理解为字符序列或单词序列，但最常见的是单词级处理。

当然，请记住，本章的这些深度学习模型都没有像人类一样真正地理解文本，而只是映射出书面语言的统计结构，但这足以解决许多简单的文本任务。深度学习用于自然语言处理是将模式识别应用于单词、句子和段落，这与计算机视觉是将模式识别应用于像素大致相同。

深度学习模型不会接收原始文本作为输入，它只能处理数值张量。
文本向量化（vectorize）是指将文本转换为数值张量的过程。它有多种实现方法。
|----将文本分割为单词，并将每个单词转换为一个向量。
|----将文本分割为字符，并将每个字符转换为一个向量。
|----提取单词或字符的 n-gram，并将每个 n-gram 转换为一个向量。n-gram 是多个连续单词或字符的集合（n-gram 之间可重叠）。

标记（token）       ----将文本分解而成的单元（单词、字符或 n-gram）
分词（tokenization）----将文本分解成标记的过程

所有文本向量化过程都是应用某种分词方案，然后将数值向量与生成的标记相关联。这些向量组合成序列张量，被输入到深度神经网络中。
        文本
         |
         v
        token
         |
         v
    token 的向量编码

将向量与标记相关联的方法有很多种。本节将介绍两种主要方法
|----one-hot 编码（one-hot encoding）
|       将每个单词与一个唯一的整数索引相关联，
|       然后将这个整数索引 i 转换为长度为 N 的二进制向量（N 是词表大小），
|       这个向量只有第 i 个元素是 1，其余元素都为 0。
|       
|       one-hot 编码的一种变体是所谓的 one-hot 散列技巧（one-hot hashing trick），如果词表中唯
|       一标记的数量太大而无法直接处理，就可以使用这种技巧。这种方法没有为每个单词显式分配
|       一个索引并将这些索引保存在一个字典中，而是将单词散列编码为固定长度的向量，通常用一
|       个非常简单的散列函数来实现。这种方法的主要优点在于，它避免了维护一个显式的单词索引，
|       从而节省内存并允许数据的在线编码（在读取完所有数据之前，你就可以立刻生成标记向量）。
|       这种方法有一个缺点，就是可能会出现散列冲突（hash collision），即两个不同的单词可能具有
|       相同的散列值，随后任何机器学习模型观察这些散列值，都无法区分它们所对应的单词。
|
|       one-hot 编码得到的向量是二进制的、稀疏的（绝大部分元素都是 0）、维度很高的（维度大小等于
|       词表中的单词个数）而词嵌入是低维的浮点数向量（即密集向量，与稀疏向量相对）
|
|----标记嵌入［token embedding，通常只用于单词，叫作词嵌入（word embedding）］
|       
|       获取词嵌入有两种方法。
|       |---在完成主任务（比如文档分类或情感预测）的同时学习词嵌入。
|       |       一开始是随机的词向量，然后对这些词向量进行学习，
|       |       其学习方式与学习神经网络的权重相同。
|       |       
|       |       要将一个词与一个密集向量相关联，最简单的方法就是随机选择向量。
|       |       这种方法的问题在于， 得到的嵌入空间没有任何结构。
|       |
|       |       说得更抽象一点，词向量之间的几何关系应该表示这些词之间的语义关系。
|       |       词嵌入的作用应该是将人类的语言映射到几何空间中。例如，在一个合理的嵌入空间中，
|       |       同义词应该被嵌入到相似的词向量中，一般来说，
|       |       任意两个词向量之间的几何距离（比如 L2 距离）应该和这两个词的语义距离有关
|       |       （表示不同事物的词被嵌入到相隔很远的点，而相关的词则更加靠近）。除了距离，
|       |       你可能还希望嵌入空间中的特定方向也是有意义的。
|       |
|       |       在真实的词嵌入空间中，常见的有意义的几何变换的例子包括“性别”向量和“复数”向量。
|       |       例如，将 king（国王）向量加上 female（女性）向量，得到的是 queen（女王）向量。
|       |
|       |       有没有一个理想的词嵌入空间，可以完美地映射人类语言，
|       |       并可用于所有自然语言处理任务？可能有，但我们尚未发现。
|       |       此外，也不存在人类语言（human language）这种东西。世界上有许多种不同的语言，
|       |       而且它们不是同构的，因为语言是特定文化和特定环境的反射。
|       |       但从更实际的角度来说，一个好的词嵌入空间在很大程度上取决于你的任务。
|       |
|       |       因此，合理的做法是对每个新任务都学习一个新的嵌入空间。
|       |       幸运的是，反向传播让这种学习变得很简单，而 Keras 使其变得更简单。
|       |       我们要做的就是学习一个层的权重，这个层就是 Embedding 层。
|       |
|       |       最好将 Embedding 层理解为一个字典，将整数索引（表示特定单词）映射为密集向量。
|       |       它接收整数作为输入，并在内部字典中查找这些整数，然后返回相关联的向量。
|       |       单词索引----> Embedding层----> 对应的词向量
|       |
|       |       Embedding 层的输入是一个二维整数张量，其形状为 (samples, sequence_length)，
|       |       每个元素是一个整数序列。
|       |
|       |       这 个 Embedding 层返回一个形状为 
|       |       (samples, sequence_length, embedding_dimensionality) 
|       |       的三维浮点数张量。然后可以用 RNN 层或一维卷积层来处理这个三维张量
|       |       （二者都会在后面介绍）。
|       |
|       |       将一个 Embedding 层实例化时，它的权重（即标记向量的内部字典）最开始是随机的，
|       |       与其他层一样。在训练过程中，利用反向传播来逐渐调节这些词向量，
|       |       改变空间结构以便下游模型可以利用。一旦训练完成，嵌入空间将会展示大量结构，
|       |       这种结构专门针对训练模型所要解决的问题。
|       |
|       |---预训练词嵌入（pretrained word embedding）
|               在不同于待解决问题的机器学习任务上预计算好词嵌入，
|               然后将其加载到模型中。
|
|               有时可用的训练数据很少，以至于只用手头数据无法学习适合特定任务的词嵌入。
|               那么应该怎么办？
|               你可以从预计算的嵌入空间中加载嵌入向量（你知道这个嵌入空间是高度结构化的，并且
|               具有有用的属性，即抓住了语言结构的一般特点），而不是在解决问题的同时学习词嵌入。
|               在自然语言处理中使用预训练的词嵌入，其背后的原理与在图像分类中使用预训练的卷积神经网络是一样的：
|               没有足够的数据来自己学习真正强大的特征，但你需要的特征应该是非常通用的，
|               比如常见的视觉特征或语义特征。在这种情况下，重复使用在其他问题上学到的特征，这种做法是有道理的。
|
|               这种词嵌入通常是利用词频统计计算得出的（观察哪些词共同出现在句子或文档中），用到
|               的技术很多，有些涉及神经网络，有些则不涉及。Bengio 等人在 21 世纪初首先研究了一种思路，
|               就是用无监督的方法计算一个密集的低维词嵌入空间，但直到最有名且最成功的词嵌入方案之
|               一 word2vec 算法发布之后，这一思路才开始在研究领域和工业应用中取得成功。word2vec 算法
|               由 Google 的 Tomas Mikolov 于 2013 年开发，其维度抓住了特定的语义属性，比如性别。
|
|               有许多预计算的词嵌入数据库，你都可以下载并在 Keras 的 Embedding 层中使用。
|               word2vec 就是其中之一。另一个常用的是 GloVe（global vectors for word representation，
|               词表示全局向量），由斯坦福大学的研究人员于 2014 年开发。这种嵌入方法基于对词共现统计矩阵进
|               行因式分解。其开发者已经公开了数百万个英文标记的预计算嵌入，它们都是从维基百科数据
|               和 Common Crawl 数据得到的。
"""

#单词级的one-hot编码
import numpy as np

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

token_index = {}
for sample in samples:
    for word in sample.split():
        if word not in token_index:
            token_index[word] = len(token_index) + 1    #注意，没有为索引编号 0 指定单词

max_length = 10 #对样本进行分词。只考虑每个样本前 max_length 个单词

results = np.zeros(shape=(len(samples), max_length, max(token_index.values()) + 1))

for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i, j, index] = 1

#字符级的one-hot编码
import string

samples = ['The cat sat on the mat.', 'The dog ate my homework.']
characters = string.printable
token_index = dict(zip(characters, range(1, len(characters) + 1)))

max_length = 50
results = np.zeros(shape=(len(samples), max_length, max(token_index.values()) + 1))
for i, sample in enumerate(samples):
    for j, character in enumerate(sample):
        index = token_index.get(character)
        results[i, j, index] = 1

#Keras 的内置函数可以对原始文本数据进行单词级或字符级的 one-hot 编码。你应该
#使用这些函数，因为它们实现了许多重要的特性，比如从字符串中去除特殊字符、只考虑数据
#集中前 N 个最常见的单词（这是一种常用的限制，以避免处理非常大的输入向量空间）。
from keras.preprocessing.text import Tokenizer

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

tokenizer = Tokenizer(num_words=1000) 
tokenizer.fit_on_texts(samples)

sequences = tokenizer.texts_to_sequences(samples)

one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary') #也可以直接得到 one-hot 二进制表示。这个分词器也支持除 one-hot 编码外的其他向量化模式

word_index = tokenizer.word_index 
print('Found %s unique tokens.' % len(word_index))


#使用散列技巧的单词级的 one-hot 编码
samples = ['The cat sat on the mat.', 'The dog ate my homework.']

dimensionality = 1000 
max_length = 10

results = np.zeros((len(samples), max_length, dimensionality))
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = abs(hash(word)) % dimensionality #将单词散列为 0~1000 范围内的一个随机整数索引
        results[i, j, index] = 1

#将一个 Embedding 层实例化
from keras.layers import Embedding

embedding_layer = Embedding(1000, 64)   #至少需要两个参数：
                                        #标记的个数（这里是 1000，即最大单词索引 +1）
                                        #嵌入的维度（这里是 64）

#加载 IMDB 数据，准备用于 Embedding 层
from keras.datasets import imdb
from keras.layers import preprocessing 

max_features = 10000 
maxlen = 20

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen) 
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

# 在IMDB 数据上使用 Embedding 层和分类器
from keras.models import Sequential 
from keras.layers import Flatten, Dense, Embedding

model = Sequential()
model.add(Embedding(10000, 8, input_length=maxlen)) #指定 Embedding 层的最大输入长度，以便后面将嵌入输入展平。
                                                    #Embedding 层激活的形状为 (samples, maxlen, 8)

model.add(Flatten())

model.add(Dense(1, activation='sigmoid')) 
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc']) 
model.summary()

history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

"""
得到的验证精度约为 76%，考虑到仅查看每条评论的前 20 个单词，这个结果还是相当不错
的。但请注意，仅仅将嵌入序列展开并在上面训练一个 Dense 层，会导致模型对输入序列中的
每个单词单独处理，而没有考虑单词之间的关系和句子结构（举个例子，这个模型可能会将 this 
movie is a bomb 和 this movie is the bomb 两条都归为负面评论 a）。更好的做法是在嵌入序列上添
加循环层或一维卷积层，将每个序列作为整体来学习特征。
"""

"""
整合在一起：从原始文本到词嵌入
将句子嵌入到向量序列中，然后将其展平，最后在上面训练一个 Dense 层。但此处将使用预训练的词嵌入。
此外，我们将从头开始，先下载 IMDB 原始文本数据，而不是使用 Keras 内置的已经预先分词的 IMDB 数据。
"""

#1. 下载 IMDB 数据的原始文本
#我们将训练评论转换成字符串列表，每个字符串对应一条评论。
#你也可以将评论标签（正面 / 负面）转换成 labels 列表。

import os

imdb_dir = '/Users/fchollet/Downloads/aclImdb'
train_dir = os.path.join(imdb_dir, 'train')

labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
        if label_type == 'neg':
            labels.append(0)
        else:
            labels.append(1)

#2. 对数据进行分词
#并将其划分为训练集和验证集。
#因为预训练的词嵌入对训练数据很少的问题特别有用（否则，针对于具体任务的嵌入可能效果更好）， 所以我们又添加了以下限制：将训练数据限定为前 200 个样本。

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

maxlen = 100 
training_samples = 200 
validation_samples = 10000 
max_words = 10000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0]) #将数据划分为训练集和验证集，但首先要打乱数据，因为一开始数据中的样本是排好序的（所有负面评论都在前面，然后是所有正面评论）
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples] 
y_val = labels[training_samples: training_samples + validation_samples]

#3. 下载 GloVe 词嵌入
#打开 https://nlp.stanford.edu/projects/glove，下载 2014 年英文维基百科的预计算嵌入。这是
#一个 822 MB 的压缩文件，文件名是 glove.6B.zip，里面包含 400 000 个单词（或非单词的标记）
#的 100 维嵌入向量。解压文件。

#4. 对嵌入进行预处理
#对解压后的文件（一个 .txt 文件）进行解析，构建一个将单词（字符串）映射为其向量表示（数值向量）的索引
glove_dir = '/Users/fchollet/Downloads/glove.6B'
embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

#接下来，需要构建一个可以加载到 Embedding 层中的嵌入矩阵。它必须是一个形状为
#(max_words, embedding_dim) 的矩阵，对于单词索引（在分词时构建）中索引为 i 的单词，
#这个矩阵的元素 i 就是这个单词对应的 embedding_dim 维向量。注意，索引 0 不应该代表任何
#单词或标记，它只是一个占位符。

#准备 GloVe 词嵌入矩阵
embedding_dim = 100

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

#5. 定义模型
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

#6. 在模型中加载 GloVe 嵌入
#Embedding 层只有一个权重矩阵，是一个二维的浮点数矩阵，其中每个元素 i 是与索引 i
#相关联的词向量。够简单。将准备好的 GloVe 矩阵加载到 Embedding 层中，即模型的第一层。

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

#外，需要冻结 Embedding 层（即将其 trainable 属性设为 False），其原理和预训练的卷
#积神经网络特征相同，你已经很熟悉了。如果一个模型的一部分是经过预训练的（如 Embedding
#层），而另一部分是随机初始化的（如分类器），那么在训练期间不应该更新预训练的部分，以
#避免丢失它们所保存的信息。随机初始化的层会引起较大的梯度更新，会破坏已经学到的特征。

#7. 训练模型与评估模型
#编译并训练模型。

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))
model.save_weights('pre_trained_glove_model.h5')

#绘制模型性能随时间的变化
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

#模型很快就开始过拟合，考虑到训练样本很少，这一点也不奇怪。出于同样的原因，验证精度的波动很大，但似乎达到了接近 60%。
#注意，你的结果可能会有所不同。训练样本数太少，所以模型性能严重依赖于你选择的200 个样本，而样本是随机选择的。如果你得到的结果很差，可以尝试重新选择 200 个不同的随机样本，你可以将其作为练习（在现实生活中无法选择自己的训练数据）。
#你也可以在不加载预训练词嵌入、也不冻结嵌入层的情况下训练相同的模型。在这种情况下，
#你将会学到针对任务的输入标记的嵌入。如果有大量的可用数据，这种方法通常比预训练词嵌
#入更加强大，但本例只有 200 个训练样本。我们来试一下这种方法。

#在不使用预训练词嵌入的情况下，训练相同的模型
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))

#验证精度停留在 50% 多一点。因此，在本例中，预训练词嵌入的性能要优于与任务一起学
#习的嵌入。如果增加样本数量，情况将很快发生变化，你可以把它作为一个练习。

#最后，我们在测试数据上评估模型。首先，你需要对测试数据进行分词。
test_dir = os.path.join(imdb_dir, 'test')

labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(test_dir, label_type)
    for fname in sorted(os.listdir(dir_name)):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
        if label_type == 'neg':
            labels.append(0)
        else:
            labels.append(1)

sequences = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences, maxlen=maxlen)
y_test = np.asarray(labels)

#加载并评估第一个模型。
model.load_weights('pre_trained_glove_model.h5')
model.evaluate(x_test, y_test)

#测试精度达到了令人震惊的 56% ！只用了很少的训练样本，得到这样的结果很不容易。

#summary
#现在你已经学会了下列内容。
#将原始文本转换为神经网络能够处理的格式。
#使用 Keras 模型的 Embedding 层来学习针对特定任务的标记嵌入。
#使用预训练词嵌入在小型自然语言处理问题上获得额外的性能提升。
