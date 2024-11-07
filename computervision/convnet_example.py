"""
使用很少的数据来训练一个图像分类模型，这是很常见的情况，如果你要从事计算机视觉方面的职业，很可能会在实践中遇到这种情况。
    很少的”样本可能是几百张图像，也可能是几 万张图像。

本节将介绍解决这一问题的基本策略，即使用已有的少量数据从头开始训练一个新模型。
|----首先，在 2000 个训练样本上训练一个简单的小型卷积神经网络，不做任何正则化，为模型目标设定一个基准。这会得到 71% 的分类精度。此时主要的问题在于过拟合。
|----然后，我们会介绍数据增强（data augmentation），它在计算机视觉领域是一种非常强大的降低过拟合的技术。使用数据增强之后，网络精度将提高到 82%。

5.3 节会介绍将深度学习应用于小型数据集的另外两个重要技巧：
|----用预训练的网络做特征提取（得到的精度范围在 90%~96%），
|----对预训练的网络进行微调（最终精度为 97%）

总而言之，这三种策略
|----从头开始训练一个小型模型、
|----使用预训练的网络做特征提取、
|----对预训练的网络进行微调
构成了你的工具箱，未来可用于解决小型数据集的图像分类问题。

深度学习与小数据问题的相关性
|----有时你会听人说，仅在有大量数据可用时，深度学习才有效。
|       |----这种说法部分正确：深度学习的一个基本特性就是能够独立地在训练数据中找到有趣的特征，无须人为的特征工程，而这只在拥有大量训练样本时才能实现。
|       |        对于输入样本的维度非常高（比如图像）的问题尤其如此。
|       |----但对于初学者来说，所谓“大量”样本是相对的，即相对于你所要训练网络的大小和深度而言。
|                只用几十个样本训练卷积神经网络就解决一个复杂问题是不可能的，但如果模型很小，并做了很好的正则化，同时任务非常简单，那么几百个样本可能就足够了。
|                由于卷积神经网络学到的是局部的、平移不变的特征，它对于感知问题可以高效地利用数据。虽然数据相对较少， 但在非常小的图像数据集上从头开始训练一个卷积神经网络，仍然可以得到不错的结果，而且无须任何自定义的特征工程。
|
|----此外，深度学习模型本质上具有高度的可复用性，
        比如，已有一个在大规模数据集上训练的图像分类模型或语音转文本模型，你只需做很小的修改就能将其复用于完全不同的问题。
        特别是在计算机视觉领域，许多预训练的模型（通常都是在 ImageNet 数据集上训练得到的）现在都可以公开下载，并可以用于在数据很少的情况下构建强大的视觉模型。这
"""

#下载数据
"""
可以从 https://www.kaggle.com/c/dogs-vs-cats/data 下载原始数据集
    这些图像都是中等分辨率的彩色 JPEG 图像。

这个数据集包含 25 000 张猫狗图像（每个类别都有 12 500 张），大小为 543MB（压缩后）。
下载数据并解压之后，你需要创建一个新数据集，其中包含三个子集：每个类别各 1000 个样本的训练集、每个类别各 500 个样本的验证集和每个类别各 500 个样本的测试集。 
创建新数据集的代码如下所示。
"""
import os, shutil

original_dataset_dir = '/Users/jieman/Desktop/ml/dl/learning_dl/computervision/kaggle_original_data/train/' 

base_dir = '/Users/jieman/Desktop/ml/dl/learning_dl/computervision/cats_and_dogs_small/' 
os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train') 
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

train_cats_dir = os.path.join(train_dir, 'cats') 
os.mkdir(train_cats_dir)

train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir, 'cats') 
os.mkdir(validation_cats_dir)

validation_dogs_dir = os.path.join(validation_dir, 'dogs') 
os.mkdir(validation_dogs_dir)

test_cats_dir = os.path.join(test_dir, 'cats') 
os.mkdir(test_cats_dir)

test_dogs_dir = os.path.join(test_dir, 'dogs') 
os.mkdir(test_dogs_dir)

fnames = ['cat.{}.jpg'.format(i) for i in range(1000)] 
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)] 
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)] 
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000)] 
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)] 
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)] 
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)

#所以我们的确有 2000 张训练图像、1000 张验证图像和 1000 张测试图像。
#每个分组中两个类别的样本数相同，这是一个平衡的二分类问题，分类精度可作为衡量成功的指标。

#构建网络
"""
我们将复用相同的总体结构，即卷积神经网络由 Conv2D 层（使用 relu 激活）和 MaxPooling2D 层交替堆叠构成。

但由于这里要处理的是更大的图像和更复杂的问题，你需要相应地增大网络，即再增加一个 Conv2D+MaxPooling2D 的组合。
这既可以增大网络容量，也可以进一步减小特征图的尺寸，使其在连接 Flatten 层时尺寸不会太大。
    本例中初始输入的尺寸为 150×150（有些随意的选择），所以最后在 Flatten 层之前的特征图大小为 7×7。
    注意 网络中特征图的深度在逐渐增大（从 32 增大到 128），而特征图的尺寸在逐渐减小（从 150×150 减小到 7×7）。这几乎是所有卷积神经网络的模式。

你面对的是一个二分类问题，所以网络最后一层是使用 sigmoid 激活的单一单元（大小为 1 的 Dense 层）。这个单元将对某个类别的概率进行编码。
"""
from keras.api import layers
from keras.api import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

"""
model.summary()

Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                      │ (None, 148, 148, 32)        │             896 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d (MaxPooling2D)         │ (None, 74, 74, 32)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_1 (Conv2D)                    │ (None, 72, 72, 64)          │          18,496 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_1 (MaxPooling2D)       │ (None, 36, 36, 64)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_2 (Conv2D)                    │ (None, 34, 34, 128)         │          73,856 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_2 (MaxPooling2D)       │ (None, 17, 17, 128)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_3 (Conv2D)                    │ (None, 15, 15, 128)         │         147,584 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_3 (MaxPooling2D)       │ (None, 7, 7, 128)           │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten (Flatten)                    │ (None, 6272)                │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 512)                 │       3,211,776 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 1)                   │             513 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 3,453,121 (13.17 MB)
 Trainable params: 3,453,121 (13.17 MB)
 Non-trainable params: 0 (0.00 B)

 在编译这一步，和前面一样，我们将使用 RMSprop 优化器。因为网络最后一层是单一 sigmoid 单元，所以我们将使用二元交叉熵作为损失函数

问题类型                    最后一层激活            损失函数

二分类问题                  sigmoiad                binary_crossentropy
多分类、单标签问题          softmax                 categorical_crossentropy 
多分类、多标签问题          sigmoid                 binary_crossentropy
回归到任意值                无                      mse
回归到0~1 范围内的值        sigmoid                 mse 或 binary_crossentropy
"""

#配置模型用于训练
model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['acc'])

#数据预处理
"""
你现在已经知道，将数据输入神经网络之前，应该将数据格式化为经过预处理的浮点数张量。
现在，数据以 JPEG 文件的形式保存在硬盘中，所以数据预处理步骤大致如下。
|----(1) 读取图像文件。
|----(2) 将 JPEG 文件解码为 RGB 像素网格。
|----(3) 将这些像素网格转换为浮点数张量。
|----(4) 将像素值（0~255 范围内）缩放到 [0, 1] 区间（正如你所知，神经网络喜欢处理较小的输入值）。

Keras有一个图像处理辅助工具的模块，位于 keras.preprocessing.image。
特别地，它包含 ImageDataGenerator 类，可以快速创建 Python 生成器，能够将硬盘上的图像文件自动转换 为预处理好的张量批量。
"""

from keras.api.preprocessing.image import ImageDataGenerator 

#将所有图像乘以 1/255 缩放
train_datagen = ImageDataGenerator(rescale=1./255) 
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=20, class_mode='binary')  #因为使用了 binary_crossentropy损失，所以需要用二进制标签
validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(150, 150), batch_size=20, class_mode='binary')

"""
我们来看一下其中一个生成器的输出：它生成了 150×150 的 RGB 图像［形状为 (20, 150, 150, 3)］与二进制标签［形状为 (20,)］组成的批量。每个批量中包含 20 个样本（批 量大小）。
注意，生成器会不停地生成这些批量，它会不断循环目标文件夹中的图像。
因此，你 需要在某个时刻终止（break）迭代循环。

>>> for data_batch, labels_batch in train_generator:
>>> print('data batch shape:', data_batch.shape)
>>> print('labels batch shape:', labels_batch.shape)
>>> break
data batch shape: (20, 150, 150, 3)
labels batch shape: (20,)
"""

#利用批量生成器拟合模型
"""
利用生成器，我们让模型对数据进行拟合。
    |----我们将使用 fit_generator 方法来拟合，它在数据生成器上的效果和 fit 相同。
    |        它的第一个参数应该是一个 Python 生成器，可以不停地生成输入和目标组成的批量，比如 train_generator。
    |            因为数据是不断生成的，所以 Keras 模型要知道每一轮需要从生成器中抽取多少个样本。
    |            这是 steps_per_epoch 参数的作用：从生成器中抽取 steps_per_epoch 个批量后（即运行了 steps_per_epoch 次梯度下降），拟合过程将进入下一个轮次。
    |                本例中，每个批量包含 20 个样本，所以读取完所有 2000 个样本需要 100 个批量。
    |
    |----使用 fit_generator 时，你可以传入一个 validation_data 参数，其作用和在 fit 方法中类似。
             值得注意的是，这个参数可以是一个数据生成器，但也可以是 Numpy 数组组成的元组。
                 如果向 validation_data 传入一个生成器，那么这个生成器应该能够不停地生成验证数据批量，因此你还需要指定 validation_steps 参数，说明需要从验证生成器中抽取多少个批次用于评估。
"""
history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=30, validation_data=validation_generator, validation_steps=50)

#始终在训练完成后保存模型，这是一种良好实践。
#保存模型
model.save('cats_and_dogs_small_1.h5')

#绘制训练过程中的损失曲线和精度曲线
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

"""
从这些图像中都能看出过拟合的特征。训练精度随着时间线性增加，直到接近 100%，而验证精度则停留在 70%~72%。验证损失仅在 5 轮后就达到最小值，然后保持不变，而训练损失则一直线性下降，直到接近于 0。
因为训练样本相对较少（2000 个），所以过拟合是你最关心的问题。

前面已经介绍过几种降低过拟合的技巧，比如 dropout 和权重衰减（L2 正则化）。

现在我们将使用一种针对于计算机视觉领域的新方法，在用深度学习模型处理图像时几乎都会用到这种方法，它就是数据增强 （data augmentation）。

使用数据增强
    |---过拟合的原因是学习样本太少，导致无法训练出能够泛化到新数据的模型。
    |       如果拥有无限的数据，那么模型能够观察到数据分布的所有内容，这样就永远不会过拟合。
    |
    |---数据增强是从现有的训练样本中生成更多的训练数据，其方法是利用多种能够生成可信图像的随机变换来增加（augment）样本。
    |   其目标是，模型在训练时不会两次查看完全相同的图像。这让模型能够观察到数据的更多内容，从而具有更好的泛化能力。
    |
    |----在 Keras 中，这可以通过对 ImageDataGenerator 实例读取的图像执行多次随机变换来实现。
"""

"""
这里只选择了几个参数（想了解更多参数，请查阅 Keras 文档）。我们来快速介绍一下这些参数的含义。
|----rotation_range 是角度值（在 0~180 范围内），表示图像随机旋转的角度范围。
|----width_shift 和 height_shift 是图像在水平或垂直方向上平移的范围（相对于总宽度或总高度的比例）。
|----shear_range 是随机错切变换的角度。
|----zoom_range 是图像随机缩放的范围。
|----horizontal_flip 是随机将一半图像水平翻转。如果没有水平不对称的假设（比如真实世界的图像），这种做法是有意义的。
|----fill_mode是用于填充新创建像素的方法，这些新像素可能来自于旋转或宽度/高度平移。
"""
datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

#显示几个随机增强后的训练图像
from keras.preprocessing import image

fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]

img_path = fnames[3] #选择一张图像进行增强

img = image.load_img(img_path, target_size=(150, 150)) 

x = image.img_to_array(img) 

x = x.reshape((1,) + x.shape) 

i = 0 
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break

plt.show()

"""
如果你使用这种数据增强来训练一个新网络，那么网络将不会两次看到同样的输入。
但网络看到的输入仍然是高度相关的，因为这些输入都来自于少量的原始图像。你无法生成新信息， 而只能混合现有信息。
因此，这种方法可能不足以完全消除过拟合。
    为了进一步降低过拟合， 你还需要向模型中添加一个 Dropout 层，添加到密集连接分类器之前。
"""

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

#　利用数据增强生成器训练卷积神经网络
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True,) 

test_datagen = ImageDataGenerator(rescale=1./255) #注意，不能增强验证数据

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=32, class_mode='binary') #因为使用了 binary_crossentropy 损失，所以需要用二进制标签

validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(150, 150), batch_size=32, class_mode='binary')

history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=100, validation_data=validation_generator, validation_steps=50)

#保存模型
model.save('cats_and_dogs_small_2.h5')

"""
通过进一步使用正则化方法以及调节网络参数（比如每个卷积层的过滤器个数或网络中的层数），你可以得到更高的精度，可以达到86%或87%。
但只靠从头开始训练自己的卷积神经网络， 再想提高精度就十分困难，因为可用的数据太少。
想要在这个问题上进一步提高精度，下一步需要使用预训练的模型，这是接下来两节的重点。
"""
