"""
卷积神经网络，也叫 convnet，它是计算机视觉应用几乎都在使用的一种深度 学习模型。

你将学到将卷积神经网络应用于图像分类问题，特别是那些训练数据集较小的问题。

先来看一个简单的卷积神经网络示例，即使用卷积神经网络对 MNIST 数字进行分 类，这个任务我们在第 2 章用密集连接网络做过（当时的测试精度为 97.8%）。虽然本例中的卷 积神经网络很简单，但其精度肯定会超过第 2 章的密集连接网络。
|----它是 Conv2D 层和 MaxPooling2D 层的堆叠。
|----卷积神经网络接收形状为 (image_height, image_width, image_channels) 的输入张量（不包括批量维度）。
        本例中设置卷积神经网络处理大小为 (28, 28, 1) 的输入张量， 
        这正是 MNIST 图像的格式。

        model.summary()
        可以看到，每个 Conv2D 层和 MaxPooling2D 层的输出都是一个形状为 (height, width, channels) 的 3D 张量。
        宽度和高度两个维度的尺寸通常会随着网络加深而变小。
        通道数量由传 入 Conv2D 层的第一个参数所控制（32 或 64）。
                >>> model.summary()
                Model: "sequential"
                ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
                ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
                ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
                │ conv2d (Conv2D)                      │ (None, 26, 26, 32)          │             320 │
                ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
                │ max_pooling2d (MaxPooling2D)         │ (None, 13, 13, 32)          │               0 │
                ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
                │ conv2d_1 (Conv2D)                    │ (None, 11, 11, 64)          │          18,496 │
                ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
                │ max_pooling2d_1 (MaxPooling2D)       │ (None, 5, 5, 64)            │               0 │
                ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
                │ conv2d_2 (Conv2D)                    │ (None, 3, 3, 64)            │          36,928 │
                └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
                 Total params: 55,744 (217.75 KB)
                 Trainable params: 55,744 (217.75 KB)
                 Non-trainable params: 0 (0.00 B)
        
        下一步是将最后的输出张量［大小为 (3, 3, 64)］输入到一个密集连接分类器网络中， 即 Dense 层的堆叠，你已经很熟悉了。
            model.add(layers.Flatten())
            model.add(layers.Dense(64, activation='relu'))

        我们将进行 10 类别分类，最后一层使用带 10 个输出的 softmax 激活。
            model.add(layers.Dense(10, activation='softmax'))
"""
from keras import layers 
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

#在 MNIST 图像上训练卷积神经网络
from keras.api.datasets import mnist
from keras.api.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)

#我们在测试数据上对模型进行评估。
test_loss, test_acc = model.evaluate(test_images, test_labels)

"""
与密集连接模型相比，为什么这个简单卷积神经网络的效果这么好？ 要回答这个问题，我 们来深入了解 Conv2D 层和 MaxPooling2D 层的作用。

卷积运算
|----密集连接层和卷积层的根本区别在于，Dense 层从输入特征空间中学到的是全局模式，（比如对于 MNIST 数字，全局模式就是涉及所有像素的模式），而卷积层学到的是局部模式（见下图），对于图像来说，学到的就是在输入图像的二维小窗口中发现的模式。在上面的例子中， 这些窗口的大小都是 3×3。

|           MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
|           MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
|           MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
|           MMMMMMMMMMMMX'''''''''''''''''''''''''''''''''''''''''''';MMMMMMMMMMMMM
|           MMMMMMMMMMMMK                                            .MMMMMMMMMMMMM
|           MMMMMMMMMMMMK                                            .MMMMMMMMMMMMM
|           MMMMMMMMMMMMK                              ;....odd;;    .MMMMMMMMMMMMM
|           MMMMMMMMMMMMK                              ;  'oKXo.:    .MMMMMMMMMMMMM
|           MMMMMMMMMMMMK                              ; d0WXO. :    .MMMMMMMMMMMMM
|           MMMMMMMMMMMMK                   ..:.       c.KXNl,  :    .MMMMMMMMMMMMM
|           MMMMMMMMMMMMK                   o0Nc. .....O0WKx,...:    .MMMMMMMMMMMMM
|           MMMMMMMMMMMMK                  .OXWc. ;   ,kWNxo    .    .MMMMMMMMMMMMM
|           MMMMMMMMMMMMK                .:kNXx;..: .;0XN:.;    .    .MMMMMMMMMMMMM
|           MMMMMMMMMMMMK               .0KWOl.   x,0XMOo  ;   .     .MMMMMMMMMMMMM
|           MMMMMMMMMMMMK      ,.....,l0NMWW0kl...0ONNd'   ;   .     .MMMMMMMMMMMMM
|           MMMMMMMMMMMMK      :   .dkWWNKxdKNMNKNWMWXl,...'   .     .MMMMMMMMMMMMM
|           MMMMMMMMMMMMK      :  'oXKxcdc  ;oKWMMMMMWO,   .   .     .MMMMMMMMMMMMM
|           MMMMMMMMMMMMK      :  .l'.  :'..,oNWWdd;;,.    .  .      .MMMMMMMMMMMMM
|           MMMMMMMMMMMMK      c........c  .d0WOx .        .  .      .MMMMMMMMMMMMM
|           MMMMMMMMMMMMK      .      c...cOWNOo  .        .  .      .MMMMMMMMMMMMM
|           MMMMMMMMMMMMK       .     ;  d0W0o.c .         .  .      .MMMMMMMMMMMMM
|           MMMMMMMMMMMMK       .     ;  cdK:. : .           .       .MMMMMMMMMMMMM
|           MMMMMMMMMMMMK       .     ;        : .          ..       .MMMMMMMMMMMMM
|           MMMMMMMMMMMMK        .    ;........, .          ..       .MMMMMMMMMMMMM
|           MMMMMMMMMMMMX        .   .          .           '.       .MMMMMMMMMMMMM
|           MMMMMMMMMMMMMWWWWWWWWNXNKWWWWWWWWWWWKWWWWWWWWWWWkWWWWWWWWWMMMMMMMMMMMMM
|           MMMMMMMMMMMMMMMMMMMMMMKXMMMMMMMMMMMWXMMMMMMMMMMMOMMMMMMMMMMMMMMMMMMMMMM
|           MMMMMMMMMMMMMMMMMMx'';lXWWxXMMMWWMkl''''NMMN''':xWMKOMMMMMMMMMMMMMMMMMM
|           MMMMMMMMMMMMMMMMMMl  kKWkl xMMMMWWKOo,..NMMX  .x0WOllMMMMMMMMMMMMMMMMMM
|           MMMMMMMMMMMMMMMMMMl  ';o,. xMMMXdoKNMNXNMMMX.dOMKk  lMMMMMMMMMMMMMMMMMM
|           MMMMMMMMMMMMMMMMMMl        xMMMx  ,lKNMMMMMWxXW0c.  lMMMMMMMMMMMMMMMMMM
|           MMMMMMMMMMMMMMMMMMX00000000NMMMWXXXNWMMMMMMMWMMNK000XMMMMMMMMMMMMMMMMMM
|           MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
|           MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
|
|   这个重要特性使卷积神经网络具有以下两个有趣的性质。
|   |----卷积神经网络学到的模式具有平移不变性（translation invariant）。
|   |           卷积神经网络在图像右下角学到某个模式之后，它可以在任何地方识别这个模式，比如左上角。
|   |           对于密集连 接网络来说，如果模式出现在新的位置，它只能重新学习这个模式。
|   |           这使得卷积神经网 络在处理图像时可以高效利用数据（因为视觉世界从根本上具有平移不变性），
|   |           它只需要更少的训练样本就可以学到具有泛化能力的数据表示。
|   |----卷积神经网络可以学到模式的空间层次结构（spatial hierarchies of patterns），见下图。
|                                                                                             cat
|                                                                             ..                                 ..                                                                                     
|                                                                          .:^:.                ^                .^^:.                                                                                  
|                                                                       .:^:.                   !                   .^^:.                                                                               
|                                                                    .:^:.                      !                      .:^:.                                                                            
|                                                                 .:^:.                         !                         .:^:.                                                                         
|                                                              .^^:.                           .!                            .:^:.                                                                      
|                                               ~::::::::::::::~^:~:                   !::::::::^::::::::~:                   ~^^^::::::::::::::^^                                                      
|                                               !                 ^^                   !                 ^^                   !         .:^.    :~                                                      
|                                               !       ....      ^^                   !    .:^^:^^:.    ^:                   !      .:^~~:~    :~                                                      
|                                               !     :!^7!~7^    ^^                   !    :5B#B#BY.    ^:                   !    :^^^^::^~.   :~                                                      
|                                               !    .J~ ?!.7?    ^^                   !      .?#7.      ^:                   !  .^:^:.   !:~   :~                                                      
|                                               !    .~7~?!~!.    ^^                   !        7        ^:                   !     ~:.   !:~   :~                                                      
|                                               !       ....      ^^                   !                 ^:                   !      .:^::^^^   :~                                                      
|                                               !                 ^^                   !                 ^:                   !         :~.!    :~                                                      
|                                               !.................~^                   !.................~:                   !....:::::..:^....^~                                                      
|                                               .:..:^~?77?!^:..:::.                   ........:7!^......:                  ..^~!777!~7?!^:.....:.                                                      
|                                                  :^:~:.~.~^::.                                ! ^^                ...::^^^~~^^:.   .~.!^::.                                                           
|                                               .^^..~: .~  ^:.:^:                              !  :~       ...:::::::^^^^:..        .!  ^^.:^:.                                                        
|                                            .:^:. :~.  .~   ^^  :^^.                           !   :!:::::::....:::::..              !   :~. .:^:.                                                     
|                                          :^^.   ^^    .~    .~.  .:^:.                     ..:7:::::~^   .:::::..                   !    .~^   .^^.                                                   
|                                       .:^:     ~:     .!      ^:    .^^:           ..:::::::..!     .~!^::.                         !      ^~.    :^:.                                                
|                                     :^:.     .~.      .!       :~      :^^...:::::::...       !.:::::.:~.                           !.      .~:     .:^:.                                             
|                                  .^^:       :~        .!        .~: .::::^~!~:.          .:::^7..       ~:                          ~.        ^~       .^^:                                           
|                               .:^:.        ^^          !   ...:::^!~...     .^^.    .:::::..  !          ^^                         ~:         :~.        :^^.                                        
|                             .^^.         .~:       ...:7:::::..    :^         .^!~^::..       !           :~.                       ~:           ~^         .:^:.                                     
|                          .:^:           .~. ..:::::::..!            .~.  ..::::...^^:         !            .~:                      ^:            ^~.          .:^:                                   
|                        :^^.         ...~!^:::...       !            .:!!::..        :^^.      !              ^^                     ^^             .~:            .^^.                                
|                     .:^:    ..::::::::~^               !       .:::::. :~             .:^:.   !               :~.                   :^               ^~              :^:.                             
|                   :^^:..:::::...    .~.                !  .:::::..      .~.              .^^. !                .~.                  :~                :~.              .:^:                           
|          ^::::::^77!~~^~:   ::::::::!^:::::^   .^::::::!!~~^^:^.   :::::::!^::::::^   .^:::^~!!::::::^.   :::::::!^::::::^   .^:::::^!::::::^.   ::::::^7^::::::^   .^::::^!^::::::^.                 
|         .~             ^^   !.             !   :~             ^^   !.             !   :~             ^^   !.            .!   :~        ..   ^^   !.            .!   :~      .:     ~:                 
|         .~      .:::.  :^   ~.             !   :~             ^^   ~.             !   :~             ^^   !.             !   :~      :^^!   ^^   !         .:   !   :^      .!     ^:                 
|         .~    .^:.     :^   ~.  .          !   :~   ::::::    ^^   ~.       :~    !   :~      ~.     ^^   !.   ~!~~!~.   !   :~   .^^:  ^^  ^^   !.      :^^.   !   :^       ^^    ^:                 
|         .~   :~        :^   ~.  ::::::.    !   :~        :^:  ^^   ~.      .~.    !   :~      Y:     ^^   !.   75##57    !   :~  .^.    .~  ^^   !.   .^^:      !   :^       .~    ^:                 
|         .~   :         :^   ~.      ...    !   :~          .  ^^   ~.   ..:^.     !   :~      :      ^^   !.     ::      !   :~         :~  ^^   !. .^:         !   :^       .!    ^:                 
|         .~             :^   ~.             !   :~             ^^   !.  .::.       !   :~             ^^   !.             !   :~         .   ^^   !              !   :^       ::    ^:                 
|         .~::::::::^~^::^:   ~^:::::::^~^:::~   .~:::::::^^::::~:   ~^::::::~^:::::~   .~::::::^::::::~:   ~^:::::~:::::::~   .~::::^~:::::::~:   ~^:::^~::::::::~   .~::^~~::::::::~:                 
|                   .::^^:..           .:^^:.             .^^:.              .~.                !.               .~:               :^^.             .:^^:.           ..:^^::.                           
|                        ..::::..          .:^::.            :^:.              ~:               !               .~.             .:^:            ..:^:.           .::^::.                                
|                             .::^::..         .:^:.           .^^:             ^^              !              :~             .^^.           .:^::.         .::^::.                                     
|                                  .::^::.        .::^:.          :^:.           ^^             !             ^^            :^:.         .:^^:.        .::^::.                                          
|                                       .::^::.       .:^::.        .^^.          :~.           !            ^^          .^^.        ..:^:.       .::^::.                                               
|                                            .::^::.      .:^:.        :^:.        .~.          !          .~:         :^:.       .:^::.     .::^::..                                                   
|                                                 .::^::.    .::^:.      .^^.        ~:         !         .~.       .:^:      .:^^:.    ..::::..                                                        
|                                                     ..::^::.   .:^::.    .:^:  ..   ^^        !        :~    .  :^^.     .:^:.   ..::::..                                                             
|                                                          ..::::..  .:^:.    .^~~^^:  :~       !       ^^  .:^:!^:    .:^:.   .::::..                                                                  
|                                                               ..::::...::::.  !:~^^^^..~.     !      ~:.:^^^^~^^ .::::...::^::.                                                                       
|                                                                    ..::::.:::~^~. .:^^^:~:  ..!... .~^^^^^:. ~:!^::.:::::.                                                                            
|                                                                         .::::!:!    .!.^^~:::::::::~^: ^^    :^~^:::.                                                                                 
|                                                                              ~:~. :^::                 .:^^. ~:!.                                                                                     
|                                                                              .!.!^:                        ^~~:~                                                                                      
|                                                                               ~:.                           . ~.                                                                                      
|                                                                               ^^      :^~^.         :^^^.     !                                                                                       
|                                                                               :~     !?:Y^7~      .?^77^J.    !                                                                                       
|                                                                               ~.     :!^J~7!      :?~77~~     ~.                                                                                      
|                                                                               ~.       .:..         .:::.:::::!^:....                                                                                 
|                                                                            .::^7^::::::::.  ~?77?7: .^:::... :~....::.                                                                                
|                                                                            ...  ^^::::::^.  ^?G#57. .^:::::^~!:.                                                                                      
|                                                                                :^^^~:.::.     ^5     :::^^^^:..::^:                                                                                   
|                                                                                .   :~~~^. ::::^^^:::..:^^::^:.                                                                                        
|                                                                                  ^^.   .:^^::.....:^^:.     .:                                                                                        
|                                                                                  .        ..::::::..
|                                                                               视觉世界形成了视觉模块的空间层次结构：
|                                                                               超局部的边缘组合成局部的对象， 比如眼睛或耳朵，
|                                                                               这些局部对象又组合成高级概念，比如“猫”
|
|               第一个卷积层将学习较小的局部模式（比如边缘），
|               第二个卷积层将学习由第一层特征组成的更大的模式，以此类推。
|               这使得卷积神经网络可以有效地学习越来越复杂、越来 越抽象的视觉概念（因为视觉世界从根本上具有空间层次结构）。           
|   
|   对于包含两个空间轴（高度和宽度）和一个深度轴（也叫通道轴）的 3D 张量，其卷积也 叫特征图（feature map）。
|           对于 RGB 图像，深度轴的维度大小等于 3，因为图像有 3 个颜色通道： 红色、绿色和蓝色。 
|           对于黑白图像（比如 MNIST 数字图像），深度等于 1（表示灰度等级）。
|
|----卷积运算从输入特征图中提取图块，并对所有这些图块应用相同的变换，生成输出特征图（output feature map）。
|   该输出特征图仍是一个 3D 张量，具有宽度和高度，其深度可以任意取值，因为 输出深度是层的参数，
|   深度轴的不同通道不再像 RGB 输入那样代表特定颜色，而是代表过滤器 （filter）。
|       过滤器对输入数据的某一方面进行编码，比如，单个过滤器可以从更高层次编码这样 一个概念：“输入中包含一张脸。
|
|       在 MNIST 示例中，第一个卷积层接收一个大小为 (28, 28, 1) 的特征图，并输出一个大小为 (26, 26, 32) 的特征图，即它在输入上计算 32 个过滤器。 对于这 32 个输出通道，每个通道都包含一个 26×26 的数值网格，它是过滤器对输入的响应图（response map），表示这个过 滤器模式在输入中不同位置的响应（见下图）。
|
|                   原始输入                                   单个过滤器               响应图，定量描述这个过滤器模式在不同的位置是否存在
|                                                                                 响应图的概念：某个模式在输入中的不同位置是否存在的二维图
| MMMMMMMMMMMMMMMXX0KWKONKOKOOWNMOWMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
| MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWMMMMM
| MMMMN........................................cMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM0........................................'MMMMM
| MMMMX........................................;MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM0........................................'MMMMM
| MMMMX........................................;MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM0........................';,'............'MMMMM
| MMMMX......................lOXMXOl...........;MMMMMMMMMMMMMWMMMMWMMMWNWWMNWMMMMMMMMMMMMMMMM0.....................'codk0No;..........'MMMMM
| MMMMX....................oOMMMMWWMc..........;MMMMMMMMMMMONNNNNN0'''''''''''''''cMMMMMMMMMM0...................,loOKXdxOWKo.........'MMMMM
| MMMMX................';lNWMMMMNxkWWXx........;MMMMMMMMMMM0MMMMMMK...............;MMMMMMMMMM0................'::ldkWWNdl;xO0:'.......'MMMMM
| MMMMX................KWMMWNKNMWolkNWK........;MMMMMMMMMMM0MMMMMMK...............;MMMMMMMMMM0...............;cddllllooc,.;o0oc.......'MMMMM
| MMMMX.............'lKWMWKl,cdx,...0MW:.......;MMMMMMMMMMMkXXXXXXO...............;MMMMMMMMMM0.............':clol;.........,dOO.......'MMMMM
| MMMMX............cXNM0kd;.........0MMXo......;MMMMMMMMMMM........KWWWWWWx.......;MMMMMMMMMM0...........',,,,,'...........'lOK:......'MMMMM
| MMMMX...........oWMKd.............0MMXd......;MMMMMMMMMMM........XMMMMMMk.......;MMMMMMMMMM0..........':;;...............'lOK:'.....'MMMMM
| MMMMX..........0NMKo..............0MMXd......;XXXXXXX0oOW........XMMMMMMk.......,XXXXXXX0lOk.........;:::;...............'cxk:'.....'MMMMM
| MMMMX.........cWMW;'.............:KWNl,......;MMMMMMMMMMM........xOOOOOOl''''''':MMMMMMMMMM0.........clc'.................,:c.......'MMMMM
| MMMMX........'lWWX.............lkNXOl........;MMMMMMMMMMM................MMMMMMM0MMMMMMMMMM0.........clc..................'''.......'MMMMM
| MMMMX........'lWN0.........,cdXN0o...........;MMMMMMMMMMM................MMMMMMM0MMMMMMMMMM0.........oo:............................'MMMMM
| MMMMX........'lWMWo:..':dxkXNNXo;............;MMMMMMMMMMM................MMMMMMM0MMMMMMMMMM0.........okKc,........'',,'.............'MMMMM
| MMMMX.........:NWMWWNXNWWWNkdl...............;MMMMMMMMMMMlcccccccccccccclNNNNNNNKMMMMMMMMMM0.........oOMNKl:;::c:;;,'...............'MMMMM
| MMMMX..........cxKMMMMNKlc;..................;MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM0.........,lXNNXOxlc:,'..................'MMMMM
| MMMMX.............''''.......................;MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM0............,;;;;'......................'MMMMM
| MMMMX........................................;MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM0........................................'MMMMM
| MMMMX........................................:MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM0........................................,MMMMM
| MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
| MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
|
|   这也是特征图这一术语的含义：深度轴的每个维度都是一个特征（或过滤器），而 2D 张量 output[:, :, n] 是这个过滤器在输入上的响应的二维空间图（map）。
|
|----卷积由以下两个关键参数所定义。
|       |----从输入中提取的图块尺寸：这些图块的大小通常是 3×3 或 5×5。
|       |       本例中为 3×3，这是 很常见的选择。
|       |----输出特征图的深度：卷积所计算的过滤器的数量。
|               本例第一层的深度为 32，最后一层的深度是 64。
|
|           对于 Keras 的 Conv2D 层，这些参数都是向层传入的前几个参数：Conv2D(output_depth, (window_height, window_width))。
|
|----卷积的工作原理：
|       |----1.在 3D 输入特征图上滑动（slide）这些 3×3 或 5×5 的窗口，在每个可能的位置停止并提取周围特征的 3D 图块［形状为 (window_height, window_width, input_depth)］。
|       |----2.然后每个 3D 图块与学到的同一个权重矩阵［叫作卷积核（convolution kernel）］做张量积，转换成形状为 (output_depth,) 的 1D 向量。
|       |----3.然后对所有这些向量进行空间重组，使其转换为形状为 (height, width, output_depth) 的 3D 输出特征图。输出特征图中的每个空间位置都对应于输入特征图中的相同位置（比如输出的右下角包含了输入右下角的信息）。
|       |       举个例子，利用 3×3 的窗口，向量 output[i, j, :] 来自 3D 图块 input[i-1:i+1, j-1:j+1, :]。整个过程详见下图。
|       |
|       |                                                                        .!?!:          .!JJ:..         height                                                                                  
|       |                                                   wideth         ..:::::~~.  ..::::..   .^.::::::..                                                                                           
|       |                                                           ..::::::.    ..::^~!~^::^~!~^::..     ..::::::..                                                                                    
|       |                                                     :!7::::..    .:^^^^~!~^:::^^^^^^:::^^~^:^^^^..     ..::^Y?^                                                                               
|       |                                                    .~!~    .:^^^^^~~~~~~~~~~~~^^^:^^^^^^^^::^^^^::^^^^:.    :::.                                                                              
|       |                                                   .   .~~~^^^~~~!~!!77??77!~~~~~^^^^^^:^^^::::^^^^^^::.:^~~~                                                                                  
|       |                                                  ~5   ^!^~~~!~~^^^~!7???7!~^^^~~~~~^^:.::^^^^^^:.:::^^^:::.7.                                                                                 
|       |                                                  !Y.  ^~...~!^^~~!!~^^^^^~~~~~^::.::^^^^^::.::^^^^::::!    !.          输入特征图                                                            
|       |                                    deepth        .~   ^!::.~~...^7^^~^~~~^:.:::^^^^^::.::^^^::::^!    !    7.                                                                                 
|       |                                                  .~   ^!^~^!!::.:!..::7:::::^^::::::^^:::::!:   .~    7::^:7.                                                                                 
|       |                                                  75.  ^~.::!!^~^~7::..!    ~~::::::::^7    ~.   :!::^:7.   !.                                                                                 
|       |                                                  ~Y   ^!::.~~.::^?^~^^7.   ^:   ^~    !    ~^:::^!.   !    7.                                                                                 
|       |                                                       .:^^^!!::.:!.::^?^::.~:   :~    !.:::7^.  .~    !::^^^                                                                                  
|       |                                                           .:^^^^~7:...!  ..!~::.^~.::^7:.  ~.   .!.:^^^:.                                                                                     
|       |                                                                .:^^^^^7    ^: ..~!..  !    ~:.:^^^:.                                                                                          
|       |                                                                   . .:~^^:.~:   :^    !.::^~^:...                                                                                             
|       |                                                                 .^:      .:~^^:.:~.::^~::.     .^:                                                                                            
|       |                                                               .^:            ..:^~::.            :^:                                                                                          
|       |                                                             .^^                 ..                 :^.                                                                                        
|       |                                                           .^^.                  ^^                   ^^.                                                                                      
|       |                                                         .^^.                    ~^                    .^^                                                                                     
|       |                                                       .^^.                      5Y                      .^:.                                                                                  
|       |                                                     .YY:                        ^^                        !P!                                                                                 
|       |                                           .::..     ^^                        .....                        .~.     ..:..                                                                      
|       |                                     .::~!~^^:^^~!~::.                   ..::~~^:..:^~~::..                   ..:^~^:...:^~^:..                                                                
|       |                               .::^!!~^::^~!777!~^::^~!!^::..      ..::^!~^...:^~~~~^:...:~~^::..       ..::~~^:...:^~~~^:...:^~~::...                                                         
|       |                             ^7!~^::^~!!!7?77777?7!!!~^::^~!7!    !~^:. .:^~~~~^.. .:~~~~^:. .:^~7:   ^!~^:. .:~~~~^:. .:^~~~~^.. :^~!~                                                        
|       |                             ~^::^!?!~^:::^!777!~:::^~!77^^::!   .! ..:!!~:.  .^~~~~^:. .:^~7^.. ^:   ~. .:^7~^:. .:^~~~~:.  .^~!!:.  !                                                        
|       |                             ~^...~~.:^~?!~^:.:^~!?~^::^!...:!   .~    !  .:~!~^.  .:~!!:.. ~.   ^:   ~.   :^ .::7~^:. .:^~7^:. .~    !    3 x 3 输入图块                                   
|       |                             ~~^:.^~...:7.:^^7~^:.!^...^!.:^~7   .7..  !    ~: .:^~:.. !    ~. ..!:   ~^.. :^    ! .::!^:. ^:   .~  .:!                                                        
|       |                             ~~^^~!7::::!....7:...~^.::~?~~^^!   .!.:::7:.  ^:   :~    !  ..7^::.~:   ~::::~!.. .~    !    ^:  .^7::::!                                                        
|       |                             ~^...~!^^~~?^::.7::::7!~^^~!...:!   .~    !::::!^.  :~  .:7:::.!.   ^:   ~.   :~.::^7..  !  ..!~::::~    !                                                        
|       |                             ~~::.^~...:7^^~~?~~^^!~...^!.::^!   .!.   !    ~^:::~!::::!    ~.  .~:   ~^.  :^   .!.:::7^::.~:   .~  .:!                                                        
|       |                              .:^^!!::.:!...:7:...~^.::~7^^::.    .::^:7.   ^:   :~    !   .!^:::.     .:::~~    ~    !.   ^:  .:!:^::.                                                        
|       |                                  ..:^^~7:::.7:.::!!^^::.              .::^:!^   :~   .7:^::.              ..:::^!.   !    ~~::::.                                                             
|       |                                       ..:^^^7^^^:..                        .::^:^!:^::.                        ..::::7:::::.                                                                  
|       |                                            ...                                  ..                                  ...                                                                       
|       |                                             ~                                   :.                                   ~                                                                        
|       |                                            .7.                                  !^                                  .7.                                                                       
|       |             与卷积核做点积                ^B#B:                                Y##?                                :B#B^                                                                      
|       |                                           .757.                                ~YY^                                .757.                                                                      
|       |                                             !                                   ~^                                   !                                                                        
|       |                                            ~B^                                  PY                                  ^B^                                                                       
|       |                                             !                                   ~:                                   !                                                                        
|       |                                           .::..                                ..:..                               .....                                                                      
|       |                                      .::^^^^:^^^^::.                      ::::::..::::::.                     .:::::...:::::.                                                                 
|       |                               .7    ^7!~^^:..::^~!!?.                    !!~^:.    ..:^~!                    .7~^:.     .:^~!^                                                                
|       |                               ~B^   ^^.:^^~~!~~^^:.!:                    !  ..:^^^::..  !                    :~  .::^~^::.  :~                                                                
|       |                               .!    ^~.:...:7....:.!:                    !.     :~      !                    .~      !.     :~                                                                
|       |                                !    ^~.:::::7.::::.~:                    ~.     :^      !.                   .~      !.     :~                                                                
|       |                                !    ^!::...:7....::7:                    !:.    :^     .!                    .!.     !.    .^~                                                                
|       |        output deepth           !    ^!~~~~^^7:^~~~~7:                    !^^^^:.^~.::^::!.                   .!:^^^:.!.::^^^^~        变换后的图块                                                  
|       |                                !    ^~..::^~?~^::..~:                    ~.   .:~!:.    !.                   .~   .::7^:.   .~                                                                
|       |                                !    ^~.::..:7...::.~:                    ~.     :^      !.                   .~      ~      :~                                                 .              
|       |                                !    ^~...:::7.:....~:                    ~.     :^      !.                   .~      !.     .~                                                                
|       |                                !    ^7~^^:::!.::^~~7:                    !~^:.  :^  ..::7.                   .7^::.  ~   .::~~                                                                
|       |                                !    ^~:^^~~~?~~~^^:~:                    ~..::^^~!:::.. !.                   .~ .:^^^7^:::. .~                                                                
|       |                               :?.   ^~....:^7::....~:                    ~.     ^~      !.                   .~      !.     .~                                                                
|       |                               ^B:   ^~.:::::7.::::.~:                    ~.     :^      ~.                   .~      !.     .~                                                                
|       |                                ^    ^!:::..:7...:::!:                    ~:     :^     .!.                   .!.     !.     :~                                                                
|       |                                      ::^^^^^7:^^^^:.                     .::^::.^~.::^::.                     .::^::.!..::^::                                                                 
|       |                                          ..:~:..                              .:^^:..                           . ..:~::.                                                                     
|       |                                                  ^^                                                           .^:                                                                             
|       |                                                   .^:                           ^^                          .^^.                                                                              
|       |                                                     :^:                         ^^                         ^^.                                                                                
|       |                                                       :^.                       5Y                       :^.                                                                                  
|       |                                                         ^^.                     ~~                     :^:                                                                                    
|       |                                                          .^~^                                       .~~:                                                                                      
|       |                                                            !P!             ..:::::::::..           .J5^                                                                                       
|       |                                                              :      ..:^^^::..       ..::^^^:.     .:                                                                                         
|       |                                                                .::::::.::::::.       .::::::.::::::.                                                                                          
|       |                                                         ..:^~~^:.           .:^^~^~^^:.           .:^~~::..                                                                                   
|       |                                                   ..::^^^^^:::^^^^:..    ..:::::...::::::.    ..:::::.. ..:::::..                                                                             
|       |                                            !     ~7!~^::....:....:^^~!~~~^:.           .:^~~~~~:.            .:^~!:                                                                           
|       |                                           ^B~    ~^::^~~~^^::::^^^^:..  ..::::..    .:::::.. ..:::::..  ..::::.. ^^                                                                           
|       |                                           .7.    ~^....:::^~7!^:.            .:^~~~~^.            .:^~!!:..      ^^                                                                           
|       |                                            !     ~^.:::::..^~ .::^::..   .:::::..  .:::::..   .:::::. .~         ^^                                                                           
|       |                                            !     ~^.::::::.^~      .::~!~^:.            .:~!!::..     .~         ^^                                                                           
|       |                                            !     ~^.::::::.^~         ^^ .::::..    .:::::..!         .~         ^^                                                                           
|       |                                            !     ~^....:::.^~         ^^     .:::~^::..     !         .~         ^^                                                      .                    
|       |                                            !     ~!~~^^::..^~         ^^         !.         !         .~    ..:::!^                                                      ^                    
|       |                       output depth         !     ~^.::^~~~^~!         ^^         ~.         !         .!.:::::.  ^^            输出特征图                                          ^                    
|       |                                            !     ~^.:...:::~7:::.     ^^         ~.         !     ..::^7..       ^^                                                                           
|       |                                            !     ~^.::::::.^~  .::^::.^^         ~.         !..::::.. .~         ^^                                                                           
|       |                                            !     ~^.::::::.^~       ..!!::..     ~.     .::^7:..      .~         ^^                                                                           
|       |                                            !     ~^..:::::.^~         ^^ .::^::. ~..::^::.. !         .~         ^^                                                                           
|       |                                            !     ~!^:::....^~         ^^      .::7^:..      !         .~       .:!^                                                                           
|       |                                            !     ~~:^~~~^::^~         ^^         ~.         !         .~ ..::^::.^^                                                                           
|       |                                            !     ~^...::^^^!7..       ^^         ~.         !       ..^7:::.     ^^                                                                           
|       |                                            !     ~^.::::...^!.::^::.  ^^         ~.         !  ..::::.:~         ^^                                                                           
|       |                                           :J:    ~^.::::::.^~    ..:::!~.        ~.       ..7^:::.    .~         ^^                                                                           
|       |                                           :G^    ~^.::::::.^~         ^~::^::.   ~.  ..:::::!         .~         ^^                                                                           
|       |                                            :     ^~::..:::.^~         ^^    .::^:7^:::..    !         .~         ^^                                                                           
|       |                                                  .^^^^^:::.^~         ^^         !:         !         .~    .::^::                                                                            
|       |                                                      .::^^^~!         ^^         ~.         !         :!.:^::..                                                                               
|       |                                                           .:^:::..    ^^         ~.         !    ..:::::.                                                                                     
|       |                                                                ..::::.~^         ~.         !.:::::.                                                                                          
|       |                                                                     ..:^:::.     ~.    ..::^^:.                                                                                               
|       |                                                                          ..::^::.!:.::::..                                                                                                    
|       |                                                                                .:^:..                                                                                                         
|       |
|       |----注意，输出的宽度和高度可能与输入的宽度和高度不同。不同的原因可能有两点。                                                                                                                                                                                                                                                                                                                                                   
|               |----边界效应，可以通过对输入特征图进行填充来抵消。
|               |       假设有一个 5×5 的特征图（共 25 个方块）。其中只有 9 个方块可以作为中心放入一个 3×3 的窗口，这 9 个方块形成一个 3×3 的网格（见下图）。
|               |
|               |                                                                                                   .^.::::::.!:.:::::.:!.::::::.^.   .^.::::::.!:.:::::.:~.::::::.^.   .^.::::::.~:.:::::.:~.::::::.^.         
|               |                                                                                                   :^        !.       .!        ^:   :^        !.       .~        ^:   :^        ~.       .~        ^:         
|               |                                                                                                   :^        !.       .!        ^:   :^        !.       .~        ^:   :^        ~.       .~        ^.         
|               |                                                                                                   :^        !.       .!        ^:   :^        !.       .~        ^:   :^        ~.       .~        ^.         
|               |                                                                                                   :~::::::::7~^^^^^^^~7^^^^^^^^!:   :!^^^^^^^^7~^^^^^^^~7^^^^^^^^!.   :!^^^^^^^^7~^^^^^^^~!::::::::~.         
|               |                                                                                                   ::        ~^:::::::^!::::::::~:   :~::::::::!^:::::::^!::::::::~:   :~::::::::!::::::::^~        ^.         
|               |          .^::::::::::::~::::::::::::~^:::::::::::^~::::::::::::!::::::::::::^.                    :^        ~^:::::::^!::::::::~:   :~::::::::!^:::::::^!::::::::~:   :~::::::::!^:::::::^~        ^.         
|               |          .^            !            ~:           :!            7            ^:                    :^        !^:::::::^!::::::::~:   :~::::::::7^:::::::^!::::::::~:   :~::::::::!^:::::::^~        ^.         
|               |          .^            !            ~:           :!            !            ^:                    :~::::::::7~~~~~~~^~7^~~~~~~^!:   :!^~~~~~~^7~~~~~~~^~7^~~~~~~^!.   :!^~~~~~~^7~~~~~~~^~!::::::::~.         
|               |          .^            !            ~:           :!            !            ^:                    :^        ~^:::::::^!::::::::~:   :~::::::::!::::::::^!::::::::~:   :~::::::::!::::::::^~        ^.         
|               |          .^            !            ~:           .~            !            ^:                    :^        ~^:::::::^!::::::::~:   :~::::::::!^:::::::^!::::::::~:   :~::::::::!^:::::::^~        ^.         
|               |          .~............!............~^...........:!............7............^:                    :^        !^:::::::^7::::::::~:   :~::::::::7^:::::::^!::::::::!:   :~::::::::!^:::::::^~        ^:         
|               |          .~::::::::::::7^^^^^^^^^^^^7~^^^^^^^^^^^~7^^^^^^^^^^^^?:::::::::::.~:                    .^::::::::^^:::::::^~::::::::^.   .^::::::::~^:::::::^~::::::::^.   .^::::::::~^:::::::^^::::::::^.         
|               |          .^            !::::::::::::!^:::::::::::^!::::::::::::7            ^:                                                                                                                                
|               |          .^            !::::::::::::!^:::::::::::^!::::::::::::7            ^:                     :........::::::::::::::::::::     :::::::::::::::::::::::::::::     ::::::::::::::::::::........:          
|               |          .^            !::::::::::::!^:::::::::::^!::::::::::::7            ^:                    :^ ...... !^^^^^^^:^7:^^^^^^:!:   :!:^^^^^^:7^^^^^^^:^7:^^^^^^:!:   :!:^^^^^^:7^^^^^^^:^~ ...... ^:         
|               |          .^            !::::::::::::!^:::::::::::^!::::::::::::7            ^:                    :^        ~^:::::::^!::::::::~:   :~::::::::!^:::::::^!::::::::~:   :~::::::::!::::::::^~        ^.         
|               |          .~::::::::::::7~~~~~~~~~~~^7!^~~~~~~~~~^!7^~~~~~~~~~~~?::::::::::::~:                    :^        ~^:::::::^!::::::::~:   :~::::::::!::::::::^!::::::::~:   :~::::::::!::::::::^~        ^.         
|               |          .^ .......... !^:::::::::::!~:::::::::::~!::::::::::::7. ......... ^:                    :~........!~^^^^^^^~7^^^^^^^^!:   :!^^^^^^^^7~^^^^^^^~7^^^^^^^^!.   :!^^^^^^^^7~^^^^^^^~!........~.         
|               |          .^            !::::::::::::!^:::::::::::^!::::::::::::7            ^:                    :^....... !^^^^^^^^^7:^^^^^^:!:   :!:^^^^^^^7^^^^^^^^^7:^^^^^^:!.   :!:^^^^^^^7^^^^^^^^^~ ...... ^.         
|               |          .^            !::::::::::::!^:::::::::::^!::::::::::::7            ^:                    :^        ~^:::::::^!::::::::~:   :~::::::::!^:::::::^!::::::::~:   :~::::::::!::::::::^~        ^.         
|               |          .^            !::::::::::::!^:::::::::::^!::::::::::::7            ^:                    :^        ~^:::::::^!::::::::~:   :~::::::::!::::::::^!::::::::~:   :~::::::::!::::::::^~        ^.         
|               |          .^            !::::::::::::!~:::::::::::^!::::::::::::7            ^:                    :~........!~^^^^^^^~7^^^^^^^^!:   :!^^^^^^^^7~^^^^^^^~7^^^^^^^^!.   :!^^^^^^^^7~^^^^^^^~!........~.         
|               |          .!::::::::::::7~~~~~~~~~~~^7!~~~~~~~~~~~!7^~~~~~~~~~~~?::::::::::::~:                    :^........!^^^^^^^^~7^^^^^^^^!:   :!^^^^^^^^7^^^^^^^^^7^^^^^^^^!.   :!^^^^^^^^7^^^^^^^^~!........~.         
|               |          .^            !::::::::::::!^:::::::::::^!::::::::::::7            ^:                    :^        ~^:::::::^!::::::::~:   :~::::::::!::::::::^!::::::::~:   :~::::::::!::::::::^~        ^.         
|               |          .^            !::::::::::::!^:::::::::::^!::::::::::::7            ^:                    :^        ~^:::::::^!::::::::~:   :~::::::::!^:::::::^!::::::::~:   :~::::::::!::::::::^~        ^:         
|               |          .^            !::::::::::::!^:::::::::::^!::::::::::::7            ^:                    :^........!^^^^^^^^^7^^^^^^^^!.   :!^^^^^^^^7^^^^^^^^^7^^^^^^^^!.   :!^^^^^^^^7^^^^^^^^~!........^.         
|               |          .^            !::::::::::::!^:::::::::::^!::::::::::::7            ^:                     .............................     .............................     .............................          
|               |          .~............7^^^^^^^^^^^^7~^^^^^^^^^^^~7^^^^^^^^^^^^?:...........~:                                                                                                                                
|               |          .~............7:...........!^...........^7.:::::::::::7:::::::::::.~:                    .^::::::::~^^^^^^^^^!^^^^^^^^~.   .~^^^^^^^^!^^^^^^^^^~^^^^^^^^~.   .~^^^^^^^^!^^^^^^^^^~::::::::^.         
|               |          .^            !            ~:           .~            !            ^:                    :^        !^:::::::^!::::::::~:   :~::::::::7^:::::::^!::::::::~:   :~::::::::!^:::::::^~        ^:         
|               |          .^            !            ~:           :!            !            ^:                    :^        ~^:::::::^!::::::::~:   :~::::::::!^:::::::^!::::::::~:   :~::::::::!^:::::::^~        ^.         
|               |          .^            !            ~:           :!            !            ^:                    :^        ~^:::::::^!::::::::~:   :~::::::::!^:::::::^!::::::::~:   :~::::::::!^:::::::^~        ^.         
|               |          .^            !            ~:           :!            7            ^:                    :~::::::::7~~~~~~~~~7^~~~~~~~!:   :!~~~~~~~~?~~~~~~~~~7^~~~~~~~!.   :!~~~~~~~~7~~~~~~~~~!::::::::~.         
|               |          .^::::::::::::!:::::::::::.~^:::::::::::^!.:::::::::::!::::::::::::^.                    :^        ~^:::::::^!::::::::~:   :~::::::::!^:::::::^!::::::::~:   :~::::::::!^:::::::^~        ^.         
|               |                   在 5×5 的输入特征图中，可以提取 3×3 图块的有效位置                              :^        ~^:::::::^!::::::::~:   :~::::::::!^:::::::^!::::::::~:   :~::::::::!^:::::::^~        ^.         
|               |                                                                                                   :^        ~^:::::::^!::::::::~:   :~::::::::!^:::::::^!::::::::~:   :~::::::::!^:::::::^~        ^.         
|               |                                                                                                   :~::::::::7~^^^^^^^~?^~~~~~~^!:   :!^^^^^^^^?~^^^^^^^~7^~~~~~~^!.   :!^^^^^^^^7~^~~~~~^~7:^^^^^^:!.         
|               |                                                                                                   :^        !.       .!        ^:   :^        !.       .~        ^:   :^        ~.       .~        ^.         
|               |                                                                                                   :^        !.       .!        ^:   :^        !.       .~        ^:   :^        ~.       .~        ^.         
|               |                                                                                                   :^        !.       .!        ^:   :^        !.       .~        ^:   :^        ~.       .~        ^:         
|               |                                                                                                   .^::::::::~:::::::::~::::::::^.   .^::::::::~:::::::::~::::::::^.   .^::::::::~:::::::::~::::::::^.         
|               |
|               |       因此，输出特征图的尺寸是 3×3。 它比输入尺寸小了一点，在本例中沿着每个维度都正好缩小了 2 个方块。
|               |       在前一个例子中你也可以看到这种边界效应的作用：开始的输入尺寸为 28×28，经过第一个卷积层之后尺寸变为 26×26。
|               |       
|               |       如果你希望输出特征图的空间维度与输入相同，那么可以使用填充（padding）。
|               |       填充是在输入特征图的每一边添加适当数目的行和列，使得每个输入方块都能作为卷积窗口的中心。
|               |           对于 3×3 的窗口，在左右各添加一列，在上下各添加一行。对于 5×5 的窗口，各添加两行和两列（见下图）。
|               |                                                                                                                                                                                                                                                                                     
|               |                                                                                                                         .............................       .............................       .............................       .............................       ..............................         
|               |                                                                                                                        !!~:::::^~?~^:::::^~?~^:::::^~7     !!~:::::^~?~^:::::^~?~^:::::^~7     !!~:::::^~?~^:::::^~?~^:::::^~7     !!~:::::^~?~^:::::^~?~^:::::^~7     !!~:::::^~?~^:::::^~?~^:::::^~7         
|               |           ::::::::::::::^::::::::::::^:::::::::::::^::::::::::::^:::::::::::::^:::::::::::::^:::::::::::::             !..:. .:: 7..:. .:. 7 .:. .:. 7     !..:. .:: 7..:. .:. 7..:. .:. 7     !..:. .:: 7..:. .:. 7..:. .:. 7     !..:...:: 7..:. .:. 7..:. .:. 7     !..:...:: !..:. .:. 7..:. .:. 7         
|               |           7^^:.......:^!!^:........^^7^^:.......:^^7^^........:^!~^:.......:^^7^^:.......:^~!^:.......:^:7.            !.  :~^.  7  .^~^.  7  .^~^.  7     !.  :~^.  7. .^~^.  7  .^~^.  7     !.  :~^.  7. .^~^.  7  .^~^.  7     !.  :~^.  7. .^~^.  7  .^~^.  7     !.  :~^.  !. .^~^.  7  .^~^.  7         
|               |           7 .::.   .:: ^~ .::   .::. !. ::.   .:: .! .::.   ::. ~: .:.   .::. 7 .::.   .:. ^~ .::   .::. !.            !:::.  :^:7:::. .:^:7:^:. .:^:7     !:::.  :^:7:::. .:^:7:^:. .:^:7     !:::.  :^:7:::. .:^:7:^:. .:^:7     !:::.  :^:7:::. .:^:7:^:. .:^:7     !:::.  :^:7:::. .:^:7:::. .:::7         
|               |           7    .^^^.   ^~   .:^^:    7.   .^^^.   .!    :^^:.   ~:   .:^^:    7    .^^^.   ^~   .:^^:    7.            77~^:^^:^!J~^^^^^^^^J~^^^^^^^~7     7!^^^^^^^^J~^^^^^^^~J~^^^^^^^~7     7!^^^^^^^^J~^^^^^^^^J~^^^^^^^~?     7!^^^^^^^^J~^^^^^^^^J~^^^^^^^~?     7!^^^^^^^^?~^^^^^^^^J7^::^::~!?         
|               |           7   .::.::.  ^~  .::.:::   7.  .::.::.  .!  .::::::.  ~:  .::.::.   7   .::.::.  ^~  .::..::   7.            !..:. .::.7         7         7     !.        7         7         7     !.        7         7         7     !.        !.        7         7     !.        !.        7.::. .:: 7         
|               |           7 ::.     .:.^~.::.    .:: !.::.     .:.:!.::.    .::.~:.::     .::.7.::.     ::.^~.::     .::.!.            !.  :~~.  7.        7         7     !.        7.        7         7     !.        7.        7         7     !.        7.        7         7     !.        7.        7  .^~^.  7         
|               |           ?!~^^^^^^^^^~77~^^^^^^^^^^~?!~^^^^^^^^^~!?~^^^^^^^^^^~7!~^^^^^^^^^^~J~^^^^^^^^^^~7?~^^^^^^^^:^~?.            !:::.  ::.7.        7         7     !.        7.        7         7     !.        7.        7         7     !.        7.        7         7     !.        7.        7.::  .::.7         
|               |           7.::.     .::~~            7.           :!            !:            7            ^~::.      ::.7.            77~^^^^^^!J^^^^^^^^^J~^^^^^^^~?     7~^^^^^^^^?~^^^^^^^^J~^^^^^^^~?     7!^^^^^^^^J~^^^^^^^^J~^^^^^^^~?     7!^^^^^^^^J~^^^^^^^^J^^^^^^^^^?     7!^^^^^^^^J^^^^^^^^^J!^:^^^^^!?         
|               |           7  .::. .::. ^~            7.           .!            ~:            7            ^~ .::. .::.  7.            !:::.  .:.7.        7:::::::::7     !.        7:::::::::?:::::::::7     !^:::::::.?:::::::::?:::::::::7     !^:::::::.?:::::::::7         7     !^:::::::.7.        7:::  .::.7         
|               |           7    .^~^.   ^~            7.           .!            ~:            7            ^~    :~~.    7.            !.  :~^.  7.        7:::::::::7     !.        7:::::::::?:::::::::7     !^.::::::.?:::::::::?:::::::::7     !^.::::::.7:.:::::::7         7     !^.::::::.7.        7  .^~^.  7         
|               |           7  .::  .::. ^~            !.           .!            ~:            7            ^~ .::.  ::.  7.            !..:. .:: 7.        7:.......:7     !.        7:.......:?:.......:7     7^........?:.......:?:.......:7     7^........?:........7         7     7^........7.        7.::. .:. 7         
|               |           7:^:.  .  .::~~ .......... 7.           :!            !^           .7........... ^~::.  .  .:^:7.            !!~^::::^~7^:^^^^^^^7~^^^^^^^~7     !^:^^^^^^:7~^^^^^^^^?~^^^^^^^~7     !~^^^^^^^^?~^^^^^^^^?~^^^^^^^~7     !~^^^^^^^^?~^^^^^^^^?^^^^^^^^^!     !~^^^^^^^^7^:^^^^^^:7~^::::^~~!         
|               |           ?~^::^^^^^::^77:^^^^^^^^^^:?~^^^^^^^^^^^!?^^^^^^^^^^^^?!^^^^^^^^^^^~J^^^^^^^^^^^:!7~^::^^^^::^~?.               ......  ......... .........       .........  ........ .........        ........  ........ .........        ........  ........ .........        ........ .........   .....            
|               |           7 ::.     .:.^~            7:...........^7............!^...........:7            ^~.::     .::.!.                                                                                                                                                                                                    
|               |           7   .::.::.  ^~            7:.:::::::::.^7.::::::::::.!~.:::::::::::7            ^~  .::..::.  7.            ...............................     ...............................     ...............................     ...............................     ...............................         
|               |           7    .^^^.   ^~            7:.:::::::::.^7.::::::::::.!~.:::::::::::7            ^~    :^^:    7.            !~^:::::^~?:::::::::?:::::::::7     !^::::::::7:::::::::?:::::::::7     !^::::::::7:::::::::?:::::::::7     !^::::::::7^::::::::?:::::::::7     !^::::::::7^::::::::?~^:::::^~7         
|               |           7  ::.   .:. ^~            7:...........^7............!^...........:7            ^~ .::.  .::. !.            !..::..:. 7.        7         7     !.        7.        7         7     !.        7.        7         7     !.        7.        7         7     !.        !.        7 .::..:. 7         
|               |           7^^::.:::.::^!!.::::::::::.?~^^^^^^^^^^^~?^^^^^^^^^^^^7!^^^^^^^^^^^^?:::::::::::.~!^^:.:::..:^^7.            !. .:^^.  7.        7         7     !.        7.        7         7     !.        7.        7         7     !.        7.        7         7     !.        !.        7  .^^^.  7         
|               |           7^^:..:::.::^!!.::::::::::.?~^^^^^^^^^^^~?^^^^^^^^^^^^7!^^^^^^^^^^^^?:::::::::::.~!^^:.::::.:^^7.            !^^:. .:^:7.........7.........7     !:........7.........7.........7     !:....... 7.........7.........7     !:........7.........7.........7     !:........7:........7:^:. .:^:7         
|               |           7 .::.   .:: ^~            7:...........^7............!^...........:7            ^~ .::    ::. !.            7!~:::::^~J^::::::::J~^^^^^^^~?     7~::::::::?~^^^^^^^^J~^^^^^^^~?     7!^^^^^^^^J~^^^^^^^^J~^^^^^^^~?     7!^^^^^^^^J~^^^^^^^^J^:::::::^7     7!^^^^^^^^J^::::::::?!~:::::^!?         
|               |           7    .^^^.   ^~            7:.:::::::::.^7.::::::::::.!~.:::::::::::7            ^~   .:^^:.   7.            !..::..:: !         7:.......:7     !.        7:........?:.......:7     !^........?:........?:.......:7     !^........7:........7         7     !^........7.        7 .:...:. 7         
|               |           7   .:::::.  ^~            7:.:::::::::.^7.::::::::::.!~.:::::::::::7            ^~   :::::.   7.            !. .:~^.  7.        7:::::::::7     !.        7:.:::::::?:::::::::7     !^.::::::.?:.:::::::?:::::::::7     !^.::::::.7:.:::::::7         7     !^.::::::.7.        7  .^~^.  7         
|               |           7 .:.     .:.^~            7:...........^7............!^...........:7            ^~ ::.    .:: !.            !:::.  .::7.        7:::::::::7     !.        7:::::::::?:::::::::7     !^::::::::?:::::::::?:::::::::7     !^::::::::?:::::::::?         7     !^::::::::7.        7:::.  :::7         
|               |           ?~^::^^^^^::^77:^^^^^^^^^^:?!~~~~~~~~~~~!J~~~~~~~~~~~^?!~~~~~~~~~~~~J^^^^^^^^^^^:!7~^::^^^^::^~?.            77~:^^^:^~J^^^^^^^^^J~~~~~~~~~?     7~^^^^^^^^J~~~~~~~~~J~~~~~~~~~?     7!~~~~~~~~J~~~~~~~~~J~~~~~~~~~?     7!~~~~~~~~J!~~~~~~~~J^^^^^^^^^?     7!~~~~~~~~J^^^^^^^^^J!^:^^^:^!?         
|               |           7:^:.  .  .::~~ .......... 7^:::::::::::^7::::::::::::7~::::::::::::7. ......... ^~::.      .::7.            !.::.  ::.7.        7:.......:7     !.        7:.......:?:.......:7     !^........?:.......:?:.......:7     !^........7:........7         7     !^........7.        7.::. .::.7         
|               |           7  .::  .::. ^~            7:.:::::::::.^7.::::::::::.!~.:::::::::.:7            ^~ .::.  .::  7.            !.  :~~.  7.        7:::::::::7     !.        7:::::::::?:::::::::7     !^.::::::.?:::::::::?:::::::::7     !^.::::::.7:.:::::::7         7     !^.::::::.7.        7  .^~^.  7         
|               |           7    .^~^.   ^~            7:.:::::::::.^7.::::::::::.!~.:::::::::::7            ^~    :~~:    7.            7:.:.  ::.7.        7:.:::::.:7     !.        7:.:::::::?:.:::::.:7     7^.::::::.?:.:::::::?:.:::::::?     7^.::::::.?:.:::::::?         7     7^.::::::.7.        7.::. .::.7         
|               |           7  .::. .::  ^~            7:.:::::::::.^7.::::::::::.!~.:::::::::::7            ^~  .::..::.  7.            ~!~^^^^^^~7^^^^^^^^^7~^^^^^^^~!     ~^^^^^^^^^7~^^^^^^^~7~^^^^^^^~!     ~~^^^^^^^^7~^^^^^^^~7~^^^^^^^~!     ~~^^^^^^^^7~^^^^^^^~7^^^^^^^^^!     ~~^^^^^^^^7^^^^^^^^^7~^^^^^^^~!         
|               |           7.::.     .::^~            7^:::::::::::^7.::::::::::.7~::::::::::::7            ^~.:.      ::.7.                                                                                                                                                                                                    
|               |           ?!~:^^^^^^^^~77:^^^^^^^^^^^?~^^^^^^^^^^^~?^^^^^^^^^^^^?!^^^^^^^^^^^~J^^^^^^^^^^^:!7~^:^^^^^^:^~?.                                                                                                                                                                                                    
|               |           7.::.     .:.^~            !.           .!            ~:            7            ^~.::      ::.!.            .::::::::::::::::::::::::::::::     .::::::::::::::::::::::::::::::     .::::::::::::::::::::::::::::::                                                                                 
|               |           7   ::..::.  ^~            7.           .!            ~:            7            ^~  .::..::.  7.            !~^:...:^^?:........?^^^^^^^^^7     !^........?^^^^^^^^^?^^^^^^^^^?     7~^^^^^^^^?^^^^^^^^^?^^^^^^^^^?                                                                                 
|               |           7    .^~^.   ^~            7.           .!            ~:            7            ^~    :~~:    7.            !. .::::. 7.        7:.......:7     !.        7:.......:?:.......:7     !^........?:.......:?:.......:7                                                                                 
|               |           7  .:.   ::. ^~            !.           .!            ~:            7            ^~ .::.  .::. !.            !. .:^^:  7.        7:.......:7     !.        7:.......:?:.......:7     !^........?:.......:?:.......:7                                                                                 
|               |           7^^:.......:^~!............7:...........:7............!^............?............~!^^:......:^^7.            !~^:...:^^?:........?^^^^^^^^^7     !:........?^^^^^^^^^?^^^^^^^^^7     7~:^^^^^^:?^^^^^^^^^?^^^^^^^^^?                                                                                 
|               |           7~^:::::::::^!7~^::::::::^^?~^:::::::::^~?~^::::::::^^7!~^::::::::^~?~^::::::::^^!7~^::::::::^^?.            7!~:::::^~?^::::::::?^^^^^^^^~?     7^::::::::?~^^^^^^^^J~^^^^^^^~?     7~^^^^^^^^J~^^^^^^^^J~^^^^^^^^?                                                                                 
|               |           7 .::    .::.^~ ::.    .:. !..::    .::..! ::.    .:: ~: ::.    .:. 7 .::    .:: ^~ ::.    .:. !.            !. ::.::. !         7:.......:7     !.        7:........?:.......:7     !^........7:........?:.......:7                                                                                 
|               |           7   .:::::   ^~   .::::.   7.  .:::::   .!   .::::.   ~:   .::::.   7   .::::.   ^~   .::::.   7.            !. .:^^:  7.        7:.:::::.:7     !.        7:.:::::.:?:.:::::.:7     !^.::::::.?:.:::::.:?:.:::::.:7                                                                                 
|               |           7   .:::::   ^~   .::::.   7.  .:::::   .!   .::::.   ~:   .::::.   7   .::::.   ^~   .::::.   7.            !^^:. .:^:7.........?:::::::::7     !: ...... 7^::::::::?:::::::::7     !^::::::::?^::::::::?:::::::::7                                                                                 
|               |           7 .::    .::.^~ ::.    .:. !..::    .::..! .:.    .:: ~: ::.    .:. 7 .:.    .:: ^~ ::.    .:. 7.            7!~:::::^~J^::::::::J~~~~~~~~~?     7~::::::::J~~~~~~~~~J~~~~~~~~~?     7!~~~~~~~~J~~~~~~~~~J~~~~~~~~~?                                                      :.:.:                      
|               |           7~~::::::::^~!!~^::::::::^^7~~::::::::^~~7~^::::::::^^!!~^::::::::^~7~^::::::::^~!!~^::::::::^~7.            !..:. .::.7.        7:.......:7     !.        7:........?:.......:7     !^........?:........?:.......:7                                                                                 
|               |           ................................................................................................             !.  :~~.  7.        7:::::::::7     !.        7:.:::::::?:::::::::7     !^.::::::.?:::::::::?:::::::::7                                                                                 
|               |                             　对 5×5 的输入进行填充，以便能够提取出 25 个 3×3 的图块                                   7:::.  ::.7.        ?:::::::::7     7.        7:::::::::?:::::::::7     7^:::::::.?:::::::::?:::::::::?                                                                                 
|               |                                                                                                                        ^~^^^^^^^~!^^^^^^^^^!^^^^^^^^~~     ^^^^^^^^^^!~^^^^^^^^!^^^^^^^^~~     ^~^^^^^^^^!~^^^^^^^^!^^^^^^^^~~                                                                                 
|               |                                                                                                                                                                                                                                                                                                                
|               |       对于 Conv2D 层，可以通过 padding 参数来设置填充，这个参数有两个取值："valid" 表示不使用填充（只使用有效的窗口位置）；"same" 表示“填充后输出的宽度和高度与输入相同”。 padding 参数的默认值为 "valid"。
|               |
|               |----使用了步幅（stride），稍后会给出其定义。
                        目前为止，对卷积的描述都假设卷积窗口的中心方块都是相邻的。
                        但两个连续窗口的距离是卷积的一个参数，叫作步幅，默认值为 1。也可 以使用步进卷积（strided convolution），即步幅大于 1 的卷积。
                        在下图中，你可以看到用步幅为 2 的 3×3 卷积从 5×5 输入中提取的图块（无填充）。

                                                                                                                             ............................................      ............................................  
                                                                                                                             ~.............~:............^^.............~      ~:............^~.............!.............~. 
                                                                                                                             ~             ~.            ::             ~      ~.            :^             ~             ~. 
                         ................................................................................                    ~             ~.            ::             ~      ~.            :^             ~             ~. 
                        .~..............:~..............^~..............^^..............^^..............~:                   ~             ~.            ::             ~      ~.            :^             ~             ~. 
                        .~              .^              :^              :^              ::              ^:                   ~             ~.            ::             ~      ~.            :^             ~             ~. 
                        .~              .^              :^              :^              ^:              ^:                   ~.............!:............^^.............~      ~:............^~.............!.............~. 
                        .~              .^              :^              :^              ^:              ^:                   !.............!^::::::::::::~~::::::::::::^!      ~^::::::::::::~!::::::::::::^!.............~. 
                        .~              .^              :^              :^              ^:              ^:                   ~             ~:............^~............:~      ~:............^~.....:......:~             ~. 
                        .~              .^              :^              ::              ::              ^:                   ~             ~::::::!~.:::.~~.::::::::::::~      ~::::::::::::.^~.::.:~7^.::::~             ~. 
                        .!::::::::::::::^!::::::::::::::~!::::::::::::::~~::::::::::::::~~::::::::::::::~:                   ~             ~:::::.~!.:::.~~.::::::::::::~      ~::::::::::::.^~.::.:77:.::::~             ~. 
                        .~              .!::::::::::::::^~::::::::::::::~~::::::::::::::~:              ^:                   ~             ~:.....::.....^~............:~      ~:............^~....::::....:~             ~. 
                        .~              .~.:::::::.::::.^~.::::::::::::.^~.:::::^^:::::.~:              ^:                   ~.............!^:^^^^::^^^^:~!:^^^^^^^^^^:^!      ~^:^^^^^^^^^^:~!:^^^::::^^^:^!.............~. 
                        .~              .~.:::::!J.::::.^~.::::::::::::.^~.:::.^~?~.:::.~:              ^:                   !.............!^:^^^^^^^^^^:~!:^^^^^^^^^^^^!      ~^:^^^^^^^^^^:~!:^^^^^^^^^^^^!.............~. 
                        .~              .~.::::.:7.::::.^~.::::::::::::.^~.:::.^?7:.:::.~:              ^:                   ~             ~:............^~............:~      ~:............^~............:~             ~. 
                        .~              .~.::::::::::::.^~.::::::::::::.^~.::::::::::::.~:              ^:                   ~             ~::::::::::::.~~.::::::::::::~      ~::::::::::::.^~.::::::::::::~             ~. 
                        .~              .!::::::::::::::^~::::::::::::::~~::::::::::::::~:              ^:                   ~             ~::::::::::::.~~.::::::::::::~      ~::::::::::::.^~.::::::::::::~             ~. 
                        .!::::::::::::::^7^^^^^^^^^^^^^^~!^^^^^^^^^^^^^^!!^^^^^^^^^^^^^^!~::::::::::::::~:                   ~             ~:............^~............:~      ~:............^~............:~             ~. 
                        .~              .~..............^~..............^~..............~:              ^:                   ~.............!^::::::::::::~~::::::::::::^!      ~^::::::::::::~!::::::::::::^!.............~. 
                        .~              .~.::::::::::::.^~.::::::::::::.^~.::::::::::::.~:              ^:                   ............................................      .............................:..............  
                        .~              .~.::::::::::::.^~.::::::::::::.^~.::::::::::::.~:              ^:                                                                                                                   
                        .~              .~.::::::::::::.^~.::::::::::::.^~.::::::::::::.~:              ^:                                                                                                                   
                        .~              .~..............^~..............^~..............~:              ^:                   ^:::::::::::::~^^^^^^^^^^^^:^^:^^^^^^^^^^^^^      ^^^^^^^^^^^^^:^~:^^^^^^^^^^^^~:::::::::::::^  
                        .~..............:!:^^^^^^^^^^^^:~!:^^^^^^^^^^^^:~!:^^^^^^^^^^^^:!^..............~:                   ~             ~::::::::::::.~~.::::::::::::!      ~::::::::::::.^~.::::::::::::!             ~. 
                        .~..............:!::::::::::::::~!::::::::::::::~!::::::::::::::~^..............~:                   ~             ~:.::::::::::.~~.::::::::::::~      ~::::::::::::.^~.::::::::::::~             ~. 
                        .~              .~......:.......^~..............^~..............~:              ^:                   ~             ~::::::::::::.~~.::::::::::::~      ~::::::::::::.^~.::::::::::::~             ~. 
                        .~              .~.:::.:!7^.:::.^~.::::::::::::.^~.:::::~?:::::.~:              ^:                   ~             ~::::::::::::.~~.::::::::::::~      ~::::::::::::.^~.::::::::::::~             ~. 
                        .~              .~.:::.:~?!.:::.^~.::::::::::::.^~.:::.~?Y^.:::.~:              ^:                   ~             ~:.::::::::::.^~.::::::::::.:~      ~:.::::::::::.^~.::::::::::.:~             ~. 
                        .~              .~.:::::^^:::::.^~.::::::::::::.^~.::::::^:::::.~:              ^:                   !:::::::::::::!~^^^^^^^^^^^^!!^^^^^^^^^^^^~!      ~~^^^^^^^^^^^^!!^^^^^^^^^^^^~7:::::::::::::!. 
                        .~              .~.:::::..:::::.^~.::::::::::::.^~.::::::.:::::.~:              ^:                   ~             ~::::::..::::.~~.::::::::::::!      ~::::::::::::.^~.:::::.::::::!             ~. 
                        .!::::::::::::::^!:^^^^^^^^^^^^^~!:^^^^^^^^^^^^^~!:^^^^^^^^^^^^:!~::::::::::::::~:                   ~             ~:.::.:~^.:::.^~.::::::::::::~      ~::::::::::::.^~.:::.:~::::::~             ~. 
                        .~              .^              :^              :^              ^:              ^:                   ~             ~::::.^!J:.::.~~.::::::::::::~      ~::::::::::::.^~.::.^7Y:.::::~             ~. 
                        .~              .^              :^              :^              ^:              ^:                   ~             ~::::.:~~::::.~~.::::::::::::~      ~::::::::::::.^~.::::^~::::::~             ~. 
                        .~              .^              :^              :^              ^:              ^:                   ~             ~:::::...::::.~~.::::::::::::~      ~::::::::::::.^~.::::..::::::~             ~. 
                        .~              .^              :^              :^              ^:              ^:                   !:::::::::::::!^^^^^^^^^^^^:~!:^^^^^^^^^^^^!      ~^^^^^^^^^^^^:~!:^^^^^^^^^^^^7:::::::::::::!. 
                        .~              .^              :^              :^              ^:              ^:                   ~             ~.            ::             ~      ~.            :^             ~             ~. 
                        .~              :~              :^              ^^              ^^              ^:                   ~             ~.            ::             ~      ~.            :^             ~             ~. 
                         ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::.                   ~             ~.            ::             ~      ~.            :^             ~             ~. 
                                                      2×2 步幅的 3×3 卷积图块                                                ~             ~.            ::             ~      ~.            :^             ~             ~. 
                                                                                                                             ~             ~.            :^             ~      ~.            :^             ~             ~. 
                                                                                                                             ^:::::::::::::~:::::::::::::^^:::::::::::::^      ^:::::::::::::^^:::::::::::::~:::::::::::::^  
                        
                        步幅为 2 意味着特征图的宽度和高度都被做了 2 倍下采样（除了边界效应引起的变化）。
                        虽然步进卷积对某些类型的模型可能有用，但在实践中很少使用。熟悉这个概念是有好处的。

                        为了对特征图进行下采样，我们不用步幅，而是通常使用最大池化（max-pooling）运算，你在第一个卷积神经网络示例中见过此运算。 下面我们来深入研究这种运算。

最大池化运算
|----这就是最大池化的作用：对特征图进行下采样，与步进卷积类似。
|       在卷积神经网络示例中，你可能注意到，在每个 MaxPooling2D 层之后，特征图的尺寸都会减半。
|           例如，在第一个 MaxPooling2D 层之前，特征图的尺寸是 26×26，但最大池化运算将其减半为 13×13。
|----最大池化是从输入特征图中提取窗口，并输出每个通道的最大值。
|       它的概念与卷积类似，但是最大池化使用硬编码的 max 张量运算对局部图块进行变换，而不是使用学到的线性变换（卷积核）。
|----最大池化与卷积的最大不同之处在于，最大池化通常使用 2×2 的窗口和步幅 2，其目的是将特征图下采样 2 倍。
        与此相对的是，卷积通常使用 3×3 窗口和步幅 1。

为什么要用这种方式对特征图下采样？为什么不删除最大池化层，一直保留较大的特征图？
    我们来这么做试一下。这时模型的卷积基（convolutional base）如下所示。
    
    model_no_max_pool = models.Sequential()
    model_no_max_pool.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model_no_max_pool.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model_no_max_pool.add(layers.Conv2D(64, (3, 3), activation='relu'))

    该模型的架构如下。


    Model: "sequential"
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
    ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
    │ conv2d (Conv2D)                      │ (None, 26, 26, 32)          │             320 │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ conv2d_1 (Conv2D)                    │ (None, 24, 24, 64)          │          18,496 │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ conv2d_2 (Conv2D)                    │ (None, 22, 22, 64)          │          36,928 │
    └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
     Total params: 55,744 (217.75 KB)
     Trainable params: 55,744 (217.75 KB)
     Non-trainable params: 0 (0.00 B)

    这种架构有什么问题？有如下两点问题。
    |----这种架构不利于学习特征的空间层级结构。
    |       第三层的 3×3 窗口中只包含初始输入的7×7 窗口中所包含的信息。
    |       卷积神经网络学到的高级模式相对于初始输入来说仍然很小， 这可能不足以学会对数字进行分类（你可以试试仅通过 7 像素×7 像素的窗口观察图像 来识别其中的数字）。
    |       我们需要让最后一个卷积层的特征包含输入的整体信息。
    |----最后一层的特征图对每个样本共有 22×22×64=30 976 个元素。
            这太多了。如果你将其展平并在上面添加一个大小为 512 的 Dense 层，那一层将会有 1580 万个参数。这对于这样一个小模型来说太多了，会导致严重的过拟合。

简而言之，使用下采样的原因，
|----一是减少需要处理的特征图的元素个数，
|----二是通过让连续卷积层的观察窗口越来越大（即窗口覆盖原始输入的比例越来越大），从而引入空间过滤器的层级结构。


注意，最大池化不是实现这种下采样的唯一方法。
|----你已经知道，还可以在前一个卷积层中使用步幅来实现。
|----此外，你还可以使用平均池化来代替最大池化，其方法是将每个局部输入图块变换为取该图块各通道的平均值，而不是最大值。

但最大池化的效果往往比这些替代方法更好。
简而言之，原因在于特征中往往编码了某种模式或概念在特征图的不同位置是否存在（因此得名特征图）， 而观察不同特征的最大值而不是平均值能够给出更多的信息。
    因此，最合理的子采样策略是首先生成密集的特征图（通过无步进的卷积），然后观察特征每个小图块上的最大激活， 而不是查看输入的稀疏窗口（通过步进卷积）或对输入图块取平均，因为后两种方法可能导致 错过或淡化特征是否存在的信息。

现在你应该已经理解了卷积神经网络的基本概念，即特征图、卷积和最大池化，并且也知道如何构建一个小型卷积神经网络来解决简单问题，比如 MNIST 数字分类。下面我们将介绍更
加实用的应用。
"""
