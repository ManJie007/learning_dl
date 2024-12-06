1.what kind of data is most naturally phrased as a graph?
2.We explore what makes graphs different from other types of data, and some of the specialized choices we have to make when using graphs. 
3.we build a modern GNN
4.provide a GNN playground

A graph represents the relations (edges) between a collection of entities (nodes).

    V Vertex (or node) attributes
        e.g., node identity, number of neighbors
    E Edge (or link) attributes and directions 
        e.g., edge identity, edge weight
    U Global (or master node) attributes 
        e.g., number of nodes, longest path

    用embedding(向量)来表示Vertex、Edge 和 Global

图可以分为有向图和无向图

data -> graph 如何转化?
    1.images
        eg: 244x244x3 -> 一个像素映射成一个顶点, 与周围像素建立边的关系       

    2.texts
        文本可以认为是一条序列
        每一个词认为是一个顶点，当前词与前一个词和后一个词建立边的关系(有向边)

    3.Molecules 分子

    4.Social networks
    
    5.Citation networks (有向图)

What types of problems have graph structured data?
    There are three general types of prediction tasks on graphs: graph-level, node-level, and edge-level.

    Graph-level task
        对图进行分类    可能不需要机器学习也可以

    Node-level task
        点分类

    Edge-level task
        图片 -> 语义分割（分割人物和背景）-> 分析人物关系
        把边的属性预测出来

challenges
    1.how we will represent graph's to be compatible with neural networks.
        nodes, edges, global-context and connectivity.
        ----------------------------    --------------
              Embedding向量              邻接矩阵？可能非常大，用稀疏矩阵？GPU处理起来低效 
                                         邻接矩阵交换行或者交换列语义都不变，设计神经网络，得确保输出相同
        
       又想要存储高效 又想要排序不影响的话 可以采取下面的存储形式
        
       Nodes    可以是标量也可以是向量
        [0, 0, 1, 0, 0, 0, 1, 1]

       Edges    可以是标量也可以是向量
        [1, 1, 1, 2, 2, 1, 2]

       Adjacency List
        第i项表示第i条边连接的哪两个顶点，打乱点和边的顺序只要把Adjacency List的顺序也调整即可
        [[1, 0], [2, 0], [4, 3], [6, 2], [7, 3], [7, 4], [7, 5]]

       Global   可以是标量也可以是向量
         0

        next step: 如何给神经网络处理?
                                         
Graph Neural Networks
    A GNN is an optimizable transformation on all attributes of the graph (nodes, edges, global-context) that preserves graph symmetries (permutation invariances). 
    GNN是一种可优化的图（节点、边、全局上下文）属性的变换，它保留了图的对称性（即对元素的排列不变性）。

    we're going to build GNNs using the "message passing neural network" framework proposed by Gilmer et al. [18]
    "graph-ig, graph-out"   
    without changing the connectivity offthe input graph.

    最简单的GNN
        Un、Vn、Fn(格式：向量) 分别构造MLP多层感知机
        三个MLP构成一个GNN层
        
        输出：属性更新过，但是图的结构是没有发生变化的

        MLP对每一个向量独自作用，不会考虑所有的连接信息，所以不管对顶点做任何排序，都不会改变结果

        可以叠加在一起做一个深一点的GNN

    最后一层的输出？
        
        如果是对每个顶点做预测
            顶点存在向量
                假如是二分类/n分类，GNN -> 全连接层(输出度为2/n) -> softmax

            顶点不存在向量
                pooling
                    拿出与该点相连的所有边的向量 + 全局向量 => 代表这个点的向量(汇聚层)

                    GNN -> 汇聚层 -> 全连接层(输出度为2/n) -> softmax

                类比：没有边向量，只有顶点向量，边相邻点的向量 + 全局向量 => 代表这条边的向量(汇聚层)
                      没有全局向量，只有顶点向量，所有顶点向量加起来 -> 全局向量(汇聚层) -> 全局向量的输出层(MLP)
                      缺乏哪一类的属性，都可以通过汇聚操作，得到那个属性的向量


        graph(U,V,F) -> GNN(每一种向量对于一个MLP，三层MLP) -> transformed Graph (保留了图的结构，属性变了) -> 缺失信息引入汇聚层 -> Classification layer -> Prediction
            缺点：当前GNN没有考虑连接信息，没有把图信息更新进属性

        改进？
            Passing messages between parts of the graph
            original:
                哪一类信息向量 -> MLP -> 
            now:
                哪一类信息向量 + 邻居的信息向量 (汇聚) -> MLP ->
                有点和图片的卷积有点像，但是你这个卷积的核窗口里的权重应该是一样的,但这里通道还是保留了(MLP的层数，对应信息的种类)
                很多层GNN放在一起，一个向量就能汇聚很多相邻向量的信息
                
                                                                                   表示
            futhur:假如缺失某种属性，我们之前的做法是用相邻其他种类属性汇聚起来 => 缺失的属性
                   不用等到最后一层做这个事情

                Learning edge representations
                    把顶点信息传递给边，                            再把边的信息传递给顶点

                    solution 1
                        (把边相邻的两个顶点向量加到边的向量里面)        (再把顶点连接边的信息加到顶点的向量中)
                        纬度不一样的话做一个投影
                    solution 2
                        concat在一起

                    返回来的话：再把边的信息传递给顶点,把顶点信息传递给边，
                    结果会不一样, 当前并不知道谁比谁好，有一个办法，交替更新

                        把顶点信息传递给边      (同时)
                        把边的信息传递给顶点
                       
                        先别加，再来一次
    
                        把顶点信息传递给边      (同时)
                        把边的信息传递给顶点

为什么要有全局信息？
    图很大 或 连接没那么紧密 (Passing message 需要走很长的步)
    解决:
        master node or context vector
        一个虚拟的点 跟所有点相连 跟所有边相连 (抽象的概念)
        Un

        更新Un     会把Vn和En汇聚到Un
        更新Vn     会把En和Un汇聚到Un
        更新En     会把Un和Vn汇聚到Un

每一个向量之间都进行了消息传递，最后做预测时
    只用本身的向量
    本身的向量 +/concat 相邻边
    本身的向量 +/concat 相邻边 相邻点
    本身的向量 +/concat 相邻边 相邻点 全局向量

有一点像attention mechanism.

汇聚操作：Mean、Sum、Max
          对应卷积池化层的max pooling 和 average pooling

GNN对超参数比较敏感

其他形式的图:
    multigraph 顶点之间有多种边
    图可能分层 一个顶点可能是一个子图
    不同图的结构可能在信息汇聚的时候有一个影响

Sampling Graphs and Batching in GNNs
    之前我们讲过 假设你有很多层的时候 
    最后那一层呢一个顶点它就算每一层里面只看它的一近邻
    但最后这个顶点因为有很多层消息传递
    所以最后顶点它其实能看到的是一个很大的图
    假设你这个图联通性够的话所以最后这个顶点可能看到的是整个图的信息

    我们知道在计算梯度的时候 我们要把整个 forward 里面 所有的中间变量给你存下来
    如果你最后一个顶点它要看整个图的话 那么意味着我对它算梯度的时候
    我得把整个图之间的中间结果都存下来 这样子导致我的计算
    可能是无法承受的 所以这个地方我们需要对图进行采样
    就是说我们把这个图 每一次采用一个小图出来 在这个小图上面做信息的汇聚
    这样子算梯度的时候呢 我只要在这个小图上的中间结果记录下来就行了
        
    采样的方法:
        第一个是说我随机采样一些点 然后把这些点 它的最近的邻居给你找出来 
            那么我们在做计算的时候 我们只在这个子图上做计算 这样的话通过控制我每次采样多少个点 我可以避免这个图不要特别大 使它内存是能存下的

        第二个是说我可以做一些随机游走 就说我从某一个顶点开始 然后我们随机在里面找到一条边 然后沿着这条边走到下一个点 节点 然后我们就沿着这个图随机走 我们可以说规定你最多随机走多少步就会得到一个子图

        第三个是说你可以结合这两种我先随机走三步 然后把这三步每个点 它的邻居也找出来

        最后一个办法是说 我取一个点 把它的一近邻二近邻三近邻然后往前走 k 步把它做一个 把它做一个宽度遍历 然后把一个子图给你拿出来

        分别是四种不同的采用方法 但是具体在你的数据集上 哪一种比较好这个是取决你这个整个图 长得是什么样子了

跟采样的一个相关的问题是做batch
    从性能上的考虑 我们不想对每一个顶点逐步逐步更新 这样子的话每一步我们计算量太小 不利于并行 我们希望像别的神经网络一样 我们能把这些小样本给你 做成一个小批量
    这里的话一个问题是我们每一个顶点 它的邻居的个数是不一样的 你怎么样把这些顶点它邻居通过合并成一个规则的一个 张量这是一个有挑战性的问题


inductive biases
    任何一个神经网络 或者任何一个机器学习的模型 都有一些假设在里面
    你如果不对这个世界做假设的话 那么你什么东西都学不出来
        比如说你的卷积神经网络假设的是你的空间变换的不变性 你的循环神经网络 假设的是你的时序的延续性
        作者说GNN的假设就是之前说的两大特点之一 它保持了图的对称性就是说你不管怎么样 交换我的顶点的一个顺序 我的GNN对它的作用都是保持不变的

Comparing aggregation operations
    比较了不同的 汇聚的时候的操作 我们之前有讲过我们可以求和求平均或者是求 max 它说其实没有一种是特别理想的

GCN as subgraph function approximators
    GCN 是 graph convolutional network
    带了汇聚的那一个图神经网络
    GCN 如果有 k 个层 然后每一层都是看一个它的邻居的话就等价是说 相类似于 它有点类似于在卷积神经网络里面我有 k 层3*3的卷积
    每一个最后的顶点 它看到的是一个子图 这个子图它的 大小是 k就是说每 最远的顶点 离我当前这个点的距离是 k 因为你 每过一层你就往前看了一步对吧 所以这样子的话它其实可认为 每个点都是看以自己为中心的 往前走 k 步那个子图的信息的汇聚
    
    所以它说从一个程度上来说 其实你这个 GCN 呢就可以认为是有 n 个这样子的子图 每个子图就是以它为中心往前走 比如说这个是走两步然后把所有的这个子图上 求一个embedding出来

Edggs and the Graph Dual
    你可以把点和边做对偶 我们知道你可以把一个点变成边 边变成点 然后它的邻接关系表示不变

Graph convolutions as matrix multiplications, and matrix multiplications as walks on a graph
    它的核心思想是说你在图上做卷积或者图上做下 random walk 呀 其等价也是说你 把它的临界矩阵拿出来然后做一个矩阵的乘法
    如果大家知道 page rank 的话 也就15年前 大家最常讨论的一个技术的话 page rank 就是在一个很大的图上面 做一个随机游走 基本上就是 拿着邻接矩阵出来 它跟一个向量不断的做乘法

Graph Attention Networks
    之前我们在讲图上的汇聚跟 卷积的关系的时候多多少少有提到过 我们在图上做汇聚的时候是每个顶点 和它的邻接的顶点 它的权重 加起来 但如果是卷积的话 我是做一个加权的一个和

    同样道理我在图上我也可以做加权合 但是呢你要注意的一点是说卷积的权重是跟位置相关的 就是说你每个窗口一个3*3的窗口在这个地方 然后在每一个 固定的点上有固定的权重

    但是呢在图来说我不需要有这个位置信息 因为每个顶点它的那个邻居个数不变而且邻居是可以随意打乱顺序的 所以我们需要整个 权重 对这个位置是不敏感的
    
    那么一个做法就是说我可以用 注意力机制的那种做法 就是说 你的这个权重取决于你那个顶点两个顶点向量之间的关系 就不再是你这个顶点是位置什么地方

    如果在使用了 attension 之后呢 我们可以得到每一个顶点给它一个权重 这样子我们再按这个权重加起来 就得到了一个 graph attention network

Graph explanations and attributions
     就是说我在图上面训练一个模型之后 我怎么去看它到底学到是什么东西比如说我可以把一些子图 里面的信息给你抓取出来 看看它学到是什么有意思的东西 

Generative modelling
