PG
    1.NN as actor
    input: observation
    output: stochastic随机的 probability of taking the action  ----|
            output layar : 有几种动作就有几个dimension             |
                                                                   |
    好处:                                                          |
        using network instead a lookup table
        generalize, 举一反三, 没有见过的东西也能有输出             |
        传统方法：Q表 输入<->输出                                  |
                                                                   |
    2.actor的好坏:                                                 |
        让actor去"玩游戏" -> total reward R = Σri                  |
        Even with the same actor, R is different each time         |
            Randomness in the actor and the game    ----------------
            actor采取的动作 和 游戏的本身 有随机性
        我们希望maximize total reward R

        An episode is considered as a trajectory τ
            τ= {s1,a1,r1,s2,a2,r2,...,st,at,rt} 一个序列
            R(τ) =  Σri
                    τ
            If you use an actor to play the game, each τhas a probability to be sampled
                The probability depends on actor parameter θ: P(τ|θ)   --- θ 其实就是神经网络的参数，策略给定了可以计算每一个trajectory τ 发生的几率

                E[R_{θ}] 总奖励的期望 =  Σ R(τ) P(τ|θ) 
                                         τ
                穷举所有 τ不现实
                让actor完N场 得到{τ1, τ2, ..., τN}
                             N
                近似= 1/N *  Σ R(τi)
                            i=1

    3.Gradient Ascent   Gradient Ascent（梯度上升）是一种优化算法，用于最大化一个函数的值。
        步骤：
            从初始点 𝑥0 开始。
            计算当前点 𝑥𝑘 处的梯度 ∇𝑓(𝑥𝑘)。
            沿着梯度的方向（正方向）更新参数。
            重复直到收敛（即，梯度变得足够小或者达到预设的最大迭代次数）。

        最大话 E[R] 其实就是调整NN的θ

        随机θ (初始actor)
                                                                                                              N
        ∇E[R_{θ}] = Σ R(τ) ∇P(τ|θ) = Σ R(τ) P(τ|θ) ∇P(τ|θ) / P(τ|θ) =  Σ R(τ) P(τ|θ) ∇logP(τ|θ)   近似= 1/N * Σ R(τ) P(τ|θ) ∇logP(τ|θ)
                    τ                τ                                 τ                                     i=1

        P(τ|θ) = P(s1)P(a1|s1, θ)P(r1, s2|s1, a1)P(a2|s2, θ)P(a1|s1, θ)...
                       T
               = p(s1) Π P(at|st, θ)P(rt, st|st, at)            其中 只有p(at|st, θ) 跟actor(NN)有关系
                      t=1

                               T
        logP(τ|θ) = logp(s1) + Σ logP(at|st, θ) + logP(rt, st|st, at)
                              t=1

                     T                   
        ∇logP(τ|θ) = Σ logP(at|st, θ)
                    t=1                 



                              N                          N      T                         N  T                    
        ∇E[R_{θ}] 近似= 1/N * Σ R(τ) ∇logP(τ|θ)  = 1/N * Σ R(τ) Σ ∇logP(at|st, θ) = 1/N * Σ  Σ  R(τ) ∇logP(at|st, θ)     注:是不是少了P(τ|θ)???
                             i=1                        i=1    t=1                       i=1t=1                   

        直觉就是: 在τi 采取 at 在st 状态下 R(τ) 是正的，改变θ 变大 P(at|st)
                                                  负的，改变θ 变小 P(at|st)

        注意这里R(τ) 是一个episode的total reward

        加log，微分之后除掉一个几率 做一个正则化

        add a baseline : 如果全是正，某个action没sample到 几率就会减小 正则化

                                  N  T                    
            ∇E[R_{θ}] 近似= 1/N * Σ  Σ  (R(τ) -b) ∇logP(at|st, θ) 
                                 i=1t=1                   

        
        
        initialize (random θ) -> play episode (data collection) -> Gradient Ascent(Update Model)    循环
                                                                    
                                                                                                                       ^
                                                                   在classification中 我们minimize corss enptory  : -Σ yi logyi

                                                                                          等价于 maximize logyi  就是 上面 logP(at|st, θ) 在当前状态采取某动作的概率 可以看成分类问题(采取某个动作)

                                                                                          但我们还乘上了 R(τ) 可以看成 classification x 对应的次数 （1 = 1次） 也可以转换为分类问题
                                                                                                是不是也可以认为是增加奖励大的状态下采取的动作

感悟：NN(actor)的参数作为策略，输入是状态，输出的动作概率分布
      知道梯度更新的公式了
                                  N  T                    
            ∇E[R_{θ}] 近似= 1/N * Σ  Σ  R(τ)∇logP(at|st, θ) 
                                 i=1t=1                   
      先用actor收集一大堆(s, a), 和每个trajectory 的 total Reward
      代入梯度更新公式 把gradient 算出来
        增加奖励大的状态下的采取的动作的概率
      更新θ

      就是收集数据 -> 更新模型 循环
      
      可以是classification 但lossfunction要乘负号和total reward

如果發現不能微分就用 policy gradient 硬 train 一发  -- 环境和reward是黑盒子

tips:
    1.add baseline      如果全是正，某个action没sample到 几率就会减小 正则化

                                  N  T                    
            ∇E[R_{θ}] 近似= 1/N * Σ  Σ  (R(τ) -b) ∇logP(at|st, θ)           b 可以取 E[R]
                                 i=1t=1                   

    2.Assign Suitable Credit
        应该给每一个action合适的credit
        在一句游戏中可能有的action是好的 有的是不好的
        把(R(τ) -b)全局的reward换成从这个action执行之后的reward

                                  N  T   T                
            ∇E[R_{θ}] 近似= 1/N * Σ  Σ ( Σr -b) ∇logP(at|st, θ)           b 可以取 E[R]
                                 i=1t=1 t'=t              

        更近一步
            未来的reward做一个discount  时间拖得越长 影响力就越小

                                  N  T   T                  
            ∇E[R_{θ}] 近似= 1/N * Σ  Σ ( Σ γ^(t' - t)r -b) ∇logP(at|st, θ)           b 可以取 E[R] 也可以是state-dependent
                                 i=1t=1 t'=t              
                                        ------------------
                                        Advantage Function Aθ(st, at)
                                        How good it is if we take afother than other actions at st.
                                        Estimated by "critic" (later)

                                  N  T                      
            ∇E[R_{θ}] 近似= 1/N * Σ  Σ Aθ(st, at) ∇logP(at|st, θ)           
                                 i=1t=1                   
                                                          
        
critic
    represented by a neural network (NN)
    input:
        state
    output:
        Vπ(s) : When using actor π, the cumulated reward expects to be obtained after seeing observation (state) s.  注意是估计直到结束会获得的reward
        给他不同的actor，同一个state，输出也是不一样的

    A critic does not determine the action.

    Given an actor π, it evaluates the how good the actor is.

    如何计算Vπ(s):
        1.蒙特卡洛:采样，估计(逼近实际数值) regression问题 需要完整回合
        2.Temporal-Difference (TD) : 不需要等待一个完整的回合（episode）结束才能更新价值估计，而是通过逐步更新来进行学习。
            TD方法的更新依据是 当前时刻的估计与下一个时刻的估计之间的差异，这就是“时间差异”。
            ...st, at, rt, st+1....
            Vπ(st) = Vπ(st+1) + rt
            训练NN 让Vπ(st) - Vπ(st+1) 逼近 rt


Another Critic
    State-action value function Qπ(s, a)

    When using actor π, the cumulated reward expects to be obtained after seeing observation s and taking a

    input:
        (s, a) or s (action空间很小的情况)

    output:
        the cumulated reward expects to be obtained after seeing observation (state) s and taking action a 
        Qπ(s, a)
        or
        the cumulated reward expects to be obtained after seeing observation (state) s and taking action a(action空间很小的情况下，展开所有action，这样输入的时候就不需要action了)
        Qπ(s, a = left) Qπ(s, a = right) Qπ(s, a = fire)


Another Way to use Critic: Q-Learning

loop:
    π interacts with the environment

            |               1.蒙特卡洛                    
            v               2.Temporal-Difference (TD) 

    Learning Qπ(s,a)

            |               "Better":Vπ'(st) >= Vπ(st) for all state s
            v               π'(s) = arg maxQπ(s, a)
                                         a
                            problem:Not suitable for continuous action a (solve it later) 因为要穷举所有的action
                            A solution : Pathwise Derivative Policy Gradient 增加一个网络(actor)产生连续的动作 (DDPG)

                                               ----
                                          s -> |  |
                                |-----|        |Qπ| -> Qπ(s, a)
                            s ->|Actor|-> a -> |  |
                                | π   |        ----
                                -------

    Find a new actor π' "better" than π    π'=π

trick:  easy-rl里面
    Double DQN
    Dueling DQN


Actor + Critic:
    Actor 不看环境的reward 去 Gradient Ascent 环境reward的随机性太大
          跟Critic学

    A2C Advantage Actor-Critic
    A3C Asynchronous Advantage Actor-Critic （开分身加速学习)


Inverse Reinforcement Learning
    env actor no reward function
    We have demonstration of the expert.
        Each τ is a trajectory of the export.
        专家玩的回合

    => 推导出Reward Function => 再用Reinforcement Learning 的方法

    Principle: The teacher is always the best.
    Basic idea:
        • Initialize an actor

        • In each iteration
            • The actor interacts with the environments to obtain some trajectories
            • Define a reward function, which makes the trajectories of the teacher better than the actor 
            • The actor learns to maximize the reward based on the new reward function.

        • Output the reward function and the actor learned from the reward function


On-policy v.s. Off-policy

• On-policy: The agent learned and the agent interacting with the environment is the same.

                              N                          N      T                         N  T                    
        ∇E[R_{θ}] 近似= 1/N * Σ R(τ) ∇logP(τ|θ)  = 1/N * Σ R(τ) Σ ∇logP(at|st, θ) = 1/N * Σ  Σ  R(τ) ∇logP(at|st, θ)     注:是不是少了P(τ|θ)???
                             i=1                        i=1    t=1                       i=1t=1                   

        用actor与环境交互采集数据，模型参数变了意味着在某个状态采取动作的概率变了，策略变了 之前采集的数据就不能用来训练模型了  非常花时间 采集一次数据 Gradient Ascent 一次
            Use πθ to collect data. When 0 is updated, we have to sample training data again.                  |
                                                                                                               |
• Off-policy: The agent learned and the agent interacting with the environment is different.                   |
                                                                                                               |
        从On-policy 到 Off-policy 的好处就是:                                                                  |
            Goal: Using the sample from πθ', to train 0. θ' is fixed, so we can re-use the samiple data.    比较有效率

            Importance Sampling 
                                  N
            E_{x~p}[f(x)] 约= 1/N Σ  f(xi)                  xi is sample from p(x) but we only have xi sampled from q(x)
                                 i=1

                            = ∫ f(x)p(x) dx = ∫ f(x)p(x)q(x)/q(x) dx = E_{x~q}[f(x)p(x)/q(x)]

            Issue of Importance Sampling:
                方差不一样  所以如果sample次数不够多，可能期望差距大

            用于从On-policy 到 Off-policy 变换 不拿actor与环境做互动


        ∇E[R_{θ}] = E_{τ~pθ(τ)}[R(τ) ∇logP(τ|θ)]

                  = E_{τ~pθ'(τ)}[R(τ) ∇logP(τ|θ) pθ(τ) / pθ'(τ)]

            Sample the data from 0'.

            Use the data to train 0 many times.

                                         
        ∇E[R_{θ}] = E_{(st, at) ~ πθ}[Aθ(st, at) ∇logP(at|st, θ)]

                  = E_{(st, at) ~ πθ'}[Aθ'(st, at) ∇logP(at|st, θ) pθ(τ) / pθ'(τ)]
                                       
                  = E_{(st, at) ~ πθ'}[Aθ'(st, at) ∇logP(at|st, θ) pθ(at|st)pθ(st) / pθ'(at|st)pθ'(st)  ]   注意：pθ(st) / pθ'(st) 这一项不好算 李宏毅说说服自己认为没影响

        ∇f(x)=f(x)∇logf(x)

        ==> Jθ'(θ) = E_{(st, at) ~ πθ'}[Aθ'(st, at) pθ(at|st) / pθ'(at|st)]   原方程 Optimization目标函数 求导后等于上面 ∇E[R_{θ}] 
                                                                              这一项是可以算的



        Proximal Policy Optimization (PPO)/TRPO

            Jθ'_PPO(θ) = Jθ'(θ) - βKL(θ, θ')    因为Issue of Importance Sampling: 方差不一样  所以如果sample次数不够多，可能期望差距大，这里有点正则化的感觉

        PPO algorithm

            • Initial policy parameters 0°

            • In each iteration

                • Using θ^k to interact with the environment to collect {st, at} and compute advantage Aθ^k (st, at) θ^k前一个training iteration得到的模型参数
                • Find θ optimizing  Jθ_PPO(θ)

                    Jθ^k_PPO(θ) = Jθ^k(θ) - βKL(θ, θ^k)    Update parameters several times

                    if KL(θ, θ^k) > KLmax, increase β   Adaptive KL Penalty
                    if KL(θ, θ^k) < KLmin, decrease β 

        PPO2

Q-Learning
    value-based
    learn Critic

    represented by a neural network (NN)
    input:
        state

    A critic does not determine the action.

    Given an actor π, it evaluates the how good the actor is.

    output:
        Vπ(s) : When using actor π, the cumulated reward expects to be obtained after seeing observation (state) s.  注意是估计直到结束会获得的reward

    强调一点:
        一个 Critic 都是绑定一个 Actor
        没有办法凭空estimate一个V 都是给定一个state假设接下来互动actor是π，会获得多少reward
        The output values of a critic depend gn the actor evaluated.
        给他不同的actor，同一个state，输出也是不一样的
            v以前的阿光（大馬步飛）=bad 
            v𤓖強的阿光（大馬步飛）=good

    How to estimate Vπ(s)
        • Monte-Carlo (MC) based approach
            The critic watches π playing the game

                  NN            
                 |--|           
            s -> |Vπ| -> Vπ(s) <-> G(accumulate reward 需要一整个回合)    (regression problem)
                 |--|           

            After seeing Sa, Until the end of the episode, the cumulated reward is Ga
            After seeing Sb, Until the end of the episode, the cumulated reward is Gb

        • Temporal-difference (TD) approach
                    ...st, at, rt, st+1,...
                    Vπ(st) = Vπ(st+1) + rt

                           NN          
                          |--|         
                    st -> |Vπ| -> Vπ(st)
                          |--|         

                                                Vπ(st) - Vπ(st+1) <-> rt   (regression problem) 

                           NN          
                          |--|         
                    st -> |Vπ| -> Vπ(st+1)
                          |--|         
        MC v.s. TD

            MC
                Larger variance     Var[kX] = k^2Var[X]
                Ga is the summation of many steps

            TD
                r smaller variance
                Vπ(s) may be inaccurate

    Another Critic
        State-action value function Qπ(s, a)

        When using actor π, the cumulated reward expects to be obtained after seeing observation s and taking a
        Qπ(s, a) = acotr在当前s采取action a 后面就由actor自己去玩 将会获得的总奖励

        input:
            (s, a) or s (action空间很小的情况)

        output:
            the cumulated reward expects to be obtained after seeing observation (state) s and taking action a 
            Qπ(s, a)
            or
            the cumulated reward expects to be obtained after seeing observation (state) s and taking action a(action空间很小的情况下，展开所有action，这样输入的时候就不需要action了)
            Qπ(s, a = left) Qπ(s, a = right) Qπ(s, a = fire)

                       ----
                  s -> |  |
                       |Qπ| -> Qπ(s, a)
                  a -> |  |
                       ----
            
                       ----
                       |  | -> Qπ(s, a=left)
                  s -> |Qπ| -> Qπ(s, a=right)       for discrete action only
                       |  | -> Qπ(s, a=fire)
                       ----

    loop:
        π interacts with the environment

                |               1.蒙特卡洛                    
                v               2.Temporal-Difference (TD) 

        Learning Qπ(s,a)        Qπ(s, a) = acotr在当前s采取action a 后面就由actor自己去玩 将会获得的总奖励

                |               "Better":Vπ'(st) >= Vπ(st) for all state s
                v               π'(s) = arg maxQπ(s, a)
                                             a
                                π' does not have extra parameters. It depends on Q

                                problem:Not suitable for continuous action a (solve it later) 因为要穷举所有的action
                                A solution : Pathwise Derivative Policy Gradient 增加一个网络(actor)产生连续的动作 (DDPG)

                                                   ----
                                              s -> |  |
                                    |-----|        |Qπ| -> Qπ(s, a)
                                s ->|Actor|-> a -> |  |
                                    | π   |        ----
                                    -------

        Find a new actor π' "better" than π    π'=π     一直循环下去policy就会更好
        
    tips:

        Target Network
            ...st, at, rt, st+1,...
            Qπ(st, at) = Qπ(st+1, at+1) + rt

                   NN          
            st -> |--|         
                  |Qπ| -> Qπ(st, at)
            at -> |--|         

                                        Qπ(st+1, π(st+1)) - Qπ(st, at) <-> rt   (regression problem) 
                                                                      去逼近
                      NN          
            st+1  -> |--|         
                     |Qπ| -> Qπ(st+1, π(st+1))
          π(st+1) -> |--|         

            不好training，因为更新网络参数，target也一直在变

                固定目标网络, After updating N times Qπ(π表示神经网络的性能参数), Target Network = Qπ

                          NN    Target Network         
                st+1  -> |--|         
                         |Qπ| -> Qπ(st+1, π(st+1))  fixed value
              π(st+1) -> |--|         

                         fixed

        Exploration
            The policy is based on Q-function

                a = arg maxQπ(s, a)
                         a
            
            与PG的区别是PG输出动作的probability distribution
            Qπ输出Q(s) or (Q(s), a)
            Q-Learning 的policy是采取Q值最大的动作
                This is not a good way for data collection.

              -> a1     Q(s, a) = 0 Never explore
            s -> a2     Q(s, a) = 1 Always sampled
              -> a3     Q(s, a) = 0 Never explore


            Epsilon Greedy  ε would decay during learning
            
                        arg maxQ(s, a),         with probability 1 - ε
                a   =        a
                        random,                 otherwise

            Boltzmann Exploration
                    
                P(a|s) = exp(Q(s, a)) / ∑ exp(Q(s, a))

        Replay Buffer
            
            π interacts with the environment    Put the experience into buffer.((st, at, rt, st+1),...)
                                                The experience in the buffer comes from different policies. 变成了Off-policy
                                                Drop the old experience if the buffer is full.

                    |               In each iteration:
                    |                   1. Sample a batch
                    |                   2. Update Q- function
                    |               1.蒙特卡洛                    
                    v               2.Temporal-Difference (TD) 

            Learning Qπ(s,a)        Qπ(s, a) = acotr在当前s采取action a 后面就由actor自己去玩 将会获得的总奖励

                    |               "Better":Vπ'(st) >= Vπ(st) for all state s
                    v               π'(s) = arg maxQπ(s, a)
                                                 a
                                    π' does not have extra parameters. It depends on Q

                                    problem:Not suitable for continuous action a (solve it later) 因为要穷举所有的action
                                    A solution : Pathwise Derivative Policy Gradient 增加一个网络(actor)产生连续的动作 (DDPG)

                                                       ----
                                                  s -> |  |
                                        |-----|        |Qπ| -> Qπ(s, a)
                                    s ->|Actor|-> a -> |  |
                                        | π   |        ----
                                        -------

            Find a new actor π' "better" than π    π'=π     一直循环下去policy就会更好

            Advantage:
                1.节约时间 减少与环境交互的次数
                2.batch data diverse
                
                我们是估计Qπ(s, a), buffer 中混杂了不是π的experience 有没有关系?李宏毅说没有


        Typical Q-Learning Algorithm

                                                         ^
            • Initialize Q-function Q, target Q-function Q = Q
            • In each episode
                • For each time step t
                • Given state st, take action a, based on Q (epsilon greedy)
                • Obtain reward rt, and reach new state st+1 
                • Store (st, at, rt, st+1) into butter
                • Sample (si, ai, ri, si+1) from buffer (usually a batch) 
                                      ^
                • Target y = ri + max Q(si+1, a)
                • Update the parameters of Q to make Q(si, ai) close to y (regression)
                                      ^
                • Every C steps reset Q = Q Created 

    tips:
        Double DQN
            Q value is usually over-estimated
            
                Q(st, at) <-> rt + max Q(st+1, a)
                                    a
                                    Tend to select the action that is over-estimated

            Double DQN: two functions Q and Q'  
                Target Network
                选action的Q function和算value的Q function不是同一个
                行政和立法分开

                Q(st, at) <-> rt + max Q'(st+1, arg maxQ(st+1, a))
                                                     a
                                    If Q over-estimate a, so it is selected. Q' would give it proper value.
                                    How about Q' overestimate? The action will not be selected by Q.

        Dueling DQN
            改了network的架构
            old output: Q(s, a)
            new output: Q(s, a) = A(s, a)                            + V(s)
                                  Vector(每一个action都有一个value)    Sclar
            
                                        
                                            state
                                          3  3  3  1
                Q(s, a)          action   1 -1  6  1
                                          2 -2  3  1

                   ||                         ||


                  V(s)                    2  0  4  1


                    +                          +

                                          1  3 -1  0
                 A(s, a)                 -1 -1  2  0
                                          0 -2 -1  0
                
                 改变V(s)会改变改列的所有Q(s, a)
                 这样就算没有sample到的(s, a), 也可以update estimate
                 会有一些constrain：让模型更倾向于改变V(s), 而不是A(s, a)
                                    V(s) Average of column
                                    A(s,a) sum of column = 0
                 实做:
                    Normalize A(s,a) before adding with V(s)!!!

        Prioritized Reply

                Experience Buffer -> (st, at, rt, st+1) batch -> estimate Q(s, a)(TD/MC) -> π'=π 

                ```
                    • Temporal-difference (TD) approach
                                    ...st, at, rt, st+1,...
                                    Vπ(st) = Vπ(st+1) + rt

                                           NN          
                                    st -> |--|         
                                          |Qπ| -> Q(st, at)
                                    at -> |--|         
                                                                          ^                                                               
                                                      Qπ(st, at) <-> rt + Q(st+1, at+1)   (regression problem) 
                                                                     ^
                                                      at+1 = arg max Q(st+1, a)
                                                                  a

                                           NN          
                                  st+1 -> |--|              
                                          |^ |    ^         
                                          |Qπ| -> Q(st+1, at+1)
                                  at+1 -> |--|         

                ```

                The data with larger TD error in previous training has higher probability to be sampled.
                Parameter update procedure is also modified.


        Multi-step          Balance between MC and TD
            
                (st, at, rt, ..., st+N, at+N, rt+N, st+N+1) in Experience buffer
                ```
                    • Temporal-difference (TD) approach
                                    ...st, at, rt, st+1,...
                                    Vπ(st) = Vπ(st+1) + rt

                                           NN          
                                    st -> |--|         
                                          |Qπ| -> Q(st, at)
                                    at -> |--|         
                                                                          t+N ^                                                               
                                                      Qπ(st, at) <-> rt' + Σ  Q(st+N+1, at+N+1)   (regression problem) 
                                                                         t'=t
                                                                       ^
                                                      at+N+1 = arg max Q(st+N+1, a)
                                                                    a

                                             NN          
                                  st+N+1   -> |--|              
                                              |^ |    ^         
                                              |Qπ| -> Q(st+1, at+1)
                                  at+N+1   -> |--|         

                ```

        Noisy Net

            • Noise on Action (Epsilon Greedy)

                Epsilon Greedy  ε would decay during learning
                
                            arg maxQ(s, a),         with probability 1 - ε
                    a   =        a
                            random,                 otherwise

                • Given the same state, the agent may takes different actions.
                • No real policy works in this way
                隨機亂試

            • Noise on Parameters

                Inject noise into the parameters of Q-function at the beginning of each episode
                每一局开始的时候加noise, 开始之后不加
                • Given the same (similar) state, the agent takes the same action.
                    • → State-dependent Exploration 
                • Explore in a consistent way
                有系統地試

                                        ~
                Q(s, a) -> add noise -> Q(s, a)

                            ~
                a = arg max Q(s,a )
                         a


        Distributional Q-function
            State-action value function Qπ(s, a)
                When using actor π, the cumulated reward expects to be obtained after seeing observation s and taking a
                Qπ(s, a)是一个期望值
                真实是一个distribution
                Different distributions can have the same values.
                output 期望值expects -> output distributions

                
                           NN          
                          |--| -> Qπ(s, a1)       
                    s ->  |Qπ| -> Qπ(s, a2)     A network with 3 outputs
                          |--| -> Qπ(s, a3)        

                           NN          
                          |--| ->  distribution               
                    s ->  |Qπ| ->  distribution A network with 15 outputs (each action has 5 bins, bins sum = 1)
                          |--| ->  distribution                               

        Rainbow
            把所有方法合并起来

Q-Learning for Continuous Actions
        没有ppo之前 policy gradient比较不稳
        Q-Learning比较容易train 只要estimate 一个Q function 就一定能找到一个更好的policy
                                    -----------------------
                                      regression problem 比较容易
        但Q-Learning不太好处理continuous problem, Action a is a continuous vector
            
            a = arg maxQ(s, a)
                     a

        Q-Learning estimate Q function 之后 一定要解出最大的a 让Q最大 如果是a是continuous的，不好解 如果是离散的直接带进去找最大值就行

        Solution 1

            Sample a set of actions: {a1, a2,.., aN}
            See which action can obtain the largest Q value

        Solution 2
            
            Using gradient ascent to solve the optimization problem.

        Solution 3
            
            Design a network to make the optimization easy.

                           NN          
                          |--| ->  μ(s) vector
                    s ->  |Qπ| ->  Σ(s) matrix
                          |--| ->  V(s) Sclar

                Q(s, a)=-(a - μ(s))^T Σ(s) (a - μ(s)) + V(s)
                μ(s) = arg max Q(s, a)
                            a

        Solution 4 
            Don't use Q-learning
                Policy-based    Learning a actor
                value-based     Learning a critic

                Policy-based + value-based      Actor + Critic


Asynchronous Advantage Actor-Critic (A3C)

                                  N  T   T                  
            ∇E[R_{θ}] 近似= 1/N * Σ  Σ ( Σ γ^(t' - t)r -b) ∇logP(at|st, θ)           b 可以取 E[R] 也可以是state-dependent
                                 i=1t=1 t'=t              
                                        ------------------
                                        Advantage Function Aθ(st, at)
                                        How good it is if we take afother than other actions at st.
                                        Estimated by "critic" (later)

                                  N  T                      
            ∇E[R_{θ}] 近似= 1/N * Σ  Σ Aθ(st, at) ∇logP(at|st, θ)           
                                 i=1t=1                   

            直觉就是: 在τi 采取 at 在st 状态下 R(τ) 是正的，改变θ 变大 P(at|st)
                                                      负的，改变θ 变小 P(at|st)

            注意这里R(τ) 是一个episode的total reward

            加log，微分之后除掉一个几率 做一个正则化
                                                               ^
           在classification中 我们minimize corss enptory  : -Σ yi logyi

                                  等价于 maximize logyi  就是 上面 logP(at|st, θ) 在当前状态采取某动作的概率 可以看成分类问题(采取某个动作)

                                  但我们还乘上了 R(τ) 可以看成 classification x 对应的次数 （1 = 1次） 也可以转换为分类问题
                                        是不是也可以认为是增加奖励大的状态下采取的动作

            感悟：NN(actor)的参数作为策略，输入是状态，输出的动作概率分布
                  知道梯度更新的公式了
                                              N  T                    
                        ∇E[R_{θ}] 近似= 1/N * Σ  Σ  R(τ)∇logP(at|st, θ) 
                                             i=1t=1                   
                  先用actor收集一大堆(s, a), 和每个trajectory 的 total Reward
                  代入梯度更新公式 把gradient 算出来
                    增加奖励大的状态下的采取的动作的概率
                  更新θ

                  就是收集数据 -> 更新模型 循环
                  
                  可以是classification 但lossfunction要乘负号和total reward

            如果發現不能微分就用 policy gradient 硬 train 一发  -- 环境和reward是黑盒子


             T            
             Σ γ^(t' - t)r
            t'=t          
            --------------
            Gt: obtained via interaction    从当前到回合结束获得的reward
                unstable
                
                互动process有随机性 total reward 也是随机的 
                可能有一个distribution, 但我们是做少量sample
            
                
                With sufficient samples, approximate the expectation of G.
            
                        -> G = 100
                        -> G = 3
                 s -> a -> G = 1
                        -> G = 2
                        -> G = -10

                 Can we estimate the expected value of G?   --  让training 更稳定
                        using a critic

                        • State value function Vπ(s)
                            • When using actor π, the cumulated reward expects to be obtained after visiting state s

                        • State-action value function Qπ(s, a)
                            • When using actor π, the cumulated reward expects to be obtained after taking a at state s
                            output:
                                the cumulated reward expects to be obtained after seeing observation (state) s and taking action a 
                                Qπ(s, a)
                                or
                                the cumulated reward expects to be obtained after seeing observation (state) s and taking action a(action空间很小的情况下，展开所有action，这样输入的时候就不需要action了)
                                Qπ(s, a = left) Qπ(s, a = right) Qπ(s, a = fire)

                                           ----
                                      s -> |  |
                                           |Qπ| -> Qπ(s, a)
                                      a -> |  |
                                           ----
                                
                                           ----
                                           |  | -> Qπ(s, a=left)
                                      s -> |Qπ| -> Qπ(s, a=right)       for discrete action only
                                           |  | -> Qπ(s, a=fire)
                                           ----

                        Estimated by TD or MC

                                  N  T   T                  
            ∇E[R_{θ}] 近似= 1/N * Σ  Σ ( Σ γ^(t' - t)r -b) ∇logP(at|st, θ)           b 可以取 E[R] 也可以是state-dependent
                                 i=1t=1 t'=t              
                                        --------------
                                        Gt: obtained via interaction    从当前到回合结束获得的reward
                                        Estimated by "critic" (later)

                                        E[Gt] = Qπθ(st, at)  有involve action

                                        b -- baseline Vπθ(st) 没有involve action


                                  N  T        
            ∇E[R_{θ}] 近似= 1/N * Σ  Σ (Qπ(st, at) - Vπ(st)) ∇logP(at|st, θ)
                                 i=1t=1     
                        Estimate two networks? We can only estimate one.

            Qπ(st, at) = E[rt + Vπ(st+1)]
                        rt random variable
                        把期望去掉  paper 里面是这样
                       = rt + Vπ(st+1)

                                  N  T        
            ∇E[R_{θ}] 近似= 1/N * Σ  Σ (rt + Vπ(st+1) - Vπ(st)) ∇logP(at|st, θ)
                                 i=1t=1     
                        Only estimate state value


            
    loop:
        π interacts with the environment

                |               1.蒙特卡洛                    
                v               2.Temporal-Difference (TD) 

        Learning Vπ(s,a)        

                |               
                v               

        Update actor from π → π'(Actor网络参数) based on Vπ(s)
            Actor网络做Gradient Ascent，推导看上面
                                  N  T        
            ∇E[R_{θ}] 近似= 1/N * Σ  Σ (rt + Vπ(st+1) - Vπ(st)) ∇logP(at|st, θ)
                                 i=1t=1     
                        Only estimate state value
                                        -----------------------
                                        Advantage function

        • Tips
            • The parameters of actor π(s) and critic Vπ(s) can be shared

                                                  NN
                                                |-----|
                                NN          ->  |Actor| -> action probability distribution
                          |--------------|      |-----|
                    s ->  |shared network| 
                          |--------------|  ->  |------|
                                                |Critic| -> Vπ(s)
                                                |------|
                                                   NN

                    shared network can be CNN : image pixel -> high level information
            
            • Use output entropy as regularization for π (s) 
                • Larger entropy is preferred → exploration


   Asynchronous Advantage Actor-Critic (A3C) 


                                           Global Network

                                        NN               NN                      
                                      |-----|         |------|                          
                                      |Actor|         |Critic|                                                             
                                      |-----|         |------|                          
                                                                                       
                                                NN                                          
                                          |--------------|                                           
                                          |shared network|                                  
                                          |--------------|                               
                                                 ^              
                                                 |              
                                              input(s)          
                                                                                                    


    
                            worker1                             worker2                 ...         workern
                      NN               NN                 NN               NN                                                                                       
                    |-----|         |------|            |-----|         |------|                                                                                    
                    |Actor|         |Critic|            |Actor|         |Critic|                                                                                    
                    |-----|         |------|            |-----|         |------|                                                                                    
                                                                                                                                                                    
                              NN                                  NN                                                                                                
                        |--------------|                    |--------------|                                                                                        
                        |shared network|                    |shared network|                                                                                        
                        |--------------|                    |--------------|                                                                                        
                               ^                                   ^                                                                                                
                               |                                   |                                                                                                
                            input(s)                            input(s)                                                                                            
                                                                                                                    
                                                                                                                    
                            Env1                                  Env2                  ....           Envn                                                                                                   
                                                                                                    
                                                                                                    
        1.worker_i Copy global parameters                                                                                                        
        2.worker_i Sampling some data
        3.worker_i compute gradients
        4.update global models  (other workers also update models)


Pathwise Derivative Policy Gradient
    可以看成Q-learning 解 Continuous action 的一种方法
    也可以看成一种特别的Actor-Critic方法
        原来的Actor-Critic方法：当前行为好还是不好
        Pathwise derivative policy gradient：告诉actor 采取什么样的方法是好的
        
    Action a is a continuous Vector
        
        a = arg max Q(s, a)
                 a
        如何解这个公式：用一个NN，s -> NN(Actor) -> a 来表示这个公式

        Actor as the solver of this optimization problem 

        

        π'(s) = arg max Qπ(s, a)    a is the output of an actor 
                     a 

                           s -> |---|
                  NN            |   |
                |-----|         |Qπ | -> Qπ(s, a)
          s ->  |Actor| -> a -> |   |
                |-----|         |---|

                -----------------------
                This is a large network

            Fixed Qπ 去调 Actor的参数

    loop:
        π interacts with the environment

                |               1.蒙特卡洛                    
                v               2.Temporal-Difference (TD) 

        Learning Qπ(s,a)

                |               "Better":Vπ'(st) >= Vπ(st) for all state s
                v               π'(s) = arg maxQπ(s, a)
                                             a
                                problem:Not suitable for continuous action a (solve it later) 因为要穷举所有的action
                                A solution : Pathwise Derivative Policy Gradient 增加一个网络(actor)产生连续的动作 (DDPG)

                                                   ----
                                              s -> |  |
                                    |-----|        |Qπ| -> Qπ(s, a)
                                s ->|Actor|-> a -> |  |
                                    | π   |        ----
                                    -------

        Find a new actor π' "better" than π    π'=π
        
        这里也可以用Reply buffer 和 exploration


        Typical Q-Learning Algorithm                                                                Pathwise Derivative Policy Gradient                                                  
                                                                                                                                                                                         
                                                         ^                                                                                           ^                            ^                       
            • Initialize Q-function Q, target Q-function Q = Q                                          • Initialize Q-function Q, target Q-function Q = Q, actor π, target actor π = π                       
            • In each episode                                                                           • In each episode                                                                
                • For each time step t                                                                      • For each time step t                                                       
                    • Given state st, take action a, based on Q (epsilon greedy)                            • Given state st, take action a, based on π  (epsilon greedy) -- first change                                         
                    • Obtain reward rt, and reach new state st+1                                            • Obtain reward rt, and reach new state st+1                                                          
                    • Store (st, at, rt, st+1) into butter                                                  • Store (st, at, rt, st+1) into butter                                                        
                    • Sample (si, ai, ri, si+1) from buffer (usually a batch)                               • Sample (si, ai, ri, si+1) from buffer (usually a batch)                                                         
                                          ^                                                                                   ^       ^                                                  
                    • Target y = ri + max Q(si+1, a)                                                        • Target y = ri + Q(si+1, π(si+1)) -- second change                                       
                    • Update the parameters of Q to make Q(si, ai) close to y (regression)                  • Update the parameters of π to maximize Q(si, π(si)) -- third change                                                         
                                          ^                                                                                       ^                                                     
                    • Every C steps reset Q = Q                                                             • Every C steps reset Q = Q                                                           
                                                                                                                                  ^                                                     
                                                                                                            • Every C steps reset π = π  -- fourth change                                               
Connection with GAN

    Method                          GANS        AC
    Freezing learning               yes         yes
    Label smoothing                 yes         no 
    Historical averaging            yes         no 
    Minibatch discrimination        yes         no
    Batch normalization             yes         yes
    Target networks                 n/a         yes
    Replay buffers                  no          yes
    Entropy regularization          no          yes
    Compatibility                   no          yes
