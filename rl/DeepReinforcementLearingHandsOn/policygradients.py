"""
policy
    we defined the entity that tells us what to do in every state

In Q-learning methods 
    when values are dictating to us how to behave, they are actually defining our policy.
    Formally, this could be written as \pi(s) = \arg\max_a Q(s, a)
, which means that the result of our policy at every state s is the action with the largest Q.

Why policy?
|---- policy is what we’re looking for when we’re solving a Reinforcement Learning (RL) problem.
|           Q-learning approach tried to answer the policy question 
|           indirectly via approximating the values of the states 
|           and trying to choose the best alternative, 
|           but if we’re not interested in values, why do extra work?
|---- due to the environments with lots of actions or, in the extreme, with a continuous action space. 
|           If our action is not a small discrete set, 
|           but has a scalar value attached to it, 
|           such as the steering wheel angle or the speed we want 
|           to run from the tiger, this optimization problem becomes
|           hard, as Q is usually represented by a highly
|           nonlinear neural network (NN), so finding the argument 
|           which maximizes the function’s values can be tricky.
|----policy is naturally represented as the probability of actions, which is a step in the same direction as the categorical DQN method
|           in categorical DQN, our agent can benefit a lot 
|           from working with the distribution of Q-values, 
|           instead of expected mean values, 
|           as our network can more precisely capture underlying 
|           probability distribution.

How do we represent the policy? 
|----In the case of Q-values, they were parametrized by the NN that returns values of actions as scalars. 
|           If we want our network to parametrize the actions, we have several options.
|           |----The first and the simplest way could be just returning the identifier of the action (in the case of a discrete set of actions).
|           |----A much more common solution, which is heavily used in classification tasks, is to return the probability distribution of our actions.
                    such representation of actions as probability has the 
                    additional advantage of smooth representation: 
                    if we change our network weights a bit, 
                    the output of the network will also change.
                    In the case of discrete numbers output, 
                    even a small adjustment of the weights can lead to a jump to the different action.
                    However, if our output is probability distribution, 
                    a small change of weights will usually lead to a small change in output distribution, 
                    such as slightly increasing the probability of one action versus the others.
                    this is a very nice property to have, 
                    as gradient optimization methods are all about 
                    tweaking the parameters of a model a bit to improve the results. 
                    In math notation, policy is usually represented as \pi(s), so we’ll use this notation as well.

Policy gradients
what we haven’t seen so far is how we’re going to change our network’s parameters to improve the policy
In the cross-entropy method, we solved a very similar problem: our network took observations as inputs and returned the probability distribution of the actions

To begin, we’ll get acquainted with the method called REINFORCE, which has only minor differences from cross-entropy

We define PG as \gamma = -Q(s, a)\log(\pi(a|s))
PG defines the direction in which we need to change our network’s parameters to improve the policy in terms of the accumulated total reward.
    The scale of the gradient is proportional to the value of the action taken, 
    which is Q(s, a) in the formula above and the gradient itself is equal to the gradient of logprobability of the action taken

    Intuitively, this means that we’re trying to increase the probability of actions that have given us good total reward and
    decrease the probability of actions with bad final outcomes. 
    Expectation E in the formula just means that we take several steps that we’ve made in the environment and average the gradient.


From a practical point of view, PG methods could be implemented as performing optimization of this loss function: -Q(s, a)\log(\pi(a|s)). 
The minus sign is important, as loss function is minimized during the Stochastic Gradient Descent (SGD), but we want to maximize our policy gradient.

The formula of PG that we’ve just seen is used by most of the policy-based methods, but the details can vary.

One very important point is how exactly gradient scales Q(s, a) are calculated. 
|----In the cross-entropy method from Chapter 4, The Cross-Entropy Method, we played several episodes, calculated the total reward for each of them, and trained on transitions from episodes with a better-than-average reward. 

The cross-entropy method worked even with those simple assumptions, but the obvious improvement will be to use Q(s, a) for training instead of just 0 and 1.

So why should it help? 
|----The answer is a more fine-grained separation of episodes.
|           For example, transitions of the episode with the total reward = 10
|           should contribute to the gradient more than transitions from the episode with the reward = 1
|----Is to increase probabilities of good actions in the beginning of the episode and decrease the actions closer to the end of episode.


REINFORCE Method
|----1.Initialize the network with random weights
|----2.Play N full episodes, saving their (s, a, r, s’) transitions
|----3.For every step t of every episode k, calculate the discounted total reward for subsequent steps Q_{k, t} = \sum_{i=0} \gamma^i r_{i}
|----4.Calculate the loss function for all transitions \gamma = - sum_{k, t} Q_{k, t} \log(\pt(s_{k, t}, a{k, t}))
|----5.Perform SGD update of weights minimizing the loss
|----6. Repeat from step 2 until converged

The algorithm above is different from Q-learning in several important aspects:
|----No explicit exploration is needed. 
|           In Q-learning, we used an epsilongreedy strategy to explore the environment and 
|           prevent our agent from getting stuck with non-optimal policy. 
|----No replay buffer is used. 
|           PG methods belong to the on-policy methods class, 
|           which means that we can’t train on data obtained from the old policy.
|----No target network is needed.
|           In DQN, we used the target network to break the correlation in Q-values approximation, 
|           but we’re not approximating it anymore
"""
