"""
we’ll consider an alternative way to handle Markov Decision Process (MDP) problems, which forms a full family of methods called Policy Gradients (PG).

We will start with a simple PG method called REINFORCE and will try to apply it to our CartPole environment, comparing this with the Deep Q-Networks (DQN) approach.

The central topic in Q-learning is the value of the state or action + state pair. 
    |---Value is defined as the discounted total reward that we can gather from this state or by issuing this particular action from the state. 
    |---If we know the value, our decision on every step becomes simple and obvious:
    |       we just act greedily in terms of value, and that guarantees us good total reward at the end of the episode.
    |---the values of states (in the case of the Value Iteration method) or state + action (in the case of Q-learning)
    |---To obtain these values, we’ve used the Bellman equation, which expresses the value on the current step via the values on the next step.

policy
    we defined the entity that tells us what to do in every state

In Q-learning methods 
    when values are dictating to us how to behave, they are actually defining our policy.
    Formally, this could be written as \pi(s) = \arg\max_a_{i} Q(s, a_{i})
        which means that the result of our policy at every state s is the action with the largest Q.

Why policy?
|---- policy is what we’re looking for when we’re solving a Reinforcement Learning (RL) problem.
|           Q-learning approach tried to answer the policy question indirectly via approximating the values of the states and trying to choose the best alternative, 
|           but if we’re not interested in values, why do extra work?
|---- due to the environments with lots of actions or, in the extreme, with a continuous action space. 
|           If our action is not a small discrete set, but has a scalar value attached to it, such as the steering wheel angle or the speed we want to run from the tiger, 
|           this optimization problem becomes hard, as Q is usually represented by a highly nonlinear neural network (NN), 
|           so finding the argument which maximizes the function’s values can be tricky.
|----policy is naturally represented as the probability of actions, which is a step in the same direction as the categorical DQN method
            in categorical DQN, our agent can benefit a lot from working with the distribution of Q-values, instead of expected mean values, 
            as our network can more precisely capture underlying probability distribution.

How do we represent the policy? 
|----In the case of Q-values, they were parametrized by the NN that returns values of actions as scalars. 
|----If we want our network to parametrize the actions, we have several options.
     |----The first and the simplest way could be just returning the identifier of the action (in the case of a discrete set of actions).
     |----A much more common solution, which is heavily used in classification tasks, is to return the probability distribution of our actions.
             In other words, for N mutually exclusive actions, we return N numbers representing the probability to take each action in the given state (which we pass as an input to the network). 

             such representation of actions as probability has the additional advantage of smooth representation: if we change our network weights a bit, the output of the network will also change.
                 In the case of discrete numbers output, even a small adjustment of the weights can lead to a jump to the different action.
                 However, if our output is probability distribution, a small change of weights will usually lead to a small change in output distribution, such as slightly increasing the probability of one action versus the others.
                 this is a very nice property to have, as gradient optimization methods are all about tweaking the parameters of a model a bit to improve the results. 

             In math notation, policy is usually represented as \pi(s), so we’ll use this notation as well.

Policy gradients
what we haven’t seen so far is how we’re going to change our network’s parameters to improve the policy
|----In the cross-entropy method, we solved a very similar problem: our network took observations as inputs and returned the probability distribution of the actions
|----To begin, we’ll get acquainted with the method called REINFORCE, which has only minor differences from cross-entropy

we need to look at some mathematical notation that we’ll use in this and the following chapters first.

We define PG as L = -Q(s, a)\log(\pi(a|s))
PG defines the direction in which we need to change our network’s parameters to improve the policy in terms of the accumulated total reward.
    The scale of the gradient is proportional to the value of the action taken, which is Q(s, a) in the formula above and the gradient itself is equal to the gradient of logprobability of the action taken
        Intuitively, this means that we’re trying to increase the probability of actions that have given us good total reward and decrease the probability of actions with bad final outcomes. 

    Expectation E in the formula just means that we take several steps that we’ve made in the environment and average the gradient.


From a practical point of view, PG methods could be implemented as performing optimization of this loss function: -Q(s, a)\log(\pi(a|s)). 
    The minus sign is important, as loss function is minimized during the Stochastic Gradient Descent (SGD), but we want to maximize our policy gradient.

The REINFORCE method

The formula of PG that we’ve just seen is used by most of the policy-based methods, but the details can vary.
    One very important point is how exactly gradient scales Q(s, a) are calculated. 
    |----In the cross-entropy method from Chapter 4, The Cross-Entropy Method, we played several episodes, calculated the total reward for each of them, and trained on transitions from episodes with a better-than-average reward. 
         This training procedure is the PG method with Q(s, a) = 1 for actions from good episodes (with large total reward) and Q(s, a) = 0 for actions from worse episodes.
            The cross-entropy method worked even with those simple assumptions, but the obvious improvement will be to use Q(s, a) for training instead of just 0 and 1.

So why should it help? 
|----The answer is a more fine-grained separation of episodes.
|           For example, transitions of the episode with the total reward = 10 should contribute to the gradient more than transitions from the episode with the reward = 1
|----Is to increase probabilities of good actions in the beginning of the episode and decrease the actions closer to the end of episode.


REINFORCE Method
|----1.Initialize the network with random weights
|----2.Play N full episodes, saving their (s, a, r, s’) transitions
|----3.For every step t of every episode k, calculate the discounted total reward for subsequent steps Q_{k, t} = \sum_{i=0} \gamma^i r_{i}
|----4.Calculate the loss function for all transitions L = - sum_{k, t} Q_{k, t} \log(\pt(s_{k, t}, a{k, t}))
|----5.Perform SGD update of weights minimizing the loss
|----6. Repeat from step 2 until converged

The algorithm above is different from Q-learning in several important aspects:
|----No explicit exploration is needed. 
|           In Q-learning, we used an epsilongreedy strategy to explore the environment and prevent our agent from getting stuck with non-optimal policy. 
|           Now, with probabilities returned by the network, the exploration is performed automatically.
|----No replay buffer is used. 
|           PG methods belong to the on-policy methods class, which means that we can’t train on data obtained from the old policy.
|               The good part is that such methods usually converge faster. 
|               The bad side is they usually require much more interaction with the environment than off-policy methods such as DQN.
|----No target network is needed.
            Here we use Q-values, but they’re obtained from our experience in the environment.
            In DQN, we used the target network to break the correlation in Q-values approximation, but we’re not approximating it anymore
            Later, we’ll see that the target network trick still can be useful in PG methods.
"""

#let’s check the implementation of the REINFORCE method on the familiar CartPole environment.
import gym
import ptan
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

"""
define hyperparameters (imports are omitted).

The EPISODES_TO_TRAIN value specifies how many complete episodes we’ll use for training.
"""
GAMMA = 0.99
LEARNING_RATE = 0.01
EPISODES_TO_TRAIN = 4


class PGN(nn.Module):
    """
    Note that despite the fact our network returns probabilities, we’re not applying softmax nonlinearity to the output. 
    The reason behind this is that we’ll use the PyTorch log_softmax function to calculate the logarithm of the softmax output at once. 
    This way of calculation is much more numerically stable, 
    but we need to remember that output from the network is not probability, but raw scores (usually called logits).
    """
    def __init__(self, input_size, n_actions):
        super(PGN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)


def calc_qvals(rewards):
    """
    It accepts a list of rewards for the whole episode and needs to calculate the discounted total reward for every step. 

    To do this efficiently, we calculate the reward from the end of the local reward list.
        Indeed, the last step of the episode will have the total reward equal to its local reward. 
        The step before the last will have the total reward of (if t is an index of the last step). 

    Our sum_r variable contains the total reward for the previous steps, so to get the total reward for the previous step, we need to multiply sum_r by gamma and sum the local reward.
    """
    res = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= GAMMA
        sum_r += r
        res.append(sum_r)
    return list(reversed(res))


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    writer = SummaryWriter(comment="-cartpole-reinforce")

    net = PGN(env.observation_space.shape[0], env.action_space.n)
    print(net)

    """
    Here we are using ptan.agent.PolicyAgent, which needs to make a decision about actions for every observation. 
    As our network now returns policy in the form of probabilities of the actions? 
        to select the action to take, we need to obtain the probabilities from the network and 
        then perform random sampling from this probability distribution.

    When we worked with DQN, the output of the network was Q-values, 
        so if some action had the value of 0.4 and another action 0.5, the second action was preferred 100% of the time. 

    In the case of probability distribution, 
        if the first action has a probability of 0.4 and the second 0.5, our agent should take the first action with 40% chance and the second with 50% chance.

    This difference is important to understand, but the change in the implementation is not large. 
        |----Our PolicyAgent internally calls the NumPy random.choice function with probabilities from the network.
        |----The argument apply_softmax argument instructs it to convert the network output to probabilities by calling softmax first. 
        |----The third argument preprocessor is a way to get around the fact that the CartPole environment in Gym returns observation as float64 instead of float32 required by PyTorch.
    """
    agent = ptan.agent.PolicyAgent(net, preprocessor=ptan.agent.float32_preprocessor,
                                   apply_softmax=True)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    total_rewards = []
    step_idx = 0
    done_episodes = 0

    batch_episodes = 0
    batch_states, batch_actions, batch_qvals = [], [], []
    """
    contains local rewards for the currently-played episode. 
    As this episode reaches the end, we calculate the discounted total rewards from local rewards using the calc_qvals function and append them to the batch_qvals list.
    """
    cur_rewards = []    

    for step_idx, exp in enumerate(exp_source):
        """
        Every experience that we get from the experience source contains state, action, local reward, and the next state. 
        If the end of the episode has been reached, the next state will be None. 
        For nonterminal experience entries, we just save state, action, and local reward in our lists. 
        At the end of the episode, we convert the local rewards into Q-values and increment the episodes counter.
        """
        batch_states.append(exp.state)
        batch_actions.append(int(exp.action))
        cur_rewards.append(exp.reward)

        if exp.last_state is None:
            batch_qvals.extend(calc_qvals(cur_rewards))
            cur_rewards.clear()
            batch_episodes += 1

        # handle new rewards
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            """
            This part of the training loop is performed at the end of the episode and 
            is responsible for the reporting of current progress and 
            writing metrics to the TensorBoard.
            """
            done_episodes += 1
            reward = new_rewards[0]
            total_rewards.append(reward)
            mean_rewards = float(np.mean(total_rewards[-100:]))
            print("%d: reward: %6.2f, mean_100: %6.2f, episodes: %d" % (
                step_idx, reward, mean_rewards, done_episodes))
            writer.add_scalar("reward", reward, step_idx)
            writer.add_scalar("reward_100", mean_rewards, step_idx)
            writer.add_scalar("episodes", done_episodes, step_idx)
            if mean_rewards > 195:
                print("Solved in %d steps and %d episodes!" % (step_idx, done_episodes))
                break

        if batch_episodes < EPISODES_TO_TRAIN:
            continue

        """
        As a first step, we need to convert states, actions, and Q-values into appropriate PyTorch form.
        """
        optimizer.zero_grad()
        states_v = torch.FloatTensor(batch_states)
        batch_actions_t = torch.LongTensor(batch_actions)
        batch_qvals_v = torch.FloatTensor(batch_qvals)

        logits_v = net(states_v) #we ask our network to calculate states into logits
        log_prob_v = F.log_softmax(logits_v, dim=1) # calculate logarithm + softmax of them
        """
        On the third line, we select log probabilities from the actions taken and scale them with Q-values.
        """
        log_prob_actions_v = batch_qvals_v * log_prob_v[range(len(batch_states)), batch_actions_t]
        """
        Once again, this minus sign is very important, as our PG needs to be maximized to improve the policy. 
        As the optimizer in PyTorch does minimization in respect to the loss function, we need to negate the PG.
        """
        loss_v = -log_prob_actions_v.mean()

        loss_v.backward() #perform backpropagation to gather gradients in our variables
        optimizer.step() # ask the optimizer to perform a SGD update.

        batch_episodes = 0
        batch_states.clear()
        batch_actions.clear()
        batch_qvals.clear()

    writer.close()

"""
As you can see, REINFORCE converges faster and requires less training steps and episodes to solve the CartPole environment.
    The Cross-Entropy Method, the cross-entropy method required about 40 batches of 16 episodes each to solve the CartPole environment, which is 640 episodes in total. 
    The REINFORCE method is able to do the same in less than 300 episodes, which is a nice improvement.

Policy-based versus value-based methods
|----Policy methods are directly optimizing what we care about: our behavior.
|       The value methods such as DQN are doing the same indirectly, learning the value first and 
|       providing to us policy based on this value.
|----Policy methods are on-policy and require fresh samples from the environment. 
|       The value methods can benefit from old data, obtained from the old policy, human demonstration, and other sources.
|----Policy methods are usually less sample-efficient, which means they require more interaction with the environment. The value methods can benefit from the large replay buffers. 
        However, sample efficiency doesn’t mean that value methods are more computationally efficient and very often it’s the opposite. 
            In the above example, during the training, we need to access our NN only once, to get the probabilities of actions. 
            In DQN, we need to process two batch of states: one for the current state and another for the next state in the Bellman update.

As you can see, there is no strong preference of one family versus another. 
In some situations, policy methods will be the more natural choice, like in continuous control problems or cases when access to the environment is cheap and fast. 
However, there are lots of situations when value methods will shine,

In the next section, we’ll talk about, REINFORCE method’s limitations, ways to improve it, and how to apply the PG method to our favorite Pong game.

REINFORCE issues

In the previous section, we discussed the REINFORCE method, which is a natural extension of cross-entropy from Chapter 4, The Cross-Entropy Method. 
Unfortunately, both REINFORCE and cross-entropy still suffer from several problems, which make both of them limited to simple environments. 
|----Full episodes are required
|       First of all, we still need to wait for the full episode to complete before we can start training. 
|       Even worse, both REINFORCE and cross-entropy behave better with more episodes used for training (just from the fact that more episodes mean more training data, which means more accurate PG). 
|       It’s equally bad from the training perspective, as our training batch becomes very large and from sample efficiency, when we need to communicate with the environment a lot just to perform a single training step.
|
|       The origin of the complete episodes requirement is to get as accurate a Q estimation as possible. 
|           When we talked about DQN, we saw that in practice, it’s fine to replace the exact value for a discounted reward with our estimation using the one-step Bellman equation . 
|           Q(s, a) = \sum_{s in S} p_{a, s}(rs + \gamma vs) ==> V_s = max_{a}Q(s, a_{i})
|                   = r_{s, a} + \gamma max_a'Q(s', a'_{i})
|                   = r_{s, a} + \gamma V(s')
|
|           To estimate V(s), we’ve used our own Q-estimation, but in the case of PG, we don’t have V(s) or Q(s, a) anymore.
|
|       To overcome this, two approaches exist. 
|           |----On the one hand, we can ask our network to estimate V(s) and use this estimation to obtain Q. 
|           |        This approach will be discussed in the next chapter and is called Actor-Critic method, which is the most popular method from the PG family.
|           |        Actor-Critic 方法引入了一个 Critic 网络，用来估计状态值V(s)。通过这个值函数，我们可以近似地计算 Q 值，从而避免等待完整回合。Actor-Critic 是 PG 中最流行的改进方法之一，将在后续章节详细讨论。
|           |
|           |----On the other hand, we can do the Bellman equation unrolling N steps ahead, which will effectively exploit the fact that value contribution is decreasing when gamma is less than 1.
|
|----High gradients variance
|       In the PG formula \nabla J \approx \mathbb{E} \left[ Q(s, a) \nabla \log \pi(a|s) \right] , we have a gradient proportional to the discounted reward from the given state. 
|       However, the range of this reward is heavily environment-dependent.
|           For example, in the CartPole environment we’re getting the reward of 1 for every timestamp we’re holding the pole vertically. 
|           If we can do this for five steps, we’ll get total (undiscounted) reward of 5. If our agent is smart and can hold the pole for, say, 100 steps, the total reward will be 100. 
|
|           The difference in value between those two scenarios is 20 times, which means that the scale between gradients of unsuccessful samples will be 20 times lower than that for more successful ones. 
|           Such a large difference can seriously affect our training dynamics, as one lucky episode will dominate in the final gradient.
|
|       In mathematical terms, our PGs have high variance and we need to do something about this in complex environments, otherwise, the training process can become unstable.
|
|       The usual approach to handle this is subtracting a value called baseline from the Q. The possible choices of the baseline are as follows:
|       |----1. Some constant value, which normally is the mean of the discounted rewards
|       |----2. The moving average of the discounted rewards
|       |----3. Value of the state V(s)
|
|----Exploration
|       Even with the policy represented as probability distribution, there is a high chance that the agent will converge to some locally-optimal policy and stop exploring the environment.
|            In DQN, we solved this using epsilon-greedy action selection:
|                with probability epsilon, the agent took some random action instead of the action dictated by the current policy
|            We can use the same approach, of course, but PG allows us to follow a better path, called the entropy bonus.
|               In the information theory, the entropy is a measure of uncertainty in some system. 
|                   Being applied to agent policy, entropy shows how much the agent is uncertain about which action to take.
|                    In math notation, entropy of the policy is defined as: H(\pi) = - sum \pi(a|s) \log \pi(a|s)
|                    The value of entropy is always greater than zero and has a single maximum when the policy is uniform.
|                       In other words, all actions have the same probability.
|                   Entropy becomes minimal when our policy has 1 for some action and 0 for all others, which means that the agent is absolutely sure what to do.
|
|               To prevent our agent from being stuck in the local minimum, we are subtracting the entropy from the loss function, punishing the agent for being too certain about the action to take.
|----Correlation between samples
        As we discussed in Chapter 6, Deep Q-Networks, training samples in one single episode are usually heavily correlated, which is bad for SGD training.
            In the case of DQN, we solved this issue by having a large replay buffer with 100k-1M observations that we sampled our training batch from.
            This solution is not applicable to the PG family anymore, due to the fact that those methods belong to the on-policy class.
                The implication is simple: using old samples generated by the old policy, we’ll get PG for that old policy, not for our current one.
                The obvious, but, unfortunately wrong solution would be to reduce the replay buffer size. 
            On-policy 方法：要求在训练中使用的数据必须由当前策略生成。
                例如，策略梯度方法（如 REINFORCE 和 A3C）是 on-policy 方法，因为它们只能使用当前策略生成的样本来更新策略。这意味着每次策略更新后，之前的数据可能不再适用，必须生成新的数据来保持训练的有效性。
            Off-policy 方法：允许使用不同策略生成的数据进行训练。
                例如 DQN 和 Q-learning 属于 off-policy 方法，可以使用旧的经验数据来更新策略。这样的优势是可以利用经验回放（replay buffer），从而提高样本利用率，减少数据生成的需求。

            It might work in some simple cases, but in general, we need fresh training data generated by our current policy. 
            To solve this, parallel environments are normally used. 
            The idea is simple: instead of communicating with one environment, we use several and use their transitions as training data.
"""

"""
PG on CartPole

Nowadays, almost nobody uses the vanilla PG method, as the much more stable Actor-Critic method exists, which will be the topic of the two following chapters. 

However, I still want to show the PG implementation, as it establishes very important concepts and metrics to check for the PG method’s performance.
"""

#!/usr/bin/env python3
import gym
import ptan
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#two new hyperparameters
GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01 # Entropy beta value is the scale of the entropy bonus.
BATCH_SIZE = 8

REWARD_STEPS = 10   # The REWARD_STEPS value specifies how many steps ahead the Bellman equation is unrolled to estimate the discounted total reward of every transition


class PGN(nn.Module):
    def __init__(self, input_size, n_actions):
        super(PGN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    writer = SummaryWriter(comment="-cartpole-pg")

    net = PGN(env.observation_space.shape[0], env.action_space.n)
    print(net)

    agent = ptan.agent.PolicyAgent(net, preprocessor=ptan.agent.float32_preprocessor,
                                   apply_softmax=True)
    #The preparation code is also the same as before, except the experience source is asked to unroll the Bellman equation for 10 steps:
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    total_rewards = []
    step_rewards = []
    step_idx = 0
    done_episodes = 0
    reward_sum = 0.0

    batch_states, batch_actions, batch_scales = [], [], []

    for step_idx, exp in enumerate(exp_source):
        reward_sum += exp.reward
        """
        In the training loop, we maintain the sum of the discounted reward for every transition and use it to calculate the baseline for policy scale.
        """
        baseline = reward_sum / (step_idx + 1)
        writer.add_scalar("baseline", baseline, step_idx)
        batch_states.append(exp.state)
        batch_actions.append(int(exp.action))
        batch_scales.append(exp.reward - baseline)

        # handle new rewards
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            done_episodes += 1
            reward = new_rewards[0]
            total_rewards.append(reward)
            mean_rewards = float(np.mean(total_rewards[-100:]))
            print("%d: reward: %6.2f, mean_100: %6.2f, episodes: %d" % (
                step_idx, reward, mean_rewards, done_episodes))
            writer.add_scalar("reward", reward, step_idx)
            writer.add_scalar("reward_100", mean_rewards, step_idx)
            writer.add_scalar("episodes", done_episodes, step_idx)
            if mean_rewards > 195:
                print("Solved in %d steps and %d episodes!" % (step_idx, done_episodes))
                break

        if len(batch_states) < BATCH_SIZE:
            continue

        states_v = torch.FloatTensor(batch_states)
        batch_actions_t = torch.LongTensor(batch_actions)
        batch_scale_v = torch.FloatTensor(batch_scales)

        """
        In the loss calculation, we use the same code as before to calculate the policy loss (which is the negated PG).
        """
        optimizer.zero_grad()
        logits_v = net(states_v)
        log_prob_v = F.log_softmax(logits_v, dim=1)
        log_prob_actions_v = batch_scale_v * log_prob_v[range(BATCH_SIZE), batch_actions_t]
        loss_policy_v = -log_prob_actions_v.mean()

        prob_v = F.softmax(logits_v, dim=1)
        entropy_v = -(prob_v * log_prob_v).sum(dim=1).mean()
        """
        然后我们通过计算批次的熵值，并将其从损失中减去，来为损失添加熵奖励。
        """
        entropy_loss_v = -ENTROPY_BETA * entropy_v
        loss_v = loss_policy_v + entropy_loss_v

        loss_v.backward()
        optimizer.step()

        # calc KL-div
        """
        KL-divergence is an information theory measurement of how one probability distribution diverges from another expected probability distribution.
        In our example, it is being used to compare the policy returned by the model before and after the optimization step.
        High spikes in KLare usually a bad sign, showing that our policy was pushed too far from the previous policy, 
            which is a bad idea most of the time (as our NN is a very nonlinear function in a high-dimension space, so large changes in the model weight could have a very strong influence on policy). 
        """
        new_logits_v = net(states_v)
        new_prob_v = F.softmax(new_logits_v, dim=1)
        kl_div_v = -((new_prob_v / prob_v).log() * prob_v).sum(dim=1).mean()
        writer.add_scalar("kl", kl_div_v.item(), step_idx)

        """
        Finally, we calculate the statistics about the gradients on this training step. 
            It’s usually good practice to show the graph of maximum and L2-norm of gradients to get an idea about the training dynamics.
        """
        grad_max = 0.0
        grad_means = 0.0
        grad_count = 0
        for p in net.parameters():
            grad_max = max(grad_max, p.grad.abs().max().item())
            grad_means += (p.grad ** 2).mean().sqrt().item()
            grad_count += 1

        #At the end of the training loop, we dump all values we’d like to monitor, to the TensorBoard.
        writer.add_scalar("baseline", baseline, step_idx)
        writer.add_scalar("entropy", entropy_v.item(), step_idx)
        writer.add_scalar("batch_scales", np.mean(batch_scales), step_idx)
        writer.add_scalar("loss_entropy", entropy_loss_v.item(), step_idx)
        writer.add_scalar("loss_policy", loss_policy_v.item(), step_idx)
        writer.add_scalar("loss_total", loss_v.item(), step_idx)
        writer.add_scalar("grad_l2", grad_means / grad_count, step_idx)
        writer.add_scalar("grad_max", grad_max, step_idx)

        batch_states.clear()
        batch_actions.clear()
        batch_scales.clear()

    writer.close()

"""
The fact that the entropy is decreasing during the training shows that our policy is moving from the uniform distribution to more deterministic actions.

Our gradients look healthy during the whole training: they are not too large and not too small, without huge spikes.

PG on Pong

As covered in the previous section, the vanilla PG method works well on a simple CartPole environment, but surprisingly badly on more complicated environments.
    Even in the relatively simple Atari game Pong, our DQN was able to completely solve it in 1M frames and showed positive reward dynamics in just 100k frames, whereas PG failed to converge.

    This doesn’t mean that the PGs are bad, because, as we’ll see in the next chapter, just one tweak of the network architecture to get the better baseline in the gradients will turn PG into one of the best methods (Asynchronous Advantage Actor-Critic (A3C) method).

The three main differences from the previous example’s code are as follows:
|----The baseline is estimated with a moving average for 1M past transitions, instead of all examples
|----Several concurrent environments are used
|----Gradients are clipped to improve training stability
"""
#!/usr/bin/env python3
import gym
import ptan
import numpy as np
import argparse
import collections
from tensorboardX import SummaryWriter

import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import torch.optim as optim

from lib import common

GAMMA = 0.99
LEARNING_RATE = 0.0001
ENTROPY_BETA = 0.01
BATCH_SIZE = 128

REWARD_STEPS = 10
BASELINE_STEPS = 1000000
GRAD_L2_CLIP = 0.1

ENV_COUNT = 32


def make_env():
    return ptan.common.wrappers.wrap_dqn(gym.make("PongNoFrameskip-v4"))


#To make moving average calculations faster, a deque-backed buffer was created.
class MeanBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.deque = collections.deque(maxlen=capacity)
        self.sum = 0.0

    def add(self, val):
        if len(self.deque) == self.capacity:
            self.sum -= self.deque[0]
        self.deque.append(val)
        self.sum += val

    def mean(self):
        if not self.deque:
            return 0.0
        return self.sum / len(self.deque)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("-n", '--name', required=True, help="Name of the run")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    envs = [make_env() for _ in range(ENV_COUNT)]
    writer = SummaryWriter(comment="-pong-pg-" + args.name)

    net = common.AtariPGN(envs[0].observation_space.shape, envs[0].action_space.n).to(device)
    print(net)

    agent = ptan.agent.PolicyAgent(net, apply_softmax=True, device=device)
    """
    The second difference in this example is working with multiple environments and this functionality is supported by the ptan library. 
    The only action we have to take is to pass the array of Env objects to the ExperienceSource class. 
        All the rest is done automatically. 
    In the case of several environments, the experience source asks them for transitions in round-robin, providing us with less-correlated training samples. 
    """
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)

    total_rewards = []
    step_idx = 0
    done_episodes = 0
    train_step_idx = 0
    baseline_buf = MeanBuffer(BASELINE_STEPS)

    batch_states, batch_actions, batch_scales = [], [], []
    m_baseline, m_batch_scales, m_loss_entropy, m_loss_policy, m_loss_total = [], [], [], [], []
    m_grad_max, m_grad_mean = [], []
    sum_reward = 0.0

    with common.RewardTracker(writer, stop_reward=18) as tracker:
        for step_idx, exp in enumerate(exp_source):
            baseline_buf.add(exp.reward)
            baseline = baseline_buf.mean()
            batch_states.append(np.array(exp.state, copy=False))
            batch_actions.append(int(exp.action))
            batch_scales.append(exp.reward - baseline)

            # handle new rewards
            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if tracker.reward(new_rewards[0], step_idx):
                    break

            if len(batch_states) < BATCH_SIZE:
                continue

            train_step_idx += 1
            states_v = torch.FloatTensor(batch_states).to(device)
            batch_actions_t = torch.LongTensor(batch_actions).to(device)

            scale_std = np.std(batch_scales)
            batch_scale_v = torch.FloatTensor(batch_scales).to(device)

            optimizer.zero_grad()
            logits_v = net(states_v)
            log_prob_v = F.log_softmax(logits_v, dim=1)
            log_prob_actions_v = batch_scale_v * log_prob_v[range(BATCH_SIZE), batch_actions_t]
            loss_policy_v = -log_prob_actions_v.mean()

            prob_v = F.softmax(logits_v, dim=1)
            entropy_v = -(prob_v * log_prob_v).sum(dim=1).mean()
            entropy_loss_v = -ENTROPY_BETA * entropy_v
            loss_v = loss_policy_v + entropy_loss_v
            loss_v.backward()
            """
            The last difference from the CartPole example is gradient clipping, which is performed using the PyTorch clip_grad_norm function from the torch.nn.utils package.
            """
            nn_utils.clip_grad_norm_(net.parameters(), GRAD_L2_CLIP)
            optimizer.step()

            # calc KL-div
            new_logits_v = net(states_v)
            new_prob_v = F.softmax(new_logits_v, dim=1)
            kl_div_v = -((new_prob_v / prob_v).log() * prob_v).sum(dim=1).mean()
            writer.add_scalar("kl", kl_div_v.item(), step_idx)

            grad_max = 0.0
            grad_means = 0.0
            grad_count = 0
            for p in net.parameters():
                grad_max = max(grad_max, p.grad.abs().max().item())
                grad_means += (p.grad ** 2).mean().sqrt().item()
                grad_count += 1

            writer.add_scalar("baseline", baseline, step_idx)
            writer.add_scalar("entropy", entropy_v.item(), step_idx)
            writer.add_scalar("batch_scales", np.mean(batch_scales), step_idx)
            writer.add_scalar("batch_scales_std", scale_std, step_idx)
            writer.add_scalar("loss_entropy", entropy_loss_v.item(), step_idx)
            writer.add_scalar("loss_policy", loss_policy_v.item(), step_idx)
            writer.add_scalar("loss_total", loss_v.item(), step_idx)
            writer.add_scalar("grad_l2", grad_means / grad_count, step_idx)
            writer.add_scalar("grad_max", grad_max, step_idx)

            batch_states.clear()
            batch_actions.clear()
            batch_scales.clear()

    writer.close()

"""
In this chapter, we saw an alternative way of solving RLproblems: 
    PG, which is different in many ways from the familiar DQN method. 
    We explored the basic method called REINFORCE, which is a generalization of our first method in RL-domain cross entropy. 
    This method is simple, but, being applied to the Pong environment, didn’t show good results.

In the next chapter, we’ll consider ways to improve the stability of PG by combining both families of value-based and policy-based methods.
"""
