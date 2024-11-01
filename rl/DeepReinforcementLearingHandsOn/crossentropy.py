"""
cross-entropy one of the RL methods

    Despite the fact that it is much less famous than other tools in the RLpractitioner's toolbox, 
    such as deep Qnetwork (DQN) or Advantage Actor-Critic, this method has its own strengths.
    |----Simplicity
    |----Good convergence

    
All methods in RLcan be classified into various aspects
|----Model-free or model-based
|----Value-based or policy-based
|       Policy 
|           is usually represented by probability distribution over the available actions.
|       value based
|           the agent calculates the value of every possible action and 
|           chooses the action with the best value.         
|----On-policy or off-policy
        it will be enough to explain off-policy as the ability of the method to
        learn on old historical data (obtained by a previous version of the agent or
        recorded by human demonstration or just seen by the same agent several
        episodes ago).

The cross-entropy method falls into the model-free and policy-based category of methods.

                    |----------------------|                            The details of the output that this function produces may depend on
Observation s --->  |Trainable function(NN)| ---> Policy \pi(a|s)       a particular method or a family of methods,
                    |----------------------|                            as described in the previous section
                                                                        (such as value-based versus policy-based methods)
 
In practice, policy is usually represented as probability distribution over actions, 
which makes it very similar to a classification problem, 
with the amount of classes being equal to amount of actions we can carry out. 

This abstraction makes our agent very simple 
|----it needs to pass an observation from the environment to the network, 
|----get probability distribution over actions, 
|----and perform random sampling using probability distribution to get an action to carry out. 
|           This random sampling adds randomness to our agent, 
|           which is a good thing, 
|           as at the beginning of the training when our weights are random, 
|           the agent behaves randomly. 
|----After the agent gets an action to issue, 
|    it fires the action to the environment and 
|    obtains the next observation and reward for the last action. 
|----Then the loop continues. 
 
Steps Of The Method
|----1.Play N number of episodes using our current model and environment
|----2. Calculate the total reward for every episode and 
|       decide on a reward boundary. 
|       Usually, we use some percentile of all rewards, such as 50th or 70th.
|----3. Throw away all episodes with a reward below the boundary
|----4. Train on the remaining "elite" episodes using observations 
|       as the input and issued actions as the desired output.
|----5. Repeat from step 1 until we become satisfied with the result
"""

import torch.nn as nn

HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70

class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)

"""
The output from the network is a probability distribution over actions, so a straightforward way to proceed would be to include softmax nonlinearity after the last layer. 

However, in the preceding network we don't apply softmax to increase the numerical stability of the training process. 
Rather than calculating softmax (which uses exponentiation) and then calculating cross-entropy loss (which uses logarithm of probabilities), we'll use the PyTorch class, nn.CrossEntropyLoss, which combines both softmax and cross-entropy in a single, more numerically stable expression.
        CrossEntropyLoss requires raw, unnormalized values from the network (also called logits), 
        and the downside of this is that we need to remember to apply
        softmax every time we need to get probabilities from our network's output.
"""

from collections import namedtuple

Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])

import torch
import numpy as np

def iterate_batches(env, net, batch_size):
    """
    we pass the observation to the net, 
    sample the action to perform, ask the environment to process the action, 
    and remember the result of this processing.

    One very important fact to understand in this function logic is that 
    the training of our network and the generation of our episodes are performed at the same time.    They are not completely in parallel, but every time our loop accumulates enough episodes (16),    
    it passes control to this function caller, 
    which is supposed to train the network using the gradient descent. 
    So, when yield is returned, the network will have different, 
    slightly better (we hope) behavior.
    """
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    sm = nn.Softmax(dim=1)  #which will be used to convert the network's output to a probability distribution of actions
    while True:
        obs_v = torch.FloatTensor([obs])
        act_probs_v = sm(net(obs_v))
        act_probs = act_probs_v.data.numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, _ = env.step(action)
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=action))
        if is_done:
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs

def filter_batch(batch, percentile):
    """
    from the given batch of episodes and percentile value, 
    it calculates a boundary reward, which is used to filter elite episodes to train on

    we will filter off our episodes
    """
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []
    for example in batch:
        if example.reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, example.steps))
        train_act.extend(map(lambda step: step.action, example.steps))

    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean

import gym
from tensorboardX import SummaryWriter
import torch.optim as optim

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    # env = gym.wrappers.Monitor(env, directory="mon", force=True)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)
    writer = SummaryWriter(comment="-cartpole")

    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        """
        These scores are passed to the objective function,
        which calculates cross-entropy between the network output and the actions that the agent took. 
        The idea of this is to reinforce our network to carry out those "elite" actions 
        which have led to good rewards.
        """
        loss_v = objective(action_scores_v, acts_v) #very important!!!
        loss_v.backward()
        optimizer.step()
        print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (
            iter_no, loss_v.item(), reward_m, reward_b))
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)
        if reward_m > 199:
            print("Solved!")
            break
    writer.close()

"""
tips:After xvfbrun to provide a virtual X11 display
"""

"""
In CartPole
    every step of the environment gives us the reward 1.0, until the moment that the pole falls. 
    So, the longer our agent balanced the pole, the more reward it obtained. 
    Due to randomness in our agent's behavior, different episodes were of different lengths, 
    which gave us a pretty normal distribution of the episodes' rewards. 
    After choosing a reward boundary, we rejected less successful episodes and learned how to repeat better ones (by training on successful episodes' data).

In the FrozenLake environment 
    episodes and their reward look different. 
    We get the reward of 1.0 only when we reach the goal, 
    and this reward says nothing about how good each episode was.
    The distribution of rewards for our episodes are also problematic
        There are only two kinds of episodes possible, 
        with zero reward (failed) and one reward (successful), 
        and failed episodes will obviously dominate in the beginning of the training. 
        So, our percentile selection of "elite" episodes is totally wrong and 
        gives us bad examples to train on. 
        This is the reason for our training failure.

This example shows us the limitations of the cross-entropy methods
|----For training, our episodes have to be finite and, preferably, short
|----The total reward for the episodes should have enough variability to separate good episodes from bad ones
|----There is no intermediate indication about whether the agent has succeeded or failed

if you're curious about how FrozenLake can be solved using cross-entropy, here is a list of tweaks
|----Larger batches of played episodes
|----Discount factor applied to reward
|----Keeping "elite" episodes for a longer time
|----Decrease learning rate
|----Much longer training time
"""
