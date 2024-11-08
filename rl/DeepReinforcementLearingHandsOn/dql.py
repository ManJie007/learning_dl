"""
1.In particular, we'll look at the application of Q-learning to so-called "grid world" environments, which is called tabular Q-learning, 
2.and then we'll discuss Q-learning in conjunction with neural networks.

Real-life value iteration

The Value iteration method on every step does a loop on all states, and for every state, it performs an update of its value with a Bellman approximation. 
The variation of the same method for Q-values (values for actions) is almost the same, but we approximate and store values for every state and action.

So, what's wrong with this process?
|----The first obvious problem is the count of environment states and our ability to iterate over them.
|       In the Value iteration, we assume that we know all states in our environment in advance, 
|       can iterate over them and can store value approximation associated with the state.
|----Another problem with the value iteration approach is that it limits us to discrete action spaces.
        Indeed, both Q(s, a) and V(s) approximations assume that 
        our actions are a mutually exclusive discrete set, 
        which is not true for continuous control problems 
        where actions can represent continuous variables

Tabular Q-learning
First of all, do we really need to iterate over every state in the state space?
    If some state in the state space is not shown to us by the environment, 
    why should we care about its value? 
        We can use states obtained from the environment to update values of states, 
        which can save us lots of work.
    
    steps:
    |----1.Start with an empty table, mapping states to values of actions.
    |----2.By interacting with the environment, obtain the tuple s, a, r, s′ (state, action, reward, and the new state). 
    |       In this step, we need to decide which action to take, 
    |       and there is no single proper way to make this decision. 
    |       We discussed this problem as exploration versus exploitation and 
    |       will talk a lot about this.
    |----3.Update the Q(s, a) value using the Bellman approximation:
    |        Q(s, a) = \sum_{s in S} p_{a, s}(rs + \gamma vs) ==> V_s = max_{a}Q(s, a_{i})
    |                = r_{s, a} + \gamma max_a'Q(s', a'_{i})
    |----4.Repeat from step 2.

As in Value iteration, the end condition could be some threshold of the update or 
we can perform test episodes to estimate the expected reward from the policy.

Another thing to note here is how to update the Q-values. 
|----As we take samples from the environment, 
|    it's generally a bad idea to just assign new values on top of existing values, 
|    as training can become unstable. 
|
|----What is usually done in practice is to update the Q(s, a) with approximations using a "blending" technique, 
     which is just averaging between old and new values of Q using learning rate \alpha with a value from 0 to 1:
        Q(s, a) = (1 - \alpha) Q(s, a) + \alpha (r + \gamma max_a'Q(s', a'_{i}))

This allows values of Q to converge smoothly, even if our environment is noisy.
The final version of the algorithm is here:
|----1.Start with an empty table for Q(s, a).
|----2.Obtain (s, a, r, s′) from the environment.
|----3.Make a Bellman update: Q(s, a) = (1 - \alpha) Q(s, a) + \alpha (r + \gamma max_a'Q(s', a'_{i}))
|----4.Check convergence conditions. If not met, repeat from step 2.
"""
from os import stat
import gym
import collections
from gym.core import RewardWrapper
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v0"
GAMMA = 0.9
ALPHA = 0.2
TEST_EPISODES = 20

class Agent:
    def __init__(self) -> None:
        """
            The initialization of our Agent class is simpler now, 
            as we don't need to track the history of rewards and transition counters, 
            just our value table. 
            This will make our memory footprint smaller, 
            which is not a big issue for FrozenLake, but can be critical for larger environments.
        """
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.values = collections.defaultdict(float)

    def sample_env(self):
        action = self.env.action_space.sample()
        old_state = self.state
        new_state, reward, is_done, _ = self.env.step(action)
        self.state = self.env.reset() if is_done else new_state
        return (old_state, action, reward, new_state)

    def best_value_and_action(self, state):
        """
        This method will be used two times: 
        first, in the test method that plays one episode using our current values table (to evaluate our policy quality), and 
        the second, in the method that performs the value update to get the value of the next state.
        """
        best_value, best_action = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values.get((state, action))
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action, best_value
    
    def value_update(self, s, a, r, next_s):
        """
        Here we update our values table using one step from the environment.
        To do this, 
        |----we're calculating the Bellman approximation for our state s and 
        |       action a by summing the immediate reward with the discounted value of the next state.
        |----Then we obtain the previous value of the state and action pair, and 
                blend these values together using the learning rate.

        The result is the new approximation for the value of state s and action a, 
        which is stored in our table.
        """
        best_v, _ = self.best_value_and_action(next_s)
        new_val = r + GAMMA * best_v
        old_val = self.values[(s, a)]
        self.values[(s, a)] = old_val * (1 - ALPHA) + new_val * ALPHA

    def play_episode(self, env):
        """
        This method is used to evaluate our current policy to check the progress of learning. 
        Note that this method doesn't alter our value table: 
            it only uses it to find the best action to take
        """
        total_reward = 0.0
        state = env.reset()
        while True:
            _, action = self.best_value_and_action(state)
            new_state, reward, is_done, _ = env.step(action)
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward

if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-q-learning")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        s, a, r, next_s = agent.sample_env()
        agent.value_update(s, a, r, next_s)

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward
        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()

"""
You may have noticed that this version used more iterations to solve the problem 
compared to the value iteration method from the previous chapter. 
The reason for that is that we're no longer using the experience obtained during testing.

(In vl.py, 
periodical tests cause an update of Q-table statistics. 
Here we don't touch Q-values during the test, 
which cause more iterations before the environment gets solved.)
"""

"""
The Q-learning method that we've just seen solves the issue with iteration over the full set of states, 
but still can struggle with situations when the count of the observable set of states is very large.

As a solution to this problem, we can use a nonlinear representation that maps both state and action onto a value. 

In machine learning this is called a "regression problem."

using a deep neural network is one of the most popular options,
    especially when dealing with observations represented as screen images.

Q-learning algorithm:
|----1.Initialize Q(s, a) with some initial approximation
|----2.By interacting with the environment, obtain the tuple (s, a, r, s′)
|----3.Calculate loss: L = (Qs,a - r) ^ 2 if episode has ended or (Qs,a - (r_{s, a} + \gamma max_a'Q(s', a'_{i})) ^ 2 otherwise
|----4.Update Q(s, a) using the stochastic gradient descent (SGD) algorithm,
|       by minimizing the loss with respect to the model parameters
|----5.Repeat from step 2 until converged

Unfortunately, it won't work very well. Let's discuss what could go wrong
|----First of all, we need to interact with the environment somehow to receive data to train on.
        but is this the best strategy to use?
            As an alternative, we can use our Q function approximation as a source of behavior 
            (as we did before in the value iteration method, 
            when we remembered our experience during testing).
            |----If our representation of Q is good, 
            |    then the experience that we get from the environment 
            |    will show the agent relevant data to train on.
            |----However, we're in trouble when our approximation is not perfect 
                 (at the beginning of the training, for example).
                    In such a case, our agent can be stuck with bad actions 
                    for some states without ever trying to behave differently.

            exploration versus exploitation dilemma
            |----On the one hand, our agent needs to explore the environment 
            |    to build a complete picture of transitions and action outcomes. 
            |----On the other hand, we should use interaction with the environment efficiently

            As you can see, random behavior is better at the beginning of the training 
            when our Q approximation is bad, 
            as it gives us more uniformly distributed information about the environment states.
            |----A method which performs such a mix of two extreme behaviors is known as an
                 epsilon-greedy method, which just means switching between random and Q
                 policy using the probability hyperparameter \epsilon.
                 |----The usual practice is to start with = 1.0 (100% random actions) and 
                 |    slowly decrease it to some small value such as 5% or 2% of random actions.
                 |----There are other solutions to the "exploration versus exploitation" problem, 
                 |    and we'll discuss some of them in part three of the book.
                 |----This problem is one of the fundamental open questions in RLand 
                      an active area of research, 
                      which is not even close to being resolved completely.

SGD optimization
The core of our Q-learning procedure is borrowed from the supervised learning.
    |----Indeed, we are trying to approximate a complex, 
    |    nonlinear function Q(s, a) with a neural network.
    |----To do this, we calculate targets for this function using the Bellman equation and 
    |    then pretend that we have a supervised learning problem at hand.
    |----but one of the fundamental requirements for SGD optimization is that the training data is independent and 
         identically distributed (frequently abbreviated as i.i.d).

In our case, data that we're going to use for the SGD update doesn't fulfill these criteria
|----1. Our samples are not independent.
|----2. Distribution of our training data won't be identical to samples provided by the optimal policy that we want to learn.
            Data that we have is a result of some other policy (our current policy, random, 
            or both in the case of \epsilon-greedy), 
            but we don't want to learn how to play randomly: 
            we want an optimal policy with the best reward.

replay buffer
To deal with this nuisance, we usually need to use a large buffer of our past experience and 
sample training data from it, instead of using our latest experience.
    |----The simplest implementation is a buffer of fixed size, 
    |    with new data added to the end of the buffer 
    |    so that it pushes the oldest experience out of it.
    |----Replay buffer allows us to train on more-or-less independent data, 
         but data will still be fresh enough to train on samples generated by our recent policy.

target network
|----Correlation between steps
     Another practical issue with the default training procedure is also related to the lack of i.i.d in our data, but in a slightly different manner.
     The Bellman equation provides us with the value of Q(s, a) via Q(s′, a′) (which has the name of bootstrapping). However, both states s and s′ have only one step between them. This makes them very similar and it's really hard for neural networks to distinguish between them. When we perform an update of our network's parameters, to make Q(s, a) closer to the desired result, we indirectly can alter the value produced for Q(s′, a′) and other states nearby. 
     This can make our training really unstable, like chasing our own tail:
         when we update Q for state s, then on subsequent states we discover that Q(s′, a′) becomes worse, but attempts to update it can spoil our Q(s, a) approximation, and so on.

|----To make training more stable, there is a trick, called target network, when we keep a copy of our network and use it for the Q(s′, a′) value in the Bellman equation.
        This network is synchronized with our main network only periodically, 
        for example, once in N steps (where N is usually quite a large hyperparameter, 
        such as 1k or 10k training iterations).

maintaining several observations from the past and using them as a state
The Markov property
Our RLmethods use MDP formalism as their basis, which assumes that the environment obeys the Markov property: observation from the environment is all that we need to act optimally
    As we've seen on the preceding Pong's screenshot, one single image from the Atari game is not enough to capture all important information (using only one image we have no idea about the speed and direction of objects, like the ball and our opponent's paddle).
    |----This obviously violates the Markov property and moves our single-frame Pong environment into the area of partially observable MDPs (POMDP).

A POMDP is basically MDP without the Markov property and they are very important in practice.
    For example, for most card games where you don't see your opponents' cards, game observations are POMDPs, because current observation (your cards and cards on the table) could correspond to different cards in your opponents' hands.

We'll not discuss POMPDs in detail in this book, so, for now, we'll use a small technique to push our environment back into the MDP domain. 
|----The solution is maintaining several observations from the past and using them as a state.!!!
     In the case of Atari games, we usually stack k subsequent frames together and use them as the observation at every state. This allows our agent to deduct the dynamics of the current state, for instance, to get the speed of the ball and its direction.

The final form of DQN training

There are many more tips and tricks that researchers have discovered to make DQN training more stable and efficient, and we'll cover the best of them in the next chapter.

\epsilon-greedy, replay buffer, and target network

The original paper (without target network) was published at the end of 2013 (Playing Atari with Deep Reinforcement Learning 1312.5602v1, Mnih and others.), and they used seven games for testing. Later, at the beginning of 2015, a revised version of the article, with 49 different games, was published in Nature (Human-Level Control Through Deep Reinforcement Learning
doi:10.1038/nature14236, Mnih and others.)

The algorithm for DQN from the preceding papers has the following steps:
|----1.Initialize parameters for Q(s, a) and Q^(s, a) with random weights, \epsilon <- 1.0 and empty replay buffer
|----2.With probability \epsilon, select a random action a, otherwise a = arg max_a_{i} Q(s, a_{i})
|----3.Execute action a in an emulator and observe reward r and the next state s′
|----4. Store transition (s, a, r, s′) in the replay buffer
|----5. Sample a random minibatch of transitions from the replay buffer
|----6. For every transition in the buffer, calculate target y = r if the episode has
ended at this step or y = r + \gamma max_a_{i} Q^(s', a_{i}) otherwise
|----7.Calculate loss: L = (Qs,a - r) ^ 2
|----8. Update Q(s, a) using the SGD algorithm by minimizing the loss in respect to model parameters
|----9.Every N steps copy weights from Q to Q^
|----10.Repeat from step 2 until converged





DQN on Pong

Another thing to note is performance. Our previous examples for FrozenLake,
or CartPole, were not demanding from a performance perspective, as
observations were small, neural network parameters were tiny, and shaving off
extra milliseconds in the training loop wasn't important. 
However, from now on, that's not the case anymore.

This example has been split into three modules due to its length, logical structure, and reusability.
|----/lib/wrappers.py: These are Atari environment wrappers mostly taken from the OpenAI Baselines project
|----/lib/dqn_model.py: This is the DQN neural net layer, with the same architecture as the DeepMind DQN from the Nature paper
|----/02_dqn_pong.py: This is the main module with the training loop, loss function calculation, and experience replay buffer
"""

#wrappers.py
"""
To make things faster, several transformations are applied to the Atari platform interaction, which are described in DeepMind's paper.
The full list of Atari transformations used by RLresearchers includes:
|----1.Converting individual lives in the game into separate episodes.
|----2.In the beginning of the game, performing a random amount (up to 30) of no-op actions. 
|       This should stabilize training, but there is no proper explanation why it is the case.
|----3.Making an action decision every K steps, where K is usually 4 or 3.
|       This allows training to speed up significantly, 
|       as processing every frame with a neural network is quite a demanding operation, 
|       but the difference between consequent frames is usually minor.
|----4.Taking the maximum of every pixel in the last two frames and using it as an observation.
|        Some Atari games have a flickering effect, 
|        which is due to the platform's limitation 
|        (Atari has a limited amount of sprites that can be shown on a single frame).
|----5.Pressing FIRE in the beginning of the game.
|       In theory, it's possible for a neural network to learn to press FIRE itself, 
|       but it will require much more episodes to be played.
|----6.Scaling every frame down from 210 × 160, with three color frames, into a single-color 84 × 84 image.
|----7.Stacking several (usually four) subsequent frames together to give the network the information about the dynamics of the game's objects.
|----8.Clipping the reward to −1, 0, and 1 values.
|       This spread in reward values makes our loss have completely different scales between the games, 
|       which makes it harder to find common hyperparameters for a set of games. 
|----9.Converting observations from unsigned bytes to float32 values. 
        The screen obtained from the emulator is encoded as a tensor of bytes with values from 0 to 255, 
        which is not the best representation for a neural network. 
        So, we need to convert the image into floats and rescale the values to the range [0.0…1.0].
"""
import cv2
import gym
import gym.spaces
import numpy as np
import collections


class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """For environments where the user need to press FIRE for the game to start."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    """
    This wrapper combines the repetition of actions during K frames and pixels from two consecutive frames.
    """
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class ImageToPyTorch(gym.ObservationWrapper):
    """
    This simple wrapper changes the shape of the observation from HWC to the CHW format required by PyTorch. 
    The input shape of the tensor has a color channel as the last dimension, 
    but PyTorch's convolution layers assume the color channel to be the first dimension.
    """
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    """
    1.The final wrapper we have in the library converts observation data from bytes to floats and 
    2.scales every pixel's value to the range [0.0...1.0].
    """
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


class BufferWrapper(gym.ObservationWrapper):
    """
    This class creates a stack of subsequent frames along the first dimension and returns them as an observation. 

    The purpose is to give the network an idea about the dynamics of the objects, 
    such as the speed and direction of the ball in Pong or how enemies are moving. 

    This is very important information, which it is not possible to obtain from a single image.
    """
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


def make_env(env_name):
    """
    At the end of the file is a simple function that creates an environment by its name and applies all the required wrappers to it.
    """
    env = gym.make(env_name)
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    return ScaledFloatFrame(env)

"""
DQN Model

The model published in Nature 
    |----has three convolution layers followed by two fully connected layers.
    |----All layers are separated by ReLU nonlinearities.
    |----The output of the model is Q-values for every action available in the environment, without nonlinearity applied (as Q-values can have any value).

The approach to have all Q-values calculated with one pass through the network helps us to increase speed significantly 
    in comparison to treating Q(s, a) literally and feeding observations and actions to the network to obtain the value of the action.
"""
import torch
import torch.nn as nn

import numpy as np


class DQN(nn.Module):
    """
    To be able to write our network in the generic way, it was implemented in two parts: convolution and sequential.
    """
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        """
        _get_conv_out() 函数的确是为了解决卷积层输出大小不确定的问题而设计的。
        在卷积神经网络中，卷积层的输出维度取决于输入的形状和卷积操作（例如卷积核的数量、大小、步幅等）。
        这一数值将作为全连接层输入的大小，用来构造模型的全连接部分。
        Another small problem is that we don't know the exact number of values in the output from the convolution layer produced with input of the given shape.
        However, we need to pass this number to the first fully connected layer constructor.
        |----One possible solution would be to hard-code this number, which is a function of input shape (for 84 × 84 input, the output from the convolution layer will have 3136 values), but it's not the best way, as our code becomes less robust to input shape change. 
        |----The better solution would be to have a simple function (_get_conv_out()) that accepts the input shape and applies the convolution layer to a fake tensor of such a shape. 
                The result of the function will be equal to the number of parameters returned by this application 
        """
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        """
        PyTorch doesn't have a 'flatter' layer which could transform a 3D tensor into a 1D vector of numbers, 
            required to feed convolution output to the fully connected layer. 
        This problem is solved in the forward() function, where we can reshape our batch of 3D tensors into a batch of 1D vectors.

        The final piece of the model is the forward() function, which accepts the 4D input tensor
            (the first dimension is batch size, 
            the second is the color channel, which is our stack of subsequent frames, 
            while the third and fourth are image dimensions).
        """

        #first we apply the convolution layer to the input and then we obtain a 4D tensor on output. 
        #This result is flattened to have two dimensions: a batch size and all the parameters returned by the convolution for this batch entry as one long vector of numbers.
        #这里view reshape 到 2D tensor，view 中-1表示通配符 即原纬度相乘 / x.size()[0] (batch)
        conv_out = self.conv(x).view(x.size()[0], -1)
        #pass this flattened 2D tensor to our fully connected layers to obtain Q-values for every batch input
        return self.fc(conv_out)

"""
Training

The third module contains the experience replay buffer, the agent, the loss function calculation, and the training loop itself.

Before going into the code, something needs to be said about the training hyperparameters. 
    |----DeepMind's Nature paper contained a table with all the details about hyperparameters used to train its model on all 49 Atari games used for evaluation. 
    |    DeepMind kept all those parameters the same for all games (but trained individual models for every game), 
    |    and it was the team's intention to show that the method is robust enough to solve lots of games with varying complexity, action space, reward structure, and other details using one single model architecture and hyperparameters. 
    |
    |----However, our goal here is much more modest: we want to solve just the Pong game.
         Pong is quite simple and straightforward in comparison to other games in the Atari test set, 
         so the hyperparameters in the paper are overkill for our task.
"""

#from lib import wrappers
#from lib import dqn_model
import argparse
import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter

#First, we import required modules and define hyperparameters.
DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19.5    #the reward boundary for the last 100 episodes to stop training.

GAMMA = 0.99                #Our gamma value used for Bellman approximation
BATCH_SIZE = 32             #The batch size sampled from the replay buffer (BATCH_SIZE)
REPLAY_SIZE = 10000         #The maximum capacity of the buffer (REPLAY_SIZE)
REPLAY_START_SIZE = 10000   #The count of frames we wait for before starting training to populate the replay buffer (REPLAY_START_SIZE)
LEARNING_RATE = 1e-4        #The learning rate used in the Adam optimizer, which is used in this example
SYNC_TARGET_FRAMES = 1000   #How frequently we sync model weights from the training model to the target model, which is used to get the value of the next state in the Bellman approximation.

EPSILON_DECAY_LAST_FRAME = 10**5
#To achieve proper exploration, at early stages of training, we start with epsilon=1.0, which causes all actions to be selected randomly.
EPSILON_START = 1.0
#Then, during first 100,000 frames, epsilon is linearly decayed to 0.02, which corresponds to the random action taken in 2% of steps.
EPSILON_FINAL = 0.02

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

class ExperienceBuffer:
    """
    The next chunk of the code defines our experience replay buffer, the purpose of
    which is to keep the last transitions obtained from the environment 
    (tuples of the observation, action, reward, done flag, and the next state).

    For training, we randomly sample the batch of transitions from the replay buffer, 
    which allows us to break the correlation between subsequent steps in the environment.

    Most of the experience replay buffer code is quite straightforward: 
    it basically exploits the capability of the deque class to maintain the given number of entries in the buffer.
    """
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        In the sample() method, 
        we create a list of random indices and 
        then repack the sampled entries into NumPy arrays for more convenient loss calculation.
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)

class Agent:
    """
    interacts with the environment and saves the result of the interaction into the experience replay buffer
    """
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device="cpu"):
        """
        The main method of the agent is to perform a step in the environment and store its result in the buffer.
        |----1.select the action
        |       With the probability epsilon (passed as an argument) 
        |       we take the random action,
        |       otherwise we use the past model to obtain the Q-values for all possible actionsand choose the best.
        |----2.As the action has been chosen, 
               we pass it to the environment to get the next observation and reward, 
               store the data in the experience buffer and the handle the end-of-episode situation.

        Returns:
            The result of the function is the total accumulated reward if we've reached the end of the episode with this step, 
            or None if not.
        """
        done_reward = None

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward

def calc_loss(batch, net, tgt_net, device="cpu"):
    """
    calculates the loss for the sampled batch

    This function is written in a form to maximally exploit GPU parallelism by processing all batch samples with vector operations, 
    which makes it harder to understand when compared with a naive loop over the batch.

    here is the loss expression we need to calculate:
        L = (Qs, a - y) ^ 2

    arguments:
        batch 
            |----as a tuple of arrays (repacked by the sample() method in the experience buffer), 
        network 
            |----that we're training 
            |----is used to calculate gradients
        target network
            |----which is periodically synced with the trained one.   
            |----is used to calculate values for the next states and this calculation shouldn't affect gradients.
                    To achieve this, we're using the detach() function of the PyTorch tensor to prevent gradients from flowing into the target network's graph. 
                    This function was described in Chapter 3, Deep Learning with PyTorch.
    """
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)

    """
    we pass observations to the first model and extract the specific Q-values for the taken actions using the gather() tensor operation.
        gather()
            The first argument to the gather() call 
                is a dimension index that we want to perform gathering on (in our case it is equal to 1, which corresponds to actions).
            The second argument is a tensor of indices of elements to be chosen.
            the result of gather() applied to tensors is a differentiable operation, which will keep all gradients with respect to the final loss value.

        Extra unsqueeze() and squeeze() calls 
            are required to fulfill the requirements of the gather functions to the index argument and 
            to get rid of extra dimensions that we created (the index should have the same number of dimensions as the data we're processing). 

        In the following image, you can see an illustration of what gather does on the example case, with a batch of six entries and four actions.
                                                                                                                                                                                                                                                                                                            
                            :                           actions
                        .!YBB^:::::::::::::::::::...::..::.:.:...:.::::..:::::::::::::::::::.Y#57:                                                                                                                                                                                                          
                        :?G&#~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^:5@BY^                                                                                                                                                                                                          
                           :^                                                                ^^                                                                                                                                                                                                             
                                                                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                                            
             ^         .!~~~~~~~~~~~~~~~~^7!~~~~~~~~~~~~~~~~~7^~~~~~~~~~~~~~~~~7~~~~~~~~~~~~~~~~~!:                          :!~~~~~~~~~~~~~~~~~!                           ::::::::::::::::::^::::::::::::::::::^:::::::::::::::::::::::::::::::::::::                           :::::::::::::::::::.      
           .Y@5.       .?                 J~                ^?                .Y:                !!                          ~7                .J.                          JP5555555555555555P?::::::::::::::::^Y^::::::::::::::::J7::::::::::::::::^?                          .P55555555555555555P^      
           ?B#BJ.      .?                 ?^                ^?                 J:                ~!                          ^7      .~~^       ?.                          J5YYYYYYYYYYYYYYYYP7                .J.                7~                .?                          .P5YYYYYYYYYYYYYYY5P^      
             ?.        .?                 ?^                ^?                 J:                ~!                          ^!     .J:.7~      ?.                          J5YYYYYYYYYYYYYYYYP7                .J.                7~                .?                          .P5YYYYYYYYYYYYYYY5P^      
             ?.        .?                 ?^                ^?                 J:                ~!                          ^!     .J:.7~      ?.                          J5YYYYYYYYYYYYYYYYP7                .J.                7~                .?                          .P5YYYYYYYYYYYYYYY5P^      
             ?.        .?                 ?^                ^?                 J:                ~!                          ^!      .^~:       ?.                          J5YYYYYYYYYYYYYYYYP7                .J.                7~                .?                          .P5YYYYYYYYYYYYYYY5P^      
             ?.        .?                 ?^                ^?                 J:                ~!                          ^!                 ?.                          J5YYYYYYYYYYYYYYYYP!                .J                 7~                .?                          .P5YYYYYYYYYYYYYYY5P^      
             ?.        .Y~~~~~~~~~~~~~~~~~Y7~~~~~~~~~~~~~~~~7Y~~~~~~~~~~~~~~~~~5!~~~~~~~~~~~~~~~~?!                          ^J~~~~~~~~~~~~~~~~~Y.                          YP5555555555555555G?::::::::::::::::^Y^::::::::::::::::J7::::::::::::::::^?                          .P55555555555555555P^      
             ?.        .?.................?~................~?.................J:................!!                          ^7.................?.                          J!^^^^^^^^^^^^^^^^?J^^^^^^^^^^^^^^^^~5~^^^^^^^^^^^^^^^^JP55555555555555555J                          .P55555555555555555P^      
             ?.        .?                 ?^                ^?                 J:                ~!                          ^!      :^^:       ?.                          ?.                ~7                .J                 7P5YYYYYYYYYYYYYYY5J                          .P5YYYYYYYYYYYYYYY5P^      
             ?.        .?                 ?^                ^?                 J:                ~!                          ^!      ^^~J.      ?.                          ?:                ~7                .J.                7P5YYYYYYYYYYYYYYY5J                          .P5YYYYYYYYYYYYYYY5P^      
             ?.        .?                 ?^                ^?                 J:                ~!                          ^!     .^^~J^      ?.                          ?:                ~7                .J.                7P5YYYYYYYYYYYYYYY5J                          .P5YYYYYYYYYYYYYYY5P^      
             ?.        .?                 ?^                ^?                 J:                ~!                          ^!      ^^^^.      ?.                          ?:                ~7                .J.                7P5YYYYYYYYYYYYYYY5J                          .P5YYYYYYYYYYYYYYY5P^      
             ?.        .?                 ?^                ^?                 J:                ~!                          ^!                 ?.                          ?.                ~7                .J.                7PYYYYYYYYYYYYYYYY5J                          .P5YYYYYYYYYYYYYYY5P^      
             ?.        .J~~~~~~~~~~~~~~~~^Y7^~~~~~~~~~~~~~~^7Y^~~~~~~~~~~~~~~~~5!~~~~~~~~~~~~~~~^?!                          ^J^~~~~~~~~~~~~~~~~J.                          ?^:::::::::::::::.7?.:::::::::::::::^Y:.:::::::::::::..?P5555555555555555PJ                          .P55555555555555555P^      
             ?.        .J.................J~................~J................:Y^................!!                          ^7................:J.                          J!^^^^^^^^^^^^^^^^?J^^^^^^^^^^^^^^^^~P5555555555555555YG?~~~~~~~~~~~~~~~~!J                          .P55555555555555555P^      
             ?.        .?                 ?^                ^?                 J:                ~!                          ^!      .^^:       ?.                          ?.                ~7                .55YYYYYYYYYYYYYYYYG~                .?                          .P5YYYYYYYYYYYYYYY5P^      
             ?.        .?                 ?^                ^?                 J:                ~!                          ^!      !::J:      ?.                          ?:                ~7                .55YYYYYYYYYYYYYYYYG~                .?                          .P5YYYYYYYYYYYYYYY5P^      
             ?.        .?                 ?^                ^?                 J:                ~!                          ^!      .~7~       ?.                          ?:                ~7                .55YYYYYYYYYYYYYYYYG~                .?                          .P5YYYYYYYYYYYYYYY5P^      
             ?.        .?                 ?^                ^?                 J:                ~!                          ^!      !7~~.      ?.                          ?:                ~7                .55YYYYYYYYYYYYYYYYG~                .?                          .P5YYYYYYYYYYYYYYY5P^      
             ?.        .?                 ?^                ^?                 J:                ~!                          ^!                 ?.                .~:       ?.                ~7                .55YYYYYYYYYYYYYYYYG~                .?                          .P5YYYYYYYYYYYYYYY5P^      
  batch      ?.        .J^^^^^^^^^^^^^^^^^Y7^^^^^^^^^^^^^^^^7Y^^^^^^^^^^^^^^^^^5!^^^^^^^^^^^^^^^^?!                          ^?^^^^^^^^^^^^^^^^~J.   .^^^^^^^^^^^^~&@G?:    ?^................!7................:555555555555555555G!................^?    .............~B57:    .P55555555555555555P^      
             ?.        .J::::::::::::::::.J!.::::::::::::::.~J.::::::::::::::::Y^:::::::::::::::.7!                          ^7.:::::::.::::::::J.   .............^BGJ~.    J!~~~~~~~~~~~~~~~^?PYYYYYYYYYYYYYYYY5P!!!!!!!!!!!!!!!!~Y?^~~~~~~~~~~~~~~~!J    :^^^^^^^^^^^^7@@G?.   .P55555555555555555P^      
             ?.        .?                 ?^                ^?                 J:                ~!                          ^!       .:        ?.                 .        ?.                ~P5Y5YYYYYYYYY55YY55                 7~                .?                 .!:      .P5YYYYYYYYYYYYYYY5P^      
             ?.        .?                 ?^                ^?                 J:                ~!                          ^!      :!5.       ?.                          ?:                ~PYYYYYYYYYYYYYYYY55.                7~                .?                          .P5YYYYYYYYYYYYYYY5P^      
             ?.        .?                 ?^                ^?                 J:                ~!                          ^!       .Y.       ?.                          ?:                ~PYYYYYYYYYYYYYYYY55.                7~                .?                          .P5YYYYYYYYYYYYYYY5P^      
             ?.        .?                 ?^                ^?                 J:                ~!                          ^!      :~?~:      ?.                          ?:                ~PYYYYYYYYYYYYYYYY55.                7~                .?                          .P5YYYYYYYYYYYYYYY5P^      
             ?.        .?                 ?^                ^?                 J:                ~!                          ^!                 ?.                          ?.                ~PYYYYYYYYYYYYYYYY55                 7~                .?                          .P5YYYYYYYYYYYYYYY5P^      
             ?.        .J^^^^^^^^^^^^^^^^^Y!^^^^^^^^^^^^^^^^!J^^^^^^^^^^^^^^^^^Y~^^^^^^^^^^^^^^^^7!                          ^?^^^^^^^^^^^^^^^^^J.                          ?:................~P555555555555555555................ ?!................:?                          .P55555555555555555P^      
             ?.        .J:::::::::::::::::J!::::::::::::::::!J:::::::::::::::::Y~::::::::::::::::7!                          ^?:::::::::::::::::J.                          J!~~~~~~~~~~~~~~~~?GPPPPPPPPPPPPPPPPPP~~~~~~~~~~~~~~~~~Y?~~~~~~~~~~~~~~~~!J                          .PPPPPPPPPPPPPPPPPPG^      
             ?.        .?                 ?^                ^?                 J:                ~!                          ^!        :        ?.                          ?.                ~PYYYYYYYYYYYYYYYY55                 7~                .?                          .P5YYYYYYYYYYYYYYY5P^      
             ?.        .?                 ?^                ^?                 J:                ~!                          ^!      :!5.       ?.                          ?:                ~PYYYYYYYYYYYYYYYY55.                7~                .?                          .P5YYYYYYYYYYYYYYY5P^      
             ?.        .?                 ?^                ^?                 J:                ~!                          ^!       .Y.       ?.                          ?:                ~PYYYYYYYYYYYYYYYY55.                7~                .?                          .P5YYYYYYYYYYYYYYY5P^      
             ?.        .?                 ?^                ^?                 J:                ~!                          ^!      ^!?~:      ?.                          ?:                ~PYYYYYYYYYYYYYYYY55.                7~                .?                          .P5YYYYYYYYYYYYYYY5P^      
             ?.        .?                 ?^                ^?                 J:                ~!                          ^!                 ?.                          ?.                ~PYYYYYYYYYYYYYYYY55                 7~                .?                          .P5YYYYYYYYYYYYYYY5P^      
             ?.        .J^^^^^^^^^^^^^^^^:J!^^^^^^^^^^^^^^^:!J:^^^^^^^^^^^^^^^^Y~^^^^^^^^^^^^^^^:7!                          ^?:^^^^^^^^^^^^^^^^J.                          ?:                ~P555555555555555555.                7!                :?                          .P5Y555555555555555P^      
             ?.        .J:^^^^^^^^^^^^^^^:J!:^^^^^^^^^^^^^^:!J:^^^^^^^^^^^^^^^^Y~:^^^^^^^^^^^^^^:7!                          ^?:^^^^^^::^^^^^^^^J.                          J!^~~~~~~~~~~~~~~^?Y7777777777777777?PJJJJJJJJJJJJJJJJJP?^~~~~~~~~~~~~~~^!J                          .PPPPPPPPPPPPPPPPPPG^      
             ?.        .?                 ?^                ^?                 J:                ~!                          ^!       ..        ?.                          ?.                ~7                .555555555555555555G~                .?                          .P5YYYYYYYYYYYYYYY5P^      
             ?.        .?                 ?^                ^?                 J:                ~!                          ^!      !~^?:      ?.                          ?:                ~7                .55YYYYYYYYYYYYYYYYG~                .?                          .P5YYYYYYYYYYYYYYY5P^      
             ?.        .?                 ?^                ^?                 J:                ~!                          ^!      ..~7.      ?.                          ?:                ~7                .55YYYYYYYYYYYYYYYYG~                .?                          .P5YYYYYYYYYYYYYYY5P^      
           ^^Y^^.      .?                 ?^                ^?                 J:                ~!                          ^!      !J7~.      ?.                          ?:                ~7                .55YYYYYYYYYYYYYYYYG~                .?                          .P5YYYYYYYYYYYYYYY5P^      
           7&@&?       .?                 ?^                ^?                 J:                ~!                          ^!                 ?.                          ?:                ~7                .55YYYYYYYYYYYYYYYYG~                .?                          .P5YYYYYYYYYYYYYYY5P^      
            !B7        .J:::::::::::::::::J!::::::::::::::::~J:::::::::::::::::Y^::::::::::::::::7!                          ^7:::::::::::::::::J.                          ?:                !7                .555555555555555555G~                :?                          .P55555555555555555P^      
             .          :^^^^^^^^^^^^^^^^:^^^^^^^^^^^^^^^^^^^^::^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^:^.                          .^^^^^^^^^^^^^^^^^^:                           !!~~~~~~~~~~~~~~~~77~~~~~~~~~~~~~~~~!??????????????????J!~~~~~~~~~~~~~~~~!!                           ???????????????????:      
                                                    Out put of model                                                           Action takens                                                           Selected g-values                                                           Result of gather                
    """
    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    """
    we apply the target network to our next state observations and calculate the maximum Q-value along the same action dimension 1.
    max(1) 表示在动作维度（第 1 维）上找出最大值。对于每个状态，都会找到该状态下所有动作的最大 Q 值。
    max(1) 函数不仅返回最大值，还返回最大值的索引（类似 argmax）。所以结果是一个包含两个元素的元组：
        第一个元素是最大 Q 值；
        第二个元素是对应的动作索引。
        由于我们只关心最大 Q 值，不关心具体的动作索引，所以只取第一个元素 [0]，即 max(1)[0]。
    """
    next_state_values = tgt_net(next_states_v).max(1)[0]
    """
    if transition in the batch is from the last step in the episode, 
    then our value of the action doesn't have a discounted reward of the next state, 
    as there is no next state to gather reward from.

    without this, training will not converge.!!!
    """
    next_state_values[done_mask] = 0.0
    """
    In this line, we detach the value from its computation graph to prevent gradients from flowing into the neural network used to calculate Q approximation for next states.

    This is important, as without this our backpropagation of the loss will start to affect both predictions for the current state and the next state.
        However, we don't want to touch predictions for the next state, as they're used in the Bellman equation to calculate reference Qvalues. 

    To block gradients from flowing into this branch of the graph, we're using the detach() method of the tensor, which returns the tensor without connection to its calculation history.
    """
    next_state_values = next_state_values.detach()

    #we calculate the Bellman approximation value and the mean squared error loss.
    expected_state_action_values = next_state_values * GAMMA + rewards_v
    """
    使用均方误差（MSE）损失函数来计算并返回当前 Q 值与目标 Q 值之间的差异
    该损失值用于指导模型更新参数，减小当前 Q 值与目标 Q 值之间的差距，从而改进策略。
    这是 Q-learning 的核心目标，即让模型学习到越来越接近最优策略的 Q 值函数。
    """
    return nn.MSELoss()(state_action_values, expected_state_action_values)

#This ends our loss function calculation, and the rest of the code is our training loop.

if __name__ == "__main__":
    """
    we create a parser of command-line arguments.
        Our script allows us to enable CUDA and train on environments that are different from the default.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("--reward", type=float, default=MEAN_REWARD_BOUND,
                        help="Mean reward boundary for stop of training, default=%.2f" % MEAN_REWARD_BOUND)
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    # env = wrappers.make_env(args.env)
    env = make_env(args.env)

    """
    Here we create our environment with all required wrappers applied, the neural network we're going to train, and our target network with the same architecture. 
    In the beginning, they'll be initialized with different random weights, 
    but it doesn't matter much as we'll sync them every 1k frames, 
    which roughly corresponds to one episode of Pong
    """
    # net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    # tgt_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    net = DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = DQN(env.observation_space.shape, env.action_space.n).to(device)
    writer = SummaryWriter(comment="-" + args.env)
    print(net)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START #Epsilon is initially initialized to 1.0, but will be decreased every iteration.

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_mean_reward = None #Every time our mean reward beats the record, we'll save the model in the file.

    while True:
        frame_idx += 1
        """
        At the beginning of the training loop, we count the number of iterations completed and decrease epsilon according to our schedule. 
        Epsilon will drop linearly during the given number of frames (EPSILON_DECAY_LAST_FRAME=100k) and then will be kept on the same level of EPSILON_FINAL=0.02.
        """
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

        reward = agent.play_step(net, epsilon, device=device)
        if reward is not None:
            """
            play_step()
            This function returns a nonNone result only if this step is the final step in the episode.
            In that case, we report our progress. Specifically, we calculate and show, both in the console and in TensorBoard, these values:
            |----Speed as a count of frames processed per second
            |----Count of episodes played
            |----Mean reward for the last 100 episodes
            |----Current value for epsilon
            """
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            mean_reward = np.mean(total_rewards[-100:])
            print("%d: done %d games, mean reward %.3f, eps %.2f, speed %.2f f/s" % (
                frame_idx, len(total_rewards), mean_reward, epsilon,
                speed
            ))
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", mean_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            if best_mean_reward is None or best_mean_reward < mean_reward:
                """
                Every time our mean reward for the last 100 episodes reaches a maximum, we report this and save the model parameters. 
                """
                torch.save(net.state_dict(), args.env + "-best.dat")
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
                best_mean_reward = mean_reward
            if mean_reward > args.reward:
                """
                 If our mean reward exceeds the specified boundary, then we stop training.
                """
                print("Solved in %d frames!" % frame_idx)
                break

        if len(buffer) < REPLAY_START_SIZE:
            """
            Here we check whether our buffer is large enough for training. 
            In the beginning, we should wait for enough data to start, which in our case is 10k transitions.
            """
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            """
            The next condition syncs parameters from our main network to the target net every SYNC_TARGET_FRAMES, which is 1k by default.
            """
            tgt_net.load_state_dict(net.state_dict())

        """
        The last piece of the training loop is very simple, but requires the most time to execute: 
        |----we zero gradients, 
        |----sample data batches from the experience replay buffer, 
        |----calculate loss, and 
        |----perform the optimization step to minimize the loss.
        """
        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()
    writer.close()

"""
In the next chapter, we'll look at various approaches, found by researchers since 2015, 
which can help to increase both training speed and data efficienc
"""

"""
Your model in action

we have a program which can load this model file and play one episode, displaying the model's dynamics.
The code is very simple, but seeing how several matrices, with a million parameters, play Pong with superhuman accuracy, by observing only the pixels, can be like magic.
"""
import gym
import time
import argparse
import numpy as np

import torch

from lib import wrappers
from lib import dqn_model

import collections

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
FPS = 25    #The preceding FPS parameter specifies the approximate speed of the shown frames.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-e", "--env", default=DEFAULT_ENV_NAME,
                        help="Environment name to use, default=" + DEFAULT_ENV_NAME)
    """
    Additionally, you can pass option -r with the name of a non existent directory, 
    which will be used to save a video of your game (using the Monitor wrapper).
    """
    parser.add_argument("-r", "--record", help="Directory to store video recording")
    parser.add_argument("--no-visualize", default=True, action='store_false', dest='visualize',
                        help="Disable visualization of the game play")
    args = parser.parse_args()

    env = wrappers.make_env(args.env)
    if args.record:
        env = gym.wrappers.Monitor(env, args.record)
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n)
    """
    load weights from the file passed in the arguments.
    """
    net.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))

    state = env.reset()
    total_reward = 0.0
    c = collections.Counter()

    while True:
        """
        This is almost an exact copy of Agent class' method play_step() from the training code, with the lack of epsilon-greedy action selection.

        We just pass our observation to the agent and select the action with maximum value.
        """
        start_ts = time.time()
        if args.visualize:
            env.render()
        state_v = torch.tensor(np.array([state], copy=False))
        q_vals = net(state_v).data.numpy()[0]
        action = np.argmax(q_vals)
        c[action] += 1
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
        if args.visualize:
            delta = 1/FPS - (time.time() - start_ts)
            if delta > 0:
                time.sleep(delta)
    print("Total reward: %.2f" % total_reward)
    print("Action counts:", c)
    if args.record:
        env.env.close()

"""
Summary

We became familiar with the limitations of value iteration in complex environments with large observation spaces and discussed how to overcome them with Qlearning. 

We checked the Q-learning algorithm on the FrozenLake environment and discussed the approximation of Q-values with neural networks and the extra complications that arise from this approximation. 

We covered several tricks for DQNs to improve their training stability and convergence, such as experience replay buffer, target networks, and frame stacking. 

In the next chapter, we'll look at a set of tricks that researchers have found, since 2015, to improve DQN convergence and quality, which (combined) can produce state-of-the-art results on most of the 54 Atari games. 
This set was published in 2017 and we'll analyze and reimplement all of the tricks.
"""
