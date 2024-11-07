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
