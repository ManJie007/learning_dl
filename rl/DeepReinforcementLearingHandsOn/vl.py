"""
Value is always calculated in the respect of some policy that our agent follows.

                 1.0             
    | 1 start | ----> | 2 end |
        |
    2.0 |
        |
        v
    | 3 end |

    what's the value of state 1?
    This question is meaningless without information about 
    our agent's behavior or, in other words, its policy.

    policy examples
    |----Agent always goes left
    |----Agent always goes down
    |----Agent goes left with a probability of 0.5 and down with a probability of 0.5
    |----Agent goes left in 10% of cases and in 90% of cases executes the "down" action

    To demonstrate how the value is calculated, let's do it for all the preceding policies
    |----The value of state 1 in the case of the "always left" agent is 1.0 (every time it goes left, it obtains 1 and the episode ends)
    |----For the "always down" agent, the value of state 1 is 2.0
    |----For the 50% left/50% down agent, the value will be 1.0*0.5 + 2.0*0.5 = 1.5
    |----In the last case, the value will be 1.0*0.1 + 2.0*0.9 = 1.9

what's the optimal policy for this agent? 
    The goal of RL is to get as much total reward as possible.

For interesting environments, the optimal policy is much harder to formulate and it's even harder to prove their optimality.

Bellman equation
    V = max(r_{i} + \gammaV_{i})

    is different from acting greedily 
        we do not only look at the immediate reward for the action, 
        but at the immediate reward plus the long-term value of the state.

let's consider one single action available from state , with three possible outcomes.
                                s0
                                |
                                |   a = 1
                                v
                             /  |   \
                        p1 / p2 |  p3\
                         /      |     \
                      v         v      v
            {s1, v1, r1}  {s2, v2, r2}  {s3, v3, r3} 

To calculate the expected value after issuing action 1, we need to sum all values, multiplied by their probabilities:

    V0(a=1) = p1(r1 + \gamma v1) + p2(r2 + \gamma v2) + p3(r3 + \gamma v3)
more formally:
    V0(a) = \sum_{s in S} p_{a, s}(rs + \gamma vs)

By combining the Bellman equation, for a deterministic case, 
with a value for stochastic actions, we get the Bellman optimality equation for a general case:

    V0 = max_{a}(\sum_{s in S} p_{a_{i}, s}(rs + \gamma vs))

the optimal value of the state is 
    equal to the action, 
    which gives us the maximum possible expected immediate reward, plus discounted long-term reward for the next state.

    You may also notice that this definition is recursive
        the value of the state is defined via the values of immediate reachable states.

These values not only give us the best reward that we can obtain, but they basically give us the optimal policy to obtain that reward 
    if our agent knows the value for every state, then it automatically knows how to gather all this reward.

Value of Action Q(s, a)
    |----it equals the total reward we can get by executing action a in state s 
    |----this quantity gave a name to the whole family of methods called "Q-learning"
 |->|----Q(s, a) = \sum_{s in S} p_{a, s}(rs + \gamma vs) ==> V_s = max_{a}Q(s, a_{i})
 |               = r_{s, a} + \gamma max_a'Q(s', a'_{i})
 |
T| give you a concrete example:
 |  we have one initial state s0, surrounded by four target states, s1, s2, s3, s4, with different rewards.
 |
 |      ----
 |      |s1|
 |  ----------  
 |  |s2|s0|s3|
 |  ----------  
 |     |s3|
 |     ----
 |
 |  Every action is probabilistic in the same way as in FrozenLake: 
 |  with a 33% chance that our action will be executed without modifications, 
 |  but with a 33% chance we will slip to the left, relatively, of our target cell and 
 |  a 33% chance we will slip to the right. 
 |  For simplicity, we use discount factor gamma=1.
 |
 |  Let's calculate the values of actions to begin with.
 |      Terminal states have no outbound connections, so Q for those states is zero for all actions. 
 |      Due to this, the values of the Terminal states are equal to their immediate reward 
 |      (once we get there, our episode ends without any subsequent states):
 |      V1 = 1, V2 = 2, V3 = 3, V4 = 4
 |
 |      The values of actions for state 0 are a bit more complicated.
 |      Let's start with the "up" action.
 |      |----Its value, according to the definition, is equal to the expected 
 |              sum of the immediate reward plus long-term value for subsequent steps.
 |      |----We have no subsequent steps for any possible transition for the "up" action, so
 |
 |      Q(s0, up) = 0.33 · V1 + 0.33 · V2 + 0.33 · V4 = 0.33 · 1 + 0.33 · 2 + 0.33 · 4 = 2.31
 |      Q(s0, left) = 0.33 · V1 + 0.33 · V2 + 0.33 · V3 = 0.33 · 1 + 0.33 · 2 + 0.33 · 3 = 1.98
 |      Q(s0, right) = 0.33 · V4 + 0.33 · V2 + 0.33 · V3 = 0.33 · 4 + 0.33 · 2 + 0.33 · 3 = 2.64
 |      Q(s0, down) = 0.33 · V3 + 0.33 · V2 + 0.33 · V4 = 0.33 · 3 + 0.33 · 2 + 0.33 · 4 = 2.97
 |
 |--The final value for state is the maximum of those actions' values, which is 2.97.

Q values are much more convenient in practice, 
as for the agent it's much simpler to make decisions about actions based on Q than based on V
|----to choose the action based on the state, 
     the agent just needs to calculate Q for all available actions, 
     using the current state and 
     choose the action with the largest value of Q.
To do the same using values of states
|----the agent needs to know not only values, 
|----but also probabilities for transitions
        In practice, we rarely know them in advance, 
        so the agent needs to estimate transition probabilities for every action and state pair.

a general way to calculate those Vs and Qs
    The value iteration method
    we had no loops in transitions, so we could start from terminal states, calculate their values and then proceed to the central state.

             r=1
       |--| ---->  |--|
       |s1|        |s2|
       |--| <----  |--|
             r=2

    We start from state s1, and the only action we can take leads us to state s2. 
    We get reward r=1,and the only transition from is an action, which brings us back to the s2. 
    So, the life of our agent is an infinite sequence of states [s1, s2, s1, s2, s1, s2, s1, s2, ...].
    To deal with this infinity loop, we can use a discount factor \gamma = 0.9.
    Now, the question is, what are the values for both the states?
        Every transition from s1 to s2 gives us a reward of 1 and every back transition gives us 2. 
        So, our sequence of rewards will be [1, 2, 1, 2, 1, 2, 1, 2, ...]. 
        As there is only one action available in every state, our agent has no choice, 
        so we can omit the max operation in formulas (there is only one alternative). 
        The value for every state will be equal to the infinite sum:
            V(s1) = 1 + \gamma (2 + \gamma (1 + \gamma (2 + ...))) = \sum_{i=0, \infinity}(1 · \gamma ^ (2i) + 2 · \gamma ^ (2i + 1))
            V(s2) = 2 + \gamma (1 + \gamma (2 + \gamma (1 + ...))) = \sum_{i=0, \infinity}(2 · \gamma ^ (2i) + 1 · \gamma ^ (2i + 1))

        Strictly speaking, we cannot calculate the exact values for our states, but with \gamma == 0.9, 
        the contribution of every transition quickly decreases over time.

        For example, after 10 steps, \gamma ^ 10 == 0.349, but after 100 steps it becomes just 0.0000266. 
        Due to this, we can stop after 50 iterations and still get quite a precise estimation

    The preceding example could be used to get a gist of a more general procedure, 
    called the "value iteration algorithm" which allows us to numerically calculate the values of states and 
    values of actions of MDPs with known transition probabilities and rewards.
    |----1.Initialize values of all states Vi to some initial value (usually zero)
    |----2.For every state s in the MDP, perform the Bellman update:
    |       Vs = max_{a}(\sum_{s in S} p_{a_{i}, s}(rs + \gamma vs))
    |----3.Repeat step 2 for some large number of steps or until changes become too small

    In the case of action values (that is Q), 
    only minor modifications to the preceding procedure are required
    |----1.Initialize all Q(s, a) to zero
    |----2.For every state s and every action a in this state, perform update:
    |       Q(s, a) = max_{a}Q(s, a_{i})
    |               = r_{s, a} + \gamma max_a'Q(s', a'_{i})
    |----3.Repeat step 2

What about the practice? In practice, this method has several obvious limitations. 
|----First of all, our state space should be discrete and small enough to perform multiple iterations over all states.
|       A potential solution for that could be discretization of our observation's values, 
|       for example, we can split the observation space of the CartPole into bins and 
|       treat every bin as an individual discrete state in space.
|----we rarely know the transition probability for the actions and rewards matrix
        We don't know (without peeking into Gym's environment code) 
        what the probability is to get into state s1 from state s0 by issuing action a0. 
        What we do have is just the history from the agent's interaction with the environment.

        However, in Bellman's update, we need both a reward for every transition and 
                                                    the probability of this transition

        So, the obvious answer to this issue is to use our agent's experience 
        as an estimation for both unknowns.

        We just need to remember what reward we've got on transition from s0 to s1, using action a, 
        but to estimate probabilities we need to maintain counters for every tuple (s0, s1, a0) and normalize them.


Value iteration in practice
the central data structures
|----Reward table 
|       A dictionary with the composite key "source state" + "action" + "target state". 
|       The value is obtained from the immediate reward.
|----Transitions table 
|       A dictionary keeping counters of the experienced transitions. 
|       The key is the composite "state" + "action" and the value is another dictionary that 
|       maps the target state into a count of times that we've seen it. 
|           For example, if in state 0 we execute action 1 ten times, 
|           after three times it leads us to state 4 and 
|           after seven times to state 5.
|           Entry with the key (0, 1) in this table will be a dict {4: 3, 5: 7}. 
|       We use this table to estimate the probabilities of our transitions.
|----Value table 
        A dictionary that maps a state into the calculated value of this state.

The overall logic of our code is simple: 
|----in the loop, we play 100 random steps from the environment, populating the reward and transition tables. 
|---After those 100 steps, we perform a value iteration loop over all states, updating our value table. 
|----Then we play several full episodes to check our improvements using the updated value table. 
        If the average reward for those test episodes is above the 0.8 boundary, then we stop training. 
|----During test episodes, we also update our reward and transition tables to use all data from the environment.
"""
#!/usr/bin/env python3
import gym
import collections
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v0"
GAMMA = 0.9
TEST_EPISODES = 20


class Agent:
    """
        keep our tables and contain functions we'll be using in the training loop
    """
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(float)

    def play_n_random_steps(self, count):
        """
            to gather random experience from the environment and
            update reward and transition tables

            Note that we don't need to wait for the end of the episode to start learning; 
            we just perform N steps and remember their outcomes.
                This is one of the differences between Value iteration and Crossentropy, which can learn only on full episodes.
                这里说明价值迭代可以不需要full episodes!!!
        """
        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, is_done, _ = self.env.step(action)
            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1
            self.state = self.env.reset() if is_done else new_state

    def calc_action_value(self, state, action):
        """
            The next function calculates the value of the action from the state, using our transition, reward and values tables.
            we use it for two purposes
            |----to select the best action to perform from the state
            |----to calculate the new value of the state on value iteration

            Its logic is illustrated in the following diagram and we do the following:

                         > s1
                    c1 /                trans[(s, a)] = {s1:c1, s2:c2}
                a    /                  total = c1 + c2
            s ----->                    Q(s, a) = c1 / total · (r_{s1} + \gamma V(s1)) + c2 / total · (r_{s2} + \gamma V(s2))
                     \
                   c2 \
                        > s2

            |----1.We extract transition counters for the given state and action from the transition table.
            |       We sum all counters to obtain the total count of times we've executed the action from the state. 
            |       We will use this total value later to go from an individual counter to probability.
            |----2.Then we iterate every target state that our action has landed on and 
                    calculate its contribution into the total action value using the Bellman equation.
                        This contribution equals to immediate reward plus discounted value for the target state. 
                        We multiply this sum to the probability of this transition and add the result to the final action value.
        """
        target_counts = self.transits[(state, action)]
        total = sum(target_counts.values())
        action_value = 0.0
        for tgt_state, count in target_counts.items():
            reward = self.rewards[(state, action, tgt_state)]
            action_value += (count / total) * (reward + GAMMA * self.values[tgt_state])
        return action_value

    def select_action(self, state):
        """
            uses the function we just described to make a decision about the best action to take from the given state.
            It iterates over all possible actions in the environment and calculates value for every action.
            This action selection process is deterministic, as the play_n_random_steps() function introduces enough exploration. 
            So, our agent will behave greedily in regard to our value approximation.
        """
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    def play_episode(self, env):
        """
            uses select_action to find the best action to take and 
            plays one full episode using the provided environment.

            This function is used to play test episodes, 
            during which we don't want to mess up with the current state of the main environment used to gather random data. 
            So, we're using the second environment passed as an argument.
        """
        total_reward = 0.0
        state = env.reset()
        while True:
            action = self.select_action(state)
            new_state, reward, is_done, _ = env.step(action)
            self.rewards[(state, action, new_state)] = reward
            self.transits[(state, action)][new_state] += 1
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward

    def value_iteration(self):
        """
            |---- What we do is just loop over all states in the environment, 
            |     then for every state we calculate the values for the states reachable from it, 
            |     obtaining candidates for the value of the state.
            |
            |---- Then we update the value of our current state with the maximum value of the action available from the stat
        """
        for state in range(self.env.observation_space.n):
            state_values = [self.calc_action_value(state, action)
                            for action in range(self.env.action_space.n)]
            self.values[state] = max(state_values)


if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-v-iteration")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        """
            The two lines in the preceding code snippet are the key piece in the training loop.
             First, we perform 100 random steps to fill our reward and transition tables with fresh data and 
             then we run value iteration over all states.
        """
        agent.play_n_random_steps(100)
        agent.value_iteration()

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
If you remember how many hours were required to achieve a 60% success ratio using Cross-entropy, 
then you can understand that this is a major improvement. 
There are several reasons for that.
|----First of all, the stochastic outcome of our actions, plus the length of the episodes (6-10 steps on average), 
|       makes it hard for the Cross-entropy method to understand what was done right in the episode and which step was a mistake.
|
|           The value iteration works with individual values of state (or action) and 
|           incorporates the probabilistic outcome of actions naturally, 
|           by estimating probability and calculating the expected value.
|           
|           So, it's much simpler for the value iteration and requires much less data from the environment
|----The second reason is the fact that the value iteration doesn't need full episodes to start learning. !!!
"""

"""
Now it's time to compare the code that learns the values of states, as we just discussed, to the code that learns the values of actions.

q-iteration

and the difference is really minor
    The most obvious change is to our value table.
    |----Now we need to store values of the Q function, 
    |    which has two parameters: state and action, 
    |    so the key in the value table is now a composite.
    |       In the previous example, we kept the value of the state, so the key in the dictionary was just a state.
    |----The second difference is in our calcactionvalue function.
    |       We just don't need it anymore, as our action values are stored in the value table.
    |---- Finally, the most important change in the code is in the agent's value_iteration method.
            Before, it was just a wrapper around the calc_action_value call, 
            which did the job of Bellman approximation. 
            Now, as this function has gone and was replaced by a value table, 
            we need to do this approximation in the value_iteration method.

    As I said, we don't have the calc_action_value method anymore, 
    so, to select action, we just iterate over the actions and look up their values in our values table.

    I don't want to say that V-functions are completely useless, 
    because they are an essential part of Actor-Critic method which we'll talk about in part three of this book.
    
    However, in the area of value learning, Q-functions is the definite favorite.
"""
#!/usr/bin/env python3
import gym
import collections
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v0"
GAMMA = 0.9
TEST_EPISODES = 20


class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(float)

    def play_n_random_steps(self, count):
        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, is_done, _ = self.env.step(action)
            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1
            self.state = self.env.reset() if is_done else new_state

    def select_action(self, state):
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        while True:
            action = self.select_action(state)
            new_state, reward, is_done, _ = env.step(action)
            self.rewards[(state, action, new_state)] = reward
            self.transits[(state, action)][new_state] += 1
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward

    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            for action in range(self.env.action_space.n):
                action_value = 0.0
                target_counts = self.transits[(state, action)]
                total = sum(target_counts.values())
                for tgt_state, count in target_counts.items():
                    reward = self.rewards[(state, action, tgt_state)]
                    best_action = self.select_action(tgt_state)
                    action_value += (count / total) * (reward + GAMMA * self.values[(tgt_state, best_action)])
                self.values[(state, action)] = action_value


if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-q-iteration")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        agent.play_n_random_steps(100)
        agent.value_iteration()

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

