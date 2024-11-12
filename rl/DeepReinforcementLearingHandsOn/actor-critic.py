"""
In particular, we focused on the method called REINFORCE and its modification that uses a discounted reward to obtain the gradient of the policy (which gives us the direction to improve the policy). 

Both methods worked well for a small CartPole problem, but for a more complicated Pong environment, the convergence dynamic was painfully slow.

In this chapter, we'll discuss one more extension to the vanilla Policy Gradient (PG) method, which magically improves the stability and convergence speed of the new method.
    Despite the modification being only minor, the new method has its own name, Actor-Critic, and it's one of the most powerful methods in deep Reinforcement Learning (RL).

Variance reduction
|   In the previous chapter, we briefly mentioned that one of the ways to improve the stability of PG methods is to reduce the variance of the gradient.
|   Now let's try to understand why this is important and what it means to reduce the variance.
|   |----In statistics, variance is the expected square deviation of a random variable from the expected value of this variable.
|   |       Var[x] = E[(x - E[x])^2]
|   |----Variance shows us how far values are dispersed from the mean.
|           When variance is high, the random variable can take values deviated widely from the mean.
|           On the following plot, there is a normal (Gaussian) distribution with the same value of mean = 10, but with different values for the variance.
|               ~J~!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!~Y
|               ~!                                                                                                                                                                                                     ?
|               ~!                                                  .~!~............!... .:  7!!^!...!:..~~7:.. .^~~^.:..:. ... .!^~:7:  :J:.?^.:... .:.  ..  ~7 ~!~                                          .....    ?
|               ~!                                                  ^P~Y!YP!J7?JY~57Y!YP7Y?7 P~JJ575!G~P~Y?P?JJ~YJY?JY7JJ7Y:^Y5Y?Y7P~57? !GYY5?5Y7JP?Y!Y ^??: ~P^5!Y                                          ~^!.^:   ?
|               ~!                                                   :^~:~~:~~^^~:~^^:~~:^:^ ~^^.^:~:^:^ ^:^^:^~^:^^:^^:^.^. ~^~.^:^:^.^ :^^^::^^:~~^^.^  ..  ^~:^~:                                          ~^~~^    ?
|               ~!                                                                                                                                                                                            ....     ?
|               ~! :^.:^^.                                                                                                                                                                                             ?
|               ~! :~:^^~: .:.........................................................................................................................................................                                 ?
|               ~!          :                                                                                                                                                                                 ^^^:^    ?
|               ~!          :                                                                                                                                                                                 :!^~!.   ?
|               ~!          :                                                                                                                                                                                  :.::.   ?
|               ~!          :                                                                                                                                                                                          ?
|               ~!  .~~:!. .^............................................................................~............................................................................                                 ?
|               ~!   :..:.  :                                                                           !?~                                                                                                   .:::.    ?
|               ~!          :                                                                          ^!.7:                                                                                                  ^^~^~    ?
|               ~!          :                                                                         .7. :7.                                                                                                  .:::    ?
|               ~!          :                                                                         7:   ~!                                                                                                          ?
|               ~! .:.:::.  :                                                                        ~!     !^                                                                                                         ?
|               ~! ^~:^~~: .^.......................................................................:?:.....:?........................................................................                        .: .:    ?
|               ~!          :                                                                       ^!       7:                                                                                               :.^.:    ?
|               ~!          :                                                                       !^       ~~                                                                                               ::..^    ?
|               ~!          :                                                                       7.       :7                                                                                                        ?
|               ~!          :                                                                      .7         7.                                                                                                       ?
|               ~!  .~^.!. .^......................................................................~!.........!~......................................................................                        :::^.    ?
|               ~!   ::.^.  :.                                                                  ...7^ .       ^7                                                                                              !^~7~.   ?
|               ~!          :                                                                     .?.         .?.                                                                                             ::!~~:   ?
|               ~!          :                                                                     ^!           !^                                                                                               ...    ?
|               ~!          :                                                                     !^           ^!                                                                                                      ?
|               ~!  . ...   :                                                                     7.           .7.                                                                                                     ?
|               ~! ^!:^~!^ .^....................................................................:7.............7^....................................................................             ..             .    ?
|               ~! .......  :                                                                    ~~             ^~                                                                         :~~~~~~^~^!!:~7!~~~~.~.:~   ?
|               ~!          :                                                                    7:             .7                                                                                   .. .......    .   ?
|               ~!          :                                                                   .?.              7:                                                                               .........     . ..   ?
|               ~!          :                                                                   ^!               !~                                                                        .:::::::~^!!:~7!~~~~.^.^~   ?
|               ~!  .^:.^.  :.                                                                . !^ .   .....     ^7                                                                                                    ?
|               ~!  .^^:~. .^...................................................................7....:^^^:^^^:....7:..................................................................     .................... . ::   ?
|               ~!          :                                                                  :!  :^:       :^:  !:                                                                       :~^^^^~:~^!!:~7~~~~~.^.^!   ?
|               ~!          :                                                                  !^.^:           :^:^~                                                                                                   ?
|               ~!          :                                                                 .?~^.              ^~7                                                                                                   ?
|               ~!          :                                                                 :7:                 :7:                                                                                                  ?
|               ~! ^!:^^!: .^................................................................:7~...................!!:................................................................                                 ?
|               ~! .:.:::.  :                                                               :^7:                   :7^:                                                                                                ?
|               ~!          :                                                              ^^:7                    .7.^:                                                                                               ?
|               ~!          :                                                             ^^ ~~                     !^ ^^                                                                                              ?
|               ~!          :                                                           .~: .?.                     :7  :^                                                                                             ?
|               ~!   :. .   :                                                          .~:  ~!                       7^  :~.                                                                                           ?
|               ~!  .~~.~. .^.........................................................:~:...7:.......................^7...:~:.........................................................                                 ?
|               ~!          :                                                        :^.   ^!                         7:   .^:                                                                                         ?
|               ~!          :                                                       :^  ...7~:^^^^^^^^^^^^^^^^^^^^^^^:~!...  ^:                                                                                        ?
|               ~!          :                                                   ..:~!^^^^^~?:::.....          ......:::?~^^^^^!~:..                                                                                    ?
|               ~!          :                                           ...:^^~^^~!^..    !^                           ~~    ..:!~^^~^^:...                                                                            ?
|               ~! :~:^~!: .^.....................................:::^~~~~^^::..^^.......^7............................:7:.......^^..::^^~~~~^:::.....................................                                 ?
|               ~! :^.::^.  :                             ..::^~~^^^:...     .:^:       :7.                             :7:       :^:      ...:^^~~~^::..                                                              ?
|               ~!          :                      ..::^~~^^^:..           .:^:        .7.                               :7.        :^:            ..:^^^~~^::..                                                       ?
|               ~!          :             ..::^^^~~^^::..                .:^:         .7:                                 :7.         :^:.                ..::^^~~^^^::..                                              ?
|               ~!          : ...:::^^^~~~^^::...                    .::^:.         .^!:                                   ^!:.         .:^::.                    ...::^^~~~^^^:::...                                  ?
|               ~!      ..  : :^^:::....                 .......:::^^::.     ....::~~^.                                     .^~~::....     ..:^^:::.......                  ...:::^^:                                  ?
|               ~!     :!^ .^..:::::::::::::::::::::::^:::^^^~!~~~^^^^^^^^^^^~^~~~^^:...:...:..::..::..:..::..::..::..:...:..:::^~~~^~^^^^:^^^^^~~~~!~^^^:::::::::::::::::::::::::::.:.                                ?
|               ~!      ..    .              .      .              .       .       .      .      ..     . .    .       . .    .         .    . .       .      ..    . .      ..    . .                                 ?
|               ~!           :!:     ^^     .!.     ~~     .!:     ~~     ^7.     ^~     :?.     !!     ~~!:   ^^~:    ~:!.   :^~~    ~^!:   :~~~    ~^7:   :^^^    ~^?:   :~!~   .!~!.                                ?
|               !!            .      ..      .      ..      .      ..      .              .      ..     ...    ....    ...     ..     ...     ...    ...     .      ...     ..     ...                                 ?
|               ~?~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~J
|
|----subtracted the mean reward from the gradient scale
     PG--the method's idea is to increase the probability of good actions and decrease the chance of bad ones.
     In math notation, our PG was written as \nabla J \approx \mathbb{E} \left[ Q(s, a) \nabla \log \pi(a|s) \right]
         The scaling factor Q(s, a) specifies how much we want to increase or decrease the probability of the action taken in the particular state.
             In the REINFORCE method
                 we used the discounted total reward as the scaling of the gradient. 
                 As an attempt to increase REINFORCE stability, we subtracted the mean reward from the gradient scale.
                     To understand why this helped, let's consider the very simple scenario of an optimization step on which we have three actions with different total discounted rewards: Q1, Q2, and Q3.
                         Now let's consider the policy gradient with regard to the relative values of those Qs.
                         As the first example, let both Q1 and Q2 be equal to some small positive number and Q3 be a large negative number. 
                             So, actions at the first and second steps led to some small reward, but the third step was not very successful. 
                             The resulted combined gradient for all three steps will try to push our policy far from the action at step three and slightly toward the actions taken at step one and two, which is a totally reasonable thing to do.
     
                         Now let's imagine that our reward is always positive, only the value is different.
                             This corresponds to adding some constant to all Q1, Q2, and Q3. 
                             In this case, Q1 and Q2 become large positive numbers and Q3 will have a small positive value.
                             However, our policy update will become different! 
                                 Next, we'll try hard to push our policy toward actions at the first and second step, and slightly push it towards an action at step three
                             So, strictly speaking, we're no longer trying to avoid the action taken for step three, despite the fact that the relative rewards are the same.
     
                         This dependency of our policy update on the constant added to the reward can slow down our training significantly, as we may require many more samples to average out the effect of such a shift in the PG. 
     
                         Even worse, as our total discounted reward changes over time, with the agent learning how to act better and better, our PG variance could also change.
     
                         To overcome this, in the previous chapter, we subtracted the mean total reward from the Q-value and called this mean baseline.
     
                         This trick normalized our PGs, as in the case of the average reward being -21, getting a reward of -20 looks like a win to the agent and it pushes its policy towards the taken actions.
                         
                         To check this theoretical conclusion in practice, let's plot the variance of the PG during the training for both the baseline version and the version without the baseline. 
"""

"""
It now accepts the command-line option --baseline, which enables the mean subtraction from the reward. By default, no baseline is used.
On every training loop, we gather the gradients from the policy loss and use this data to calculate the variance.
"""
#!/usr/bin/env python3
import gym
import ptan
import argparse
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 8

REWARD_STEPS = 10


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default=False, action='store_true', help="Enable mean baseline")
    args = parser.parse_args()

    env = gym.make("CartPole-v0")
    writer = SummaryWriter(comment="-cartpole-pg" + "-baseline=%s" % args.baseline)

    net = PGN(env.observation_space.shape[0], env.action_space.n)
    print(net)

    agent = ptan.agent.PolicyAgent(net, preprocessor=ptan.agent.float32_preprocessor,
                                   apply_softmax=True)
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
        baseline = reward_sum / (step_idx + 1)
        writer.add_scalar("baseline", baseline, step_idx)
        batch_states.append(exp.state)
        batch_actions.append(int(exp.action))
        if args.baseline:
            batch_scales.append(exp.reward - baseline)
        else:
            batch_scales.append(exp.reward)

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
        We calculate the policy loss as before, by calculating the log from the probabilities of taken actions and multiply it by policy scales 
        (which are the total discounted reward, if we're not using the baseline or the total reward minus the baseline).
        """
        optimizer.zero_grad()
        logits_v = net(states_v)
        log_prob_v = F.log_softmax(logits_v, dim=1)
        log_prob_actions_v = batch_scale_v * log_prob_v[range(BATCH_SIZE), batch_actions_t]
        loss_policy_v = -log_prob_actions_v.mean()

        """
        we ask PyTorch to backpropagate the policy loss, calculating the gradients and keeping them in our model's buffers
        One tricky thing here is the retain_graph=True option when we called backward(). 
            It instructs PyTorch to keep the graph structure of the variables.

        However, our parameter update should take into account not only policy gradient but also the gradient provided by our entropy bonus. 
        To achieve this, we calculate the entropy loss and call backward() again. 
        To be able to do this the second time, we need to pass retain_graph=True.
        about entropy loss, see policygradients.py
        """
        loss_policy_v.backward(retain_graph=True)
        """
        In general, retaining the graph could be useful when we need to backpropagate loss multiple times before the call to the optimizer.
        """
        grads = np.concatenate([p.grad.data.numpy().flatten()
                                for p in net.parameters()
                                if p.grad is not None])

        prob_v = F.softmax(logits_v, dim=1)
        entropy_v = -(prob_v * log_prob_v).sum(dim=1).mean()
        entropy_loss_v = -ENTROPY_BETA * entropy_v
        entropy_loss_v.backward()
        optimizer.step()

        loss_v = loss_policy_v + entropy_loss_v

        # calc KL-div
        new_logits_v = net(states_v)
        new_prob_v = F.softmax(new_logits_v, dim=1)
        kl_div_v = -((new_prob_v / prob_v).log() * prob_v).sum(dim=1).mean()
        writer.add_scalar("kl", kl_div_v.item(), step_idx)

        writer.add_scalar("baseline", baseline, step_idx)
        writer.add_scalar("entropy", entropy_v.item(), step_idx)
        writer.add_scalar("batch_scales", np.mean(batch_scales), step_idx)
        writer.add_scalar("loss_entropy", entropy_loss_v.item(), step_idx)
        writer.add_scalar("loss_policy", loss_policy_v.item(), step_idx)
        writer.add_scalar("loss_total", loss_v.item(), step_idx)

        writer.add_scalar("grad_l2", np.sqrt(np.mean(np.square(grads))), step_idx)
        writer.add_scalar("grad_max", np.max(np.abs(grads)), step_idx)
        writer.add_scalar("grad_var", np.var(grads), step_idx)

        batch_states.clear()
        batch_actions.clear()
        batch_scales.clear()

    writer.close()

"""
As you can see, variance for the version with the baseline is two-to-three orders of magnitude lower than the version without one, 
which helps the system to converge faster.

|----Actor-critic
"""
