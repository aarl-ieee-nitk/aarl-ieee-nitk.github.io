---
layout: post
title:  "Understanding Policy Gradients"
date:   2019-12-26 22:19:00 +0530
categories: reinforcement-learning, policy-gradients
---

# Understanding Policy Gradients
In reinforcement learning, we have an agent interacting with the environment. The agent acts, the environment gives feedback in the for of rewards. The agent wants to maximize the total rewards it gets.

The goal goal of policy gradients: make an agent (a Neural Network, to be concrete) that maximizes its expected return. To be precise, the agent must: for every state it might be in, assign a probability distribution for actions it might take, so that it gets the maximum possible return on average from then onwards.

## Vanilla Policy Gradient (REINFORCE)
If we want the agent to get the maximum return on average, a natural first step is to directly optimize the policy(a NN that takes in state and outputs action probs).

Find Ө to maximize Utility

$$\mathop{maximize}_{\theta}\ \mathop{\mathbb{E}}_{\tau \char`\~ \pi_{\theta}}[R(\tau)])$$

$$(Utility\ U(\pi_{\theta}) =\mathop{\mathbb{E}}_{\tau \char`\~  \pi_{\theta}}[R(\tau)]) $$

Let $$\inline \pi_{\theta}$$ be our policy network. We can do gradient _ascent_ on $$\inline U(\pi)$$, wrt the policy net’s parameters, taking small steps in a direction which is likely to increase the return

$$\theta_{k+1} \leftarrow \theta_{k} + \eta \nabla_{\theta} U(\pi_{\theta})$$

We need the gradient term to be in a more useful form:

$$\nabla_{\theta} U(\pi_{\theta})\ = \nabla_{\theta}\mathop{\mathbb{E}}_{\tau \char`\~ \pi_{\theta}}[R(\tau)])$$

Expanding the expectation,

$$\nabla_{\theta} U(\pi_{\theta})\ = \nabla_{\theta} \sum_{\tau} P(\tau)R(\tau)$$

We can take the gradient inside the integral, since the limits don't depend on $$\inline \theta$$:

$$\nabla_{\theta} U(\pi_{\theta})\ =\sum_{\tau}\nabla_{\theta} \Big (P(\tau)R(\tau) \Big )$$

Returns don't depend on $$\inline \theta$$. Also, $$\inline P(\tau) = P(s_0) \prod_t P(s_{t+1}|s_t, a_t) \pi_{\theta}(a_t|s_t)$$.

Here's an expression containing products. But differentiating long products is hard, so we use the log-derivative trick: $$\inline {d \over dx} log(f(x)) = {1 \over f(x)}{d \over dx} f(x)$$

$$\nabla_{\theta} U(\pi_{\theta})\ = \sum_{\tau}R(\tau) P(\tau)\nabla_{\theta} log P(\tau) $$

log of product is sum of logs:

$$\nabla_{\theta} U(\pi_{\theta})\ = \sum_{\tau}R(\tau) P(\tau) \Big ( \sum_t \nabla_{\theta} log \pi_{\theta}(a_t | s_t) \Big  \ + \ \nabla_{\theta}\rho_{0}(s_{0}) \sum_t \nabla_{\theta} log P(s_{t+1} | s_t, a_t) ) $$

Getting rid of gradient terms which don't depend on $$\inline \theta $$, and converting back to an expectation, we get an expression that we can use:

$$\nabla_{\theta} U(\pi)\ = \mathop{\mathbb{E}}_{\tau} \Big[\sum_{t=0}^{H-1} R(\tau) \nabla_{\theta}\ log\ \pi (a_{t}|s_{t})\Big] $$

Since this is an expectation, we can estimate it with SGD. If we have $$\inline m$$ samples, 

$$\nabla_{\theta} U(\pi)\ \approx {1 \over m}\sum_{i=0}^{m-1}\sum_{t=0}^{H-1} R(\tau) \nabla_{\theta}\ log\ \pi (a_{t}|s_{t})$$


In practice, this performs poorly, which I think is for two main reasons:

#### 1. Multiplying every gradient term in a trajectory $$\inline R(\tau)$$ doesn’t discriminate between good harmless actions and bad actions which actually led to bad reward.

R(T) is the sum of rewards which came _after_ the action was taken. So we are trying to form a cause-effect relationship between actions and rewards.
This kind of return is called **_return-to-go_**. Let's denote the returns recieved in trajectory $$\inline \tau$$ after and including time $$\inline t$$ with $$\inline RTG(\tau, t)$$


#### 2. Returns can have high variance. 
Say, we're in a state where there are 3 possible actions from state $$\inline s$$

action $$\inline a_{1}$$ : return -10

action $$\inline a_{2}$$ : return -60

action $$\inline a_{3}$$ : return -80

and our agent takes action $$\inline a_{1}$$

The agent gets a negative return of -10, so our basic algorithm will end up _discouraging_ this action. But this is not the correct thing to do! $$\inline a_{1}$$ is the best action we can take if we end up in this state, so the algorithm _should encourage_ it. Gotta make lemonade when life gives us lemons, right?

We need to somehow find if an action was _better than average_ among all actions in that state. **_Baselines_** help us achieve this. Instead of using $$\inline R(\tau)$$ as weights, we use $$\inline R(\tau) - b$$. Here,  $$\inline b$$ can be any function, the only constraint is it shouldn’t depend on $$\inline \theta$$. (If it did, the gradient wouldn’t remain the same)

So our gradient term becomes: 

$$\nabla_{\theta} U(\pi)\ \approx {1 \over m}\sum_{i=0}^{m-1}\sum_{t=0}^{H-1} \Big(RTG(\tau, t) - b(s)\Big) \nabla_{\theta}\ log\ \pi (a_{t}|s_{t})$$

Some common choices for b:
1. Mean return for all $$\inline (s, a) $$ in the current batch. This isn't the best option for b, but this already performs better and is more stable than the basic version. For a sample implementation of this, check out [this notebook on colab](https://colab.research.google.com/drive/1KwHIlm7xesrm3wg0ks-TlvHy09br4PAk)
2. Time based: $$\inline b(s_{t}) = average\ returns\ after\ s_{t}$$ (in the current batch)
3. Learned baselines: $$\inline b(s_{t}) = V_{\phi}(s_{t}) $$ (these kinds have a separate critic, which is, in a sense, evaluating the agent’s actions. Discussed next)


## Actor-Critic methods


These are policy gradient methods where the baseline is also learned.
Two networks: Critic $$\inline V_{\phi}(s)$$ and Actor $$\inline \pi_{\theta}(a|s)$$
Both may be separate networks, or may share initial layers and branch out at a later layer.

Critic: did this state turn out more rewarding than I expected?

Actor: did this action send us into a better situation than we thought it would?

If we use this value function as a baseline, our gradient term becomes: 

$$\nabla_{\theta} U(\pi)\ \approx {1 \over m}\sum_{i=0}^{m-1}\sum_{t=0}^{H-1} (RTG(\tau, t) - V(s_{t})) \nabla_{\theta}\ log\ \pi (a_{t}|s_{t})$$

There's something to notice here. For the gradient, along with the grad log probability term, we are using a _weight_ multiplier, to indicate whether we want to increase (+ve weight) or decrease (-ve weight) the probability of that action in that state.

The notion of “better than average” is captured nicely by the _Advantage_ function:

$$A(s, a) = Q(s, a) - V(s)$$



$$\nabla_{\theta} U(\pi)\ \approx {1 \over m}\sum_{i=0}^{m-1}\sum_{t=0}^{H-1} (Q(s_{t}^{(i)}, a_{t}^{(i)}) - V(s_{t}^{(i)})) \nabla_{\theta}\ log\ \pi (a_{t}^{(i)}|s_{t}^{(i)})$$

Using Q-values instead of raw returns helps to reduce variance, since the Q-value captures information from multiple branches following this $$\inline (s, a)$$ choice, whereas $$\inline R(\tau)$$ only gives information about one path.

Now the problem is, we don't have an explicit Q-network, so we'll need to estimate the Q-values somehow. One simple way to estimate A is using the _TD error_:

$$A(s_{t}, a_{t}) = r_{t} + V(s_{t+1}) - V(s_{t})$$

The same TD error serves two purposes: as the advantage function for the actor, and as the value error for the critic.

The A3C paper takes A(s, a) as

$$A(s_t, a_t) = [r_t + \gamma ^1r_{t+1} + … + \gamma ^kr_{t+k} + V(s_{t+k+1})] - V(s_t)$$

(for some fixed k, or until terminal state, whichever comes first. $$\inline \gamma$$ is the discount factor)

### Sources
* [Spinning up in Deep RL by OpenAI](https://spinningup.openai.com)
* [Deep RL Bootcamp by Deepmind](https://sites.google.com/view/deep-rl-bootcamp/lectures)
* [Daniel Takeshi's blog](https://danieltakeshi.github.io/). Specifically, [this post](https://danieltakeshi.github.io/2017/03/28/going-deeper-into-reinforcement-learning-fundamentals-of-policy-gradients/) and [this post](https://danieltakeshi.github.io/2018/06/28/a2c-a3c/)
* [Lilian Weng's blog](https://lilianweng.github.io/). [This post](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html) in particular.

_(I plan on adding more algorithms as soon as I understand them a bit better. TRPO, PPO, SAC, DDPG, etc.)_

