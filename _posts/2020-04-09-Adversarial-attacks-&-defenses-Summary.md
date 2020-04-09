---
layout: post
title:  "Adversarial attacks and defenses in Reinforcement Learning - a Summary"
author: "Videh Raj Nema"
author_github: vrn25
github_username: 'vrn25'
date:   2020-04-9 18:10:45 +0530
categories: reinforcement-learning, adversarial attacks, defense mechanisms
---

## Adversarial attacks and defenses in Reinforcement Learning

To a large extent, robustness in RL depends upon the robustness of the function approximator network which is used to output the Q-values or the probability of taking various actions in a state (policy). The most common function approximators which are used in RL are **Neural Networks** (NN). Hence, the research works which aim to exploit the vulnerabilities and explore some potential defenses for neural nets in supervised learning are also applicable in RL.

Apart from that, there are few other attacks that are specifically applicable in RL settings. We'll see attacks in both the settings.

Broadly there are two types of attacks -


### Pixel-based attacks 

These types of attacks change the input observation of the RL function approximator (NN in our case). Assuming the input to the neural network is in the form of pixels, we make carefully calculated perturbations to the pixels, so that the RL agent picks an incorrect action, which might lead the agent to a **bad** state. The perturbation is made such that the change in the input image is imperceptible to humans. This type of attack was initially introduced in Computer Vision settings and later extended to RL. It tends to work for RL because the attack exploits the vulnerabilities in general ML models which are used to approximate functions like neural nets and other linear models.

### Attacking by training an adversarial policy

This type of attack has been introduced recently. The adversary trained an adversarial policy for the opponent which took certain actions that generated natural observations that were adversarial to the victim policy. This attack is discussed in detail later.

## Pixel-based attacks

These types of attacks focus on attacking the vulnerabilities in the function approximator. Most of these attacks are studied in Computer Vision but can be applied in a similar fashion in Reinforcement Learning, where the state-space is the pixels. The papers follow different approaches, but a common goal. The goal is to -

*  Find the input features that are most sensitive to class change (change in the action in RL).
 
*  Make a perturbation, i.e, change the pixels slightly such that the network misclassifies the input but the change is visually imperceptible to humans.

In short, they focus on making small $L_p$ norm perturbations to the image. Also, all the attacks that we discuss do not tamper with the training process in any way. All these attacks are made during the test time.

### [INTRIGUING PROPERTIES OF NEURAL NETWORKS](https://arxiv.org/pdf/1312.6199.pdf) (Szegedy et al.)

This is probably the first paper discovering such attacks in the context of neural networks. The authors generate adversarial examples using **box-constrained L-BFGS**. They provide results on MNIST and ImageNet datasets. They argue that adversarial examples represent low probability pockets in the manifold represented by the network, which are hard to efficiently find by simply randomly sampling the input around a given example. They also found that the same adversarial example would be misclassified by different models trained with different hyperparameter settings, or trained on disjoint training sets. They refer to it as the **transferability** property of adversarial examples.

However, the authors do not provide a concrete explanation of why these adversarial examples occur. If the model is able to generalize well, how can it get confused with the inputs that look almost similar to the clean examples. They say that the adversarial examples in the input space are like rational numbers in the real number space, *sparse yet dense*. They argue that an adversarial example has very less probability (hence rarely occurs in the test set), and yet dense because we can find an adversarial example corresponding to almost every clean example in the test set.

### [EXPLAINING AND HARNESSING ADVERSARIAL EXAMPLES](https://arxiv.org/pdf/1412.6572.pdf) (Goodfellow et al.)

This paper proposes a novel way of creating adversarial examples very fast. They propose **Fast Gradient Sign Method** (FGSM) to harness adversarial samples. This method works by taking the sign of the gradient of the loss with respect to the input pixels. They use the max norm constraint in which each pixel is changed by an amount no more than $\epsilon$. This perturbation is then added to or subtracted from each pixel (depending upon the sign of the gradient). This process can be repeated until the distortion becomes human-recognizable or the network misclassifies the image.

FGSM works as a result of the hypothesis the authors propose for the occurence of adversarial examples. Earlier, it was thought that adversarial examples are a result of overfitting and the model being highly non-linear. However, empirical results showed something else. If the above notion of overfitting was true, then we should get different random adversarial example points in the input space if we retrain the model with different hyperparameters or train a slightly different architecture model. **But it was found that the same example would be misclassified by many models and they would assign the same class to it**.

Following is a paragraph from the paper.


"""
*Why should multiple extremely non-linear models with excess capacity consistently label out-of-distribution points in the same way? This behavior is especially surprising from the view of the hypothesis that adversarial examples finely tile space like the rational numbers among the reals, because in this view adversarial examples are common but occur only at very precise locations*.

*Under the linear view, adversarial examples occur in broad subspaces. The direction η need only have a positive dot product with the gradient of the cost function, and need only be large enough.*
"""


So the authors proposed a new hypothesis for the cause of adversarial examples to be these models being highly linear or *piecewise linear functions of the inputs*, and extrapolating in a linear fashion, thus exhibiting high confidence at the points it has not seen in the training set. The adversarial examples exist in broad subspaces of the input space. The adversary need not know the exact location of the point in the space, but just needs to find a direction (gradient of the cost function) giving a large positive dot product with the perturbation. In fact, if we take the difference between an adversarial example and a clean example, we have a direction in the input space and adding that to a clean example would almost always result in an adversarial example.

In summary, this paper gives a novel method of generating adversarial examples very fast and gives a possible explanation for existence and generalizability of adversarial examples.

Another explanation for FGSM in simple words is -

In Machine Learning, we use the gradient of the cost function to optimize the parameters of the model to minimize the cost keeping the input to the model fixed, whereas, 

For generating adversarial examples, we do the opposite. We try to maximize the cost by moving the input values in the direction of gradient of the cost function, keeping the parameters of the model fixed.

**This applies to every linear model and not just deep neural networks.** The paper also suggests a tradeoff between the robustness of the network and the ease of training it.

### [ADVERSARIAL ATTACKS ON NEURAL NETWORK POLICIES](https://arxiv.org/pdf/1702.02284.pdf) (Huang et al.)  

This is the first paper showing the existence of adversarial attacks in Reinforcement Learning. Here they show attacks on three RL algorithms - DQN, TRPO, and A3C in Atari environments in both White-box and Black-box settings. The approach followed is the same as FGSM (as the state input to the neural network function approximator are raw pixels), and they do it with different $L_p$ norms. Here the loss is calculated as the cross-entropy loss function between the output action probabilities and the action with highest action probability, in case of policy-based methods. For DQN, the output Q-values are converted to a probability distribution using the softmax function with temperature. These attacks are able to decrease the rewards per episode by a considerable amount by making the RL agent take the wrong action in certain situations.

The attack is similar to the gradient attack in Computer Vision (as discussed above). There the objective was to make the network misclassify the image, while here the same thing holds with actions. The authors provide the results for white-box (i.e, the adversary has access to the training environment, neural network architecture, parameters, hyperparameters, and the training algorithm) as well as black-box settings. The latter is divided into two parts -

*  Transferability across policies - Adversary has all the knowledge as in the case of white-box attacks except the random initialization of the target policy network.
    
*  Transferability across algorithms - Adversary has no knowledge of the training algorithm or hyperparameters.

Some important conclusions from this paper -

* The attack methods like FGSM (and ones further we will discuss) are applicable in RL domains also. This is because the root cause of these attacks is not related to any specific field, instead they apply to all the models in Machine Learning which are linear or literally piecewise linear functions of the inputs. Neural Networks come under this category, hence are vulnerable to them. Since RL can only be applied in real-world problems with continuous state and action spaces using these function approximators, these attacks can also confuse well-trained RL agents which take state input in the form of pixels.
    
 * DQN tends to be more vulnerable to adversarial attacks as compared to policy gradient methods like TRPO and A3C.
 
### [THE LIMITATIONS OF DEEP LEARNING IN ADVERSARIAL SETTINGS](https://arxiv.org/pdf/1511.07528.pdf) (Papernot et al.)

This paper proposes a slightly different method to craft adversarial samples. Unlike [Goodfellow et al](https://arxiv.org/pdf/1412.6572.pdf). this method does not alter all the pixels by equal amounts. The adversaries craft the examples by carefully finding the input dimensions changing which will have the most impact on class change. Hence this attack fools the network in minimum required perturbation only to the pixels that can cause class change with higher probability.

Here, instead of computing the gradient of the loss by backpropagation, the adversary finds the gradient of the DNN output with respect to the input as a forward derivative, i.e, the Jacobian matrix of the function learned by the DNN by recursively computing the gradient for each layer in a forward run. The authors do this as this enables finding input components that lead to significant changes in the network outputs. This is followed by constructing adversarial saliency maps which act as heuristics to estimate the pixels to perturb.

One important thing to note in this approach is that unlike [Goodfellow et al.](https://arxiv.org/pdf/1412.6572.pdf), it implicitly enables source-target misclassification, i.e, forcing the output classification of a specified input to be a specific target class.

The reason for this being that here we compute the Jacobian of the function which the network is trying to learn. This derivative (Jacobian) basically tells how much an output neuron changes if we change an input pixel, which gives us a direction in which we can increase or decrease the probability of any output class. Hence we can increase or decrease an input pixel in such a way that increases the probability of a certain target class and decreases the probabilities of others. This gives a stronger advantage to the adversary in not only making the DNN misclassify the input, but also leading it to a target class. Hence the adversary can have a significant impact when applying this attack in RL as it is a closed loop problem.

### [TACTICS OF ADVERSARIAL ATTACK ON DEEP REINFORCEMENT LEARNING AGENTS](https://www.ijcai.org/Proceedings/2017/0525.pdf) (Lin et al.)

Here the authors argue that the previous work for adversarial attacks in Deep RL ([Huang et al.](https://arxiv.org/pdf/1702.02284.pdf)) ignores certain important factors. The attack needs to be minimal in both spatial as well as temporal domains and previous work says nothing about minimizing the number of timesteps in which attack is made. Hence they give two novel tactics for attacking Deep RL.

#### Strategically-timed attack

This attack accounts for the minimal total temporal perturbation of the input state. It is based on a simple observation. The adversary should attack the RL agent only in critical situations where the agent has a very high probability of taking a particular action. A simple analogy can be given in case of Pong. At the time instants when the ball is away from the agent and moving towards the adversary, intuitively, the probability of the agent taking any action should be more or less uniform. Making a perturbation at these timesteps does not make sense.

Instead, the timesteps when the ball is very near to the agent padel, its probability of taking a certain action is quite high. Hence attacking at this time instant is likely to provoke the agent to take another action, thereby missing the ball and getting a negative reward.

The authors make this attack by framing a mixed integer programming problem which is difficult to solve. Instead they use a heuristic to measure in which time steps to attack. They measure the difference between the action probabilities of the most preferred and the least preferred action. If it is more than a certain threshold, then they attack, otherwise not. For DQN, this is calculated by passing Q-values through the softmax function with temperature. Once the difference is more than the threshold, the adversary uses the [Carlini-Wagner](https://arxiv.org/pdf/1608.04644.pdf) method to attack, where the target class is the least preferred action before adding the perturbation.

For limiting the total number of attacks, there is an upper bound to the total number of timesteps when the adversary can attack. Results show that it is possible to reach the same attack success rate as in [Huang et al.](https://arxiv.org/pdf/1702.02284.pdf) while attacking only 25% of the total time steps per episode on average.

#### Enchanting attack
    

This attack is based on the closed loop and sequential property in RL where the current action affects the next state and so on. It lures the agent to a particular state of the adversary’s choice. This is accomplished by using a planning algorithm (to plan the sequence of actions that would ultimately lead the agent to the target state) and a deep generative model (for simulation and predicting the model). In the paper, the authors assume that the adversary can take full control of the agent and make it take any action at any time step.

The first step is to force the agent to take the first planned action in the current state. This is achieved by the [Carlini-Wagner](https://arxiv.org/pdf/1608.04644.pdf) attack to craft the perturbed state leading the agent to a new state from where this process is repeated until it reaches the target state.

This type of attack can be extremely beneficial for the adversary in some cases like autonomous driving where it can lead the car (agent) to a **bad** state like hitting an obstacle and causing an accident. It also seems to be a much more promising and strategic attack than others to allow the adversary to exploit the agent to its fullest potential.

## Attacking by training an Adversarial Policy

### [ADVERSARIAL POLICIES: ATTACKING DEEP REINFORCEMENT LEARNING](https://arxiv.org/pdf/1905.10615.pdf) (Gleave et al.)
```
Note: Related Work section of this paper contains many other existing approaches
```
This paper follows a completely different approach towards attacking well-trained RL agents. Unlike an indirect approach of attacking Deep RL by exploiting the vulnerabilities in the DNN, this paper trains an adversarial policy in competitive multi-agent zero-sum environments directly to confuse/fool the agent (also referred to as the victim), thereby decreasing the positive reward gained. The paper is able to successfully attack the state-of-the-art agents trained via self-play to be robust to opponents, especially in high-dimensional environments, despite the adversarial policy trained for less than 3% of the time steps than the victim and generating seemingly random behavior.

Let us look at a detailed analysis of the paper -

Previous works on adversarial attacks in Deep RL have used the same image perturbation approach which was initially proposed in the context of CV. However, real-world deployments of RL agents are in natural environments populated by other agents including humans. So a more realistic attack would be to train an adversarial policy that can take actions which generate observations that are adversarial to the victim.

A motivation for this attack from the paper is -


"""
*RL has been applied in settings as varied as autonomous driving (Dosovitskiy et al., 2017), negotiation (Lewis et al., 2017) and automated trading (Noonan, 2017). In domains such as these, the attacker cannot usually directly modify the victim policy’s input. For example, in autonomous driving pedestrians and other drivers can take actions in the world that affect the camera image, but only in a physically realistic fashion. They cannot add noise to arbitrary pixels, or make a building disappear. Similarly, in financial trading an attacker can send orders to an exchange which will appear in the victim’s market data feed, but the attacker cannot modify observations of a third party’s orders.*
"""


The method in this paper seems to be more appropriate for attacking Deep RL in actual deployment domains. A key point here is that the adversary trains its policy in black-box scenario where it can just give observation input to the victim (by its actions) and receive the output from it. **The key difference in this approach is using a physically realistic threat model that disallows direct modifications of the victim’s observations.**

#### Setup

The victim is modelled as playing against an opponent in a two-player Markov game. The opponent is called an adversary when it can control the victim. The authors frame a two-player MDP for this. The adversary is allowed unlimited black-box access to the actions sampled from the victim policy, which is kept as fixed while making an attack. Hence the problem reduces a single-agent game as the victim policy is fixed.

The experiments are carried out in zero-sum robotics game environments introduced in [Bansal et al. (2018a)](https://arxiv.org/pdf/1710.03748.pdf). The environments and the rules for the game are described in the paper. The victim policies are also taken from the pre-trained parameters in [Bansal et al. (2018a)](https://arxiv.org/pdf/1710.03748.pdf), which are trained to be robust to opponents via self-play. The environments used in the paper are - Kick and Defend, You shall not pass, Sumo Humans, and Sumo Ants.

#### Training the adversary

Although [Bansal et al. (2018a)](https://arxiv.org/pdf/1710.03748.pdf) have pre-trained opponents as well which try to fail the victim from doing their tasks in the respective environments, the authors trained a new adversarial policy. They did this using the [PPO](https://aarl-ieee-nitk.github.io/reinforcement-learning,/policy-gradient-methods,/sampled-learning,/optimization/theory/2020/03/25/Proximal-Policy-Optimization.html) algorithm, giving the adversary sparse positive reward when it wins against the victim and negative if it loses or ties. They train it for 20 million time steps which is less than 3% of the time steps the victims were trained.

#### Results and Observations

The adversarial policy succeeds in most environments and performs better than Rand, Zero, and Zoo* opponents. Here an important observation is that the adversarial policy wins against the victims not by becoming strong opponents (i.e, performing the intended actions like blocking the goal) in general, but instead taking actions which generated natural observations that were adversarial to the victim policy. For instance, in Kick and Defend and You shall not pass, the adversary never stands up and learns to lie down in contorted positions on the ground. This confuses the victim and forces it to take wrong actions. In Sumo Humans, if the adversary does the same, it loses the game so it learns an even interesting strategy to kneel down in the center of the sumo ring in a stable position. In the former two environments, the adversary's win rate surpasses all the other types of opponents, and in Sumo Humans, it is competitive and performs close to the Zoo opponents.

The victim does not fail just because the observations generated by the adversary are off its training distribution. This is confirmed by using two off-distribution opponent policies - *Rand* (takes random actions) and *Zero* (lifeless policy exerting zero control). The victim is fairly robust to such off-distribution policies, until they are specifically adversarially optimized.

#### Conclusions from Masked Policies

This is an interesting methodology adopted in the paper to prove that the adversaries win by natural observations adversarial to the victim and activations are very different from the ones generated by normal opponents.

The authors compare the performance of the masked and the normal victims against the adversarial and the normal opponents. Masked victim means that the observation input of the victim is set to some static initial opponent position. One would normally expect the victim to perform badly when the position of the opponent and the actions it takes are “invisible”. This is indeed the case with the normal opponent. The masked victim performs poorly against it. However, the performance of the masked victim against the adversary is seen to improve as compared to the unmasked victim.

This also leads us to an important observation. All these games discussed above are transitive in nature, i.e, a high-ranked professional player should intuitively outperform a low-ranked amateur player. However, here the adversary wins not by physically interfering with the victim or becoming strong opponents, but instead placing it into positions of which the victim is not able to make sense. This suggests highly non-transitive relationships between adversarial policies, victims, and masked victims. This is similar to an out-of-syllabus that is specifically designed to confuse the student.

#### Effects of Dimensionality

In Pixel-based attacks, it has been shown that classifiers with high-dimensional inputs are more vulnerable to adversarial examples. Similar thing is observed here. The authors compare the performance of the adversary in two types of environments in the Sumo game. One with Sumo Humanoid and the other as Sumo Ant quadrupedal robots. In the former case, the adversary has access to $P$ = 24 dimensions of the total observation space that it can influence, while the latter has only $P$ = 15. $P$ is the adversary’s joints position. The damage caused by the adversary in the first case is more than in the second.

#### Why are victim observations adversarial?

The authors plot the activations of the victim policy using [Gaussian Mixture Model](https://towardsdatascience.com/gaussian-mixture-models-explained-6986aaf5a95) (GMM) and [t-SNE](http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) visualization. They observe that the activations induced by adversarial policy substantially differ from the ones induced by normal opponents. More details can be found in the paper.

#### Defenses against Adversarial policies

The authors tried basic Adversarial training as a defense to these policies, i.e, training the victim explicitly against the adversary. The authors use two types of adversarial training - *single* and *dual*. The victim is just trained against the adversary in the former, while against both an adversarial and normal zoo policy randomly picked at the beginning of each episode. However, they found that even if the victim becomes robust to that adversary, they are able to find another adversarial policy which failed the victim. Moreover, the new adversarial policy also transfers to the previous victim (i.e, before adversarial training).

Till now, there is no concrete work in defenses in such attacks and is left to future work. Adversarial training seems to be promising but does not generalize well to all kinds of adversaries. The authors also suggest *population-based training* where the victims are continually trained against new types of opponents to promote diversity, and trained against a single opponent for a prolonged time to avoid local equilibria.

Lastly following is a paragraph from the paper which gives a few important points to keep in mind regarding future work -

"""

*While it may at first appear unsurprising that a policy trained as an adversary against another RL policy would be able to exploit it, we believe that this observation is highly significant. The policies we have attacked were explicitly trained via self-play to be robust. Although it is known that self-play with deep RL may not converge, or converge only to a local rather than global Nash, self-play has been used with great success in a number of works focused on playing adversarial games directly against humans ([Silver et al., 2018](https://deepmind.com/research/publications/general-reinforcement-learning-algorithm-masters-chess-shogi-and-go-through-self-play); [OpenAI, 2018](https://openai.com/blog/openai-five/)). Our work shows that even apparently strong self-play policies can harbor serious but hard to find failure modes, demonstrating these theoretical limitations are practically relevant and highlighting the need for careful testing.*

"""

## Defenses against Adversarial attacks

Many defenses have been proposed to counter these adversarial attacks and make our function approximators robust. However, there is no single defense yet which can successfully guarantee protection against all the attacks introduced till now. Here we discuss two such defenses -

###  Adversarial training  

This, in its most elementary form means training the network/policy by generating adversarial examples. This tends to work upto some extent but is not applicable in all situations. Many other efficient forms of Adversarial training have also been introduced. These can be found in this [paper](https://link.springer.com/content/pdf/10.1186/s42400-019-0027-x.pdf).

An interesting observation is that training the network against these adversarial examples not only overcomes the attack by the adversary, but also helps the network to generalize better. Networks trained using adversarial training are seen to perform better than their previous versions on naturally occuring test sets with no adversarial examples.

### [Distillation as a Defense to Adversarial Perturbations against Deep Neural Networks](https://arxiv.org/pdf/1511.04508.pdf) (Papernot et al.)

Defensive Distillation is one of the prominent defense methods that have been proposed against pixel-based adversarial attacks. It uses the well known method of Distillation. In a way, it does not remove adversarial examples, but helps the model to generalize better and increases the chances of its perception by the human eye.

Defensive Distillation is basically the follow-up work after [Papernot et al.](https://arxiv.org/pdf/1511.07528.pdf)’s Jacobian approach for crafting adversarial examples, which was discussed before. We call it the Jacobian-Based-Saliency-Map (JSMA) attack. First of all let us look at the adversarial samples crafting approach that they have used in the paper. It consists of two steps -

#### Direction Sensitivity Estimation

Here the authors estimate the sensitivity of the input pixels to a class change. The approach can be similar to the one adopted in the JSMA, i.e, using saliency maps or based on FGSM attacks introduced by [Goodfellow et al.](https://arxiv.org/pdf/1412.6572.pdf)

#### Perturbation Selection -

This focuses on finding the appropriate perturbation for the pixels. Some possible approaches for this are as done in FGSM method by changing each pixel by small amounts or in JSMA by carefully selecting sensitive pixels and attacking only those.

#### More on Defensive Distillation

Distillation in its original form is a training procedure enabling deployment of Deep Neural Network (DNN) trained models in resource constrained devices like Smartphones. Defensive Distillation, in contrast to using two neural networks, feeds the soft probability outputs from the DNN to the same network and trains it again from scratch.

#### Training Procedure

First of all, the authors train the input images on a DNN at high temperature. The outputs of this DNN are taken as the soft probability outputs after the softmax layer. The DNN parameters are then reinitialized and it is re-trained with the Y-labels as the soft probability outputs from the first training procedure at high temperature. This network is then called Distilled Network and is robust to adversarial examples. At the test time, the temperature is again set back to 1 to get high-confidence discrete probability outputs.

#### Why does this tend to work?

As discussed in the paper, the intuition behind this procedure working is as follows -

#### Effect of High Temperature 

If we compute the Jacobian of the function that the DNN tries to learn, it comes out to be inversely proportional to the temperature $T$. Hence, training the DNN at high $T$ decreases the sensitivity of the pixels to class change and to some extent smooths out the gradient. This makes it harder for the adversary to make the network misclassify the input by increasing the minimum required distortion.

#### Model now generalizes better

One of the main reasons behind the existence of adversarial examples is the lack of generalization in our models. Defensive Distillation improves on this. In the normal cases we use the network’s output probabilities and the true hard labels to compute the cross-entropy loss for the network. This essentially reduces to computing the negative of the log of the probability corresponding to the true hard label. A necessary thing to note here is that when performing the updates, this will constrain any neuron different from the one corresponding to the true hard label to output $0$. This can result in the model making overly confident predictions in the sample class. In the cases like choice between $3$ and $8$, it is likely that the probabilities for both classes is high and close to $0.5$ (one less and other more). Just modifying a few pixels in these inputs would confuse the network and confidently predict the other wrong class.

Defensive Distillation overcomes this problem by using soft probabilities as labels in the second round. This ensures that the optimization algorithm constrains all the neurons proportionally according to their contribution in the probability distribution. This improves the generalizability of the model outside the training set by avoiding the circumstances where the model is forced to make hard predictions when the characteristics of two classes are quite similar (ex in case of $3$ and $8$, or $1$ and $7$). Hence it helps the DNN to capture the structural similarities between two very similar classes.

Another explanation is that by training the network with the soft labels obtained from the first round, we avoid overfitting the actual training data. However, this does not tend to work if we consider the linearity hypothesis for adversarial examples as proposed by [Goodfellow et al.
](https://arxiv.org/pdf/1412.6572.pdf)

As a keynote, when Defensive Distillation was introduced, it failed all the attacks that existed till then. However, one year later, the stronger [Carlini-Wagner](https://arxiv.org/pdf/1608.04644.pdf) attack was introduced which Defensive Distillation was not able to counter.

There are some key points in the [Carlini-Wagner](https://arxiv.org/pdf/1608.04644.pdf) attack paper which explain the failure of Defensive Distillation -

"""

*The key insight here is that by training to match the first network, we will hopefully avoid overfitting against any of the training data. If the reason that neural networks exist is because neural networks are highly non-linear and have “blind spots” [[46](https://arxiv.org/pdf/1312.6199.pdf)] where adversarial examples lie, then preventing this type of over-fitting might remove those blind spots.*

*In fact, as we will see later, defensive distillation does not remove adversarial examples. One potential reason this may occur is that others [[11](https://arxiv.org/pdf/1412.6572.pdf)] have argued the reason adversarial examples exist is not due to blind spots in a highly non-linear neural network, but due only to the locally-linear nature of neural networks. This so-called linearity hypothesis appears to be true [[47](https://ieeexplore.ieee.org/document/8093865)], and under this explanation it is perhaps less surprising that distillation does not increase the robustness of neural networks.*

"""

## End thoughts

Having seen these attack approaches, a nice explanation of the existence of such attacks is given by the story of **Cleverhans** -

"""

*So Cleverhans was a horse and its owner had trained it to solve arithmetic problems. The owner would say “Cleverhans, what is 1+1?”, and it would tap its feet two times and the crowd would cheer for him or its owner would give it some fruit to eat as a reward. However, a psychologist examined the situation and figured out that the horse actually did not learn to do arithmetic. He discovered that if the horse was placed in a dark room with no audience and a person wearing a mask would ask it the arithmetic problem, it would just continue to tap feet on the floor waiting to receive some signal to stop.*

"""

This is exactly what is happening with our Machine Learning models. They fit to the training and the test set with very high accuracies, but if an adversary intentionally designs some example to fail them, they get fooled. In the story of *Cleverhans*, the horse assumed the correct signal to be the cheerings of the crowd or the fruit that its owner would give when it reached the correct number of taps. This is similar to a Weather predicting ML model. **Instead of knowing that the clouds actually cause rains, it believes that the “presence” of clouds causes rains**.

There is a small difference between the two things, but it is this difference that only fools our ML models. These models associate the occurrence of events but do not know what caused them.

It is not that our models do not generalize to test sets. Like in the story of *Cleverhans*, the model (horse) did generalize to solving the problem correctly in front of the crowd, which indeed is a naturally occuring test set. But it did not generalize to the case when the adversary specifically creates a test case to fail it. More importantly, this test example need not be off the distribution. Like in the story, the scenario created by the psychologist is totally legitimate. Similarly, these adversarial examples are also valid test cases for the model.

Also, it is not that these adversarial examples are a characteristic property of Deep Learning or Neural Networks. The linear nature of our ML models is problematic and because Neural Networks build up on their architecture, they inherit this flaw. Ease of optimization in our models have come at the cost of these intrinsically flawed models that can be easily fooled.

These arguments generalize to both types of attacks we have discussed. In pixel-based attacks, we are able to find adversarial examples that look absolutely legit but still confuse the network. In the adversarial policy attack too, the victim policies were used to seeing certain naturally occuring opponents (test cases) where say in You shall not pass, the opponents would try to push the victim in some way to prevent crossing the line. However, the adversaries fooled the victim by adjusting themselves into non-conventional positions, in which a human would be easily able to win. This leaves us with the following question -

**Are we really close to human-level performance in Machine Learning?**

## Further reading (non-exhaustive)

### [TOWARDS EVALUATING THE ROBUSTNESS OF NEURAL NETWORKS](https://arxiv.org/pdf/1608.04644.pdf) (Carlini-Wagner attack)

### [DEEP NEURAL NETWORKS ARE EASILY FOOLED: HIGH CONFIDENCE PREDICTIONS FOR UNRECOGNIZABLE IMAGES](https://arxiv.org/pdf/1412.1897.pdf) (Nguyen et al.)

### [DeepFool: a simple and accurate method to fool deep neural networks](https://arxiv.org/pdf/1511.04599.pdf)

### [CopyCAT: Taking Control of Neural Policies with Constant Attacks](https://arxiv.org/pdf/1905.12282.pdf)

### [Stealthy and Efficient Adversarial Attacks against Deep Reinforcement Learning](https://yanzzzzz.github.io/files/5040.pdf)

### [Targeted Attacks on Deep Reinforcement Learning Agents through Adversarial Observations](https://arxiv.org/pdf/1905.12282v1.pdf)

Below paper gives a collection of most of the attacks and defenses introduced till now
### [Adversarial attack and defense in reinforcement learning-from AI security view](https://link.springer.com/content/pdf/10.1186/s42400-019-0027-x.pdf)

### [Adversarial Examples: Attacks and Defenses for Deep Learning](https://arxiv.org/pdf/1712.07107.pdf)

### [Adversarial Machine Learning Reading List](https://nicholas.carlini.com/writing/2018/adversarial-machine-learning-reading-list.html)

### [ADVERSARIAL MACHINE LEARNING AT SCALE](https://arxiv.org/pdf/1611.01236.pdf)

### [Adversarial perturbations of Deep Neural Networks](https://ieeexplore.ieee.org/document/8093865)

### [DELVING INTO ADVERSARIAL ATTACKS ON DEEP POLICIES](https://arxiv.org/pdf/1705.06452.pdf) (Kos and Song)

### [On the Effectiveness of Defensive Distillation](https://arxiv.org/pdf/1607.05113.pdf)
