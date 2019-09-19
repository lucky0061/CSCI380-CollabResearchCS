# Title
## Intro
## Background on methods
### Reinforcment learning
It is common practice to use certain language when describing what RL is in machine learning. We must understand what state, reward, environment, policy, and action are in regards to RL. State refers to the progress and data that has been collected at a given point in a learning process. Each step, or iteration bring the program to the next state. The environment is the way that a programmer initializes the problem at hand in the code. There are many different types of machine learning environments. A policy refers to the way that the program applies a reward. The reward is essential how we keep track of the action taken by the program, and if it was productive in the learning process.
In other words, RL is a style of machine learning that requires a reward. We use the reward in RL by assigning a policy that applies to a given environment. An environment is an instance of the problem, or obstacles that you want your program to gain insight on. When the program makes an action oppon a state and that action makes progress towards the end goal, then the reward policy will provide reward. The program will then reference this data for the next step in order to achieve the end goal.
### Meta-RL

Reinforcement Learning (RL) has a tendency to "overfit". It struggles when faced with similar environments and need to completely relearn in order to adjust. Meta Reinforcement Learning (Meta-RL) aims to fix this problem by programming a computer how to learn in more general environments so it does not have to completely relearn. For example, say we have a two-armed bandit, where one arm gives a reward and the other doesn't. RL is easily able to learn which hand gives the reward, but will fail if we introduce it to another two-armed bandit where the rewarding arm has switched. It had grown accustumed to pull a specific arm rather than considering the possibility that the situation had changed. Meta-RL would learn how to approach a general two-armed bandit problem rather than a specific one in order to adjust when the bandit is different.

### Quantum Computing
## Our Method
### Meta-RL + QNN
## The problem / testing our method
### Stock market problem

The environment we are testing our algorithm on is trading on the stock market. The goal of our method is to train our model on one stock to maximize our returns. Then after our algorithm is trained on one stock, we want to see if it can generalize quickly on a stock that is similar to the one we trained on. For example, if we trained on the stock GOOGL and got good returns from our model, we should be able to run our model on a similar stock, say another stock in the same technology industry, like AMZN, and with little training the Meta-RL algorithm should get good results as well.

### Compare Performance from other methods
## Conclusion
## Bibliography
