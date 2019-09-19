# Title
## Intro
## Background on methods
### Reinforcment learning
### Meta-RL

Reinforcement Learning (RL) has a tendency to "overfit". It struggles when faced with similar environments and need to completely relearn in order to adjust. Meta Reinforcement Learning (Meta-RL) aims to fix this problem by programming a computer how to learn in more general environments so it does not have to completely relearn. For example, say we have a two-armed bandit, where one arm gives a reward and the other doesn't. RL is easily able to learn which hand gives the reward, but will fail if we introduce it to another two-armed bandit where the rewarding arm has switched. It had grown accustumed to pull a specific arm rather than considering the possibility that the situation had changed. Meta-RL would learn how to approach a general two-armed bandit problem rather than a specific one in order to adjust when the bandit is different.

### Quantum Computing
## Our Method
### Meta-RL + QNN
## The problem / testing our method
### Stock market problem
### Compare Performance from other methods
## Conclusion
## Bibliography