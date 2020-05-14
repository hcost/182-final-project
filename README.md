README
CS 182 FINAL PROJECT
-Chandler Jones
-Harrison Costantino
-Daniel Grimshaw

Abstract: 
Techniques used in reinforcement learning (RL) to teach a model to play a complex game and to 
generalize what it has learned can be used in non-trivial, real-world contexts. Generalization is an 
important area of RL as real world environments are not deterministic and a model cannot train on every 
possible environment it will encounter. We can adapt the methods used in teaching agents to play games 
to teach them how to perform tasks that humans either cannot or do not want to do. Here we evaluate 
the performance of various model architectures -- namely DQN and PPO -- on Procgen Benchmark's FruitBot 
environment.  Each model was implemented using TensorFlow tf-agents, OpenAI baselines, and Stable Baselines.

The best model was the OpenAI-based PPO agent, found in the PPO/openai section.
All requirements are listed in each model's respective requirements.txt file, but generally tf-agents required TensorFlow 2.x
and the OpenAI baselines ran on TensorFlow 1.15. Most of the other requirements were those needed to run Procgen.
