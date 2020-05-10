import gym
import stable_baselines as sb


env = gym.make(id='procgen:procgen-fruitbot-v0', num_levels=100, use_sequential_levels=True, distribution_mode='easy')


print("Beginning Training...")




try:
	model = sb.PPO2('CnnPolicy', env, verbose=1, tensorboard_log='stats/')
	model.learn(7500000)
except:
	print("Error occured; saving model")

model.save('models/model_2')
