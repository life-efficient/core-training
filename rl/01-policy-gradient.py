import gym
env = gym.make("CartPole-v1")
obs = env.reset()
print(obs)
for step in range(1000):
    env.render()
    action = env.action_space.sample()
