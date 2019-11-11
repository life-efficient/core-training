import torch
import gym
from time import sleep
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

# hyperparams = {
#     e
    
# }

nodes = 32

class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(4, nodes),
            torch.nn.ReLU(),
            torch.nn.Linear(nodes, 2),
            torch.nn.Softmax()
        )

    def forward(self, s):
        return self.layers(s)

# episodes_per_update = 100

epochs = 1000
episodes = 30
lr = 0.001
weight_decay = 1

env = gym.make('CartPole-v0')

policy = Policy()
# .double()

optimiser = torch.optim.SGD(policy.parameters(), lr=lr, weight_decay=weight_decay)

# env.reset()
# for step in range(100):              # for 1000 steps
#     action = env.action_space.sample()    # randomly sample an action to take
#     obs, reward, done, info = env.step(action)   # take the action and one timestep
#     print('Observation:', obs, '\tReward:', reward, '\tDone?', done, '\tInfo:', info, '\tPrevious action:', action)
#     env.render()                     # show the env
#     sleep(0.01)    

# SET UP PLOTTING
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111)
ax.set_title('Rewards')
plt.ion()
plt.show()

all_rewards = []
for epoch in range(epochs):
    obj = 0
    for episode in range(episodes):
        done = False
        state = torch.Tensor(env.reset())
        log_policy = []
        rewards = []
        step = 0
        while not done:     # while the episode is not terminated
            state = torch.Tensor(state)
            # print('STATE:', state)
            action_distribution = policy(state)
            # print('ACTION DISTRIBUTION:', action_distribution)

            action = torch.distributions.Categorical(action_distribution).sample()
            action = int(action)
            # print('ACTION:', action)
            
            state, reward, done, info = env.step(action)

            rewards.append(reward)
            log_policy.append(torch.log(action_distribution[action]))
            
            step += 1
            if done:
                break
            if step > 10000000:
                # break
                pass
        
        obj += sum(log_policy) * sum(rewards)

    obj /= episodes
    all_rewards.append(obj)
    obj *= -1

    writer.add_scalar('Reward/Train', obj, epoch)

    # UPDATE POLICY
    # print('updating policy')
    print('EPOCH:', epoch, 'REWARD:', int(obj))
    obj.backward()
    optimiser.step()
    optimiser.zero_grad()
    
    # VISUALISE
    ax.plot(all_rewards)
    fig.canvas.draw()
    state = env.reset()
    done = False
    while not done:
        env.render()
        state = torch.Tensor(state)
        action_distribution = policy(state)
        action = torch.distributions.Categorical(action_distribution).sample()
        action = int(action)
        state, reward, done, info = env.step(action)
        sleep(0.01)

plt.plot(all_rewards)
plt.show()



env.close()