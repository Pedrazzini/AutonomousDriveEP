import numpy as np
import matplotlib.pyplot as plt

# load data for evaluations
data = np.load("evaluations.npz", allow_pickle=True)

# visualize available keys
print(list(data.keys()))

# extract data of reward and timestep
rewards = data['results']
timesteps = data['timesteps']

# create a plot
plt.figure(figsize=(10, 6))
plt.plot(timesteps, np.mean(rewards, axis=1))
plt.fill_between(timesteps,
                 np.mean(rewards, axis=1) - np.std(rewards, axis=1),
                 np.mean(rewards, axis=1) + np.std(rewards, axis=1),
                 alpha=0.2)
plt.xlabel('Timesteps')
plt.ylabel('Mean Reward')
plt.title('Evaluation Rewards During Training')
plt.grid(True)
plt.savefig('reward_plot.png')
plt.show()