import numpy as np
import matplotlib.pyplot as plt

# Carica i dati di valutazione
data = np.load("evaluations.npz", allow_pickle=True)

# Visualizza le chiavi disponibili
print(list(data.keys()))

# Estrai i dati delle reward e i timestep
rewards = data['results']
timesteps = data['timesteps']

# Crea un grafico
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