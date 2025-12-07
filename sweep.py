from project import ZombieApocolypse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
"""vaccine_supply = np.linspace(0.25,0.75,16)
for r in vaccine_supply:
    Z = ZombieApocolypse(n=500, k=2, vaccine_supply=r)

    for t in tqdm(range(50)):
        Z.update_step()
    
    Z.show(f"{50}_5_{int(r*100)}")"""

plt.close()
vaccine_supply = np.linspace(0.2,0.8,16)
vaccinate_rate = list(range(1,17))

fig, axs = plt.subplots(2,2, figsize=(12,12))
cmap = get_cmap('Spectral')  # or 'plasma', 'cool', 'Spectral', etc.

last_populations = []
last_infected_gradient = []
healthy_r_gradient = [0]
infected_t_gradient_r_gradient = [0]
for idx, r in tqdm(enumerate(vaccinate_rate)):
    color = cmap(idx / (len(vaccinate_rate) - 1))  # Normalize index to [0,1]
    instances = []
    for i in tqdm(range(10)): # initializing the graphs takes up the most time
        instances.append(ZombieApocolypse(n=500, k=2, vaccine_supply=0.5, vaccinate_rate=r, verbose=False))


    for t in tqdm(range(50)):

        for i in instances:
            i.update_step()
        
    new_populations = np.average(list(map(lambda x: np.transpose(x.pops), instances)), axis=0)   
    if idx > 0:
        healthy_r_gradient.append(np.sum((new_populations[0]-last_populations[0])))
    
    last_populations = new_populations
    t = 50
    populations = np.average(list(map(lambda x: np.transpose(x.populations_history), instances)), axis=0)

    axs[0][0].plot(list(range(t)), populations[0], label=f"{np.round(r,2)}",color=color)
    axs[0][0].set_title(f"Healthy Population vs. Time vs. Vaccinate Rate")
    axs[0][0].grid(True)
    axs[0][0].legend()
    axs[0][0].set_ylabel("Healthy Population")
    axs[0][0].set_xlabel("Time Step")
    infected_gradient = np.gradient(populations[1])

    if idx > 0:
        infected_t_gradient_r_gradient.append(np.sum((infected_gradient-last_infected_gradient)**2))
    last_infected_gradient = infected_gradient
    
    axs[1][0].plot(list(range(t)), infected_gradient, label=f"{np.round(r,2)}",color=color)
    axs[1][0].set_title("Infected Population Growth vs. Time vs. Vaccinate Rate")
    axs[1][0].grid(True)
    axs[1][0].set_ylabel("Infected Growth Rate")
    axs[1][0].set_xlabel("Time Step")

axs[0][1].plot(vaccinate_rate, healthy_r_gradient, label=f"{np.round(r,2)}",color=color)
axs[0][1].set_title("Healthy Population Change vs. Vaccinate Rate")
axs[0][1].grid(True)
axs[0][1].set_ylabel("Final Healthy Population Change")
axs[0][1].set_xlabel("Vaccinate Rate")

axs[1][1].plot(vaccinate_rate, infected_t_gradient_r_gradient, label=f"{np.round(r,2)}",color=color)
axs[1][1].set_title("Infected Growth Change vs. Vaccinate Rate")
axs[1][1].grid(True)
axs[1][1].set_ylabel("Infected Growth Change")
axs[1][1].set_xlabel("Vaccinate Rate")

plt.savefig(f"infected_growth_50_50_vr.png")