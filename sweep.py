from project import ZombieApocolypse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

"""vaccine_supply = np.linspace(0.25,0.75,16)
for r in vaccine_supply:
    Z = ZombieApocolypse(n=500, k=2, vaccine_supply=r)

    for t in tqdm(range(50)):
        Z.update_step()
    
    Z.show(f"{50}_5_{int(r*100)}")"""

plt.close()
vaccine_supply = np.linspace(0.25,0.75,16)
fig, axs = plt.subplots(2, figsize=(6,10))
for r in tqdm(vaccine_supply):

    instances = []
    for i in tqdm(range(2)): # initializing the graphs takes up the most time
        instances.append(ZombieApocolypse(n=500, k=2, vaccine_supply=r, vaccinate_rate=6, verbose=False))

    for t in tqdm(range(50)):
        
        for i in instances:
            i.update_step()
    t = 50
    populations = np.average(list(map(lambda x: np.transpose(x.populations_history), instances)), axis=0)

    print(populations)
    axs[0].plot(list(range(t)), populations[0], label=f"{np.round(r,2)}")
    axs[0].set_title(f"Healthy Population vs. Time vs. Vaccinate Supply")
    axs[0].grid(True)
    axs[0].legend()
    infected_gradient = np.gradient(populations[1])
    axs[1].plot(list(range(t)), infected_gradient, label=f"{np.round(r,2)}")
    axs[1].set_title("Infected Population Growth vs. Time vs. Vaccinate Supply")
    axs[1].grid(True)
    axs[1].legend()

plt.savefig(f"infected_growth_50_50.png")