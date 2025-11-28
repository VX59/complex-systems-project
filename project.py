import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import random

class SocialNetowrk():
    def __init__(self, n=100, k=1):
        self.n=n
        self.k=k
        self.pops = [0]*4
        Insample = nx.barabasi_albert_graph(n, k)        
        InAmatrix = self.weight_Amatrix_Poisson(np.triu(nx.to_numpy_array(Insample)))
        
        Outsample = nx.barabasi_albert_graph(n, k)        
        OutAmatrix = self.weight_Amatrix_Poisson(np.tril(nx.to_numpy_array(Outsample)))

        SampleAmatrix = InAmatrix + OutAmatrix
        
        self.G = nx.from_numpy_array(SampleAmatrix,create_using=nx.DiGraph)
        for id in self.G.nodes:
            self.G.nodes[id]["state"] = 0
            self.G.nodes[id]["immunity"] = 0
            self.G.nodes[id]["i2z"] = 0
        
        self.layout = nx.spring_layout(self.G,seed=10)

        self.fig, self.axs = plt.subplots(3, figsize=(5,12))
        self.node_artists = nx.draw_networkx_nodes(
            self.G,
            pos=self.layout,
            ax=self.axs[0],
            node_size=35
        )
        nx.draw_networkx_edges(self.G, pos=self.layout, ax=self.axs[0], node_size=35)

    def weight_Amatrix_Poisson(self,Amatrix):
            for i,node in enumerate(Amatrix):

                neighbors = np.where(node == 1)[0]
                weights = np.zeros(self.n)

                poisson_vals = np.random.poisson(self.k, len(neighbors))
                weights[neighbors] = poisson_vals

                if weights.sum() == 0:
                    continue

                Amatrix[i] = weights / weights.sum()
            return Amatrix
    
    def show(self, fname):
        state_to_color = {
            0: "gray",
            1: "orange",
            2: "green",
            3: "brown",
        }
        new_colors = [
            state_to_color[self.G.nodes[i]["state"]]
            for i in self.G.nodes
        ]
        self.node_artists.set_color(new_colors)

        ax1 = self.axs[1]
        ax1.cla()

        degree_dist = nx.degree_histogram(self.G)
        ax1.plot(range(len(degree_dist)), degree_dist, label="overall")

        for i in range(4):
            if i in self.degree_distributions:
                d = self.degree_distributions[i]
                max_k = max(d.keys())
                domain = list(range(max_k + 1))
                codomain = np.zeros(max_k + 1)
                for k, v in d.items():  # FAST clear
                    codomain[k] = v
                ax1.plot(
                    domain, codomain,
                    label=f"state {i}",
                    color=["gray", "orange", "green", "brown"][i]
                )

        ax1.set_title("degree distribution")
        ax1.set_xlabel("degree")
        ax1.set_ylabel("frequency")
        ax1.grid(True)
        ax1.legend()

        ax2 = self.axs[2]
        ax2.cla()
        ax2.bar(
            [0, 1, 2, 3],
            self.pops,
            color=["gray", "orange", "green", "brown"],
        )
        ax2.set_title("population by state")

        ax0 = self.axs[0]
        ax0.collections[-1].set_color(new_colors)  # update nodes
        ax0.legend(
            handles=[
                mpatches.Patch(color="gray", label="Healthy (0)"),
                mpatches.Patch(color="orange", label="Infected (1)"),
                mpatches.Patch(color="green", label="Zombie (2)"),
                mpatches.Patch(color="brown", label="Dead (3)")
            ]
        )

        self.fig.suptitle(fname)
        self.fig.tight_layout()
        self.fig.savefig(fname, dpi=150)

    def show_degree_centrality(self):
        centrality = nx.degree_centrality(self.G)
        plt.grid(True)
        plt.title("degree centrality")
        plt.xlabel("node")
        plt.ylabel("centrality")
        plt.plot(centrality.keys(),centrality.values())
        plt.show()

class ZombieApocolypse(SocialNetowrk):
    def __init__(self,initial_infected_num=1,n=100,k=1):
        super().__init__(n,k)
        initial_infected = random.sample(range(self.n), initial_infected_num)

        for node in initial_infected:
            self.G.nodes[node]["state"] = 1
            self.G.nodes[node]["ttl"] = max(1, int(np.random.exponential(100)))
    
    def populations(self):
        self.pops = [0]*4
        self.degree_distributions = {}
        for n in list(range(0,self.n)):
            node = self.G.nodes[n]
            node_state = node.get("state",0)
            self.pops[node_state] += 1
            if not self.degree_distributions.keys().__contains__(node_state):
                self.degree_distributions[node_state] = {}

            degree = self.G.degree(n)
            if not self.degree_distributions[node_state].keys().__contains__(degree):
                self.degree_distributions[node_state][degree] = 1
            else:
                self.degree_distributions[node_state][degree] += 1

    def show(self,fname):
        super().show(fname)

    def update_step(self):
        interaction_sequence = list(range(0,self.n))
        random.shuffle(interaction_sequence)
        for id in interaction_sequence:
            node_state = self.G.nodes[id].get("state",0)
            if node_state == 3: # dead
                continue
            
            if node_state == 2:
                self.G.nodes[id]["ttl"] -= 1
                if self.G.nodes[id]["ttl"] < 0:
                    self.G.nodes[id]["state"] = 3  # becomes dead
                    node_state = 3

                continue

            if node_state == 1:
                self.G.nodes[id]["ttl"] -= 1
                if self.G.nodes[id]["ttl"] <= 0:
                    self.G.nodes[id]["state"] = 2  # becomes zombie
                    node_state = 2  
                    self.G.nodes[id]["ttl"] = max(1, int(np.random.exponential(30)))

            valid_neighbors = [
                nbr for nbr in self.G[id]
                if self.G.nodes[nbr].get("state",0) < 2
            ]

            if not valid_neighbors:
                continue

            weights = [self.G[id][nbr].get("weight", 1) for nbr in valid_neighbors]

            choice = random.choices(population=valid_neighbors, weights=weights, k=1)[0]
            choice_state = self.G.nodes[choice].get("state",0)

            if node_state == 0 and choice_state == 1:
                new_state = int(np.random.choice([0,1]))
                self.G.nodes[id]["state"] = new_state
                if new_state == 1:
                    self.G.nodes[id]["ttl"] = max(1, int(np.random.exponential(3)))

            if node_state == 1 and choice_state == 0:
                new_state = int(np.random.choice([0,1]))
                self.G.nodes[choice]["state"] = new_state
                if new_state == 1:
                    self.G.nodes[choice]["ttl"] = max(1, int(np.random.exponential(3)))
                
            if node_state == 2 and choice_state == 0:
                neighbors_1 = set(self.G[id])
                neighbors_2 = set()
                for nbr in neighbors_1:
                    neighbors_2.update(self.G[nbr])

                all_neighbors = (neighbors_1 | neighbors_2) - {id}
                valid_neighbors = [
                    nbr for nbr in all_neighbors
                    if self.G.nodes[nbr].get("state", 0) == 0
                ]

                choice = random.choices(population=valid_neighbors, k=1)[0]
                choice_state = self.G.nodes[choice].get("state",0)

                self.G.nodes[choice]["state"] = 1
                self.G.nodes[choice]["ttl"] = max(1, int(np.random.exponential(7)))

        self.populations()

import imageio.v2 as imageio
from tqdm import tqdm

def make_gif(zombie_sim:ZombieApocolypse, steps=20, outfile="zombie.gif"):
    frames = []

    for t in tqdm(range(steps)):
        zombie_sim.update_step()
        fname = f"frames/_frame_{t}.png"
        zombie_sim.show(fname)
        frames.append(imageio.imread(fname))

    imageio.mimsave(outfile, frames, fps=10)
    print(f"Saved GIF → {outfile}")

Z = ZombieApocolypse(n=100, k=2)
make_gif(Z, steps=100)