from project import ZombieApocolypse

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

Z = ZombieApocolypse(n=150, k=2, vaccine_supply=0.35, vaccinate_rate=5)
make_gif(Z, steps=50)