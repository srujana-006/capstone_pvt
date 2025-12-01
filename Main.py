'''from MAP_env import MAP
from Controller import Controller



# 1) Load grid environment
env = MAP("Data/grid_config_2d.json")

# 2) Create EVs
env.init_evs()

# 3) Create Controller (with your real CSV path)
controller = Controller(
    env,
    ticks_per_ep=180,
    csv_path="Data/5Years_SF_calls_latlong.csv",  # adjust to your actual path
)

# 4) Run the short debug episode (5 ticks)
controller.run_one_episode()
print("Reposition buffer size:", len(controller.buffer_reposition))
'''
'''
from MAP_env import MAP
from Controller import Controller

env = MAP("Data/grid_config_2d.json")
env.init_evs()
env.init_hospitals("D:\\Downloads\\hospitals_latlong.csv")
ctrl = Controller(
    env,
    ticks_per_ep=180,
    csv_path="D:\\Downloads\\5Years_SF_calls_latlong.csv"
)

n_episodes = 500
all_stats = []
all_loss = []
for ep in range(1, n_episodes + 1):
    #dispatched = 0
    stats = ctrl.run_training_episode(ep)
    episode_loss = stats["average ep loss"]
    all_loss.append(episode_loss)
    all_stats.append(stats)
    
import matplotlib.pyplot as plt

plt.plot(all_loss)
plt.xlabel("Episode")
plt.ylabel("Average Navigation Loss")
plt.title("Navigation Training Loss Curve")
plt.grid(True)
plt.show()
'''

import matplotlib.pyplot as plt
import pandas as pd
from MAP_env import MAP
from Controller import Controller

# Initialize Environment
env = MAP("Data/grid_config_2d.json")
env.init_evs()
env.init_hospitals("D:\\Downloads\\hospitals_latlong.csv")
#env.init_hospitals("Data/hospitals_latlong.csv")

# Initialize Controller
ctrl = Controller(
    env,
    ticks_per_ep=180,
    csv_path="D:\\Downloads\\5Years_SF_calls_latlong.csv"
    #csv_path="Data/5Years_SF_calls_latlong.csv"
)

n_episodes = 500
all_stats = []
all_nav_loss = []
all_repo_loss = [] # New list for repositioning


for ep in range(1, n_episodes + 1):
    stats = ctrl.run_training_episode(ep)
    
    # Get both losses
    nav_loss = stats["average ep loss"]
    repo_loss = stats["average repo loss"]
    
    all_nav_loss.append(nav_loss)
    all_repo_loss.append(repo_loss)
    all_stats.append(stats)
    
    
# === NEW PLOTTING SECTION ===
plt.figure(figsize=(10, 8)) # Make the figure taller

# Plot 1: Navigation Loss
plt.subplot(2, 1, 1) # 2 rows, 1 column, plot #1
plt.plot(all_nav_loss, color='blue', label='Nav Loss')
plt.ylabel("Navigation Loss")
plt.title("Training Loss Curves")
plt.grid(True)
plt.legend()

# Plot 2: Repositioning Loss
plt.subplot(2, 1, 2) # 2 rows, 1 column, plot #2
plt.plot(all_repo_loss, color='orange', label='Reposition Loss')
plt.xlabel("Episode")
plt.ylabel("Reposition Loss")
plt.grid(True)
plt.legend()

plt.tight_layout() # Prevents overlap
plt.show()

