#==============Checking Entities=================
#--------------Checking EV-----------------------
'''
from Entities.ev import EV, EvState

ev = EV(id=1, gridIndex=5, location=(10.0, 20.0))
#print(ev)

#ev.assign_incident(99)
#print(ev.assignedPatientId, ev.status, ev.aggIdleTime, ev.aggIdleEnergy)
#ev.release_incident()
#print(ev)
ev.move_to(3, (1, 1))
#print(ev.gridIndex, ev.location)    

ev.add_idle(8)
#print(ev.aggIdleTime)

before_e = ev.aggIdleEnergy
before_t = ev.aggIdleTime
ev.execute_reposition()
print(ev.aggIdleEnergy - before_e, ev.aggIdleTime - before_t)

print(ev.to_dict())
'''

#---------------Checking Grid-----------------
'''
from Entities.GRID import Grid
from Entities.ev import EV, EvState
from Entities.Incident import Incident, IncidentStatus
from datetime import datetime

g = Grid(index=0)
evs = {
    1: EV(1, 0, (0,0)),
    2: EV(2, 0, (0,0))
}
incs = {
    10: Incident(10, 0, datetime.now(), (0,0)),
    11: Incident(11, 0, datetime.now(), (0,0))
}

g.evs = [1,2]
g.incidents = [10,11]

evs[1].state = EvState.IDLE
evs[1].status = "available"
evs[1].sarns["action"] = 0

evs[2].state = EvState.BUSY
evs[2].status = "Navigation"
#print(g.count_idle_available_evs(evs))
incs[10].assignedEvId = 1
#print(g.count_unassigned_incidents(incs))
#print(g.calculate_imbalance(evs, incs))
#print(g.get_eligible_idle_evs(evs))
#print(g.get_pending_incidents(incs))

#g = Grid(index=0)

# Add neighbour 1
#g.add_neighbour(1)
#print(g.neighbours)   # Expected: [1]

# Add same neighbour again
#g.add_neighbour(1)
#print(g.neighbours)   # Expected: still [1]

#g = Grid(index=0)

#g.add_incident(100)
#print(g.incidents)     # Expected: [100]

#g.add_incident(100)
#print(g.incidents)     # Expected: still [100]
print(g.to_dict)
'''
#-----------------Check Incident--------------
'''
from Entities.Incident import Incident, Priority
from datetime import datetime

i = Incident(1, 5, datetime.now(), (10,10))

#print(i)
i.assign_ev(3)
#print(i.status, i.assignedEvId)
i.add_wait(10)
#print(i.waitTime)

print(i.get_urgency_score())
i.waitTime = 100
print(i.get_urgency_score())
'''
#=======================SERVICES===========================
#-----------------------Repositioning----------------------
'''
from Entities.ev import EV, EvState
from Entities.GRID import Grid
from services.repositioning import RepositioningService

# Grids
g0 = Grid(index=0)
g1 = Grid(index=1)
g0.add_neighbour(1)
g1.add_neighbour(0)

grids = {0: g0, 1: g1}

# EV
ev = EV(id=1, gridIndex=0, location=(0.0, 0.0))
ev.state = EvState.IDLE
ev.status = "available"
ev.aggIdleTime = 30.0
ev.aggIdleEnergy = 5.0
ev.sarns["action"] = 1  # wants to go to grid 1

g0.add_ev(ev.id)
evs = {ev.id: ev}

incidents = {}  # <- no incidents, so imbalance = 0

rep = RepositioningService()

print("before:", ev.nextGrid, ev.sarns.get("reward"), ev.status)
rep.accept_reposition_offers(evs, grids, incidents)
print("after:", ev.nextGrid, ev.sarns.get("reward"), ev.status)

from datetime import datetime
from Entities.Incident import Incident, Priority

# 1) Create one incident in grid 1
inc = Incident(
    id=1,
    gridIndex=1,
    timestamp=datetime.now(),
    location=(0.0, 0.0),
    priority=Priority.MED,
)
incidents = {1: inc}
g1.add_incident(inc.id)

# 2) Now run accept_reposition_offers again
rep.accept_reposition_offers(evs, grids, incidents)

print("after:", ev.nextGrid, ev.sarns.get("reward"), ev.status)
'''
#-------------------Dispatching------------------------------
'''
from services.dispatcher import DispatcherService
from Entities.GRID import Grid
from Entities.ev import EV, EvState
from Entities.Incident import Incident
from datetime import datetime

g = Grid(0)
ev = EV(1, 0, (0,0))
ev.sarns["action"] = 0
ev.status = "available"
ev.state = EvState.IDLE

inc = Incident(10,0,datetime.now(),(1,1),waitTime=8)

g.evs=[1]
g.incidents=[10]

evs={1:ev}
incs={10:inc}
grids={0:g}

d = DispatcherService()
print(d.dispatch_gridwise(grids,evs,incs))
print(ev)
'''
#===================TESTING MAP_env=============================
'''
from MAP_env import MAP
from Entities.ev import EvState
from Entities.Incident import IncidentStatus
from utils.Helpers import P_MAX

env = MAP("Data/grid_config_2d.json")   # your real path
env.init_evs()
#env.create_incident(0, (10,10))
ev = env.create_ev(0)

#print(env.grids[0].evs)
#print(env.grids[0].incidents)
print("nRows, nCols:", env.nRows, env.nCols)
print("num grids:", len(env.grids))

mid = list(env.grids.keys())[len(env.grids)//2]
g = env.grids[mid]
print("Grid", mid, "neighbours:", sorted(g.neighbours))


all_evs = env.evs
print("EV count:", len(all_evs))

# Check each EV exists in its grid’s ev list
for eid, ev in all_evs.items():
    assert eid in env.grids[ev.gridIndex].evs, f"EV {eid} not in its grid list!"
print("✅ EVs correctly placed in grids.")


ev = env.create_ev(0)
print("New EV:", ev.id, ev.gridIndex, ev.location)
print("Grid[0] evs:", env.grids[0].evs)

env.move_ev_to_grid(ev.id, 1)
print("After move:", ev.id, ev.gridIndex)
print("Grid[0] evs:", env.grids[0].evs)
print("Grid[1] evs:", env.grids[1].evs)



inc = env.create_incident(grid_index=0, location=(10.0, 20.0))
print("Incident:", inc.id, inc.gridIndex, inc.location)
print("Grid[0] incidents:", env.grids[0].incidents)

#------------Micro test - Idle EV in place---------------

ev.state = EvState.IDLE
ev.status = "available"
ev.sarns["action"] = ev.gridIndex  # <- add this
print("Before:", ev.aggIdleTime)

env.update_after_timeslot(dt_minutes=8.0)

print("After:", ev.aggIdleTime)  # Expect +8
'''
#------------------Micro Test - Dispatching EV---------------
'''
ev.status = "Dispatching"
ev.state = EvState.IDLE

# Incident in grid 5
inc = env.create_incident(grid_index=5, location=(0.0,0.0))
ev.assignedPatientId = inc.id
ev.sarns["reward"] = None  # no reward yet

print("Before:", ev.gridIndex, ev.state, ev.status)

env.update_after_timeslot(dt_minutes=8.0)

print("After:", ev.gridIndex, ev.state, ev.status)
'''
#------------------Repositiong-------------------
'''
ev.state = EvState.IDLE
ev.status = "Repositioning"
ev.nextGrid = 3
ev.sarns["reward"] = 0.8
ev.aggIdleTime = 0.0
ev.aggIdleEnergy = 0.0

print("Before:", ev.gridIndex, ev.aggIdleTime, ev.aggIdleEnergy)

env.update_after_timeslot(dt_minutes=8.0)

print("After:", ev.gridIndex, ev.aggIdleTime, ev.aggIdleEnergy)
'''
#-------------------Incident cancellation---------------------
'''
inc = env.create_incident(grid_index=0, location=(0.0,0.0))
inc.waitTime = P_MAX + 1  # Force over threshold

print("Before:", env.incidents.keys(), env.grids[0].incidents)

env.update_after_timeslot(dt_minutes=8.0)

print("After:", env.incidents.keys(), env.grids[0].incidents)
'''
#=================CHECKING NEIGHBOUR ALLOTMENT==================#
'''
from MAP_env import MAP
from Controller import Controller
from Entities.ev import EvState
from Entities.Incident import IncidentStatus
from utils.Helpers import P_MAX

env = MAP("Data/grid_config_2d.json")   # your real path
env.init_evs()

# 1. Build env + controller (use your real paths)

ctrl = Controller(env, csv_path="Data/5Years_SF_calls_latlong.csv")

# 2. Pick a cell that should be interior (has all 8 neighbours)
center_idx = (env.nRows // 2) * env.nCols + (env.nCols // 2)

neighs = ctrl._get_direction_neighbors_for_index(center_idx)
print("Center index:", center_idx)
print("Neighbours in order [N, NE, E, SE, S, SW, W, NW]:")
print(neighs)
print("Length:", len(neighs))

corner = 0
print("Corner 0 neighbours:", ctrl._get_direction_neighbors_for_index(corner))
'''

#======================Build_state===================#
'''
from MAP_env import MAP
from Entities.ev import EvState
from Controller import Controller
env = MAP("Data/grid_config_2d.json")

env.init_evs()
# reuse env, ctrl from above
ev = next(iter(env.evs.values()))  # just grab the first EV
ev.state = EvState.IDLE
ev.status = "Idle"
ctrl = Controller(env, csv_path="Data/5Years_SF_calls_latlong.csv")

state = ctrl._build_state(ev)
print("State length:", len(state))
print("State vector:", state)

'''

#====================Directional poke itseems==================#
'''
from Entities.Incident import Incident, Priority, IncidentStatus
from datetime import datetime
from MAP_env import MAP
from Entities import GRID, ev
from Controller import Controller
env = MAP("Data/grid_config_2d.json")   # your real path
env.init_evs()

ctrl = Controller(env, csv_path="Data/5Years_SF_calls_latlong.csv")
ev = next(iter(env.evs.values()))
gi = ev.gridIndex
print("EV grid index:", gi)

# Get neighbour indices by direction order
neighs = ctrl._get_direction_neighbors_for_index(gi)
print("Neighbour indices [N, NE, E, SE, S, SW, W, NW]:", neighs)

# Build baseline state
base_state = ctrl._build_state(ev)
print("Baseline neighbour imbalances:", base_state[2:10])

# Now create an incident in some neighbour that exists, say E (index 2)
east_idx = neighs[2]
if east_idx != -1:
    inc = env.create_incident(grid_index=east_idx, location=(0.0,0.0), priority="MED")
    # Recompute imbalance
    for g in env.grids.values():
        g.imbalance = g.calculate_imbalance(env.evs, env.incidents)

    new_state = ctrl._build_state(ev)
    print("New neighbour imbalances:", new_state[2:10])

    print("Change at E slot (index 4):", base_state[4], "->", new_state[4])
else:
    print("E neighbour does not exist for this EV grid; try another direction.")
'''

#=========================Action_Check========================#
'''
import numpy as np
from Entities.Incident import Incident, Priority, IncidentStatus
from datetime import datetime
from MAP_env import MAP
from Entities import GRID, ev
from Controller import Controller
env = MAP("Data/grid_config_2d.json")   # your real path
env.init_evs()

ctrl = Controller(env, csv_path="Data/5Years_SF_calls_latlong.csv")

ctrl.epsilon = 1.0  # force random actions

ev = next(iter(env.evs.values()))
gi = ev.gridIndex

neighs = ctrl._get_direction_neighbors_for_index(gi)
valid_grids = {gi} | {idx for idx in neighs if idx != -1}
print("EV grid:", gi)
print("Neighbour grids:", neighs)
print("Valid destination set:", valid_grids)

bad = 0
for _ in range(100):
    s = ctrl._build_state(ev)
    dest = ctrl._select_action(s, gi)
    if dest not in valid_grids:
        bad += 1
        print("Bad destination:", dest)

print("Bad destinations count:", bad)
'''

#==========================checkin da DQN=========================#
'''
import torch
import torch.nn as nn
import numpy as np

from MAP_env import MAP
from Controller import Controller
from Entities.ev import EvState  # only if you want to fiddle with state/status later


# 1) Build environment and controller
env = MAP("Data/grid_config_2d.json")     # adjust path if needed
env.init_evs()

ctrl = Controller(
    env,
    csv_path="Data/5Years_SF_calls_latlong.csv"  # adjust path if needed
)

print("EV count:", len(env.evs))


# 2) Dummy network that always returns q = [0,1,2,3,4,5,6,7,8]
class DummyNet(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        # Return [0,1,2,3,4,5,6,7,8] for every input in the batch
        batch_size = x.size(0)
        q = torch.arange(9, dtype=torch.float32, device=x.device).unsqueeze(0).expand(batch_size, -1)
        return q


# 3) Replace the reposition main DQN with DummyNet
ctrl.dqn_reposition_main = DummyNet().to(ctrl.device)   # type: ignore
ctrl.epsilon = 0.0  # force greedy (no random exploration)


# 4) Pick one EV and test action mapping
ev_obj = next(iter(env.evs.values()))   # grab first EV object
gi = ev_obj.gridIndex

neighs = ctrl._get_direction_neighbors_for_index(gi)
print("EV grid index:", gi)
print("Neighbours [N, NE, E, SE, S, SW, W, NW]:", neighs)

# Build its state
s = ctrl._build_state(ev_obj)
print("State length:", len(s))
print("State vector:", s)

# 5) Ask controller for an action with DummyNet
dest = ctrl._select_action(s, gi)
print("Chosen dest:", dest)

# With strictly increasing q-values, best slot is 8
# → action slot 8 → direction index 7 → NW neighbour (neighs[7])
nw_idx = neighs[7]
expected = nw_idx if nw_idx != -1 else gi
print("Expected dest (NW or stay if -1):", expected)
'''

#======================PushRepositionBuffer=================#

import numpy as np
from Entities.Incident import Incident, Priority, IncidentStatus
from datetime import datetime
from MAP_env import MAP
from Entities import GRID
from Entities.ev import EvState
from Controller import Controller
env = MAP("Data/grid_config_2d.json")   # your real path
env.init_evs()

ctrl = Controller(env, csv_path="Data/5Years_SF_calls_latlong.csv")

print("Buffer size before:", len(ctrl.buffer_reposition))

ev = next(iter(env.evs.values()))
ev.state = EvState.IDLE
ev.status = "Repositioning"
ev.aggIdleTime = 10.0
ev.aggIdleEnergy = 1.5

# Fake s, a, r as if DQN + service have run
ev.sarns["state"] = ctrl._build_state(ev)
ev.sarns["action"] = ev.gridIndex   # say it chose to stay; any int is fine for test
ev.sarns["reward"] = 0.7

ctrl._push_reposition_transition(ev)

print("Buffer size after:", len(ctrl.buffer_reposition))
