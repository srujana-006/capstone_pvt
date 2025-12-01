# MAP_env.py (Refactored - OOP-first design)
"""
Environment orchestrator for emergency response simulation.

MAP (Map) class serves as a thin orchestrator that:
1. Manages grid topology and geometric conversions
2. Manages collections (EVs, incidents, hospitals, grids)
3. Delegates domain logic to entity methods and services

Domain logic is distributed to:
- Entity classes (EV, Incident, Grid, Hospital) for entity-specific behavior
- Service classes (DispatcherService, RepositioningService, NavigationService) for cross-entity logic
"""
import random
from typing import Tuple, Dict, List, Optional
from datetime import datetime
import math
import numpy as np
from utils.Helpers import (
    point_to_grid_index,
    load_grid_config_2d, P_MAX,
)

from Entities.GRID import Grid
from Entities.ev import EV, EvState
from Entities.Incident import Incident, Priority, IncidentStatus
from Entities.Hospitals import Hospital

# Import services
from services.dispatcher import DispatcherService
from services.repositioning import RepositioningService
from services.navigation import NavigationService


class MAP:
    """
    Environment orchestrator for emergency response simulation.
    
    Responsibilities:
    - Initialize and manage grids, EVs, incidents, hospitals
    - Provide geometric/topology operations (grid conversions, neighbors)
    - Coordinate algorithms via service layer
    - Manage simulation state
    """
    
    def __init__(self, grid_config_path: str):
        """Initialize the MAP environment with grid configuration."""
        self.grids: Dict[int, Grid] = {}
        self.evs: Dict[int, EV] = {}
        self.incidents: Dict[int, Incident] = {}
        self.hospitals: Dict[int, Hospital] = {}

        self._incidentCounter = 0
        self._evCounter = 0
        self._hospitalCounter = 0
        self.dispatcher: DispatcherService

        # Load grid configuration
        self.lat_edges, self.lng_edges, _ = load_grid_config_2d(grid_config_path)
        self.nRows = len(self.lat_edges) - 1
        self.nCols = len(self.lng_edges) - 1

        # Initialize services
        self.dispatcher = DispatcherService()
        self.repositioner = RepositioningService()
        self.navigator = NavigationService()

        # Build grid topology
        self.build_grids(self.lat_edges, self.lng_edges)

    # ========== GRID GEOMETRY & TOPOLOGY ==========
    
    def build_grids(self, lat_edges, lng_edges) -> None:
        """Build grid cells and establish 8-connected neighbor relationships."""
        n_rows = len(lat_edges) - 1
        n_cols = len(lng_edges) - 1

        # Create all grid cells
        for r in range(n_rows):
            for c in range(n_cols):
                idx = r * n_cols + c
                self.grids[idx] = Grid(index=idx)

        # Connect 8-neighbors
        for r in range(n_rows):
            for c in range(n_cols):
                idx = r * n_cols + c
                nbs = []
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < n_rows and 0 <= nc < n_cols:
                            nbs.append(nr * n_cols + nc)
                self.grids[idx].neighbours = nbs

    def index_to_rc(self, idx: int) -> Tuple[int, int]:
        """Convert 1D grid index to (row, col) coordinates."""
        return idx // self.nCols, idx % self.nCols

    #def rc_to_index(self, r: int, c: int) -> int:
        #"""Convert (row, col) coordinates to 1D grid index."""
        #return r * self.nCols + c

    def grid_center(self, idx: int) -> Tuple[float, float]:
        """Get the (lat, lng) center of a grid cell."""
        r, c = self.index_to_rc(idx)
        lat = (self.lat_edges[r] + self.lat_edges[r + 1]) / 2.0
        lng = (self.lng_edges[c] + self.lng_edges[c + 1]) / 2.0
        return lat, lng

    # ========== EV MANAGEMENT ==========
    
    def init_evs(self, seed: int = 42) -> None:
        """Initialize EVs with reproducible seeded placement."""
        self.evs.clear()
        self._evCounter = 0
        for g in self.grids.values():
            g.evs.clear()

        rng = random.Random(seed)
        n_evs = 27
        all_idx = list(self.grids.keys())

        for _ in range(n_evs):
            gi = rng.choice(all_idx)
            self.create_ev(gi)

    def create_ev(self, grid_index: int) -> EV:
        """Create a new EV and place it in a grid."""
        self._evCounter += 1
        loc = self.grid_center(grid_index)
        ev = EV(id=self._evCounter, gridIndex=grid_index, location=loc)
        self.evs[ev.id] = ev
        self.grids[grid_index].add_ev(ev.id)
        return ev

    def move_ev_to_grid(self, ev_id: int, new_grid_index: int) -> None:
        """Move an EV from its current grid to a new grid."""
        ev = self.evs[ev_id]
        old_idx = ev.gridIndex
        if old_idx in self.grids:
            self.grids[old_idx].remove_ev(ev_id)
        self.grids[new_grid_index].add_ev(ev_id)
        ev.move_to(new_grid_index, self.grid_center(new_grid_index))

    # ========== INCIDENT MANAGEMENT ==========
    
    def create_incident(self, grid_index: int, location: Tuple[float, float], priority: str = "MED") -> Incident:
        """Create a new incident and place it in a grid."""
        self._incidentCounter += 1
        inc = Incident(
            id=self._incidentCounter,
            gridIndex=grid_index,
            timestamp=datetime.now(),
            location=location,
            priority=Priority[priority],
        )
        self.incidents[inc.id] = inc
        self.grids[grid_index].add_incident(inc.id)
        return inc

    # ========== HOSPITAL MANAGEMENT ==========
    
    def init_hospitals(
        self,
        csv_path: str,
        *,
        lat_col: str = "Latitude",
        lng_col: str = "Longitude",
        name_col: str = "Name",
    ) -> None:
        """Load hospitals from CSV and place them in the grid."""
        if getattr(self, "_hospitals_initialized", False):
            return

        import pandas as pd

        df = pd.read_csv(csv_path)
        if lat_col not in df.columns or lng_col not in df.columns:
            raise ValueError(f"Missing hospital columns: {lat_col}/{lng_col}")

        # Clear existing hospital links
        for g in self.grids.values():
            g.hospitals.clear()

        # Create and place hospitals
        for _, row in df.iterrows():
            lat = float(row[lat_col])
            lng = float(row[lng_col])
            name = str(row[name_col]) if name_col in df.columns else f"Hospital_{self._hospitalCounter+1}"

            gi = point_to_grid_index(lat, lng, self.lat_edges, self.lng_edges)

            self._hospitalCounter += 1
            hid = self._hospitalCounter
            hc = Hospital(id=hid, loc=(lat, lng), gridIndex=gi, waitTime=0.0, services=[])
            self.hospitals[hid] = hc
            self.grids[gi].hospitals.append(hid)

        self._hospitals_initialized = True
        print(f"[MAP] Hospitals placed: {len(self.hospitals)} fixed locations.")

        print("Hospitals per grid (non-empty):")
        for gi, g in self.grids.items():
            if g.hospitals:
                ids = [self.hospitals[h].id for h in g.hospitals]
                print(f"  Grid {gi}: {ids}")


    def tick_hospital_waits(self, low_min: float = 5.0, high_min: float = 45.0, seed: int | None = None) -> None:
        """Reset hospital wait times to random values in range."""
        rng = random.Random(seed)
        if not getattr(self, "hospitals", None):
            print("[MAP] No hospitals to reset waits for.")
            return
        for hc in self.hospitals.values():
            #hc.waitTime = math.exp(low_min + high_min/2)
            rng = np.random.default_rng()
            lam = low_min + high_min / 2.0 # mean
            hc.waitTime = rng.poisson(lam) #poisson dist with mean
            #print(f"[MAP] Hospital waits initialised in [{hc.id}, {hc.waitTime}] minutes.")

    '''def tick_hospital_waits(self, lam: float = 0.04, wmin: float = 5.0, wmax: float = 90.0, seed: int | None = None) -> None:
        """Update hospital wait times with random exponential drift."""
        if not getattr(self, "hospitals", None):
            return
        rng = random.Random(seed)
        for hc in self.hospitals.values():
            eps = rng.uniform(-lam, lam)
            hc.waitTime = max(wmin, min(wmax, hc.waitTime * math.exp(eps)))'''



    def next_grid_towards(self, from_idx: int, to_idx: int) -> int:

        if from_idx == to_idx:
            return from_idx

        n_rows = len(self.lat_edges) - 1
        n_cols = len(self.lng_edges) - 1

        # current cell
        row_from = from_idx // n_cols
        col_from = from_idx % n_cols

        # target cell
        row_to = to_idx // n_cols
        col_to = to_idx % n_cols

        # step direction in row/col: -1, 0, or 1
        dr = 0
        if row_to > row_from:
            dr = 1
        elif row_to < row_from:
            dr = -1

        dc = 0
        if col_to > col_from:
            dc = 1
        elif col_to < col_from:
            dc = -1

        # take one step
        new_row = row_from + dr
        new_col = col_from + dc

        # safety clamp (should already be in bounds)
        new_row = max(0, min(n_rows - 1, new_row))
        new_col = max(0, min(n_cols - 1, new_col))

        return new_row * n_cols + new_col


    # ========== ALGORITHMS (delegated to services) ==========
    
    def accept_reposition_offers(self) -> None:
        """
        Algorithm 1: Accept or reject repositioning offers from idle EVs.
        
        Delegates to RepositioningService.
        See services.repositioning.RepositioningService.accept_reposition_offers()
        """
        self.repositioner.accept_reposition_offers(self.evs, self.grids, self.incidents)

    '''def step_reposition(self) -> None:
        """
        Apply accepted reposition moves and clear pending decisions.
        
        Delegates to RepositioningService and handles physical grid moves.
        """
        #self.repositioner.execute_repositions(self.evs, self.grids)
        
        # Apply physical grid moves (MAP manages topology)
        for ev in self.evs.values():
            if ev.state != EvState.IDLE:
                continue
            dst = ev.nextGrid
            if dst is None:
                continue
            if dst != ev.gridIndex:
                self.move_ev_to_grid(ev.id, dst)
            ev.nextGrid = None'''

    def dispatch_gridwise(self, beta: float = 0.5) -> List[Tuple[int, int, float]]:
        """
        Algorithm 2: Gridwise dispatch of EVs to incidents.
        
        For each grid:
        1. Collect idle EVs that stayed in this grid
        2. For each unassigned incident:
           - If no local EVs, borrow from neighbors
           - Select EV with highest dispatch utility
           - Assign and remove from available lists
        
        Delegates to DispatcherService.
        
        Args:
            beta: Weight for vehicle idle time utility (vs patient wait time utility)
            
        Returns:
            List of (ev_id, incident_id, utility) assignments
        """
        return self.dispatcher.dispatch_gridwise(
            self.grids,
            self.evs,
            self.incidents,
            beta=beta,
        )

    '''def choose_hospital_for_ev(self, ev_id: int, inc_id: int) -> None:
        """
        Select the best (nearest) hospital for a patient incident.
        
        Delegates to NavigationService.
        
        Args:
            ev_id: EV ID (for context; not used in selection)
            inc_id: Incident ID
            
        Returns:
            Tuple of (hospital_id, eta_minutes)
        """
        inc = self.incidents[inc_id]
        evs = self.evs[ev_id]
        return self.navigator.select_hospital_for_incident(inc, self.hospitals,evs)'''

    '''def get_nav_candidates(self, inc_id: int, max_k: int = 8) -> Tuple[List[int], List[float], List[float]]:
        """
        Get top K candidate hospitals for an incident (sorted by proximity).
        
        Useful for decision-making systems (e.g., RL agents) that need multiple options.
        
        Delegates to NavigationService.
        
        Args:
            inc_id: Incident ID
            max_k: Maximum number of hospitals to return
            
        Returns:
            Tuple of (hospital_ids, etas_minutes, wait_times)
        """
        inc = self.incidents[inc_id]
        return self.navigator.get_candidate_hospitals(inc, self.hospitals, max_k=max_k)'''

    def update_after_tick(self, dt_minutes: float = 8.0) -> None:
        # EV updates
        for ev in self.evs.values():
            if ev.nextGrid is not None:
                if ev.state == EvState.BUSY and ev.gridIndex == ev.navdstGrid and ev.assignedPatientId is not None:
                    inc = self.incidents.get(ev.assignedPatientId)
                    if inc is not None:
                        inc.mark_resolved()
                        g = self.grids.get(inc.gridIndex)
                        if g is not None:
                            g.remove_incident(inc.id)
                            del inc
                       
                        ev.release_incident()


                self.move_ev_to_grid(ev.id,ev.nextGrid)

            # 1) EV staying idle in its chosen grid
            if ev.state == EvState.IDLE and ev.gridIndex == ev.sarns.get("action"):
                ev.add_idle(dt_minutes)

            # 2) EV has been dispatched but no reward yet: move it to patient's grid
            elif ev.status == "Dispatching" and ev.assignedPatientId is not None:
                ev.state = EvState.BUSY
                #dispatched += 1
                #print("changed the status after dispatch for the EV", ev.id)
                #inc = self.incidents.get(ev.assignedPatientId)
        
            # 3) Accepted reposition: execute energy/time cost + move
            elif ev.status == "Repositioning" and ev.sarns.get("reward") is not None:
                ev.execute_reposition()

                # optional: reset status after move
                # ev.status = "available"
                # ev.nextGrid = None

            elif ev.state == EvState.BUSY:
                ev.add_busy(8)

                '''
                hc_id = ev.navTargetHospitalId
                if hc_id is not None:
                    hospital = self.hospitals.get(hc_id)
                    if hospital is not None and getattr(hospital, "gridIndex", None) is not None:
                        ev.nextGrid = self.next_grid_towards(ev.gridIndex, hospital.gridIndex)
                        '''


        # Incident updates
        to_delete = []
        for inc_id, inc in self.incidents.items():
            if inc.status == IncidentStatus.UNASSIGNED and inc.waitTime < P_MAX:
                inc.add_wait(dt_minutes)
            elif inc.waitTime > P_MAX:
                inc.status = IncidentStatus.CANCELLED

                grid_idx = inc.gridIndex
    
                if grid_idx in self.grids:
                    g = self.grids[grid_idx]

                    if inc_id in g.incidents:
                        g.incidents.remove(inc_id)
                to_delete.append(inc_id)

        for inc_id in to_delete:
            del self.incidents[inc_id]

        # Recompute grid imbalances
        for g in self.grids.values():
            g.imbalance = g.calculate_imbalance(self.evs, self.incidents)
        
    '''def update_Navigation(self, dt_minutes: float = 8.0) -> None:
        for ev in self.evs.values():
            if ev.state == EvState.BUSY:
                #ev.add_busy(8)
                hc_id = ev.navTargetHospitalId
                if hc_id is not None:
                    hospital = self.hospitals.get(hc_id)
                    if hospital is not None and getattr(hospital, "gridIndex", None) is not None:
                        ev.nextGrid = self.next_grid_towards(ev.gridIndex, hospital.gridIndex)
                        
        '''
    #def update_after_timeslot(self, dt_minutes: float = 8.0) -> None:



                    
                


