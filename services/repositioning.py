# services/repositioning.py
"""
Repositioning service: handles EV repositioning offers and acceptance.
Manages Algorithm 1: accept reposition offers.
"""
from typing import Dict, List, Tuple
from collections import defaultdict

from Entities.GRID import Grid

from Entities.ev import EV, EvState
from Entities.GRID import Grid
from Entities.Incident import Incident
from utils.Helpers import utility_repositioning


class RepositioningService:
   
    
    def accept_reposition_offers(
        self,
        evs: Dict[int, EV],
        grids: Dict[int, Grid],
        incidents: Dict) -> None:
        # Group offers by destination grid
        offers_by_g = defaultdict(list)  # g_idx -> list[(utility, ev_id)]
        for g_idx, g in grids.items():
            neighbour_evs = []
            
            for nb in g.neighbours:
                if nb not in grids:
                    continue
                neigh_grid = grids[nb]
                for ev_id in neigh_grid.evs:
                    neighbour_evs.append(evs[ev_id])
            # 2) Build offers_g: offers from neighbour EVs that want THIS grid
            offers_g = []   # list of tuples (utility, ev_id, ev_obj)
            for v in neighbour_evs:
                if v.state != EvState.IDLE:
                    continue
                dst = v.sarns.get("action")
                #u = v.sarns.get("utility")
                u = utility_repositioning(v.aggIdleTime,v.aggIdleEnergy)
                if dst is None or u is None:
                        continue
                if dst == g_idx:
                    offers_g.append((float(u), v.id, v))

            #if not offers_g:
                #continue
            offers_g.sort(key=lambda x: x[0], reverse=True)
            # 4) Capacity: how many EVs this grid "needs"
            imbalance = g.imbalance
            cap = max(0, imbalance)
            accepted = 0
            while accepted < cap and offers_g:
                u_val, ev_id, v_obj = offers_g.pop(0)
                # and record the reposition utility as reward
                #v_obj.execute_reposition()
                v_obj.status = "Repositioning"
                v_obj.sarns["reward"] = u_val
                #print("reward griven after acceptance", v_obj.sarns["reward"])
                v_obj.nextGrid = g_idx
                accepted += 1
                #g.add_ev(ev_id)
        

    '''def execute_repositions(
        self,
        evs: Dict[int, EV],
        grids: Dict[int, Grid],
    ) -> None:
        """
        Step repositioning: apply accepted moves and clear pending decisions.
        
        Args:
            evs: Dict mapping EV IDs to EV objects
            grids: Dict mapping grid indices to Grid objects
        """
        for ev in evs.values():            
            if ev.state != EvState.IDLE:
                ev.nextGrid = None
                continue
            
            dst = ev.nextGrid
            if dst is None:
                # Not decided; treat as stay
                ev.nextGrid = None
                continue
            
            if dst != ev.gridIndex:
                # Execute the move (caller should use move_ev_to_grid from MAP)
                # Here we just mark it; MAP will handle the actual grid list updates
                pass
            
            # Clear pending reposition
            ev.execute_reposition()'''
