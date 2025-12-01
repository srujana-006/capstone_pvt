# entities/ev.py
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Tuple, Dict, Any

LatLng = Tuple[float, float]

class EvState(Enum):
    IDLE = auto()
    
    BUSY = auto()

@dataclass
class EV:
    id: int
    gridIndex: int
    location: LatLng
    state: EvState = EvState.IDLE
    nextGrid: Optional[int] = None
    status: str = "Idle"
    assignedPatientId: Optional[int] = None

    aggIdleTime: float = 0.0
    aggIdleEnergy: float = 0.0
    aggBusyTime: float = 0.0
    navTargetHospitalId: int | None = None  # hospital currently chosen for navigation
    navdstGrid: int | None = None        # grid index of that hospital
    navEtaMinutes: float = 0.0              # latest ETA to that hospital
    navUtility: float = 0.0                 

    # sarns now a dict, as requested
    sarns: Dict[str, float ] = field(default_factory=dict)

    def assign_incident(self, patient_id: int) -> None:
        self.assignedPatientId = patient_id
        #self.state = EvState.BUSY
        self.status = "Dispatching"
        self.aggIdleEnergy = 0.0
        self.aggIdleTime = 0.0
        

    def release_incident(self) -> None:
        self.assignedPatientId = None
        self.status = "Idle"
        self.state = EvState.IDLE
        self.nextGrid = None
        self.navTargetHospitalId = None
        self.navEtaMinutes = 0.0
        self.navUtility = 0.0

    def move_to(self, grid_index: int, new_loc: LatLng) -> None:
        self.gridIndex = grid_index
        self.location = new_loc
    
    def set_state(self, new_state: EvState) -> None:
        self.state = new_state

    def add_idle(self, dt: float) -> None:
        self.aggIdleTime += dt
        
    def add_busy(self, dt: float) -> None:
        self.aggBusyTime += dt


    # ========== Repositioning logic ==========
   
    def execute_reposition(self) -> None:
        """
        Execute the reposition decision made in this tick.
        This should be called after move_to() has been invoked by MAP.
        """
        
        self.aggIdleEnergy += 0.12  # Fixed energy cost for repositioning from one grid to another
        self.aggIdleTime += 8.0       # Fixed time cost for repositioning from one grid to another
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "gridIndex": self.gridIndex,
            "location": self.location,
            "nextGrid": self.nextGrid,
            "state": self.state.name,
            "status": self.status,
            "assignedPatientId": self.assignedPatientId,
            "aggIdleTime": self.aggIdleTime,
            "aggIdleEnergy": self.aggIdleEnergy,
            "sarns": self.sarns,
        }
