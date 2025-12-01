# entities/incident.py
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple
from datetime import datetime


class IncidentStatus(Enum):
    UNASSIGNED = auto()
    ASSIGNED = auto()
    SERVICING = auto()

    RESOLVED = auto()
    CANCELLED = auto()


class Priority(Enum):
    LOW = 1
    MED = 2
    HIGH = 3
    CRIT = 4


LatLng = Tuple[float, float]


@dataclass
class Incident:
    id: int
    gridIndex: int
    timestamp: datetime
    location: LatLng
    dropLocation: Optional[LatLng] = None
    priority: Priority = Priority.MED
    status: IncidentStatus = IncidentStatus.UNASSIGNED
    waitTime: float = 0.0
    serviceTime: float = 0.0
    remainingWaitTime: Optional[float] = None
    assignedEvId: Optional[int] = None
    

    def assign_ev(self, ev_id: int) -> None:
        self.assignedEvId = ev_id
        self.status = IncidentStatus.ASSIGNED
        

    '''def start_service(self) -> None:
        self.status = IncidentStatus.SERVICING

    def start_drop(self) -> None:
        self.status = IncidentStatus.SERVICING'''

    def mark_resolved(self) -> None:
        self.status = IncidentStatus.RESOLVED
        self.remainingWaitTime = 0.0

    def cancel_incident(self) -> None:
        self.status = IncidentStatus.CANCELLED

    def add_wait(self, dt: float) -> None:
        self.waitTime += dt
        if self.remainingWaitTime is not None:
            self.remainingWaitTime = max(0.0, self.remainingWaitTime - dt)
    
    # ========== Domain logic for incident state ==========
    
    def is_unassigned(self) -> bool:
        """Check if this incident is waiting for assignment."""
        return self.assignedEvId is None or self.status == IncidentStatus.UNASSIGNED
    
    def get_wait_minutes(self) -> float:
        """Get accumulated wait time in minutes."""
        return self.waitTime
    
    def get_urgency_score(self) -> float:
        """
        Return urgency based on priority and wait time.
        Higher priority and longer waits = higher urgency.
        """
        priority_weight = {
            Priority.LOW: 1.0,
            Priority.MED: 2.0,
            Priority.HIGH: 3.0,
            Priority.CRIT: 4.0,
        }
        return priority_weight.get(self.priority, 1.0) * (1.0 + self.waitTime / 30.0)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "gridIndex": self.gridIndex,
            "timestamp": self.timestamp.isoformat(),
            "location": self.location,
            "dropLocation": self.dropLocation,
            "priority": self.priority.name,
            "status": self.status.name,
            "waitTime": self.waitTime,
            "serviceTime": self.serviceTime,
            "remainingWaitTime": self.remainingWaitTime,
            "assignedEvId": self.assignedEvId,
        }
