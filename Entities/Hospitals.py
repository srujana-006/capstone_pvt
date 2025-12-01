# entities/hospital.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import math

LatLng = Tuple[float, float]


@dataclass
class Hospital:
    id: int
    loc: LatLng
    gridIndex: int
    waitTime: float = 0.0
    services: List[str] = field(default_factory=list)

    queue: List[int] = field(default_factory=list)          # patient ids
    currentEvId: Optional[int] = None

    def enqueue_patient(self, patient_id: int) -> None:
        self.queue.append(patient_id)

    def start_service(self, ev_id: int) -> None:
        self.currentEvId = ev_id

    def finish_service(self) -> Optional[int]:
        self.currentEvId = None
        if self.queue:
            return self.queue.pop(0)
        return None

    def add_wait(self, dt: float) -> None:
        self.waitTime += dt

    def add_service(self, name: str) -> None:
        if name not in self.services:
            self.services.append(name)

    # ========== Domain logic for hospital selection & navigation ==========
    
    def haversine_distance_km(self, lat2: float, lng2: float) -> float:
        """
        Calculate great-circle distance in km from this hospital to a point (lat2, lng2).
        Uses Haversine formula.
        """
        R = 6371.0  # Earth radius in km
        lat1, lng1 = self.loc
        
        dlat = math.radians(lat2 - lat1)
        dlng = math.radians(lng2 - lng1)
        a = (math.sin(dlat/2)**2 + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlng/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c
    
    def estimate_eta_minutes(self, lat2: float, lng2: float, kmph: float = 40.0) -> float:
        """
        Estimate ETA (in minutes) from this hospital to a point (lat2, lng2)
        at a constant average speed.
        """
        km = self.haversine_distance_km(lat2, lng2)
        return 60.0 * km / max(kmph, 1e-6)
    
    @staticmethod
    def select_nearest_hospitals(
        hospitals_dict: Dict[int, Hospital],
        patient_lat: float,
        patient_lng: float,
        max_k: int = 8
    ) -> Tuple[List[int], List[float], List[float]]:
        """
        Find the nearest hospitals to a patient location and return their details.
        
        Args:
            hospitals_dict: Dict mapping hospital IDs to Hospital objects
            patient_lat, patient_lng: Patient location
            max_k: Maximum number of hospitals to return
            
        Returns:
            Tuple of (hospital_ids, etas_minutes, wait_times)
        """
        items = []
        for hid, hc in hospitals_dict.items():
            eta = hc.estimate_eta_minutes(patient_lat, patient_lng, kmph=40.0)
            wait = float(getattr(hc, "waitTime", 0.0))
            items.append((hid, eta, wait))
        
        # Sort by ETA (nearest first)
        items.sort(key=lambda x: x[1])
        items = items[:max_k]
        
        hids = [hid for (hid, _, _) in items]
        etas = [eta for (_, eta, _) in items]
        waits = [w for (_, _, w) in items]
        
        return hids, etas, waits
    
    def select_best_hospital(self,
        hospitals_dict: Dict[int, Hospital],
        patient_lat: float,
        patient_lng: float
    ) -> Tuple[int, float]:
        """
        Select the best (nearest) hospital for a patient.
        
        Returns:
            Tuple of (hospital_id, eta_minutes)
        """
        best_hid, best_eta = -1, float("inf")
        for hid, hc in hospitals_dict.items():
            eta = hc.estimate_eta_minutes(patient_lat, patient_lng, kmph=40.0)
            if eta < best_eta:
                best_eta, best_hid = eta, hid
        return best_hid, best_eta

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "loc": self.loc,
            "gridIndex": self.gridIndex,
            "waitTime": self.waitTime,
            "services": list(self.services),
            "queueLen": len(self.queue),
            "currentEvId": self.currentEvId,
        }
