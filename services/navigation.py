# services/navigation.py
"""
Navigation service: handles hospital selection and route planning for EVs.
"""
from typing import Dict, Tuple
from Entities.Incident import Incident
from Entities.Hospitals import Hospital
from Entities.ev import EV
from utils.Helpers import utility_navigation


class NavigationService:
    """Service for managing hospital navigation and selection."""
    
    @staticmethod
    def select_hospital_for_incident(
        incident: Incident,
        hospitals: Dict[int, Hospital],
        evs: EV
    ) -> None:

        best_hid, best_w_busy = -1, float("inf")
        patient_lat, patient_lng = incident.location
        
        for hid, hc in hospitals.items():
            eta = hc.estimate_eta_minutes(patient_lat, patient_lng, kmph=40.0)
            wait = float(getattr(hc, "waitTime", 0.0))
            W_Busy = eta+wait
            if W_Busy < best_w_busy:
                best_w_busy, best_hid = eta, hid
            
        if best_hid is not None:
            best_hc = hospitals[best_hid]
            best_hc.currentEvId = evs.id
            evs.navTargetHospitalId = best_hid
            evs.nextGrid = best_hc.gridIndex
            evs.sarns["reward"] = utility_navigation(best_w_busy)


    
    @staticmethod
    def get_candidate_hospitals(
        incident: Incident,
        hospitals: Dict[int, Hospital],
        max_k: int = 8,
    ) -> Tuple[list[int], list[float], list[float]]:
        """
        Get the top K nearest hospitals for a patient incident.
        
        Useful for decision-making systems that need multiple options
        (e.g., reinforcement learning agents).
        
        Args:
            incident: The incident/patient location
            hospitals: Dict mapping hospital IDs to Hospital objects
            max_k: Maximum number of hospitals to return
            
        Returns:
            Tuple of (hospital_ids, etas_minutes, wait_times)
        """
        patient_lat, patient_lng = incident.location
        return Hospital.select_nearest_hospitals(
            hospitals,
            patient_lat,
            patient_lng,
            max_k=max_k,
        )
