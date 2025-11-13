# Controller.py
import random
from typing import Optional, List

import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np

from MAP_env import MAP
from Entities.ev import EvState
from utils.Helpers import (
    build_daily_incident_schedule,
    point_to_grid_index,
    W_MIN, W_MAX, E_MIN, E_MAX,H_MIN, H_MAX,
    utility_repositioning,
)

from DQN import DQNetwork, ReplayBuffer
from Entities.ev import EvState
  
NAV_K = 8
class Controller:
    def __init__(
        self,
        env: MAP,
        ticks_per_ep: int = 180,
        seed: int = 123,
        csv_path: str = "Data/5Years_SF_calls_latlong.csv",
        time_col: str = "Received DtTm",
        lat_col: Optional[str] = "Latitude",
        lng_col: Optional[str] = "Longitude",
        wkt_col: Optional[str] = None,
        
    ):
        self.env = env
        self.ticks_per_ep = ticks_per_ep
        self.rng = random.Random(seed)

        # agent params
        self.epsilon = 0.2
        self.busy_fraction = 0.5

        # DQNs (state=19, action=9 [stay + 8 neighbours])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        state_dim = 19
        action_dim = 9
        self.dqn_reposition_main = DQNetwork(state_dim, action_dim).to(self.device)
        self.dqn_reposition_target = DQNetwork(state_dim, action_dim).to(self.device)
        self.dqn_reposition_target.load_state_dict(self.dqn_reposition_main.state_dict())
        self.opt_reposition = torch.optim.Adam(self.dqn_reposition_main.parameters(), lr=1e-3)
        self.buffer_reposition = ReplayBuffer(100_000)

        state_dim_nav = 2 * NAV_K
        action_dim_nav = NAV_K
        self.nav_step = 0
        self.nav_target_update = 500  # soft update every N training steps
        self.nav_tau = 0.005          # Polyak factor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dqn_navigation_main = DQNetwork(state_dim_nav, action_dim_nav).to(self.device)
        self.dqn_navigation_target = DQNetwork(state_dim_nav, action_dim_nav).to(self.device)
        self.dqn_navigation_target.load_state_dict(self.dqn_navigation_main.state_dict())
        self.opt_navigation = torch.optim.Adam(self.dqn_navigation_main.parameters(), lr=1e-3)
        self.buffer_navigation = ReplayBuffer(100_000)

        print("[Controller] DQNs initialised:")
        print("  Reposition main / target:", sum(p.numel() for p in self.dqn_reposition_main.parameters()))
        print("  Navigation main / target:", sum(p.numel() for p in self.dqn_navigation_main.parameters()))
        print("  Device:", self.device)

        # dataset (for incident schedule per episode)
        self.df = pd.read_csv(csv_path)
        self.time_col = time_col
        self.lat_col = lat_col
        self.lng_col = lng_col
        self.wkt_col = wkt_col

        self._schedule = None
        self._current_day = None

        # EV randomisation bounds (already enforced in Helpers via constants)
        self.max_idle_minutes = W_MAX
        self.max_idle_energy = E_MAX

    # ---------- state/action helpers ----------
    def _pad_neighbors(self, nbs: List[int]) -> tuple[list[int], list[int]]:
        N = 8
        n = (nbs[:N] if len(nbs) >= N else nbs + [-1] * (N - len(nbs)))
        mask = [1 if x != -1 else 0 for x in n]
        n_feat = [0 if x == -1 else x for x in n]
        return n_feat, mask

    def _build_state(self, ev) -> list[float]:
        gi = ev.gridIndex
        nbs = self.env.grids[gi].neighbours
        n8, _ = self._pad_neighbors(nbs)

        vec = [gi]
        for nb in n8:
            if nb == 0:
                imb = 0.0
            else:
                # Use new Grid method to calculate imbalance (OOP refactor)
                imb = float(self.env.grids[nb].calculate_imbalance(self.env.evs, self.env.incidents))
            vec.extend([nb, imb])

        vec.extend([float(ev.aggIdleTime), float(ev.aggIdleEnergy)])
        return vec  # length 19

    def _select_action(self, state_vec: list[float], gi: int) -> int:
        # 9 slots: [stay] + 8 neighbours (padded)
        nbs = self.env.grids[gi].neighbours
        n8, mask8 = self._pad_neighbors(nbs)
        actions = [gi] + n8
        mask = [1] + mask8

        if self.rng.random() < self.epsilon:
            valid = [i for i, m in enumerate(mask) if m == 1]
            slot = self.rng.choice(valid) if valid else 0
            return actions[slot]

        s = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.dqn_reposition_main(s).detach().cpu().numpy().ravel()
        for i, m in enumerate(mask):
            if m == 0:
                q[i] = -1e9
        slot = int(np.argmax(q))
        return actions[slot]

    # ---------- episode reset ----------
    def _reset_episode(self) -> None:
        import pandas as pd

        # 1) clear incidents from env + grids
        self.env.incidents.clear()
        for g in self.env.grids.values():
            g.incidents.clear()

        # 2) randomise EV placement and base fields (no offers yet)
        all_idx = list(self.env.grids.keys())
        for ev in self.env.evs.values():
            gi = self.rng.choice(all_idx)
            self.env.move_ev_to_grid(ev.id, gi)

            ev.aggIdleTime = self.rng.uniform(0.0, self.max_idle_minutes)
            ev.aggIdleEnergy = self.rng.uniform(0.0, self.max_idle_energy)

            # reset per-episode state
            ev.set_state(EvState.IDLE)
            ev.status = "available"
            ev.nextGrid = ev.gridIndex  # default: stay unless accepted later
            # clear NAV fields
            ev.navTargetHospitalId = None
            ev.navEtaMinutes = 0.0
            ev.navUtility = 0.0
            # clear SARNS
            ev.sarns.clear()
            ev.sarns["state"] = None
            ev.sarns["action"] = None
            ev.sarns["utility"] = None
            ev.sarns["reward"] = None
            ev.sarns["next_state"] = None

        # 3) busy/idle split only (do NOT build offers here)
        for ev in self.env.evs.values():
            if self.rng.random() < self.busy_fraction:
                ev.set_state(EvState.SERVICE)
                ev.status = "busy"
                ev.nextGrid = ev.gridIndex

        # 4) pick a random day and build schedule
        series = pd.to_datetime(self.df[self.time_col], errors="coerce").dt.normalize().dropna()
        days = series.unique()
        if len(days) == 0:
            raise RuntimeError(f"No valid dates in dataset for {self.time_col}")
        self._current_day = pd.Timestamp(self.rng.choice(list(days)))
        self._schedule = build_daily_incident_schedule(
            self.df,
            self._current_day,
            time_col=self.time_col,
            lat_col=self.lat_col,
            lng_col=self.lng_col,
            wkt_col=self.wkt_col,
        )

        # 5) initialise hospital waits for this episode (random range)
        #    signature: reset_hospital_waits(low_min, high_min, seed)
        self.env.reset_hospital_waits(low_min=H_MIN, high_min=H_MAX, seed=self.rng.randint(1, 10_000))

        # 6) log
        total_today = 0 if not self._schedule else sum(len(v) for v in self._schedule.values())
        print(f"[Controller] _reset_episode ready: day={self._current_day.date()} incidents_today={total_today}")

    # ---------- per-tick ----------
    def _spawn_incidents_for_tick(self, t: int) -> None:
        todays_at_tick = self._schedule.get(t, []) if self._schedule else []
        for (lat, lng) in todays_at_tick:
            gi = point_to_grid_index(lat, lng, self.env.lat_edges, self.env.lng_edges)
            self.env.create_incident(grid_index=gi, location=(lat, lng), priority="MED")

    def _tick(self, t: int) -> None:
        # 1) spawn incidents for this tick
        self._spawn_incidents_for_tick(t)

        # 2) build offers for idle EVs (state, action, utility) — “stay” is not an offer
        n_offers = self._build_offers_for_idle_evs()

        # 3) Algorithm 1: accept offers (sets nextGrid and reward; no movement yet)
        self.env.accept_reposition_offers()

        # 4) Gridwise dispatch (Algorithm 2) using EVs that stayed/rejected
        dispatches = self.env.dispatch_gridwise(beta=0.5)

        # 5) Debug snapshot so you can see it running
        todays = self._schedule.get(t, []) if self._schedule else []
        accepted = sum(
            1
            for ev in self.env.evs.values()
            if ev.state == EvState.IDLE and ev.sarns.get("reward") not in (None, 0.0)
        )
        print(
            f"Tick {t:03d} | incidents+{len(todays):2d} | offers={n_offers:2d} | "
            f"accepted={accepted:2d} | dispatched={len(dispatches):2d}"
        )

        # 6) NAV per tick (only for dispatched EVs)
        #    Update hospital waits, then choose hospital via DQN (action), reward = U^N, store transition.
        self.env.tick_hospital_waits(lam=0.04, wmin=5.0, wmax=90.0)

        for (eid, inc_id, _Ud) in dispatches:
            ev = self.env.evs[eid]

            # If EV already at its hospital's grid (when you implement movement), skip NAV
            hid_prev = getattr(ev, "navTargetHospitalId", None)
            if hid_prev is not None:
                hc_prev = self.env.hospitals.get(hid_prev)
                if hc_prev is not None and ev.gridIndex == hc_prev.gridIndex:
                    continue

            # Build state over candidates (patient -> hospital ETAs, hospital waits)
            s_vec, cand_hids, mask = self._build_nav_state(inc_id)
            if sum(mask) == 0:
                continue  # no valid hospitals

            # Epsilon-greedy action over candidates
            slot = self._select_nav_action(s_vec, mask)
            hid = cand_hids[slot]

            # Reward = U^N for this choice
            hc = self.env.hospitals[hid]
            # recompute current eta & wait for the chosen one
            hids, etas, waits = self.env.get_nav_candidates(inc_id, max_k=NAV_K)
            j = hids.index(hid)
            eta_ph, wait_h = etas[j], waits[j]
            r_nav = self._compute_un(eta_ph, wait_h)

            # Record on EV (not movement yet)
            ev.navTargetHospitalId = hid
            ev.navEtaMinutes = eta_ph
            ev.navUtility = r_nav

            # Build next-state immediately (you can also defer to next tick)
            s2_vec, _, _ = self._build_nav_state(inc_id)
            done = 0.0  # no terminal yet (arrival handled when you add movement)

            # Push to replay
            s_t = torch.tensor(s_vec, dtype=torch.float32)
            a_t = torch.tensor(slot, dtype=torch.int64)
            r_t = torch.tensor(r_nav, dtype=torch.float32)
            s2_t = torch.tensor(s2_vec, dtype=torch.float32)
            d_t = torch.tensor(done, dtype=torch.float32)
            self.buffer_navigation.push(s_t, a_t, r_t, s2_t, d_t)

        # Single nav training step per tick
        self._train_navigation(batch_size=64, gamma=0.99)



    def _debug_print_navigation_utility(self, dispatches, H_min: float, H_max: float):
        from utils.Helpers import travel_minutes, utility_navigation_un

        if not getattr(self.env, "hospitals", None):
            print("  NAV: no hospitals initialised; skip navigation utility check.")
            return

        shown = 0
        for (eid, inc_id, _Ud) in dispatches:
            inc = self.env.incidents[inc_id]
            lat_p, lng_p = inc.location

            best_hc, best_eta = None, float("inf")
            for hc in self.env.hospitals.values():
                eta = travel_minutes(lat_p, lng_p, hc.loc[0], hc.loc[1], kmph=40.0)
                if eta < best_eta:
                    best_eta, best_hc = eta, hc

            if best_hc is None:
                print("  NAV: no hospital found (unexpected).")
                continue

            wait_at_hc = float(getattr(best_hc, "waitTime", 0.0))  # ← camelCase
            W_busy = max(0.0, H_max - (best_eta + wait_at_hc))
            U_N = utility_navigation_un(W_busy, H_min=H_min, H_max=H_max)

            print(
                f"  NAV: EV{eid:02d} -> Inc {inc_id} via HC{best_hc.id:02d} | "
                f"ETA_to_HC={best_eta:.1f}m wait@HC={wait_at_hc:.1f}m | "
                f"W_busy={W_busy:.1f} -> U^N={U_N:.3f}"
            )
            shown += 1
            if shown >= 3:
                break


    # ---------- run one episode ----------
    def run_one_episode(self) -> None:
        print("[Controller] Resetting episode...")
        self._reset_episode()

        # Safe day print (avoid NoneType)
        if self._current_day is not None:
            print(f"[Controller] Day selected: {self._current_day.date()}")
        else:
            print("[Controller] Warning: No day selected (dataset may be empty or invalid).")

        # Safe schedule summary
        if self._schedule:
            total_incidents = sum(len(v) for v in self._schedule.values())
            print(f"[Controller] Total incidents today: {total_incidents}")
        else:
            print("[Controller] Warning: Schedule not built — no incidents will spawn.")


        # Run all ticks
        for t in range(self.ticks_per_ep):
            self._tick(t)
            if t % 30 == 0:
                print(f"Tick {t:03d}: incidents so far = {len(self.env.incidents)}")

        print(f"[Controller] Episode complete. Total incidents created: {len(self.env.incidents)}")

    def _build_offers_for_idle_evs(self) -> int:
        offers = 0
        for ev in self.env.evs.values():
            if ev.state != EvState.IDLE or ev.status != "available":
                ev.nextGrid = ev.gridIndex
                ev.sarns["state"] = None
                ev.sarns["action"] = None
                ev.sarns["utility"] = None
                ev.sarns["reward"] = 0.0
                ev.sarns["next_state"] = None
                continue

            s_vec = self._build_state(ev)
            a_gi  = self._select_action(s_vec, ev.gridIndex)

            ev.sarns["state"] = s_vec
            ev.sarns["action"] = a_gi

            if a_gi == ev.gridIndex:
                # “Stay” is NOT a reposition offer
                ev.nextGrid = ev.gridIndex
                ev.sarns["utility"] = None
                ev.sarns["reward"] = 0.0   # no reward from acceptance step
                ev.sarns["next_state"] = None
                continue  # do not count as an offer

            # Only moving proposals get a utility and enter acceptance
            u = utility_repositioning(
                W_idle=ev.aggIdleTime, E_idle=ev.aggIdleEnergy,
                W_min=W_MIN, W_max=W_MAX, E_min=E_MIN, E_max=E_MAX
            )
            ev.sarns["utility"] = float(u)
            ev.sarns["reward"] = None
            ev.sarns["next_state"] = None
            ev.nextGrid = ev.gridIndex  # pending; only changes if accepted
            offers += 1

        return offers
    def _build_nav_state(self, inc_id: int) -> tuple[list[float], list[int], list[int]]:
        
        hids, etas, waits = self.env.get_nav_candidates(inc_id, max_k=NAV_K)

        # normalise by H_MAX so values are ~0..1
        feats: list[float] = []
        for i in range(NAV_K):
            if i < len(hids):
                feats.append(etas[i] / max(H_MAX, 1e-6))
                feats.append(waits[i] / max(H_MAX, 1e-6))
            else:
                feats.extend([0.0, 0.0])

        mask = [1]*len(hids) + [0]*(NAV_K - len(hids))
        return feats, hids, mask

    def _select_nav_action(self, s_vec: list[float], mask: list[int]) -> int:
        """
        Epsilon-greedy over NAV_K slots. Returns slot index 0..NAV_K-1.
        Masked slots are invalid.
        """
        import numpy as np
        if self.rng.random() < self.epsilon:
            valid = [i for i, m in enumerate(mask) if m == 1]
            return self.rng.choice(valid) if valid else 0

        s = torch.tensor(s_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.dqn_navigation_main(s).detach().cpu().numpy().ravel()
        for i, m in enumerate(mask):
            if m == 0:
                q[i] = -1e9
        return int(np.argmax(q))

    def _compute_un(self, eta_ph: float, wait_h: float) -> float:
        # U^N per Eq. (14), using Helpers util you already added
        from utils.Helpers import utility_navigation_un
        # remaining slack style: larger slack ⇒ higher utility
        W_busy = max(0.0, H_MAX - (eta_ph + wait_h))
        return utility_navigation_un(W_busy, H_MIN, H_MAX)
    def _train_navigation(self, batch_size: int = 64, gamma: float = 0.99):
        # need enough samples
        if len(self.buffer_navigation) < batch_size:
            return

        # sample a batch (ReplayBuffer.sample should accept device=... and return tensors)
        try:
            s, a, r, s2, done = self.buffer_navigation.sample(batch_size, device=self.device)
        except TypeError:
            # fallback if your sample() returns python lists/np arrays
            batch = self.buffer_navigation.sample(batch_size)
            s   = torch.stack([torch.as_tensor(x, dtype=torch.float32, device=self.device) for x in batch[0]])
            a   = torch.as_tensor(batch[1], dtype=torch.long,   device=self.device)
            r   = torch.as_tensor(batch[2], dtype=torch.float32, device=self.device)
            s2  = torch.stack([torch.as_tensor(x, dtype=torch.float32, device=self.device) for x in batch[3]])
            done= torch.as_tensor(batch[4], dtype=torch.float32, device=self.device)

        # target: r + γ (1-done) max_a' Q_target(s')
        with torch.no_grad():
            q2 = self.dqn_navigation_target(s2).max(dim=1).values
            y  = r + gamma * (1.0 - done) * q2

        # current Q(s,a)
        q = self.dqn_navigation_main(s).gather(1, a.view(-1, 1)).squeeze(1)

        loss = torch.nn.functional.smooth_l1_loss(q, y)
        self.opt_navigation.zero_grad()
        loss.backward()
        self.opt_navigation.step()

        # soft-update target
        self.nav_step += 1
        if self.nav_step % self.nav_target_update == 0:
            with torch.no_grad():
                for p_t, p in zip(self.dqn_navigation_target.parameters(),
                                self.dqn_navigation_main.parameters()):
                    p_t.data.mul_(1.0 - self.nav_tau).add_(self.nav_tau * p.data)

        if self.nav_step % 500 == 0:
            print(f"[Controller] NAV train step={self.nav_step} loss={loss.item():.4f}")

    def _build_state_for_grid(self, ev, grid_index: int) -> list[float]:

        # neighbour list of target grid
        nbs = self.env.grids[grid_index].neighbours
        # pack (nb_index, nb_imbalance) pairs into a fixed-size slice
        pairs = []
        for gi in nbs:
            pairs.append(gi)
            # Use new Grid method to calculate imbalance (OOP refactor)
            pairs.append(self.env.grids[gi].calculate_imbalance(self.env.evs, self.env.incidents))
        # pad to a fixed length if you use a cap (e.g., 16 values)
        MAX_PAIRS = 8  # 8 neighbours
        while len(pairs) < 2 * MAX_PAIRS:
            pairs.append(0)
        pairs = pairs[: 2 * MAX_PAIRS]

        # add own grid and EV accumulators
        s = [
            grid_index,
            # Use new Grid method to calculate imbalance (OOP refactor)
            self.env.grids[grid_index].calculate_imbalance(self.env.evs, self.env.incidents),
            ev.aggIdleTime,
            ev.aggIdleEnergy,
        ] + pairs
        return [float(x) for x in s]

    def _push_reposition_transition(self, ev, accepted: bool) -> None:
        """
        Take what we stored in ev.sarns, build s', and push (s,a,r,s').
        """
        s  = ev.sarns.get("state")
        a  = ev.sarns.get("action")
        r  = ev.sarns.get("reward", 0.0) or 0.0
        if s is None or a is None:
            return
        # next-state is built wrt the EV's chosen nextGrid if accepted,
        # otherwise its current grid (stay)
        next_g = ev.nextGrid if accepted else ev.gridIndex
        s2 = self._build_state_for_grid(ev, next_g)
        done = 0.0  # not terminal at this stage

        # push to replay (tensorise once; buffer will normalise if needed)
        import torch
        s_t  = torch.tensor(s,  dtype=torch.float32)
        a_t  = torch.tensor(a,  dtype=torch.int64)
        r_t  = torch.tensor(r,  dtype=torch.float32)
        s2_t = torch.tensor(s2, dtype=torch.float32)
        d_t  = torch.tensor(done, dtype=torch.float32)
        self.buffer_reposition.push(s_t, a_t, r_t, s2_t, d_t)
