import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class DQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)
import random
import torch

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0    # indexing for circular buffer

    def push(self, s, a, r, s2, done):
        """Store one transition: (state, action, reward, next_state, done)"""
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)  # expand buffer

        self.buffer[self.pos] = (s, a, r, s2, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, device):
        """Return a batch of transitions converted to tensors."""
        
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)
        rewards_clean = [0.0 if r is None else float(r) for r in rewards]
        S  = torch.stack(states).to(device)
        A  = torch.tensor(actions, dtype=torch.long, device=device)
        R  = torch.tensor(rewards_clean, dtype=torch.float32, device=device)
        S2 = torch.stack(next_states).to(device)
        D  = torch.tensor(dones, dtype=torch.float32, device=device)
        #if any(r is None for r in rewards):
            #print("[ReplayBuffer] WARNING: None reward found in batch:", rewards)

        return S, A, R, S2, D

    def __len__(self):
        return len(self.buffer)


'''class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, s, a, r, s2, done):
        """
        Store transitions as-is. They can be lists/np arrays/tensors.
        We'll normalize to torch tensors on sample().
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (s, a, r, s2, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, device: str | torch.device = "cpu"):
        """
        Return (S, A, R, S2, D) as torch tensors on the requested device.
        Works even if items were pushed as tensors.
        """
        #batch = random.sample(self.buffer, batch_size)
        actual_size = min(len(self.buffer), batch_size)
        batch = random.sample(self.buffer, actual_size)
        return batch

        states, actions, rewards, next_states, dones = [], [], [], [], []
        for s, a, r, s2, d in batch:
            # states
            if isinstance(s, torch.Tensor): s = s.detach().cpu()
            if isinstance(s2, torch.Tensor): s2 = s2.detach().cpu()
            states.append(torch.as_tensor(s, dtype=torch.float32))
            next_states.append(torch.as_tensor(s2, dtype=torch.float32))

            # scalars
            if isinstance(a, torch.Tensor): a = int(a.item())
            if isinstance(r, torch.Tensor): r = float(r.item())
            if isinstance(d, torch.Tensor): d = float(d.item())
            actions.append(int(a))
            rewards.append(float(r))
            dones.append(float(d))
        if len(states) or len(next_states) == 0:
            raise ValueError("Replay buffer returned an empty batch!")
            
        else: 
            S  = torch.stack(states, dim=0).to(device)
            S2 = torch.stack(next_states, dim=0).to(device)
            A  = torch.as_tensor(actions, dtype=torch.long,    device=device)
            R  = torch.as_tensor(rewards, dtype=torch.float32, device=device)
            D  = torch.as_tensor(dones,   dtype=torch.float32, device=device)
        return S, A, R, S2, D

    def __len__(self):
        return len(self.buffer)
'''

