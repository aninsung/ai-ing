"""DDQN policy for cost-aware sequential region observation."""
from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn


class RegionQNetwork(nn.Module):
    def __init__(self, state_dim, actions=10):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(),
                                 nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, actions))

    def forward(self, state):
        return self.net(state)


class TransitionBuffer:
    def __init__(self, capacity=100_000): self.data = deque(maxlen=capacity)
    def add(self, *transition): self.data.append(transition)
    def sample(self, size): return random.sample(self.data, size)
    def __len__(self): return len(self.data)


def crop_grid(images, grid=3):
    """Return full image followed by nine square grid crops, padded to square."""
    _, _, height, width = images.shape
    size = min(height, width)
    top, left = (height-size)//2, (width-size)//2
    square = images[:, :, top:top+size, left:left+size]
    crops = [square]
    step = size // grid
    for row in range(grid):
        for col in range(grid):
            y, x = row*step, col*step
            crops.append(square[:, :, y:y+step, x:x+step])
    return crops


def make_state(global_logits, current_logits, visited, remaining):
    return torch.cat([global_logits, current_logits,
                      torch.as_tensor(visited, dtype=torch.float32),
                      torch.tensor([remaining], dtype=torch.float32)])


def select_action(q_net, state, valid, epsilon, device):
    valid = np.asarray(valid, dtype=bool)
    if random.random() < epsilon:
        return random.choice(np.flatnonzero(valid).tolist())
    with torch.no_grad():
        values = q_net(state.to(device).unsqueeze(0))[0].detach().cpu().numpy()
    values[~valid] = -np.inf
    return int(values.argmax())
