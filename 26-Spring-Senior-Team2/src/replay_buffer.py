import numpy as np
import torch
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
    
    def push(self, state, action, reward, next_state, done):
        """경험 튜플 (s_t, a_t, r_t, s_{t+1})을 최대 우선순위로 버퍼에 저장"""
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size):
        """PER 방식을 사용한 샘플 추출 (TD 에러 비례)"""
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:len(self.buffer)]
            
        probs  = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        batch = [self.buffer[idx] for idx in indices]
        
        # PER 가중치 계산 (Importance Sampling)
        total = len(self.buffer)
        self.beta = np.min([1., self.beta + self.beta_increment])
        
        weights  = (total * probs[indices]) ** (-self.beta)
        if weights.max() > 0:
            weights /= weights.max()
            
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 텐서 배치 변환
        state_batch = torch.cat([s for s in states])
        next_state_batch = torch.cat([s for s in next_states])
        action_batch = torch.tensor(actions)
        reward_batch = torch.tensor(rewards)
        done_batch = torch.tensor(dones)
        weights_batch = torch.tensor(weights, dtype=torch.float32)
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch, indices, weights_batch

    def update_priorities(self, batch_indices, batch_priorities):
        """TD 에러에 기반하여 우선순위 업데이트"""
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = max(float(prio), 1e-5) # 최소한의 샘플링 확률 보장
            
    def __len__(self):
        return len(self.buffer)
