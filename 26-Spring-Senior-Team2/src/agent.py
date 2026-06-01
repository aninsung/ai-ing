import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import random
import numpy as np
import pandas as pd
import os
import math
from .networks import DDQNNetwork

class DDQNAgent:
    def __init__(self, num_classes, config, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.num_classes = num_classes
        
        # Phase 3: Agent Architecture parameters
        self.gamma = config['agent']['gamma_factor']
        self.epsilon = config['agent']['epsilon_start']
        self.epsilon_min = config['agent']['epsilon_end']
        self.epsilon_decay = config['agent']['epsilon_decay']
        
        # MainNet (학습용) 및 TargetNet (타겟 Q값 고정용) 초기화
        self.main_net = DDQNNetwork(num_classes, config['agent']['backbone']).to(self.device)
        self.target_net = DDQNNetwork(num_classes, config['agent']['backbone']).to(self.device)
        self.target_net.load_state_dict(self.main_net.state_dict())
        self.target_net.eval() # 타깃 네트워크는 학습하지 않음
        
        self.optimizer = optim.Adam(self.main_net.parameters(), lr=config['train']['learning_rate'])
        self.criterion = nn.MSELoss()
        # Reduce LR on plateau based on best score (max mode)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=2)

    def select_action(self, state):
        """
        입력 상태(State)에 대해 epsilon-greedy 전략으로 행동(Action) 선택
        """
        if random.random() < self.epsilon:
            # 탐험 (Exploration)
            return random.randint(0, self.num_classes - 1)
        
        # 활용 (Exploitation)
        with torch.no_grad():
            state = state.to(self.device).float()
            q_values = self.main_net(state)
            return q_values.argmax(dim=1).item()

    def decay_epsilon(self):
        """
        학습 진행에 따른 epsilon 값 점진적 감소
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def update_target_network(self):
        """
        주기적으로 MainNet 가중치를 TargetNet에 복사 (Overestimation 방지)
        """
        self.target_net.load_state_dict(self.main_net.state_dict())

    def train_step(self, states, actions, rewards, next_states, dones, weights=None):
        """
        Replay Buffer에서 샘플링한 배치 데이터(Tuple)를 사용해 MainNet 가중치 업데이트
        """
        states = states.to(self.device).float()
        actions = actions.to(self.device).long().unsqueeze(1)
        rewards = rewards.to(self.device).float()
        next_states = next_states.to(self.device).float()
        dones = dones.to(self.device).float()
        if weights is not None:
            weights = weights.to(self.device).float()

        # 현재 상태의 Q-value 계산 (MainNet)
        q_values = self.main_net(states)
        current_q = q_values.gather(1, actions).squeeze(1)

        # 다음 상태의 최대 Q-value 계산 (Double DQN 로직: MainNet으로 행동 선택, TargetNet으로 가치 평가)
        with torch.no_grad():
            next_actions = self.main_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + (self.gamma * next_q * (1 - dones))

        # TD 에러 및 PER 가중치가 적용된 손실 계산
        td_errors = torch.abs(target_q - current_q).detach()
        if weights is not None:
            loss = (weights * (target_q - current_q) ** 2).mean()
        else:
            loss = self.criterion(current_q, target_q)
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), td_errors.cpu().numpy()
