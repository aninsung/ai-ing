class MedicalImageEnv:
    def __init__(self, dataloader, class_weights):
        """
        강화학습 환경(Environment) (Phase 4)
        """
        self.dataloader = dataloader
        self.class_weights = class_weights # 계산된 {class_id: gamma_k}
        self.iterator = iter(self.dataloader)
        
        self.current_state = None
        self.current_label = None
        
    def reset(self):
        """
        환경 초기화 및 새로운 상태(의료 이미지 State) 반환
        """
        try:
            batch = next(self.iterator)
        except StopIteration:
            # 1 에폭 종료 시 다시 반복
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
            
        self.current_state = batch['image']
        self.current_label = batch['label']
        return self.current_state
        
    def step(self, action):
        """
        선택된 행동에 대한 보상(Reward) 계산 (Phase 4 적응형 보상 메커니즘)
        """
        # 논문의 Adaptive Reward 계산
        true_label = self.current_label.item()
        gamma_k = self.class_weights.get(true_label, 1.0)
        
        # 예측과 정답 비교
        if action == true_label:
            reward = gamma_k # 정답 (소수 클래스일수록 높은 보상)
        else:
            reward = -gamma_k # 오답 (소수 클래스일수록 높은 비용 페널티)
            
        # 다음 상태(Next State) 샘플 리로드 (Contextual Bandit에 가까운 MDP 설계)
        next_state = self.reset()
        done = True # 매 이미지가 하나의 에피소드 스텝 (순차적 종속성 없이 분류 즉시 보상)
        
        return next_state, reward, done, {}
