import os
import torch
import datetime
from src.utils import get_config, _set_seed
from src.dataset import load_and_filter_data, get_dataloader
from src.environment import MedicalImageEnv
from src.agent import DDQNAgent
from src.replay_buffer import ReplayBuffer

def train_agent():
    # 1. 환경 및 설정 초기화
    _set_seed(42)
    config = get_config("configs/config.yaml")
    
    # 2. 데이터 준비 및 환경 구성
    data_dicts, class_weights = load_and_filter_data(config['data']['train_csv_path'], config['data']['image_dir'])
    train_loader = get_dataloader(data_dicts, batch_size=1) # 환경 조회를 위해 배치 1
    env = MedicalImageEnv(train_loader, class_weights)
    
    # 3. 강화학습 DDQN 구성요소 초기화
    num_classes = config['agent']['num_classes']
    agent = DDQNAgent(num_classes=num_classes, config=config)
    replay_buffer = ReplayBuffer(config['agent']['buffer_capacity'])
    
    num_episodes = config['train']['num_episodes']
    batch_size = config['agent']['batch_size']
    target_update_freq = config['agent']['target_update_freq']
    
    global_step = 0
    
    # 3.5 모델 저장소 초기화 (시간대별 타임스탬프)
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = config.get('experiment_name', 'BicDRL_Exp')
    save_dir = os.path.join(config['train']['save_dir'], f"{exp_name}_{current_time}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Directory: Parameters will be saved in [{save_dir}]!")
    
    # 역대 최고 점수 기록용
    best_score = -float('inf')
    
    # Early stopping parameters
    early_stop_patience = config['train'].get('early_stop_patience', 5)
    epochs_without_improve = 0
    
    # 학습 루프 시작
    print("Starting Training...")
    from tqdm import tqdm
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        loss_history = []
        
        # tqdm 진행바 추가
        pbar = tqdm(range(config['train']['steps_per_episode']), desc=f"Episode {episode+1}/{num_episodes}")
        
        tp, tn, fp, fn = 0, 0, 0, 0
        
        for step in pbar:
            true_label = env.current_label.item()
            
            # Phase 4/5: 상태 관찰 및 행동 선택 (a_t)
            action = agent.select_action(state)
            
            # 행동 환경에 투여 -> 보상(r_t) 및 다음 상태 획득
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            # Confusion Matrix
            if true_label == 1 and action == 1:
                tp += 1
            elif true_label == 0 and action == 0:
                tn += 1
            elif true_label == 0 and action == 1:
                fp += 1
            elif true_label == 1 and action == 0:
                fn += 1
                
            pbar.set_postfix({
                "Target": true_label, 
                "Pred": action, 
                "Result": f"TP:{tp} TN:{tn} FP:{fp} FN:{fn}",
                "Score": f"{episode_reward:.1f}"
            })
            
            # Replay Buffer에 튜플 저장 (PER 기반으로 확장 여지)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            
            # 버퍼 용량이 차면 학습 시작
            if len(replay_buffer) > batch_size:
                # PER 샘플링
                states, actions, rewards, next_states, dones, indices, weights = replay_buffer.sample(batch_size)
                # 모델 가중치 업데이트 (MainNet) 및 TD 에러 반환
                loss, td_errors = agent.train_step(states, actions, rewards, next_states, dones, weights)
                
                # TD 에러를 기반으로 버퍼의 우선순위 업데이트
                replay_buffer.update_priorities(indices, td_errors)
                
                loss_history.append(loss)
                agent.decay_epsilon()
                
            # 타깃 네트워크(TargetNet) 동기화
            if global_step % target_update_freq == 0:
                agent.update_target_network()
                
            global_step += 1
            
        # Logging & MLOps 파라미터 저장
        avg_loss = sum(loss_history) / len(loss_history) if loss_history else 0
        
        # === 평가지표(Metrics) 계산 ===
        total_steps = tp + tn + fp + fn
        accuracy = (tp + tn) / total_steps if total_steps > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall (결절 적중률)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
        
        import math
        g_mean = math.sqrt(sensitivity * specificity)
        
        # Macro F1-score
        prec_pos = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_pos = 2 * (prec_pos * sensitivity) / (prec_pos + sensitivity) if (prec_pos + sensitivity) > 0 else 0
        
        prec_neg = tn / (tn + fn) if (tn + fn) > 0 else 0
        f1_neg = 2 * (prec_neg * specificity) / (prec_neg + specificity) if (prec_neg + specificity) > 0 else 0
        macro_f1 = (f1_pos + f1_neg) / 2
        
        # 만약을 대비한 매 에피소드 당 최신 상태 저장 (백업)
        torch.save(agent.main_net.state_dict(), os.path.join(save_dir, 'latest_model.pth'))
        
        is_best = False
        # 내 최고 성적 갱신 시 베스트 모델 파일 저장
        if episode_reward > best_score:
            best_score = episode_reward
            is_best = True
            torch.save(agent.main_net.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            pbar.write(f"[Best Score!] New record ({best_score:.2f}) - best_model.pth saved.")
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            
        # 스케줄러 업데이트 (에피소드 보상을 기반으로 학습률 조정)
        agent.scheduler.step(episode_reward)
        current_lr = agent.optimizer.param_groups[0]['lr']
        
        print(f"Episode {episode+1}/{num_episodes} | Score: {episode_reward:.1f} | Acc: {accuracy:.3f} | F1: {macro_f1:.3f} | G-Mean: {g_mean:.3f} | LR: {current_lr:.6f}")
        print(f"  -> Pos (Nodule): Prec: {prec_pos:.3f}, Rec: {sensitivity:.3f} (TP:{tp} FN:{fn})")
        print(f"  -> Neg (NoFinding): Prec: {prec_neg:.3f}, Rec: {specificity:.3f} (TN:{tn} FP:{fp})")
        
        # 텍스트 로그 파일에 모델 평가지표 세부 누적 기록!
        log_file = os.path.join(save_dir, 'training_log.txt')
        with open(log_file, "a", encoding="utf-8") as f:
            best_mark = "[BEST]" if is_best else ""
            f.write(f"Episode {episode+1:03d} | Score: {episode_reward:7.2f} | Loss: {avg_loss:.5f} | "
                    f"TP:{tp:3d} TN:{tn:3d} FP:{fp:3d} FN:{fn:3d} | "
                    f"Acc: {accuracy:.4f} | F1: {macro_f1:.4f} | G-Mean: {g_mean:.4f} | LR: {current_lr:.6f} {best_mark}\n")
                    
        # Early stopping check inside the loop
        if epochs_without_improve >= early_stop_patience:
            print(f"\n[Early Stopping] Triggered after {early_stop_patience} episodes without improvement.")
            break
            
    print(f"Training Complete. All models saved in [{save_dir}]/")

if __name__ == "__main__":
    train_agent()