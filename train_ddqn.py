"""Train a DDQN policy that selects useful X-ray regions before classification."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import yaml
from torch.utils.data import DataLoader
from src.multiclass_dataset import read_manifest, NIHMultiTaskDataset
from src.multiclass_model import MultiTaskResNet50
from src.ddqn_region import RegionQNetwork, TransitionBuffer, crop_grid, make_state, select_action


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def load_rows(path, split):
    rows = [r for r in read_manifest(path, split) if r["primary_class_id"]]
    if not rows: raise ValueError(f"No single-label rows in {split}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/multiclass.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/multiclass/best_model.pth")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--max-views", type=int, default=3)
    parser.add_argument("--view-cost", type=float, default=0.005)
    args = parser.parse_args()
    if args.episodes < 1 or args.max_views < 1 or args.max_views > 9: raise ValueError("Invalid episodes/max-views")
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); seed_all(config["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskResNet50(config["data"]["num_classes"], config["data"]["multilabel_classes"], pretrained=False).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"]); model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    train_rows = load_rows(config["data"]["manifest"], "train")
    dataset = NIHMultiTaskDataset(train_rows, config["data"]["image_root"], augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    iterator = iter(loader)
    state_dim, action_count = 15 + 15 + 9 + 1, 10
    q_net = RegionQNetwork(state_dim, action_count).to(device)
    target = RegionQNetwork(state_dim, action_count).to(device); target.load_state_dict(q_net.state_dict()); target.eval()
    optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-4); buffer = TransitionBuffer()
    gamma, batch_size, target_update = 0.95, 64, 250
    epsilon, epsilon_min, epsilon_decay = 1.0, 0.05, 0.995
    view_cost, reward_scale = args.view_cost, 1.0
    history, steps = [], 0
    save_dir = Path(config["train"]["save_dir"]).parent / "ddqn_region"; save_dir.mkdir(parents=True, exist_ok=True)
    for episode in range(args.episodes):
        try: batch = next(iterator)
        except StopIteration: iterator = iter(loader); batch = next(iterator)
        image = batch["image"].to(device)
        true_label = int(batch["primary"].item())
        with torch.no_grad():
            crops = crop_grid(image)
            # Full image plus nine regions in one forward pass.
            logits, _ = model(torch.cat([torch.nn.functional.interpolate(c, (224,224), mode="bilinear", align_corners=False) for c in crops]))
            logits = logits.detach().float().cpu()
        global_logits, region_logits = logits[0], logits[1:]
        current = global_logits.clone(); visited = np.zeros(9, dtype=np.float32); remaining = args.max_views
        episode_reward, done = 0.0, False
        while not done:
            valid = np.ones(10, dtype=bool); valid[:9] = visited == 0
            if remaining <= 0: valid[:9] = False
            state = make_state(global_logits, current, visited, remaining)
            action = select_action(q_net, state, valid, epsilon, device)
            if action < 9:
                visited[action] = 1; remaining -= 1; current = torch.maximum(current, region_logits[action]); reward = -view_cost; done = remaining == 0
            else:
                reward = reward_scale if int(current.argmax()) == true_label else -reward_scale; done = True
            next_state = make_state(global_logits, current, visited, remaining)
            buffer.add(state.numpy(), action, reward, next_state.numpy(), float(done)); episode_reward += reward; steps += 1
            if len(buffer) >= batch_size:
                batch_trans = buffer.sample(batch_size); states, actions, rewards, next_states, dones = zip(*batch_trans)
                states = torch.tensor(np.asarray(states), dtype=torch.float32, device=device)
                actions = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=device); next_states = torch.tensor(np.asarray(next_states), dtype=torch.float32, device=device); dones = torch.tensor(dones, dtype=torch.float32, device=device)
                current_q = q_net(states).gather(1, actions).squeeze(1)
                with torch.no_grad():
                    next_actions = q_net(next_states).argmax(1, keepdim=True); next_q = target(next_states).gather(1, next_actions).squeeze(1); target_q = rewards + gamma * next_q * (1-dones)
                loss = torch.nn.functional.smooth_l1_loss(current_q, target_q)
                optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(q_net.parameters(), 5.0); optimizer.step()
                if steps % target_update == 0: target.load_state_dict(q_net.state_dict())
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        if (episode + 1) % 100 == 0:
            record = {"episode": episode+1, "reward": episode_reward, "epsilon": epsilon, "buffer": len(buffer), "steps": steps}
            history.append(record); print(json.dumps(record), flush=True)
    torch.save({"q_network": q_net.state_dict(), "target_network": target.state_dict(), "config": config, "episodes": args.episodes, "max_views": args.max_views, "history": history}, save_dir / "ddqn_region.pth")
    (save_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoint": str(save_dir / "ddqn_region.pth"), "episodes": args.episodes, "epsilon": epsilon}, indent=2), flush=True)


if __name__ == "__main__": main()
