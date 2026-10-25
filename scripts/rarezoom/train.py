"""Stage 3: Double DQN chooses which raw ROI to inspect. The fusion network stays fixed."""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rarezoom.data import load_records, positive_weights
from src.rarezoom.metrics import summarize
from src.rarezoom.model import QNetwork, RareZoomNet, focus_bce
from src.rarezoom.roi import BOXES, RAW_SIZE, crop_tensor, load_raw, view_tensor
from src.rarezoom.rollout import build_state, run_policy
from src.rarezoom.data import DISEASES


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Replay:
    def __init__(self, capacity, state_dim, actions=10):
        self.capacity = capacity
        self.state = np.zeros((capacity, state_dim), np.float32)
        self.next_state = np.zeros((capacity, state_dim), np.float32)
        self.action = np.zeros((capacity,), np.int64)
        self.reward = np.zeros((capacity,), np.float32)
        self.done = np.zeros((capacity,), np.float32)
        self.mask = np.zeros((capacity, actions), np.bool_)
        self.index = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done, mask):
        count = len(action)
        slots = (np.arange(count) + self.index) % self.capacity
        self.state[slots] = state
        self.next_state[slots] = next_state
        self.action[slots] = action
        self.reward[slots] = reward
        self.done[slots] = done
        self.mask[slots] = mask
        self.index = int((self.index + count) % self.capacity)
        self.size = min(self.capacity, self.size + count)

    def sample(self, batch, rng):
        slots = rng.integers(0, self.size, batch)
        return (self.state[slots], self.action[slots], self.reward[slots],
                self.next_state[slots], self.done[slots], self.mask[slots])


def choose(q_net, state, valid, epsilon):
    with torch.no_grad():
        greedy = q_net(state).masked_fill(~valid, -1e9).argmax(dim=1)
    if epsilon <= 0:
        return greedy
    explore = torch.rand(state.shape[0], device=state.device) < epsilon
    probabilities = valid.float()
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
    sampled = torch.multinomial(probabilities, 1).squeeze(1)
    return torch.where(explore, sampled, greedy)


def learn(replay, online, target, optimizer, gamma, batch, device, rng, clip):
    state, action, reward, next_state, done, mask = replay.sample(batch, rng)
    state = torch.tensor(state, device=device)
    next_state = torch.tensor(next_state, device=device)
    action = torch.tensor(action, device=device)
    reward = torch.tensor(reward, device=device)
    done = torch.tensor(done, device=device)
    mask = torch.tensor(mask, device=device)
    current = online(state).gather(1, action.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_online = online(next_state).masked_fill(~mask, -1e9)
        next_action = next_online.argmax(dim=1, keepdim=True)
        bootstrap = target(next_state).gather(1, next_action).squeeze(1)
        goal = reward + gamma * bootstrap * (1 - done)
    loss = torch.nn.functional.smooth_l1_loss(current, goal)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online.parameters(), clip)
    optimizer.step()
    return float(loss.detach())


def calibrate_penalty(model, records, device, pos_weight, focus_index, max_zooms, samples=512):
    """Set the zoom cost inside the scale of focus-loss reductions on the training split."""
    generator = np.random.default_rng(0)
    chosen = [records[int(index)] for index in generator.choice(len(records), size=min(samples, len(records)), replace=False)]
    deltas = []
    for start in range(0, len(chosen), 32):
        chunk = chosen[start:start + 32]
        images = [load_raw(record["path"])[0] for record in chunk]
        globals_ = torch.stack([view_tensor(image) for image in images]).to(device)
        labels = torch.stack([torch.from_numpy(record["label"]) for record in chunk]).to(device)
        features, _ = model.global_observation(globals_)
        hidden = model.initial_state(features.shape[0], features)
        before = focus_bce(model.predict(features, hidden), labels, pos_weight, focus_index)
        rois = generator.integers(0, 9, len(chunk))
        crops = torch.stack([crop_tensor(images[index], int(rois[index])) for index in range(len(chunk))]).to(device)
        hidden = model.gru_step(model.local_features(crops), hidden)
        after = focus_bce(model.predict(features, hidden), labels, pos_weight, focus_index)
        deltas.append((before - after).detach().float().cpu().numpy())
    delta = np.concatenate(deltas)
    scale = float(np.std(delta))
    penalty = float(max(0.25 * scale, 0.0))
    stats = {
        "samples": int(delta.size),
        "mean_reduction": float(delta.mean()),
        "std_reduction": scale,
        "p25": float(np.quantile(delta, 0.25)),
        "p50": float(np.quantile(delta, 0.50)),
        "p75": float(np.quantile(delta, 0.75)),
    }
    return penalty, stats


def collect_batch(model, q_net, records, device, max_zooms, penalty, pos_weight, focus_index, epsilon):
    images = []
    for record in records:
        image, _, _ = load_raw(record["path"])
        images.append(image)
    globals_ = torch.stack([view_tensor(image) for image in images]).to(device)
    labels = torch.stack([torch.from_numpy(record["label"]) for record in records]).to(device)
    features, spatial = model.global_observation(globals_)
    batch = features.shape[0]
    hidden = model.initial_state(batch, features)
    logits = model.predict(features, hidden)
    visited = torch.zeros(batch, 9, device=device)
    remaining = torch.full((batch,), max_zooms, device=device, dtype=features.dtype)
    active = torch.ones(batch, dtype=torch.bool, device=device)
    transitions = []
    episode_reward = torch.zeros(batch, device=device)
    zooms = torch.zeros(batch, device=device)
    for _ in range(max_zooms + 1):
        if not active.any():
            break
        valid = torch.ones(batch, 10, dtype=torch.bool, device=device)
        valid[:, :9] = (visited == 0) & active.unsqueeze(1) & (remaining > 0).unsqueeze(1)
        valid[:, 9] = True
        acting = active
        state = build_state(
            features, torch.sigmoid(logits), hidden, visited, remaining / max_zooms, spatial,
        )
        actions = choose(q_net, state, valid, epsilon)
        actions = torch.where(active, actions, torch.zeros_like(actions))
        previous = focus_bce(logits, labels, pos_weight, focus_index)
        zoom = active & (actions < 9)
        reward = torch.zeros(batch, device=device)
        next_hidden, next_logits, next_visited, next_remaining = hidden, logits, visited, remaining
        if zoom.any():
            indexes = zoom.nonzero(as_tuple=False).flatten().tolist()
            crops = torch.stack([crop_tensor(images[index], int(actions[index])) for index in indexes]).to(device)
            local = features.new_zeros(batch, features.shape[1])
            local[indexes] = model.local_features(crops)
            updated = model.gru_step(local, hidden)
            next_hidden = torch.where(zoom.view(1, batch, 1), updated, hidden)
            refreshed = model.predict(features, next_hidden)
            next_logits = torch.where(zoom.unsqueeze(1), refreshed, logits)
            current = focus_bce(next_logits, labels, pos_weight, focus_index)
            reward = torch.where(zoom, previous - current - penalty, reward)
            next_visited = visited.clone()
            for index in indexes:
                next_visited[index, int(actions[index])] = 1
            next_remaining = remaining - zoom.to(remaining.dtype)
            zooms = zooms + zoom.to(zooms.dtype)
        stop = active & (actions == 9)
        done = stop | (zoom & (next_remaining == 0)) | (~active)
        still = active & ~done
        next_valid = torch.ones(batch, 10, dtype=torch.bool, device=device)
        next_valid[:, :9] = (next_visited == 0) & still.unsqueeze(1) & (next_remaining > 0).unsqueeze(1)
        next_valid[:, 9] = still
        next_state = build_state(
            features, torch.sigmoid(next_logits), next_hidden, next_visited, next_remaining / max_zooms, spatial,
        )
        transitions.append((
            state[acting].float().cpu().numpy(),
            actions[acting].cpu().numpy(),
            reward[acting].float().cpu().numpy(),
            next_state[acting].float().cpu().numpy(),
            done[acting].float().cpu().numpy(),
            next_valid[acting].cpu().numpy(),
        ))
        episode_reward = episode_reward + reward
        hidden, logits, visited, remaining = next_hidden, next_logits, next_visited, next_remaining
        active = still
    return transitions, float(episode_reward.mean()), float(zooms.mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rarezoom.yaml")
    parser.add_argument("--stage2", default="checkpoints/rarezoom/v2/stage2/best_model.pth")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--batch-episodes", type=int, default=32)
    parser.add_argument("--penalty", default="auto")
    parser.add_argument("--tag", default="v2")
    parser.add_argument("--save-name", default="ddqn.pth")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    seed = int(config["train"]["seed"])
    seed_all(seed)
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records, summary = load_records(ROOT / "data", seed=seed)
    weights, _, _ = positive_weights(records, config["reward"]["class_weight_clip"])
    train_rows = [record for record in records if record["split"] == "train"]
    val_rows = [record for record in records if record["split"] == "val"]
    model = RareZoomNet(config["dataset"]["num_classes"], config["model"]["gru_hidden"]).to(device)
    checkpoint = torch.load(ROOT / args.stage2, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    pos_weight = torch.tensor(checkpoint.get("pos_weight", weights.tolist()), device=device)
    hidden = int(config["model"]["gru_hidden"])
    state_dim = 2048 + 14 + hidden + 9 + 1 + 9
    focus_index = torch.tensor(
        [DISEASES.index(name) for name in config["dataset"]["focus_classes"]], device=device,
    )
    actions = int(config["agent"]["num_actions"])
    online = QNetwork(state_dim, actions).to(device)
    target = QNetwork(state_dim, actions).to(device)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=float(config["agent"]["learning_rate"]))
    replay = Replay(int(config["agent"]["replay_capacity"]), state_dim, actions)
    episodes = args.episodes or int(config["train"]["stage3_episodes"])
    max_zooms = int(config["roi"]["max_zooms"])
    if args.penalty == "auto":
        penalty, penalty_stats = calibrate_penalty(
            model, train_rows, device, pos_weight, focus_index, max_zooms,
        )
    else:
        penalty, penalty_stats = float(args.penalty), {"source": "fixed"}
    print(json.dumps({"penalty": penalty, "penalty_stats": penalty_stats}), flush=True)
    gamma = float(config["agent"]["gamma"])
    epsilon = float(config["agent"]["epsilon_start"])
    epsilon_end = float(config["agent"]["epsilon_end"])
    decay = float(config["agent"]["epsilon_decay"])
    batch = int(config["agent"]["batch_size"])
    target_every = int(config["agent"]["target_update_freq"])
    clip = float(config["agent"]["grad_clip"])
    save_dir = ROOT / config["train"]["save_dir"] / args.tag / "stage3"
    save_dir.mkdir(parents=True, exist_ok=True)
    history = save_dir / f"{Path(args.save_name).stem}_history.jsonl"
    steps = 0
    synced = 0
    print(json.dumps({"stage": 3, "episodes": episodes, "train_images": summary["images"]["train"]}), flush=True)
    completed = 0
    while completed < episodes:
        count = min(args.batch_episodes, episodes - completed)
        sample = [train_rows[int(index)] for index in rng.integers(0, len(train_rows), count)]
        transitions, reward, zooms = collect_batch(
            model, online, sample, device, max_zooms, penalty, pos_weight, focus_index, epsilon,
        )
        loss = None
        for state, action, rewards, next_state, done, mask in transitions:
            replay.add(state, action, rewards, next_state, done, mask)
            steps += len(action)
            if replay.size >= batch:
                loss = learn(replay, online, target, optimizer, gamma, batch, device, rng, clip)
            if steps // target_every > synced:
                synced = steps // target_every
                target.load_state_dict(online.state_dict())
        epsilon = max(epsilon_end, epsilon * (decay ** count))
        completed += count
        if completed % 640 == 0 or completed == episodes:
            record = {"episodes": completed, "reward": reward, "zooms": zooms, "epsilon": epsilon,
                      "replay": replay.size, "loss": loss}
            print(json.dumps(record), flush=True)
            with history.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record) + "\n")
    torch.save({
        "q_network": online.state_dict(),
        "state_dim": state_dim,
        "episodes": completed,
        "penalty": penalty,
        "focus_classes": config["dataset"]["focus_classes"],
    }, save_dir / args.save_name)
    print(json.dumps({"checkpoint": str(save_dir / args.save_name), "episodes": completed, "penalty": penalty}), flush=True)


if __name__ == "__main__":
    main()
