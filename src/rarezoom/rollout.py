"""Run a frozen fusion model under a region-selection policy."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from .roi import BOXES, crop_tensor, load_raw, view_tensor


def build_state(global_features, probabilities, hidden, visited, remaining_fraction, spatial):
    return torch.cat([
        global_features,
        probabilities,
        hidden.squeeze(0),
        visited,
        remaining_fraction.unsqueeze(1),
        spatial,
    ], dim=1)


def _load(record):
    image, width, height = load_raw(record["path"])
    return image, width, height


@torch.no_grad()
def run_policy(model, records, mode, device, max_zooms=3, batch_size=64, q_net=None):
    """mode is stop, random, or greedy. Returns final probabilities, zoom counts, and ROI lists."""
    model.eval()
    if q_net is not None:
        q_net.eval()
    scores, zooms, selected, labels, names, sizes = [], [], [], [], [], []
    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]
        with ThreadPoolExecutor(max_workers=8) as pool:
            loaded = list(pool.map(_load, chunk))
        images = [item[0] for item in loaded]
        globals_ = torch.stack([view_tensor(image) for image in images]).to(device)
        features, spatial = model.global_observation(globals_)
        batch = features.shape[0]
        hidden = model.initial_state(batch, features)
        logits = model.predict(features, hidden)
        visited = torch.zeros(batch, 9, device=device)
        remaining = torch.full((batch,), max_zooms, device=device, dtype=features.dtype)
        active = torch.ones(batch, dtype=torch.bool, device=device)
        choices = [[] for _ in range(batch)]
        if mode != "stop":
            for _ in range(max_zooms):
                if not active.any():
                    break
                valid = torch.ones(batch, 10, dtype=torch.bool, device=device)
                valid[:, :9] = (visited == 0) & active.unsqueeze(1) & (remaining > 0).unsqueeze(1)
                valid[:, 9] = True
                if mode == "random":
                    probabilities = valid.float()
                    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-8)
                    actions = torch.multinomial(probabilities, 1).squeeze(1)
                    actions = torch.where(active, actions, torch.full_like(actions, 9))
                else:
                    state = build_state(
                        features, torch.sigmoid(logits), hidden, visited, remaining / max_zooms, spatial,
                    )
                    values = q_net(state).masked_fill(~valid, -1e9)
                    actions = values.argmax(dim=1)
                zoom = active & (actions < 9)
                if zoom.any():
                    indexes = zoom.nonzero(as_tuple=False).flatten().tolist()
                    crops = torch.stack([
                        crop_tensor(images[index], int(actions[index])) for index in indexes
                    ]).to(device)
                    local = features.new_zeros(batch, features.shape[1])
                    local[indexes] = model.local_features(crops)
                    updated = model.gru_step(local, hidden)
                    hidden = torch.where(zoom.view(1, batch, 1), updated, hidden)
                    refreshed = model.predict(features, hidden)
                    logits = torch.where(zoom.unsqueeze(1), refreshed, logits)
                    for index in indexes:
                        roi = int(actions[index])
                        visited[index, roi] = 1
                        choices[index].append(roi)
                    remaining = remaining - zoom.to(remaining.dtype)
                active = active & (actions != 9) & (remaining > 0)
        scores.append(torch.sigmoid(logits).float().cpu().numpy())
        zooms.append(np.array([len(item) for item in choices], dtype=np.int64))
        selected.extend(choices)
        labels.append(np.stack([record["label"] for record in chunk]))
        names.extend(record["image_index"] for record in chunk)
        sizes.extend((item[1], item[2]) for item in loaded)
    return {
        "score": np.concatenate(scores),
        "label": np.concatenate(labels),
        "zooms": np.concatenate(zooms),
        "selected": selected,
        "names": names,
        "sizes": sizes,
    }


@torch.no_grad()
def run_matched(model, records, budgets, device, max_zooms=3, batch_size=64):
    """Open exactly `budgets[i]` random ROIs, matching a policy's zoom count."""
    model.eval()
    scores, labels = [], []
    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]
        quota = torch.tensor(budgets[start:start + len(chunk)], device=device, dtype=torch.float32)
        with ThreadPoolExecutor(max_workers=8) as pool:
            loaded = list(pool.map(_load, chunk))
        images = [item[0] for item in loaded]
        globals_ = torch.stack([view_tensor(image) for image in images]).to(device)
        features, _ = model.global_observation(globals_)
        batch = features.shape[0]
        hidden = model.initial_state(batch, features)
        logits = model.predict(features, hidden)
        visited = torch.zeros(batch, 9, device=device)
        taken = torch.zeros(batch, device=device)
        for _ in range(max_zooms):
            active = taken < quota
            if not active.any():
                break
            valid = torch.zeros(batch, 10, dtype=torch.bool, device=device)
            valid[:, :9] = (visited == 0) & active.unsqueeze(1)
            valid[:, 9] = ~valid[:, :9].any(dim=1)
            probabilities = valid.float()
            probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
            actions = torch.multinomial(probabilities, 1).squeeze(1)
            zoom = active & (actions < 9)
            if zoom.any():
                indexes = zoom.nonzero(as_tuple=False).flatten().tolist()
                crops = torch.stack([
                    crop_tensor(images[index], int(actions[index])) for index in indexes
                ]).to(device)
                local = features.new_zeros(batch, features.shape[1])
                local[indexes] = model.local_features(crops)
                updated = model.gru_step(local, hidden)
                hidden = torch.where(zoom.view(1, batch, 1), updated, hidden)
                refreshed = model.predict(features, hidden)
                logits = torch.where(zoom.unsqueeze(1), refreshed, logits)
                for index in indexes:
                    visited[index, int(actions[index])] = 1
                taken = taken + zoom.to(taken.dtype)
        scores.append(torch.sigmoid(logits).float().cpu().numpy())
        labels.append(np.stack([record["label"] for record in chunk]))
    return {"score": np.concatenate(scores), "label": np.concatenate(labels)}
