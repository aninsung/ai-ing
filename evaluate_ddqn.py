"""Compare global-only classification with greedy DDQN region observation."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.multiclass_dataset import read_manifest, NIHMultiTaskDataset
from src.multiclass_model import MultiTaskResNet50
from src.ddqn_region import RegionQNetwork, crop_grid, make_state


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/multiclass.yaml")
    parser.add_argument("--model", default="checkpoints/multiclass/best_model.pth")
    parser.add_argument("--policy", default="checkpoints/ddqn_region/ddqn_region.pth")
    parser.add_argument("--max-samples", type=int, default=2000)
    args = parser.parse_args()
    config = __import__("yaml").safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskResNet50(15, 14, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True)["model"]); model.eval()
    policy_state = torch.load(args.policy, map_location=device, weights_only=True)
    policy = RegionQNetwork(40, 10).to(device); policy.load_state_dict(policy_state["q_network"]); policy.eval()
    rows = [r for r in read_manifest(config["data"]["manifest"], "test") if r["primary_class_id"]]
    if args.max_samples > 0:
        rows = rows[:args.max_samples]
    loader = DataLoader(NIHMultiTaskDataset(rows, config["data"]["image_root"]), batch_size=32, shuffle=False, num_workers=0)
    global_correct = ddqn_correct = total = views = 0
    for batch in loader:
        images = batch["image"].to(device); labels = batch["primary"].to(device)
        crops = crop_grid(images)
        logits, _ = model(torch.cat([torch.nn.functional.interpolate(c, (224,224), mode="bilinear", align_corners=False) for c in crops]))
        logits = logits.reshape(10, len(images), 15).permute(1,0,2).float().cpu()
        labels = labels.cpu()
        global_logits = logits[:,0]; global_pred = global_logits.argmax(1)
        global_correct += (global_pred == labels).sum().item()
        for i in range(len(images)):
            current, visited, remaining = global_logits[i].cpu(), np.zeros(9, dtype=np.float32), 3
            while True:
                valid = np.ones(10, dtype=bool); valid[:9] = visited == 0
                if remaining <= 0: valid[:9] = False
                state = make_state(global_logits[i].cpu(), current, visited, remaining).to(device)
                q = policy(state.unsqueeze(0))[0].cpu().numpy(); q[~valid] = -np.inf; action = int(q.argmax())
                if action == 9 or remaining <= 0: break
                visited[action] = 1; remaining -= 1; current = torch.maximum(current, logits[i, action+1])
            ddqn_correct += int(current.argmax().item() == int(labels[i])); views += int(visited.sum()); total += 1
    result = {"samples": total, "global_accuracy": global_correct/total, "ddqn_accuracy": ddqn_correct/total,
              "average_region_views": views/total, "policy": str(Path(args.policy).resolve())}
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
