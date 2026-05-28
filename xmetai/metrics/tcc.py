import torch

from xmetai.builder import METRICS
from xmetai.core import DatasetEvaluator
from xmetai.layers import to_1tuple
from xmetai.utils import comm

__all__ = [
    "TCC",
]


@METRICS.register()
class TCC(DatasetEvaluator):
    def __init__(self, test_names, test_times):
        super().__init__()
        self.test_names = test_names
        self.test_times = to_1tuple(test_times)
        self.interval = 168  # hours in one week
        self.weekly_data = {}

    def get_key(self, name, t):
        return f"{name} @ {t:03d}h"

    def get_week_key(self, name, t):
        week = (t - min(self.test_times)) // self.interval
        return f"week_{week}_{name}"

    def reset(self):
        self.weekly_data.clear()
        min_time = min(self.test_times)
        max_time = max(self.test_times)
        self.num_weeks = (max_time - min_time) // self.interval + 1

        for week in range(self.num_weeks):
            for name in self.test_names:
                self.weekly_data[f"week_{week}_{name}"] = {
                    "outputs": [],
                    "targets": [],
                    "times": [],
                    "weights": None,
                }

    def process(self, inputs, outputs):
        output = outputs["output"]  # (B, T, N, H, W)
        target = outputs["target"]
        weight = outputs.get("weight", None)  # (1, H, 1) or (1, H, W)

        for i, t in enumerate(self.test_times):
            for j, name in enumerate(self.test_names):
                out = output[:, i, j]  # (B, H, W)
                tgt = target[:, i, j]
                week_key = self.get_week_key(name, t)
                self.weekly_data[week_key]["outputs"].append(out)
                self.weekly_data[week_key]["targets"].append(tgt)
                self.weekly_data[week_key]["times"].append(t)
                if weight is not None and self.weekly_data[week_key]["weights"] is None:
                    self.weekly_data[week_key]["weights"] = weight.to(out.device)

    @torch.no_grad()
    def evaluate(self):
        comm.synchronize()
        raw_results = {}
        min_time = min(self.test_times)

        for week in range(self.num_weeks):
            for name in self.test_names:
                week_key = f"week_{week}_{name}"
                if not self.weekly_data[week_key]["outputs"]:
                    continue

                # Stack tensors in GPU to minimize memory transfers
                outputs = torch.cat(self.weekly_data[week_key]["outputs"], dim=0)  # (N, H, W)
                targets = torch.cat(self.weekly_data[week_key]["targets"], dim=0)
                times = self.weekly_data[week_key]["times"]
                weight = self.weekly_data[week_key]["weights"]

                unique_times = sorted(set(times))
                for t in unique_times:
                    indices = [i for i, time in enumerate(times) if time == t]
                    if not indices:
                        continue

                    key = self.get_key(name, t)
                    pred = outputs[indices]  # (B, H, W)
                    true = targets[indices]

                    B, H, W = pred.shape
                    pred = pred.view(B, -1)
                    true = true.view(B, -1)

                    pred_mean = pred.mean(dim=0, keepdim=True)
                    true_mean = true.mean(dim=0, keepdim=True)
                    pred_anom = pred - pred_mean
                    true_anom = true - true_mean

                    pred_std = pred_anom.std(dim=0, unbiased=False)
                    true_std = true_anom.std(dim=0, unbiased=False)
                    denom = pred_std * true_std
                    mask = denom > 1e-6

                    corr = torch.zeros_like(denom)
                    corr[mask] = (pred_anom * true_anom).mean(dim=0)[mask] / denom[mask]

                    if weight is not None:
                        weight_used = weight.squeeze()  # (H, W) or (H,)
                        if weight_used.dim() == 1:
                            weight_used = weight_used.unsqueeze(1).expand(H, W)
                        elif weight_used.dim() == 2:
                            weight_used = weight_used.expand(H, W)
                        weight_used = weight_used.contiguous().view(-1).to(corr.device)
                        weight_used = weight_used / (weight_used.sum() + 1e-8)
                        tcc_score = (corr[mask] * weight_used[mask]).sum()
                    else:
                        tcc_score = corr[mask].mean() if mask.any() else torch.tensor(0.0, device=pred.device)

                    raw_results[key] = tcc_score.item()

                # Release memory for this week
                self.weekly_data[week_key] = {
                    "outputs": [],
                    "targets": [],
                    "times": [],
                    "weights": None,
                }

        if not comm.is_main_process():
            return

        # Aggregate to weekly mean
        week_results = {"Week\\TCC": [f"Week{i+1}" for i in range(self.num_weeks)]}
        for name in self.test_names:
            week_results[name] = []
            for week in range(self.num_weeks):
                start_t = min_time + week * self.interval
                end_t = start_t + self.interval
                vals = [raw_results[self.get_key(name, t)] for t in self.test_times if start_t <= t < end_t and self.get_key(name, t) in raw_results]
                avg_tcc = sum(vals) / len(vals) if vals else 0.0
                week_results[name].append(float(avg_tcc))

        return {"TCC": week_results}
