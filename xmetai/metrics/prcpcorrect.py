"""Masked precipitation RMSE and pooled CSI metrics."""

import numpy as np
import torch

from xmetai.metrics.mse import MSE
from xmetai.registries import METRICS
from xmetai.utils import comm


@METRICS.register()
class PrcpMaskedRMSE(MSE):
    """Native MSE output format, using per-case valid-cell RMSE for each lead."""

    def _process_output(self, output, target, weight, prefix="", frame_start=0):
        if not torch.is_tensor(weight) or weight.dtype != torch.bool or weight.shape != target.shape or output.shape != target.shape:
            raise ValueError("PrcpMaskedRMSE requires a boolean valid mask matching output and target.")
        test_times = self.test_times[frame_start : frame_start + output.shape[1]]
        if len(test_times) != output.shape[1]:
            raise ValueError("PrcpMaskedRMSE forecast leads do not match test_times.")
        for i, t in enumerate(test_times):
            for j, name in enumerate(self.test_names):
                valid = weight[:, i, j]
                count = valid.sum(dim=(-2, -1))
                if torch.any(count == 0):
                    raise ValueError("Every evaluated lead must contain valid observation cells.")
                error = torch.where(valid, output[:, i, j].double() - target[:, i, j].double(), 0.0)
                rmse = (error.square().sum(dim=(-2, -1)) / count).sqrt()
                if not torch.isfinite(rmse).all():
                    raise ValueError("Non-finite precipitation prediction at valid observation cells.")
                key = (f"{prefix}/" if prefix else "") + self.get_key(name, t)
                self.rmse.setdefault(key, []).extend(rmse.tolist())


@METRICS.register()
class PrcpMaskedCSI(MSE):
    """Pool valid-cell contingency counts over all cases, leads and ranks."""

    thresholds = (0.1, 1, 5, 10, 25, 50)

    def __init__(self, test_names, test_times):
        super().__init__(test_names=test_names, test_times=test_times)
        if list(test_names) != ["tp"]:
            raise ValueError("PrcpMaskedCSI requires the single precipitation channel tp.")

    def reset(self):
        self.counts = {}

    def _process_output(self, output, target, weight, prefix="", frame_start=0):
        if not torch.is_tensor(weight) or weight.dtype != torch.bool or weight.shape != target.shape or output.shape != target.shape:
            raise ValueError("PrcpMaskedCSI requires a boolean valid mask matching output and target.")
        if target.ndim != 5 or target.shape[2] != 1 or frame_start < 0 or frame_start + target.shape[1] > len(self.test_times):
            raise ValueError("PrcpMaskedCSI requires matching precipitation forecast leads.")
        if not torch.isfinite(output[weight]).all() or not torch.isfinite(target[weight]).all():
            raise ValueError("Non-finite precipitation at valid observation cells.")
        counts = self.counts.setdefault(prefix, np.zeros((len(self.thresholds), 3), dtype=np.int64))
        for i, threshold in enumerate(self.thresholds):
            predicted = (output >= threshold) & weight
            observed = (target >= threshold) & weight
            hits = (predicted & observed).sum()
            counts[i] += torch.stack((hits, observed.sum() - hits, predicted.sum() - hits)).cpu().numpy()

    def evaluate(self):
        shards = comm.gather(self.counts, dst=0)
        if not comm.is_main_process():
            return None
        counts = {}
        for shard in shards:
            for mode, values in shard.items():
                counts[mode] = counts.get(mode, 0) + values
        results = {}
        scores_by_mode = {}
        for i, threshold in enumerate(self.thresholds):
            table = {}
            for mode, values in sorted(counts.items()):
                hits, misses, false_alarms = map(int, values[i])
                union = hits + misses + false_alarms
                score = hits / union if union else float("nan")
                table[f"{mode}/tp" if mode else "tp"] = [score]
                scores_by_mode.setdefault(mode, []).append(score)
            results[f"CSI{threshold:g}"] = table
        results["CSImean6"] = {}
        for mode, scores in sorted(scores_by_mode.items()):
            # An unavailable threshold invalidates the fixed six-threshold mean.
            # Do not use nanmean, which would silently change its weights.
            mean_score = float(np.mean(scores))
            results["CSImean6"][f"{mode}/tp" if mode else "tp"] = [mean_score]
            if mode == "corrected":
                # The native BestCheckpointer minimizes the configured scalar.
                results["Selection"] = {"csi_mean6_error": [1.0 - mean_score]}
        return results
