import itertools
import numpy as np
import torch.nn.functional as F
from xmetai.builder import METRICS
from xmetai.core import DatasetEvaluator
from xmetai.utils import comm
from xmetai.layers import to_1tuple


@METRICS.register()
class MSE(DatasetEvaluator):
    def __init__(
        self,
        test_names,
        test_times,
    ):
        super().__init__()
        self.test_names = test_names
        self.test_times = to_1tuple(test_times)

    def get_key(self, name, t):
        return "{} @ {:03d}h".format(name, t)

    def reset(self):
        self.rmse = {}

        for t in self.test_times:
            for name in self.test_names:
                key = self.get_key(name, t)
                self.rmse[key] = []

    def process(self, inputs, outputs):
        output = outputs["output"].cpu()
        target = outputs["target"].cpu()

        if "weight" in outputs:
            weight = outputs["weight"].cpu()
        else:
            weight = 1

        for i, t in enumerate(self.test_times):
            for j, name in enumerate(self.test_names):
                output_j = output[:, i, j]
                target_j = target[:, i, j]

                err = F.mse_loss(target_j, output_j, reduction="none")
                rmse = (err * weight).mean().sqrt()
                key = self.get_key(name, t)
                self.rmse[key].append(rmse.item())

    def evaluate(self):
        comm.synchronize()

        for t in self.test_times:
            for name in self.test_names:
                key = self.get_key(name, t)
                rmse = comm.gather(self.rmse[key], dst=0)
                self.rmse[key] = list(itertools.chain(*rmse))

        if not comm.is_main_process():
            return

        results = {}
        for t in self.test_times:
            for name in self.test_names:
                key = self.get_key(name, t)
                rmse = np.mean(self.rmse[key])
                results[key] = rmse

        flatten_results = {"Times": self.test_times}

        for name in self.test_names:
            flatten_results[name] = []
            for k, v in results.items():
                if name in k:
                    flatten_results[name].append(float(v))

        return {"RMSE": flatten_results}
