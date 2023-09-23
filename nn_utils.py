from typing import Any, Callable, Dict

import numpy as np
from catalyst.dl import Callback, CallbackOrder
from scipy.special import expit

import torch
import torch.nn as nn
import catalyst

from catalyst import dl, metrics
from os.path import join as pjoin
from catalyst import callbacks

class MetricWrapper(Callback):
    def __init__(
        self,
        metric_func: Callable,
        metric_name: str,
        output_key: str = "logit",
        input_key: str = "target",
        aggregation_policy_pred: str = "raw",
        aggregation_policy_target: str = "raw",
        metric_kwargs: Dict[str, str] = {},
    ):
        super().__init__(CallbackOrder.Metric)

        if aggregation_policy_pred not in ["raw", "argmax", "many_hot"]:
            raise ValueError("Invalid aggregation_policy_pred")
        if aggregation_policy_target not in ["raw", "argmax", "many_hot"]:
            raise ValueError("Invalid aggregation_policy_target")

        self.metric_func = metric_func
        self.metric_name = metric_name
        self.aggregation_policy_pred = aggregation_policy_pred
        self.aggregation_policy_target = aggregation_policy_target
        self.metric_kwargs = metric_kwargs
        self.input_key = input_key
        self.output_key = output_key

        self.running_preds = []
        self.running_targets = []

    def on_batch_end(self, state):
        y_hat = state.output[self.output_key].detach().cpu().numpy()
        y = state.input[self.input_key].detach().cpu().numpy()

        self.running_preds.append(y_hat)
        self.running_targets.append(y)

    def _compute_metric(self, y_true, y_pred):
        if self.aggregation_policy_pred == "raw":
            pass
        elif self.aggregation_policy_pred == "argmax":
            y_pred = y_pred.argmax(-1)
        elif self.aggregation_policy_pred == "many_hot":
            y_pred = (expit(y_pred) > 0.5).astype(int)

        if self.aggregation_policy_target == "raw":
            pass
        elif self.aggregation_policy_target == "argmax":
            y_true = y_true.argmax(-1)
        elif self.aggregation_policy_target == "many_hot":
            y_true = (y_true > 0.5).astype(int)

        return self.metric_func(y_true, y_pred, **self.metric_kwargs)

    def on_loader_end(self, state):
        y_true = np.concatenate(self.running_targets)
        y_pred = np.concatenate(self.running_preds)

        score = self._compute_metric(y_true, y_pred)

        state.loader_metrics[self.metric_name] = score

        self.running_preds = []
        self.running_targets = []

class TableDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        input_df,
        float_cols,
        int_cols,
        target_cols
    ):
        self.input_df = input_df.reset_index(drop=True)
        
        self.float_fs = self.input_df[float_cols].values.astype(float)
        self.int_fs = self.input_df[int_cols].values.astype(int)
        self.tgts = self.input_df[target_cols].values.astype(float)
        
        self.float_fs = torch.FloatTensor(self.float_fs)
        self.int_fs = torch.LongTensor(self.int_fs)
        self.tgts = torch.FloatTensor(self.tgts)
        
    def __len__(self):
        return len(self.input_df)
        
    def __getitem__(self, idx):         
        return self.float_fs[idx], self.int_fs[idx], self.tgts[idx]

class SAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input, target):
        return (input - target).abs().sum()


class CustomRunner(dl.Runner):
    def _dynamic_meters_updated(self, batch_metrics_dict):
        if len(batch_metrics_dict) > len(self.meters.keys()):
            additional_loss_metric_names = list(
                set(batch_metrics_dict.keys()) - set(self.meters.keys())
            )
            for add_key in additional_loss_metric_names:
                self.meters[add_key] = metrics.AdditiveMetric(
                    compute_on_call=False
                )
        for key in batch_metrics_dict.keys():
            self.meters[key].update(
                self.batch_metrics[key].item(), self.batch_size
            )

    def on_loader_start(self, runner):
        super().on_loader_start(runner)
        self.meters = {}

    def on_loader_end(self, runner):
        for key in self.meters.keys():
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)

    def on_epoch_start(self, runner: "IRunner"):
        if hasattr(runner.criterion, "on_epoch_start"):
            runner.criterion.on_epoch_start(runner)
        return super().on_epoch_start(runner)

    def handle_batch(self, batch):

        float_f, int_f, tgt = batch
        float_f = float_f.float()
        tgt = tgt.float()
        int_f = int_f.int()
                
        pred = self.model(int_f, float_f)
        
        loss = self.criterion(pred, tgt)
        
        losses = {
            "loss": loss,
            "mean_pred": pred.mean(),
            "target_mean": tgt.mean()
        }
        inputs = {
            "target": tgt
        }
        outputs = {
            "logit": pred 
        }
        
        self.batch_metrics.update(losses)
        self._dynamic_meters_updated(losses)
        self.input = inputs
        self.output = outputs
        
    def predict_batch(self, batch):
        float_f, int_f, tgt = batch
        float_f = float_f.float().to("cuda")
        tgt = tgt.float().to("cuda")
        int_f = int_f.int().to("cuda")
                
        pred = self.model(int_f, float_f)
        
        return pred.detach().cpu().numpy()

    def on_loader_end(self, runner):
        for key in self.meters.keys():
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)