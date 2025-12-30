import os
import time
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from .alarm import get_alarm
from .utils.beta_scheduler import build_beta_scheduler

class AlarmHook:
    def __init__(self, logger, result_dir, 
        alarm={'type': 'silent', 'target': 'step'}, end=False):
        self.logger = logger
        if not isinstance(alarm, list):
            alarm = [alarm]
        self.alarms = [get_alarm(logger=logger, **a) for a in alarm]
        self.end = end
    def __call__(self, batch, model):
        ring = False
        for alarm in self.alarms:
            if alarm(batch):
                ring = True
        if ring or ('end' in batch and self.end):
            self.ring(batch=batch, model=model)
    def ring(self, batch, model):
        raise NotImplementedError

class SaveAlarmHook(AlarmHook):
    def __init__(self, logger, result_dir, 
        alarm={'type': 'silent', 'target': 'step'}, end=False):
        super().__init__(logger, result_dir, alarm, end=end)
        self.models_dir = f"{result_dir}/models"
        os.makedirs(self.models_dir, exist_ok=True)
    def ring(self, batch, model):
        self.logger.info(f"Saving model at step {batch['step']:2>}...")
        path = f"{self.models_dir}/{batch['step']}"
        model.save_state_dict(path)

class AccumulateHook:
    def __init__(
        self,
        logger,
        result_dir,
        names,
        cols,
        save_alarm,
        checkpoint=None,
        fname='steps',
        mode='chunk',
    ):
        os.makedirs(result_dir, exist_ok=True)
        self.path_df = result_dir+f"/{fname}.csv"
        self.save_alarm = get_alarm(logger=logger, **save_alarm)
        self.dfs = []
        if checkpoint is not None:
            self.dfs.append(pd.read_csv(checkpoint))
        self.lists = defaultdict(list)
        self.names = names
        self.cols = cols
        if mode not in {'chunk', 'snapshot'}:
            raise ValueError(f"Unsupported accumulate mode: {mode}")
        self.mode = mode

    def __call__(self, batch, model):
        should_save = self.save_alarm(batch)

        if self.mode == 'chunk':
            if 'end' not in batch:
                for name, col in zip(self.names, self.cols):
                    item = batch[name]
                    if isinstance(item, torch.Tensor):
                        item = item.detach().cpu().numpy()
                    self.lists[col].append(item)
            if should_save or ('end' in batch and self.lists):
                self.dfs.append(pd.DataFrame(self.lists))
                self.lists.clear()
        else:  # snapshot mode
            if should_save or 'end' in batch:
                row = {}
                for name, col in zip(self.names, self.cols):
                    item = batch[name]
                    if isinstance(item, torch.Tensor):
                        item = item.detach().cpu()
                        item = item.item() if item.numel() == 1 else item.numpy()
                    row[col] = item
                self.dfs.append(pd.DataFrame([row]))

        if should_save or ('end' in batch and self.dfs):
            pd.concat(self.dfs, ignore_index=True).to_csv(self.path_df, index=False)

class AbortHook:
    def __init__(self, logger, result_dir, 
        target, threshold):
        """
        Parameters
        ----------
        target: str
            Target value in batch
        threshold: int or float
            When batch[target] >= threshold, batch['end'] is added.
        """
        self.target = target
        self.threshold = threshold
    def __call__(self, batch, model):
        if self.target in batch and \
            batch[self.target] >= self.threshold:
            batch['end'] = True
class StepAbortHook(AbortHook):
    def __init__(self, logger, result_dir, threshold):
        super().__init__(logger, result_dir, 'step', threshold)
class EpochAbortHook(AbortHook):
    def __init__(self, logger, result_dir, threshold):
        super().__init__(logger, result_dir, 'epoch', threshold)
class TimeAbortHook:
    def __init__(self, logger, result_dir, 
        threshold):
        """
        Parameters
        ----------
        threshold: int
            Second to continue        
        """
        self.end = time.time() + threshold
    def __call__(self, batch, model):
        if time.time() > self.end:
            batch['end'] = True


class KLBetaSchedulerHook(AlarmHook):
    """Adjust the KL loss weight (`-d_kl_factor`) according to a beta schedule."""

    def __init__(
        self,
        logger,
        result_dir,
        module='-d_kl_factor',
        base_beta=None,
        max_step=None,
        scheduler=None,
        alarm={'type': 'silent', 'target': 'step'},
        end=False,
        **kwargs,
    ):
        super().__init__(logger, result_dir, alarm=alarm, end=end, **kwargs)
        self.module_name = module
        self.base_beta = float(base_beta) if base_beta is not None else 0.0
        self.max_step = max_step
        self.scheduler_config = scheduler
        self.schedule_fn = self._build_schedule()
        self.cached_module = None

    def _build_schedule(self):
        if not self.max_step:
            self.logger.warning(
                "KLBetaSchedulerHook requires max_step; falling back to constant beta %s.",
                self.base_beta,
            )
            return lambda _step: self.base_beta
        return build_beta_scheduler(
            self.scheduler_config,
            base_beta=self.base_beta,
            max_step=self.max_step,
            logger=self.logger,
        )

    def _get_target(self, model):
        if self.cached_module is not None:
            return self.cached_module
        if self.module_name not in model:
            self.logger.warning(
                "Module '%s' not found; disabling beta scheduler.",
                self.module_name,
            )
            self.schedule_fn = lambda _step: self.base_beta
            return None
        module = model[self.module_name]
        if not hasattr(module, 'weight'):
            self.logger.warning(
                "Module '%s' has no weight parameter; disabling beta scheduler.",
                self.module_name,
            )
            self.schedule_fn = lambda _step: self.base_beta
            return None
        self.cached_module = module
        return module

    def _set_weight(self, module, beta):
        weight = getattr(module, 'weight', None)
        if weight is None:
            return
        if isinstance(weight, torch.Tensor):
            with torch.no_grad():
                weight.fill_(beta)
        else:
            module.weight = float(beta)
        bias = getattr(module, 'bias', None)
        if isinstance(bias, torch.Tensor):
            with torch.no_grad():
                bias.zero_()
        elif bias is not None:
            module.bias = 0.0

    def ring(self, batch, model):
        if 'step' not in batch:
            self.logger.warning("Batch missing 'step'; beta scheduler skipped.")
            return
        target = self._get_target(model)
        if target is None:
            return
        beta = float(self.schedule_fn(batch['step']))
        self._set_weight(target, beta)
        batch['beta'] = beta


class LossTrackerHook(AlarmHook):
    """Persist selected loss values (e.g. reconstruction, KL) to CSV."""

    def __init__(
        self,
        logger,
        result_dir,
        *,
        names,
        cols=None,
        filename: str = "loss.csv",
        alarm={'type': 'silent', 'target': 'step'},
        end=False,
        **kwargs,
    ):
        super().__init__(logger, result_dir, alarm=alarm, end=end, **kwargs)
        if not names:
            raise ValueError("LossTrackerHook requires at least one field name")
        cols = cols or names
        if len(names) != len(cols):
            raise ValueError("Length mismatch between 'names' and 'cols'")
        self.names = list(names)
        self.cols = list(cols)
        self.path = os.path.join(result_dir, filename)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._initialised = False

    def _ensure_header(self):
        if self._initialised:
            return
        with open(self.path, 'w', encoding='utf-8') as handle:
            handle.write(','.join(self.cols) + '\n')
        self._initialised = True

    @staticmethod
    def _normalise_value(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            if value.numel() == 1:
                return value.item()
            return value.numpy().tolist()
        if isinstance(value, np.ndarray):
            if value.size == 1:
                return value.item()
            return value.tolist()
        if isinstance(value, (np.generic,)):
            return value.item()
        return value

    def ring(self, batch, model):
        missing = [name for name in self.names if name not in batch]
        if missing:
            self.logger.warning(
                "Batch missing %s; loss tracker skipped.",
                ', '.join(missing),
            )
            return
        self._ensure_header()
        values = [self._normalise_value(batch[name]) for name in self.names]
        line = ','.join(str(v) for v in values)
        with open(self.path, 'a', encoding='utf-8') as handle:
            handle.write(line + '\n')


hook_type2class = {
    'save_alarm': SaveAlarmHook,
    'accumulate': AccumulateHook,
    'abort': AbortHook,
    'step_abort': StepAbortHook,
    'epoch_abort': EpochAbortHook,
    'time_abort': TimeAbortHook,
    'kl_beta_scheduler': KLBetaSchedulerHook,
    'loss_tracker': LossTrackerHook,
}
def get_hook(type, **kwargs):
    return hook_type2class[type](**kwargs)
