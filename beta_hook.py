from src.utils.beta_scheduler import build_beta_scheduler


class KLBetaSchedulerHook(AlarmHook):
    """Adjust the KL loss weight (`-d_kl_factor`) according to a beta schedule."""

    def __init__(self, logger, result_dir,
        module='-d_kl_factor', base_beta=None, max_step=None, scheduler=None,
        alarm={'type': 'silent', 'target': 'step'}, end=False, **kwargs):
        super().__init__(logger, result_dir, alarm=alarm, end=end, **kwargs)
        self.module_name = module
        self.base_beta = float(base_beta) if base_beta is not None else 0.0
        self.max_step = max_step
        self.scheduler_config = scheduler
        self.schedule_fn = self._build_schedule()
        self.cached_module = None

    def _build_schedule(self):
        if not self.max_step:
            self.logger.warning("KLBetaSchedulerHook requires max_step; falling back to constant beta %s.", self.base_beta)
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
            self.logger.warning("Module '%s' not found; disabling beta scheduler.", self.module_name)
            self.schedule_fn = lambda _step: self.base_beta
            return None
        module = model[self.module_name]
        if not hasattr(module, 'weight'):
            self.logger.warning("Module '%s' has no weight parameter; disabling beta scheduler.", self.module_name)
            self.schedule_fn = lambda _step: self.base_beta
            return None
        self.cached_module = module
        return module

    def ring(self, batch, model):
        if 'step' not in batch:
            self.logger.warning("Batch missing 'step'; beta scheduler skipped.")
            return
        target = self._get_target(model)
        if target is None:
            return
        beta = float(self.schedule_fn(batch['step']))
        with torch.no_grad():
            target.weight.fill_(beta)
            if hasattr(target, 'bias') and target.bias is not None:
                target.bias.zero_()
        batch['beta'] = beta


# When ready to integrate, append to `src/hooks.py`:
# hook_type2class['kl_beta_scheduler'] = KLBetaSchedulerHook
