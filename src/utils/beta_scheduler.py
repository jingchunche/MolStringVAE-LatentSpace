import logging
import math
from typing import Callable, Optional, Union, Mapping, Any

__all__ = ["build_beta_scheduler"]
_VALID_SCHEDULES = {"linear", "cosine"}

def build_beta_scheduler(
    config: Optional[Mapping[str, Any]],
    *,
    base_beta: Union[int, float, None],
    max_step: Optional[int],
    logger: Optional[logging.Logger] = None,
) -> Callable[[int], float]:
    log = logger or logging.getLogger(__name__)
    base_value = float(base_beta) if base_beta is not None else 0.0

    if not config:
        log.debug("Beta scheduler not configured; using constant beta %s", base_value)
        return lambda _step: base_value

    if not isinstance(config, Mapping):
        log.warning("Unsupported scheduler config type %s; using constant beta %s", type(config), base_value)
        return lambda _step: base_value

    cfg = config
    mode_raw = cfg.get("mode")
    if not mode_raw:
        log.debug("Scheduler lacks 'mode'; using constant beta %s", base_value)
        return lambda _step: base_value

    mode = str(mode_raw).strip().lower()
    if max_step is None or max_step <= 0:
        log.warning(
            "Invalid or missing max_step (%s) for beta scheduler (mode=%s); falling back to constant beta.",
            max_step,
            mode,
        )
        return lambda _step: base_value

    builders = {
        "warmup": _build_warmup_scheduler,
        "cycling": _build_cycling_scheduler,
    }
    builder = builders.get(mode)
    if builder is None:
        log.warning("Unsupported beta scheduler mode '%s'; using constant beta %s", mode_raw, base_value)
        return lambda _step: base_value

    return builder(cfg, base_value, max_step, log)


def _build_warmup_scheduler(
    cfg: Mapping[str, Any], base_beta: float, max_step: int, logger: logging.Logger
) -> Callable[[int], float]:
    if "beta_max" not in cfg:
        logger.warning("Beta scheduler missing required parameter 'beta_max'.")
        return lambda _step: base_beta
    beta_max = float(cfg["beta_max"])
    if beta_max <= base_beta:
        logger.warning(
            "beta_max (%s) must be greater than base_beta (%s) for warmup; using constant beta.",
            beta_max,
            base_beta,
        )
        return lambda _step: base_beta

    warmup_steps_raw = cfg.get("warmup_steps")
    start_step = end_step = None
    if warmup_steps_raw is not None:
        start_step, end_step = _parse_warmup_steps(warmup_steps_raw, logger)
        if start_step is None or end_step is None:
            logger.warning(
                "Invalid warmup_steps config '%s'; falling back to warmup_ratio.",
                warmup_steps_raw,
            )
            start_step = end_step = None
        else:
            start_step = max(0, int(start_step))
            if start_step >= max_step:
                logger.warning(
                    "warmup_steps start (%s) must be < max_step (%s); clamping to max_step-1.",
                    start_step,
                    max_step,
                )
                start_step = max(0, max_step - 1)
            end_step = max(start_step + 1, int(end_step))
            if end_step > max_step:
                logger.warning(
                    "warmup_steps end (%s) exceeds max_step (%s); clamping to max_step.",
                    end_step,
                    max_step,
                )
                end_step = max_step

    if start_step is None or end_step is None:
        warmup_ratio_raw = cfg.get("warmup_ratio", 0.1)
        start_ratio, end_ratio = _parse_warmup_ratios(warmup_ratio_raw, logger)
        if start_ratio < 0:
            logger.warning(
                "warmup_ratio start (%s) must be >= 0; clamping to 0.",
                start_ratio,
            )
            start_ratio = 0.0
        if end_ratio <= start_ratio:
            logger.warning(
                "warmup_ratio end (%s) must be greater than start (%s); using constant beta.",
                end_ratio,
                start_ratio,
            )
            return lambda _step: base_beta
        if end_ratio > 1:
            logger.warning("warmup_ratio end (%s) exceeds 1.0; clamping to 1.0.", end_ratio)
            end_ratio = 1.0

        start_step = max(0, int(round(start_ratio * max_step)))
        end_step = max(start_step + 1, int(round(end_ratio * max_step)))
        span_desc = f"ratios {start_ratio}~{end_ratio}"
    else:
        span_desc = f"steps {start_step}~{end_step}"

    ramp_steps = max(1, end_step - start_step)
    schedule_name = str(cfg.get("schedule", "linear")).strip().lower()
    schedule_fn = _get_progress_fn(schedule_name, logger)
    delta = beta_max - base_beta

    logger.info(
        "Initialised KL beta warmup: base=%s max=%s span=%s schedule=%s",
        base_beta,
        beta_max,
        span_desc,
        schedule_name,
    )

    def _schedule(step: int) -> float:
        step_idx = max(0, int(step))
        if step_idx < start_step:
            return base_beta
        if step_idx >= start_step + ramp_steps:
            return beta_max
        progress = schedule_fn(step_idx - start_step, ramp_steps)
        return base_beta + delta * progress

    return _schedule


def _parse_warmup_ratios(value, logger: logging.Logger) -> tuple:
    """Return (start_ratio, end_ratio) given user config.

    The config can be a float (legacy behaviour), a sequence of two
    numbers, or a string such as "0.5 0.8" / "0.5,0.8".
    """

    def _as_float(val):
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    start_ratio = 0.0
    end_ratio = None

    if isinstance(value, (list, tuple)) and value:
        if len(value) >= 2:
            start_ratio = _as_float(value[0]) or 0.0
            end_ratio = _as_float(value[1])
        else:
            end_ratio = _as_float(value[0])
    elif isinstance(value, str):
        tokens = value.replace(",", " ").split()
        if len(tokens) >= 2:
            start_ratio = _as_float(tokens[0]) or 0.0
            end_ratio = _as_float(tokens[1])
        elif tokens:
            end_ratio = _as_float(tokens[0])
    else:
        end_ratio = _as_float(value)

    if end_ratio is None:
        logger.warning(
            "Invalid warmup_ratio value '%s'; defaulting to 0.1 for end ratio.",
            value,
        )
        end_ratio = 0.1

    return float(start_ratio), float(end_ratio)


def _parse_warmup_steps(value, logger: logging.Logger) -> tuple:
    """Return (start_step, end_step) parsed from user config."""

    def _as_int(val):
        try:
            return int(val)
        except (TypeError, ValueError):
            return None

    start_step = 0
    end_step = None

    if isinstance(value, (list, tuple)) and value:
        if len(value) >= 2:
            parsed_start = _as_int(value[0])
            start_step = parsed_start if parsed_start is not None else 0
            end_step = _as_int(value[1])
        else:
            end_step = _as_int(value[0])
    elif isinstance(value, str):
        tokens = value.replace(",", " ").split()
        if len(tokens) >= 2:
            parsed_start = _as_int(tokens[0])
            start_step = parsed_start if parsed_start is not None else 0
            end_step = _as_int(tokens[1])
        elif tokens:
            end_step = _as_int(tokens[0])
    else:
        end_step = _as_int(value)

    if end_step is None:
        logger.warning("Invalid warmup_steps value '%s'; expected one or two integers.", value)
        return None, None

    return start_step, end_step


def _build_cycling_scheduler(
    cfg: Mapping[str, Any], base_beta: float, max_step: int, logger: logging.Logger
) -> Callable[[int], float]:
    if "beta_max" not in cfg:
        logger.warning("Beta scheduler missing required parameter 'beta_max'.")
        return lambda _step: base_beta
    beta_max = float(cfg["beta_max"])
    if beta_max <= base_beta:
        logger.warning(
            "beta_max (%s) must be greater than base_beta (%s) for cycling; using constant beta.",
            beta_max,
            base_beta,
        )
        return lambda _step: base_beta

    num_cycles = int(cfg.get("num_cycles", 1))
    if num_cycles < 1:
        logger.warning("num_cycles (%s) must be >= 1; clamping to 1.", num_cycles)
        num_cycles = 1
    cycle_ratio = float(cfg.get("cycle_ratio", 0.3))
    if cycle_ratio <= 0:
        logger.warning("cycle_ratio (%s) must be positive; using constant beta.", cycle_ratio)
        return lambda _step: base_beta
    if cycle_ratio > 1:
        logger.warning("cycle_ratio (%s) exceeds 1.0; clamping to 1.0.", cycle_ratio)
        cycle_ratio = 1.0

    schedule_name = str(cfg.get("schedule", "linear")).strip().lower()
    schedule_fn = _get_progress_fn(schedule_name, logger)

    cycle_length = max(1, int(math.ceil(max_step / num_cycles)))
    ramp_length = max(1, int(round(cycle_ratio * cycle_length)))

    delta = beta_max - base_beta

    logger.info(
        "Initialised KL beta cycling: base=%s max=%s cycles=%s cycle_len=%s ramp_len=%s schedule=%s",
        base_beta,
        beta_max,
        num_cycles,
        cycle_length,
        ramp_length,
        schedule_name,
    )

    def _schedule(step: int) -> float:
        step_idx = max(0, int(step))
        if step_idx >= max_step:
            return beta_max
        cycle_index = step_idx // cycle_length
        cycle_start = cycle_index * cycle_length
        # Adjust last cycle length if max_step does not align perfectly.
        remaining = max_step - cycle_start
        span = max(1, min(cycle_length, remaining))
        ramp = max(1, min(ramp_length, span))
        position = step_idx - cycle_start
        if position >= ramp:
            return beta_max
        progress = schedule_fn(position, ramp)
        return base_beta + delta * progress

    return _schedule


def _get_progress_fn(name: str, logger: logging.Logger) -> Callable[[int, int], float]:
    if name not in _VALID_SCHEDULES:
        logger.warning("Unknown beta scheduler curve '%s'; falling back to linear.", name)
        name = "linear"

    if name == "linear":
        return _linear_progress
    return _cosine_progress


def _linear_progress(step: int, span: int) -> float:
    if span <= 0:
        return 1.0
    return min(max(float(step) / float(span), 0.0), 1.0)


def _cosine_progress(step: int, span: int) -> float:
    if span <= 0:
        return 1.0
    ratio = min(max(float(step) / float(span), 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * ratio)
