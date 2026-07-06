"""Unit tests for EarlyStoppingHook and TrainerBase early-stop integration."""

import math

import pytest

from xmetai.core.hooks import EarlyStoppingHook

# ─── Lightweight trainer / storage mocks ──────────────────────────────────────


class _FakeStorage:
    """Minimal EventStorage stand-in."""

    def __init__(self):
        self._data: dict[str, tuple[float, int]] = {}

    def set(self, key: str, value: float, iteration: int = 0) -> None:
        self._data[key] = (value, iteration)

    def latest(self) -> dict[str, tuple[float, int]]:
        return dict(self._data)


class _FakeTrainer:
    """Minimal TrainerBase stand-in."""

    def __init__(self, max_iter: int = 100, start_iter: int = 0):
        self.iter = start_iter
        self.max_iter = max_iter
        self.storage = _FakeStorage()
        self._early_stop_requested = False


def _make_hook(**overrides) -> EarlyStoppingHook:
    kwargs = dict(eval_period=10, val_metric="loss/val")
    kwargs.update(overrides)
    return EarlyStoppingHook(**kwargs)


def _attach(hook: EarlyStoppingHook, trainer: _FakeTrainer) -> None:
    hook.trainer = trainer  # type: ignore[assignment]


def _step(hook: EarlyStoppingHook, trainer: _FakeTrainer, metric: float) -> None:
    """Simulate one evaluation step: put metric, increment iter, call after_step."""
    trainer.storage.set("loss/val", metric, trainer.iter)
    hook.after_step()
    trainer.iter += hook._period  # jump to next eval point


# ─── Constructor validation ───────────────────────────────────────────────────


def test_invalid_mode():
    with pytest.raises(ValueError, match="mode"):
        EarlyStoppingHook(eval_period=10, val_metric="x", mode="unknown")


def test_invalid_patience():
    with pytest.raises(ValueError, match="patience"):
        EarlyStoppingHook(eval_period=10, val_metric="x", patience=0)


def test_invalid_eval_period():
    with pytest.raises(ValueError, match="eval_period"):
        EarlyStoppingHook(eval_period=0, val_metric="x")


# ─── Mode "min" — stop when metric stops decreasing ──────────────────────────


def test_min_mode_improves_resets_counter():
    trainer = _FakeTrainer(max_iter=1000)
    hook = _make_hook(mode="min", patience=3)
    _attach(hook, trainer)

    # Two improving steps → counter stays 0
    trainer.iter = 9
    trainer.storage.set("loss/val", 1.0)
    hook.after_step()
    trainer.iter = 19
    trainer.storage.set("loss/val", 0.8)
    hook.after_step()

    assert hook._num_no_improve == 0
    assert hook.best_metric == pytest.approx(0.8)
    assert not trainer._early_stop_requested


def test_min_mode_triggers_after_patience():
    trainer = _FakeTrainer(max_iter=1000)
    hook = _make_hook(mode="min", patience=3)
    _attach(hook, trainer)

    # One improvement followed by 3 non-improvements
    trainer.iter = 9
    trainer.storage.set("loss/val", 1.0)
    hook.after_step()

    for i in range(3):
        trainer.iter = 19 + i * 10
        trainer.storage.set("loss/val", 1.0)  # same value — not an improvement
        hook.after_step()

    assert hook._num_no_improve == 3
    assert trainer._early_stop_requested


def test_min_mode_does_not_trigger_before_patience():
    trainer = _FakeTrainer(max_iter=1000)
    hook = _make_hook(mode="min", patience=3)
    _attach(hook, trainer)

    trainer.iter = 9
    trainer.storage.set("loss/val", 1.0)
    hook.after_step()

    for i in range(2):  # only 2 non-improvements, need 3
        trainer.iter = 19 + i * 10
        trainer.storage.set("loss/val", 1.0)
        hook.after_step()

    assert hook._num_no_improve == 2
    assert not trainer._early_stop_requested


# ─── Mode "max" — stop when metric stops increasing ──────────────────────────


def test_max_mode_triggers_after_patience():
    trainer = _FakeTrainer(max_iter=1000)
    hook = _make_hook(mode="max", patience=2, val_metric="acc/val")
    _attach(hook, trainer)

    trainer.iter = 9
    trainer.storage.set("acc/val", 0.8)
    hook.after_step()

    for i in range(2):
        trainer.iter = 19 + i * 10
        trainer.storage.set("acc/val", 0.8)  # no improvement
        hook.after_step()

    assert trainer._early_stop_requested


def test_max_mode_improves():
    trainer = _FakeTrainer(max_iter=1000)
    hook = _make_hook(mode="max", patience=2, val_metric="acc/val")
    _attach(hook, trainer)

    for val, iter_ in [(0.7, 9), (0.8, 19), (0.9, 29)]:
        trainer.iter = iter_
        trainer.storage.set("acc/val", val)
        hook.after_step()

    assert hook._num_no_improve == 0
    assert not trainer._early_stop_requested


# ─── min_delta ────────────────────────────────────────────────────────────────


def test_min_delta_ignores_tiny_improvement():
    trainer = _FakeTrainer(max_iter=1000)
    hook = _make_hook(mode="min", patience=2, min_delta=0.05)
    _attach(hook, trainer)

    trainer.iter = 9
    trainer.storage.set("loss/val", 1.0)
    hook.after_step()

    # A decrease of only 0.03 (< min_delta=0.05) should NOT count as improvement.
    trainer.iter = 19
    trainer.storage.set("loss/val", 0.97)
    hook.after_step()

    assert hook._num_no_improve == 1


def test_min_delta_accepts_sufficient_improvement():
    trainer = _FakeTrainer(max_iter=1000)
    hook = _make_hook(mode="min", patience=2, min_delta=0.05)
    _attach(hook, trainer)

    trainer.iter = 9
    trainer.storage.set("loss/val", 1.0)
    hook.after_step()

    trainer.iter = 19
    trainer.storage.set("loss/val", 0.90)  # decrease of 0.10 >= min_delta
    hook.after_step()

    assert hook._num_no_improve == 0
    assert hook.best_metric == pytest.approx(0.90)


# ─── Edge cases ───────────────────────────────────────────────────────────────


def test_missing_metric_does_not_crash(caplog):
    trainer = _FakeTrainer(max_iter=1000)
    hook = _make_hook()
    _attach(hook, trainer)

    trainer.iter = 9
    # Storage has no metric → warning, no crash, no stop
    hook.after_step()

    assert not trainer._early_stop_requested
    assert any("not found" in r.message for r in caplog.records)


def test_nan_metric_is_skipped():
    trainer = _FakeTrainer(max_iter=1000)
    hook = _make_hook()
    _attach(hook, trainer)

    trainer.iter = 9
    trainer.storage.set("loss/val", float("nan"))
    hook.after_step()

    assert not trainer._early_stop_requested
    assert hook.best_metric is None


def test_inf_metric_is_skipped():
    trainer = _FakeTrainer(max_iter=1000)
    hook = _make_hook()
    _attach(hook, trainer)

    trainer.iter = 9
    trainer.storage.set("loss/val", math.inf)
    hook.after_step()

    assert not trainer._early_stop_requested


def test_hook_does_not_fire_between_eval_periods():
    trainer = _FakeTrainer(max_iter=1000)
    hook = _make_hook(eval_period=10)
    _attach(hook, trainer)

    # iter=4 → (4+1) % 10 != 0
    trainer.iter = 4
    trainer.storage.set("loss/val", 99.0)
    hook.after_step()

    assert hook.best_metric is None


def test_hook_skips_at_final_iter():
    """At max_iter the final eval belongs to EvalHook.after_train, not us."""
    trainer = _FakeTrainer(max_iter=100)
    hook = _make_hook(eval_period=10, patience=1)
    _attach(hook, trainer)

    # First eval: set a good metric
    trainer.iter = 9
    trainer.storage.set("loss/val", 1.0)
    hook.after_step()

    # Final iteration (iter=99, next_iter=100==max_iter) → skip
    trainer.iter = 99
    trainer.storage.set("loss/val", 2.0)  # would trigger stop if evaluated
    hook.after_step()

    assert not trainer._early_stop_requested


def test_improvement_resets_no_improve_counter():
    trainer = _FakeTrainer(max_iter=1000)
    hook = _make_hook(patience=2)
    _attach(hook, trainer)

    trainer.iter = 9
    trainer.storage.set("loss/val", 1.0)
    hook.after_step()

    trainer.iter = 19
    trainer.storage.set("loss/val", 1.0)  # no improve → count=1
    hook.after_step()

    assert hook._num_no_improve == 1

    trainer.iter = 29
    trainer.storage.set("loss/val", 0.5)  # improvement → count reset
    hook.after_step()

    assert hook._num_no_improve == 0
    assert not trainer._early_stop_requested


# ─── state_dict / load_state_dict ─────────────────────────────────────────────


def test_state_dict_round_trip():
    trainer = _FakeTrainer(max_iter=1000)
    hook = _make_hook(patience=3)
    _attach(hook, trainer)

    trainer.iter = 9
    trainer.storage.set("loss/val", 0.5)
    hook.after_step()
    trainer.iter = 19
    trainer.storage.set("loss/val", 0.6)
    hook.after_step()

    sd = hook.state_dict()
    assert sd["best_metric"] == pytest.approx(0.5)
    assert sd["num_no_improve"] == 1

    # Restore into a fresh hook
    hook2 = _make_hook(patience=3)
    _attach(hook2, _FakeTrainer(max_iter=1000))
    hook2.load_state_dict(sd)

    assert hook2.best_metric == pytest.approx(0.5)
    assert hook2._num_no_improve == 1


# ─── TrainerBase integration ──────────────────────────────────────────────────


def test_trainer_breaks_on_early_stop_flag():
    """TrainerBase must check _early_stop_requested and break the training loop."""
    import weakref

    from xmetai.core.trainer import TrainerBase

    class _MinimalTrainer(TrainerBase):
        def __init__(self):
            super().__init__()
            self.steps_run = 0

        def run_step(self):
            self.steps_run += 1

    trainer = _MinimalTrainer()
    hook = EarlyStoppingHook(eval_period=2, val_metric="v", patience=1)
    hook.trainer = weakref.proxy(trainer)
    trainer.register_hooks([hook])

    # Inject declining metrics via a wrapper hook
    class _InjectMetric:
        trainer = None

        def before_train(self):
            pass

        def after_train(self):
            pass

        def before_step(self):
            pass

        def after_backward(self):
            pass

        def after_step(self):
            # Always write a non-improving value so early stop fires quickly
            self.trainer.storage.put_scalar("v", 1.0, smoothing_hint=False)

        def state_dict(self):
            return {}

    injector = _InjectMetric()
    injector.trainer = weakref.proxy(trainer)
    trainer._hooks.insert(0, injector)

    trainer.train(start_iter=0, max_iter=100)

    # Early stopping should have triggered well before iteration 100
    assert trainer.steps_run < 100
    assert trainer._early_stop_requested
