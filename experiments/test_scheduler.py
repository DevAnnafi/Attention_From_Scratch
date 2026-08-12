"""
Verify the Noam warmup schedule matches the paper's formula.

Run: python experiments/test_scheduler.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math
import torch
from src.scheduler import WarmupScheduler, get_optimizer_and_scheduler
from src.model import Transformer


def paper_lr(step, d_model, warmup_steps):
    """Section 5.3 formula."""
    if step == 0:
        return 0.0
    return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)


def test_scheduler_matches_formula():
    d_model = 512
    warmup = 4000

    model = Transformer(32, 2, 64, 2, 100, 50)
    optimizer, scheduler = get_optimizer_and_scheduler(model, d_model, warmup)

    errors = []
    for step in [1, 100, 4000, 4001, 10000, 50000]:
        scheduler.step_num = step - 1
        scheduler.step()
        actual = optimizer.param_groups[0]['lr']
        expected = paper_lr(step, d_model, warmup)
        err = abs(actual - expected)
        errors.append(err)
        print(f"step {step:6d}  expected {expected:.6e}  actual {actual:.6e}  err {err:.2e}")

    assert all(e < 1e-10 for e in errors), "Scheduler does not match paper formula"
    print("\nPASS — scheduler matches section 5.3 formula")


if __name__ == '__main__':
    test_scheduler_matches_formula()