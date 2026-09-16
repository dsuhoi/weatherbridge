from __future__ import annotations

import torch
from torch import nn

from tools.eval.capmatched_loader import CapMatchedEnsemble


class _Member(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value
        self.delta_t = 6.0

    def forward(self, x0, xT, tau, cond=None, static=None):
        del xT, tau, cond, static
        return torch.full_like(x0, self.value), {}


def test_capmatched_ensemble_averages_member_predictions() -> None:
    ensemble = CapMatchedEnsemble([_Member(1.0), _Member(3.0)])
    x0 = torch.zeros(2, 3, 4, 5)
    output, auxiliary = ensemble(x0, x0, torch.tensor([0.25, 0.75]))

    torch.testing.assert_close(output, torch.full_like(x0, 2.0))
    assert int(auxiliary["ensemble_size"].item()) == 2


def test_capmatched_ensemble_rejects_one_member() -> None:
    try:
        CapMatchedEnsemble([_Member(1.0)])
    except ValueError as error:
        assert "at least two" in str(error)
    else:
        raise AssertionError("single-member ensemble was accepted")
