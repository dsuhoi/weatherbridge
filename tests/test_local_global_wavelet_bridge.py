from __future__ import annotations

import pytest
import torch

from weather_time_interp.model.local_global_wavelet_bridge_model import (
    LocalGlobalWaveletBridgeModel,
    haar_dwt2,
    haar_iwt2,
    haar_pyramid,
    haar_reconstruct,
)


def _small_model() -> LocalGlobalWaveletBridgeModel:
    return LocalGlobalWaveletBridgeModel(
        in_channels=4,
        out_channels=4,
        n_static_features=3,
        widths=(24, 16, 8),
        endpoint_widths=(8, 6, 4),
        band_widths=(10, 8, 6),
        static_widths=(4, 4, 2),
        blocks=(1, 1, 1),
        time_dim=16,
        global_width=0,
        global_modes=0,
        base_grid=(16, 32),
    )


def test_haar_analysis_is_orthonormal_and_invertible() -> None:
    torch.manual_seed(11)
    field = torch.randn(2, 5, 16, 32)

    low, detail = haar_dwt2(field)
    reconstructed = haar_iwt2(low, detail)

    torch.testing.assert_close(reconstructed, field, atol=1e-6, rtol=1e-6)
    input_energy = field.square().sum()
    coefficient_energy = low.square().sum() + detail.square().sum()
    torch.testing.assert_close(
        coefficient_energy,
        input_energy,
        atol=2e-4,
        rtol=2e-6,
    )


def test_three_level_pyramid_reconstructs_input() -> None:
    torch.manual_seed(12)
    field = torch.randn(1, 4, 16, 32)

    low, details = haar_pyramid(field)

    assert low.shape == (1, 4, 2, 4)
    assert [detail.shape for detail in details] == [
        (1, 12, 8, 16),
        (1, 12, 4, 8),
        (1, 12, 2, 4),
    ]
    torch.testing.assert_close(
        haar_reconstruct(low, details),
        field,
        atol=1e-6,
        rtol=1e-6,
    )


def test_zero_initialized_bridge_starts_from_linear_interpolation() -> None:
    torch.manual_seed(13)
    model = _small_model().eval()
    x0 = torch.randn(2, 4, 16, 32)
    xT = torch.randn_like(x0)
    static = torch.randn(2, 3, 16, 32)
    tau = torch.tensor([0.25, 0.7])

    with torch.no_grad():
        output, auxiliary = model(x0, xT, tau, static=static)

    expected = (1.0 - tau[:, None, None, None]) * x0
    expected = expected + tau[:, None, None, None] * xT
    torch.testing.assert_close(output, expected, atol=2e-6, rtol=2e-6)
    assert auxiliary["coarse_low_delta"].abs().max().item() == 0.0


def test_bridge_has_exact_endpoint_contract() -> None:
    torch.manual_seed(14)
    model = _small_model().eval()
    x0 = torch.randn(2, 4, 16, 32)
    xT = torch.randn_like(x0)
    static = torch.randn(2, 3, 16, 32)

    with torch.no_grad():
        output, _ = model(
            x0,
            xT,
            torch.tensor([0.0, 1.0]),
            static=static,
        )

    assert torch.equal(output[0], x0[0])
    assert torch.equal(output[1], xT[1])


def test_detail_bands_include_latitude_reflection_sign() -> None:
    model = LocalGlobalWaveletBridgeModel(
        widths=(24, 16, 8),
        endpoint_widths=(8, 6, 4),
        band_widths=(10, 8, 6),
        static_widths=(4, 4, 2),
        blocks=(1, 1, 1),
        time_dim=16,
        global_width=0,
        global_modes=0,
        base_grid=(16, 32),
    )
    expected = torch.ones(24)
    expected[4:12] = -1.0
    expected[21:23] = -1.0

    torch.testing.assert_close(model.field_pole_parity, expected)
    lon_high, lat_high, diagonal = model.detail_pole_parity.chunk(3)
    torch.testing.assert_close(lon_high, expected)
    torch.testing.assert_close(lat_high, -expected)
    torch.testing.assert_close(diagonal, -expected)


def test_bridge_backward_reaches_coefficient_heads() -> None:
    torch.manual_seed(15)
    model = _small_model().train()
    x0 = torch.randn(1, 4, 16, 32)
    xT = torch.randn_like(x0)
    static = torch.randn(1, 3, 16, 32)
    output, _ = model(x0, xT, torch.tensor([0.5]), static=static)

    output.square().mean().backward()

    assert model.low_head.weight.grad is not None
    assert torch.isfinite(model.low_head.weight.grad).all()
    for head in model.detail_heads:
        assert head.weight.grad is not None
        assert torch.isfinite(head.weight.grad).all()


@pytest.mark.gpu
@pytest.mark.prod_grid
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_canonical_bridge_full_grid_forward_backward() -> None:
    torch.manual_seed(16)
    model = LocalGlobalWaveletBridgeModel().cuda().train()
    x0 = torch.randn(1, 24, 360, 720, device="cuda")
    xT = torch.randn_like(x0)
    static = torch.randn(1, 3, 360, 720, device="cuda")

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output, _ = model(x0, xT, torch.tensor([0.5], device="cuda"), static=static)
        loss = output.square().mean()
    loss.backward()

    assert output.shape == x0.shape
    assert torch.isfinite(output).all()
    assert model.low_head.weight.grad is not None
    assert torch.isfinite(model.low_head.weight.grad).all()
