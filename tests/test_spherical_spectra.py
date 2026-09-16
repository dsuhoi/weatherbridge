import numpy as np
import pytest
import torch

from weather_time_interp.grid import wb2_block_average_latitudes
from weather_time_interp.metrics.spherical_spectra import (
    CellCenteredRealSHT,
    CellCenteredRealVectorSHT,
    aligned_sht_band_objective,
    coefficient_power,
    fejer1_nodes_weights,
)


def test_fejer1_nodes_match_half_cell_latitudes_and_integrate_polynomials():
    nlat = 16
    theta, weights = fejer1_nodes_weights(nlat)
    np.testing.assert_allclose(
        theta,
        (np.arange(nlat) + 0.5) * np.pi / nlat,
        rtol=0.0,
        atol=0.0,
    )
    x = np.cos(theta)
    for degree in range(nlat):
        expected = 0.0 if degree % 2 else 2.0 / (degree + 1)
        np.testing.assert_allclose(
            np.sum(weights * x**degree),
            expected,
            rtol=0.0,
            atol=2.0e-13,
        )


def test_cell_centered_sht_recovers_a_pure_spherical_harmonic():
    pytest.importorskip("torch_harmonics")
    special = pytest.importorskip("scipy.special")
    sph_harm_y = getattr(special, "sph_harm_y", None)
    if sph_harm_y is None:
        pytest.skip("SciPy does not expose sph_harm_y")

    nlat, nlon = 32, 64
    ell, m = 8, 3
    theta, quadrature = fejer1_nodes_weights(nlat)
    longitude = np.arange(nlon) * 2.0 * np.pi / nlon
    field = np.sqrt(2.0) * np.real(
        sph_harm_y(ell, m, theta[:, None], longitude[None, :])
    )
    sht = CellCenteredRealSHT(nlat, nlon, lmax=16, mmax=16).double()
    coefficients = sht(torch.from_numpy(field))
    power = coefficient_power(coefficients)

    assert int(power.argmax()) == ell
    assert float(power[ell] / power.sum()) > 1.0 - 1.0e-12
    spatial_energy = (2.0 * np.pi / nlon) * np.sum(
        quadrature[:, None] * field**2
    )
    np.testing.assert_allclose(float(power.sum()), spatial_energy, atol=2.0e-12)


def test_aligned_objective_penalizes_antiphase_even_when_power_matches():
    class FakeSHT:
        lmax = 3

        def __call__(self, field):
            sign = field[:, :, 0, 0].reshape(field.shape[0], field.shape[1], 1, 1)
            base = torch.ones(
                field.shape[0],
                field.shape[1],
                self.lmax,
                2,
                dtype=torch.complex128,
            )
            return sign * base

    target = torch.ones(2, 3, 1, 1)
    prediction = -target
    result = aligned_sht_band_objective(
        prediction,
        target,
        FakeSHT(),
        ell_min=1,
    )

    dtype = result["total"].dtype
    torch.testing.assert_close(result["amplitude"], torch.zeros((), dtype=dtype))
    torch.testing.assert_close(result["shape"], torch.zeros((), dtype=dtype))
    torch.testing.assert_close(result["phase"], torch.full((), 2.0, dtype=dtype))


def test_vector_sht_power_is_invariant_to_longitude_translation():
    pytest.importorskip("torch_harmonics")
    torch.manual_seed(3)
    field = torch.randn(2, 3, 2, 32, 64, dtype=torch.float64)
    vector_sht = CellCenteredRealVectorSHT(
        32,
        64,
        lmax=16,
        mmax=16,
    ).double()

    original = coefficient_power(vector_sht(field))
    translated = coefficient_power(vector_sht(torch.roll(field, 7, dims=-1)))

    torch.testing.assert_close(original, translated, rtol=3.0e-13, atol=3.0e-14)


def test_vector_sht_recovers_degree_of_zonal_harmonic_gradient():
    pytest.importorskip("torch_harmonics")
    special = pytest.importorskip("scipy.special")

    nlat, nlon = 64, 128
    ell = 8
    theta, _ = fejer1_nodes_weights(nlat)
    x = np.cos(theta)
    legendre = special.eval_legendre(ell, x)
    previous = special.eval_legendre(ell - 1, x)
    derivative_x = ell * (x * legendre - previous) / (np.square(x) - 1.0)
    normalization = np.sqrt((2 * ell + 1) / (4.0 * np.pi))
    northward_gradient = normalization * np.sin(theta) * derivative_x
    meridional = np.broadcast_to(northward_gradient[:, None], (nlat, nlon))
    field = np.stack((np.zeros_like(meridional), meridional), axis=0)

    vector_sht = CellCenteredRealVectorSHT(
        nlat,
        nlon,
        lmax=16,
        mmax=16,
    ).double()
    power = coefficient_power(vector_sht(torch.from_numpy(field)))
    total_by_degree = power.sum(dim=0)

    assert int(total_by_degree.argmax()) == ell
    assert float(total_by_degree[ell] / total_by_degree.sum()) > 1.0 - 1.0e-12
    assert float(power[:, ell].min() / power[:, ell].max()) < 1.0e-24


def test_explicit_latitude_sht_uses_actual_wb2_rows():
    pytest.importorskip("torch_harmonics")
    latitude = wb2_block_average_latitudes()
    sht = CellCenteredRealSHT(
        360,
        720,
        lmax=8,
        mmax=8,
        latitude_degrees=latitude,
    ).double()
    coefficients = sht(torch.ones(1, 1, 360, 720, dtype=torch.float64))
    assert sht.grid == "explicit-latitude-strip-area"
    assert torch.isfinite(torch.view_as_real(coefficients)).all()
