import torch

from tools.eval.capmatched_loader import (
    normalize_weatherbridge_decoder_state_dict,
)
from tools.train.capmatched_to_bare import checkpoint_arch_meta, resolve_bare_arch


def test_public_weatherbridge_bare_aliases_select_paper_variants() -> None:
    assert resolve_bare_arch("weatherbridge") == "flow_pp3"
    assert resolve_bare_arch("weatherbridge_detail") == "flow_pp3_detail"
    assert resolve_bare_arch("weatherbridge_flow_spectral") == "flow_pp3"
    assert (
        resolve_bare_arch("weatherbridge_universal_latent_refine")
        == "flow_universal_latent_refine"
    )


def test_legacy_single_block_decoder_keys_are_normalized() -> None:
    weight = torch.empty(4, 4, 3, 3)
    state = {
        "encoder.0.0.weight": torch.empty(4, 75, 3, 3),
        "dec.0.0.weight": weight,
        "dec.0.1.weight": torch.empty(4),
    }

    normalized = normalize_weatherbridge_decoder_state_dict(state)

    assert normalized["dec.0.0.0.weight"] is weight
    assert "dec.0.0.weight" not in normalized
    assert normalized["encoder.0.0.weight"] is state["encoder.0.0.weight"]


def test_current_decoder_keys_are_unchanged() -> None:
    state = {"dec.0.0.0.weight": torch.empty(4, 4, 3, 3)}

    assert normalize_weatherbridge_decoder_state_dict(state) is state


def test_mixed_decoder_layout_is_rejected() -> None:
    state = {
        "dec.0.0.weight": torch.empty(4, 4, 3, 3),
        "dec.1.0.0.weight": torch.empty(4, 4, 3, 3),
    }

    try:
        normalize_weatherbridge_decoder_state_dict(state)
    except ValueError as error:
        assert "mixed or unrecognised" in str(error)
    else:
        raise AssertionError("mixed decoder layout was accepted")


def test_legacy_skip_bare_metadata_follows_checkpoint_shape() -> None:
    legacy = checkpoint_arch_meta(
        "wb_skip",
        {"encoder.conv_in.weight": torch.empty(128, 51, 1, 1)},
    )
    strict = checkpoint_arch_meta(
        "wb_skip",
        {"encoder.conv_in.weight": torch.empty(64, 51, 1, 1)},
    )

    assert legacy["kwargs"]["block_out_channels"] == (128, 128, 256, 256)
    assert legacy["kwargs"]["latent_channels"] == 32
    assert strict["kwargs"]["block_out_channels"] == (64, 128, 256)
    assert strict["kwargs"]["latent_channels"] == 256


def test_spherical_flow_metadata_enables_only_new_geometry() -> None:
    meta = checkpoint_arch_meta("flow_spherical_ep", {})

    assert meta["cls"] == "WeatherBridgeModel"
    assert meta["kwargs"]["spherical_ops"] is True
    assert meta["kwargs"]["endpoint_preserving"] is True
    assert meta["kwargs"]["query_independent_trajectory"] is True
    assert meta["kwargs"]["spectral_branch"] is True
    assert meta["kwargs"]["hydro_couple"] is True


def test_spherical_pp3_metadata_preserves_temporal_logic() -> None:
    meta = checkpoint_arch_meta("flow_pp3_spherical", {})

    assert meta["cls"] == "WeatherBridgeModel"
    assert meta["kwargs"]["spherical_ops"] is True
    assert meta["kwargs"]["endpoint_preserving"] is False
    assert meta["kwargs"]["query_independent_trajectory"] is False
    assert meta["kwargs"]["spectral_branch"] is True
    assert meta["kwargs"]["hydro_couple"] is True


def test_flow_pp3_spectral_variant_bare_metadata() -> None:
    multiband = checkpoint_arch_meta("flow_pp3_multiband", {})
    detail = checkpoint_arch_meta("flow_pp3_detail", {})

    for meta in (multiband, detail):
        assert meta["cls"] == "WeatherBridgeModel"
        assert meta["kwargs"]["use_accel"] is True
        assert meta["kwargs"]["spectral_branch"] is True
        assert meta["kwargs"]["hydro_couple"] is True
    assert multiband["kwargs"]["multiband_calibration"] is True
    assert multiband["kwargs"]["anchor_detail_bypass"] is False
    assert detail["kwargs"]["multiband_calibration"] is False
    assert detail["kwargs"]["anchor_detail_bypass"] is True


def test_flow_matching_bare_metadata_separates_universal_controls() -> None:
    detail = checkpoint_arch_meta("flow_pp3_detail_fm", {})
    deterministic = checkpoint_arch_meta("flow_universal_detail", {})
    universal = checkpoint_arch_meta("flow_universal_fm", {})

    for meta in (detail, universal):
        assert meta["kwargs"]["flow_matching"] is True
        assert meta["kwargs"]["flow_matching_steps"] == 4
        assert meta["kwargs"]["endpoint_preserving"] is True
        assert meta["kwargs"]["anchor_detail_bypass"] is True
    assert detail["kwargs"]["shared_field_controls"] is False
    assert detail["kwargs"]["hydro_couple"] is True
    assert universal["kwargs"]["shared_field_controls"] is True
    assert universal["kwargs"]["hydro_couple"] is False
    assert universal["kwargs"]["latent_transformer_tokens"] == 16
    assert deterministic["kwargs"]["flow_matching"] is False
    assert deterministic["kwargs"]["shared_field_controls"] is True
    assert deterministic["kwargs"]["hydro_couple"] is False
    assert deterministic["kwargs"]["endpoint_preserving"] is True

    latent = checkpoint_arch_meta("flow_universal_latent", {})
    assert latent["kwargs"]["flow_matching"] is False
    assert latent["kwargs"]["shared_field_controls"] is True
    assert latent["kwargs"]["hydro_couple"] is False
    assert latent["kwargs"]["latent_transformer_tokens"] == 16
    assert latent["kwargs"]["latent_transformer_dim"] == 128

    refine = checkpoint_arch_meta("flow_universal_latent_refine", {})
    assert refine["kwargs"]["shared_field_controls"] is True
    assert refine["kwargs"]["decoder_blocks_per_level"] == 2
