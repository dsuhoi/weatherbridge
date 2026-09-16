from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import numpy as np

from scripts.paper_plot_style import (
    MODEL_COLORS,
    MODEL_LINESTYLES,
    MODEL_MARKERS,
)

ROOT = Path(__file__).resolve().parents[2]


def test_full_query_legend_does_not_overlap_panel_titles(tmp_path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    from make_fig_full_vs_heldout import ARCHITECTURES, YEARS, plt, summary_figure

    rows = [
        {
            "architecture": architecture,
            "year": str(year),
            "hours": split,
            "heldout_nrmse": str(0.07 + index * 0.01),
            "full_nrmse": str(0.068 + index * 0.01),
            "full_gain_pct": "2.5",
        }
        for index, architecture in enumerate(ARCHITECTURES)
        for year in YEARS
        for split in ("trained", "held_out")
    ]
    summary_figure(rows, tmp_path / "summary.pdf")
    figure = plt.gcf()
    try:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        legend = figure.legends[0].get_window_extent(renderer)
        for axes in figure.axes:
            assert not legend.overlaps(axes._left_title.get_window_extent(renderer))
    finally:
        plt.close(figure)


def test_paper_plot_style_embeds_truetype_fonts() -> None:
    assert mpl.rcParams["pdf.fonttype"] == 42
    assert mpl.rcParams["ps.fonttype"] == 42


def test_weatherbridge_has_one_fixed_visual_identity() -> None:
    assert MODEL_COLORS["WeatherBridge"] == "#D62728"
    assert MODEL_MARKERS["WeatherBridge"] == "X"
    assert MODEL_LINESTYLES["WeatherBridge"] == "-"
    assert "Detail-bypass ablation" not in MODEL_COLORS
    assert MODEL_COLORS["Frozen coefficient adapter"] != MODEL_COLORS["WeatherBridge"]


def test_active_comparison_generators_include_weatherbridge() -> None:
    generators = (
        "make_fig1_rmse_per_tau.py",
        "make_fig2_acc_per_tau.py",
        "make_fig_main_scores.py",
        "make_fig3_spectra.py",
        "make_fig_specific_channels_phys.py",
        "make_fig_rmse_maps_5tau_per_field.py",
        "make_fig_case_ida_wind.py",
        "make_fig5_haishen_local.py",
        "make_fig_supp_laura_wind.py",
        "make_hres_field_comparison.py",
        "make_nwp_adapter_2022_figure.py",
    )
    for filename in generators:
        source = (ROOT / "scripts" / filename).read_text()
        assert '"WeatherBridge"' in source, filename
        assert "Detail-bypass ablation" not in source, filename


def test_hres_figures_require_the_canonical_model_set() -> None:
    generator = (ROOT / "scripts" / "make_hres_field_comparison.py").read_text()
    queue = (ROOT / "scripts" / "run_hres_field_figures_cloudru.sh").read_text()
    for label in (
        "Linear Interp.",
        "WeatherDCAE-14M",
        "WeatherBridge",
    ):
        assert f'"{label}"' in generator
        assert f'--artifact "{label}=' in queue
    assert "set(artifacts) != set(MODEL_ORDER)" in generator
    assert "MODEL_COLORS[label]" in generator
    assert 'linewidth=1.35 if label == "WeatherBridge" else 1.0' in generator
    assert "minimum - 0.12 * span" in generator
    assert 'f"{chr(ord(\'a\') + channel_index)})"' in generator


def test_main_score_panels_use_npj_lettering() -> None:
    source = (ROOT / "scripts" / "make_fig_main_scores.py").read_text()
    manuscript = (ROOT / "paper" / "main.tex").read_text()
    assert '"a)  Error"' in source
    assert '"b)  Anomaly correlation"' in source
    assert '"a  Error"' not in source
    assert '"b  Anomaly correlation"' not in source
    assert "\\textbf{a)} Normalised RMSE" in manuscript
    assert "\\textbf{b)} ACC" in manuscript
    assert "(\\textbf{a--d})" not in manuscript
    assert "(\\textbf{e--h})" not in manuscript


def test_spectral_field_choice_is_disclosed_as_developmental() -> None:
    manuscript = " ".join((ROOT / "paper" / "main.tex").read_text().split())
    assert "Specific humidity (Q850) and zonal wind (U850)" in manuscript
    assert "were chosen during development" in manuscript
    assert "are interpreted descriptively" in manuscript
    assert "Power and cospectrum must be read together" in manuscript
    assert "The figure shows the component curves" in manuscript


def test_case_generators_use_canonical_model_colors() -> None:
    for filename in (
        "make_fig_case_ida_wind.py",
        "make_fig5_haishen_local.py",
    ):
        source = (ROOT / "scripts" / filename).read_text()
        assert "from paper_plot_style import MODEL_COLORS" in source, filename
        assert '"WeatherBridge"' in source, filename
        assert "Detail-bypass ablation" not in source, filename
        assert "MODEL_COLORS[" in source, filename
        assert "linewidths=1.0" in source, filename


def test_main_score_figure_marks_held_out_query_hours() -> None:
    generator = (ROOT / "scripts" / "make_fig_main_scores.py").read_text()
    manuscript = (ROOT / "paper" / "main.tex").read_text()
    assert "for held_tau in (2, 4)" in generator
    assert "ax.axvspan" in generator
    assert "Grey bands mark the held-out query hours" in manuscript


def test_haishen_figure_keeps_geographic_context() -> None:
    source = (ROOT / "scripts" / "make_fig5_haishen_local.py").read_text()
    assert "LON_TICKS = (105, 120, 135)" in source
    assert "LAT_TICKS = (25, 40, 55)" in source
    assert 'f"{value}°E"' in source
    assert 'f"{value}°N"' in source


def test_laura_figure_shows_wind_speed_direction_and_vector_error() -> None:
    wrapper = (ROOT / "scripts" / "make_fig_supp_laura_wind.py").read_text()
    renderer = (ROOT / "scripts" / "make_fig_supp_ciara_wind.py").read_text()
    assert "np.hypot" in renderer
    assert "axis.quiver" in renderer
    assert "_weighted_vector_rmse" in renderer
    assert 'EXPECTED_INIT_TIME = "2020-08-27T03:00:00"' in wrapper
    assert 'renderer.EVENT_NAME = "Hurricane Laura"' in wrapper
    assert "not selected by model error" in wrapper
    assert '"selection_note": SELECTION_NOTE' in renderer


def test_ida_figure_keeps_full_domain_with_geographic_context() -> None:
    source = (ROOT / "scripts" / "make_fig_case_ida_wind.py").read_text()
    assert "LON_TICKS = (240, 265, 290)" in source
    assert "LAT_TICKS = (25, 40, 55)" in source
    assert 'f"{360 - value}°W"' in source
    assert 'expected = base[key] + np.float32(0.125)' in source
    assert 'provenance.get("schema_version") != 3' in source


def test_case_exporter_declares_block_average_cell_centres() -> None:
    source = (ROOT / "tools" / "eval" / "export_weatherbridge_cases.py").read_text()
    assert '"schema_version": 3' in source
    assert '"wb2_0p25_pair_average_cell_centres_v1"' in source
    assert "lat=np.arange(54.875, 19.874, -0.5" in source
    assert "lon=np.arange(100.125, 145.126, 0.5" in source


def test_spatial_atlas_uses_declared_grid_and_core_models() -> None:
    source = (
        ROOT / "scripts" / "make_fig_rmse_maps_5tau_per_field.py"
    ).read_text()
    for model_name in (
        "Linear Interp.",
        "WeatherDCAE-14M",
        "WeatherBridge",
    ):
        assert model_name in source
    assert "PLOT_MODEL_NAMES" in source
    assert "89.875, -89.625" in source
    assert "0.125, 359.625" in source
    assert "figsize=(COLUMN_WIDTH_IN, 6.8 * COLUMN_WIDTH_IN / 10.0)" in source
    assert "ax.set_xticks([60, 180, 300])" in source
    assert "labelleft=ci == 0" in source
    assert "labelbottom=ri == n_rows - 1" in source
    assert "missing RMSE-map entries" in source
    assert "exp_weatherdcae_14m_6h_6yr_refinev1_s202707_bs4" in source
    assert "skip_ablation_noskip" not in source


def test_paper_uses_canonical_model_names() -> None:
    source = (ROOT / "paper" / "main.tex").read_text()
    assert "WeatherBridge-Detail" not in source
    assert "Flow-Spectral" not in source
    assert "Detail-bypass ablation" not in source
    assert "WeatherBridge" in source
    code_availability = source.split("\\bmhead{Code availability}", 1)[1]
    code_availability = code_availability.split(
        "\\bmhead{Acknowledgements}", 1
    )[0]
    assert "canonical WeatherBridge configuration" in code_availability
    assert "flow\\_pp3" not in code_availability


def test_production_manifest_uses_canonical_model_name() -> None:
    source = (
        ROOT / "production" / "weatherbridge_app" / "worker.py"
    ).read_text()

    assert '"model": "WeatherBridge"' in source
    assert "WeatherBridge-Detail" not in source


def test_paper_uses_published_dcae_reference() -> None:
    references = (ROOT / "paper" / "references.bib").read_text()
    entry = references.split("@inproceedings{cai2024dcae,", 1)[1].split("\n}", 1)[0]
    assert "author = {Chen, Junyu and Cai, Han" in entry
    assert "International Conference on Learning Representations" in entry
    assert "year = {2025}" in entry
    assert "https://openreview.net/forum?id=wH8XXUOUZU" in entry
    assert "arXiv preprint" not in entry


def test_paper_spectral_objective_matches_training_code() -> None:
    paper = (ROOT / "paper" / "main.tex").read_text()
    normalized_paper = " ".join(paper.split())
    trainer = (
        ROOT / "tools" / "train" / "train_capacity_matched_6h.py"
    ).read_text()
    assert "All 24 channels are scored" in paper
    assert (
        "FFT term is restricted to the declared advected-channel mask"
        in normalized_paper
    )
    assert "periodic-longitude and replicated-latitude padding" in normalized_paper
    assert '"flow_pp3"' in trainer
    assert "SPECTRAL_MASK_PROFILES" in trainer


def test_paper_separates_additive_and_transport_residual_gains() -> None:
    paper = " ".join((ROOT / "paper" / "main.tex").read_text().split())
    assert "scalar in WeatherDCAE and channelwise in PixelAttn-VFI" in paper
    assert "WeatherBridge uses the distinct transport-and-fusion path below" in paper
    assert "its residual gain is also channelwise" in paper


def test_reference_architecture_figure_matches_evaluated_baselines() -> None:
    figure = (
        ROOT / "paper" / "figs" / "fig_reference_architectures.tex"
    ).read_text()
    main = (ROOT / "paper" / "main.tex").read_text()
    supplement = (ROOT / "paper" / "supplementary.tex").read_text()

    assert "images/fig_reference_architectures.pdf" in supplement
    assert "8 blocks total" in figure
    assert "W-MSA" in figure and "SW-MSA" in figure
    assert "Cross-frame attention" in figure
    assert "mean endpoint features at each decoder scale" in figure
    assert "64 ch" in figure and "128 ch" in figure and "256 ch" in figure
    assert "no encoder skips" in figure
    assert figure.count("Exact linear scaffold") == 3
    assert "rstage,text width=24mm] (dcae-e0)" in figure
    assert "rstage,text width=27mm] (dcae-e1)" in figure
    assert "rstage,text width=29mm] (dcae-dec)" in figure
    assert "identity skip: add the unchanged input" in figure
    assert r"(dcae-lin.north) -- ++(0,0.30) -| (dcae-mix.south)" in figure
    assert r"(dcae-lin.east) -| (dcae-mix.south)" not in figure
    assert r"z\mapsto\gamma(t)\odot z+\beta(t)" in figure
    assert "Conv--SiLU, FiLM, Conv and RMSNorm act sequentially" in supplement
    for heading in (
        "SwinV2.",
        "PixelAttn-VFI.",
        "WeatherDCAE-14M.",
    ):
        assert heading in main
    assert "256-dimensional tokens" in main
    assert "72, 144, 288 and" in main and "576 channels" in main
    normalized_main = " ".join(main.split())
    assert "Eight latitude rows are" in normalized_main
    assert "no cross-scale encoder--decoder" in normalized_main
    assert "without a displacement field" in normalized_main
    assert r"\section{Supplementary Methods}" not in supplement
    assert r"\subsection{SwinV2 patch-token interpolator}" not in supplement


def test_main_and_supplement_have_the_same_authors() -> None:
    main = (ROOT / "paper" / "main.tex").read_text()
    supplement = (ROOT / "paper" / "supplementary.tex").read_text()
    authors = (
        r"\fnm{Daniil} \sur{Sukhorukov}",
        r"\fnm{Andrei} \sur{Zakharov}",
        r"\fnm{Yuri} \sur{Maximov}",
        r"\fnm{Ilya} \sur{Makarov}",
        r"\fnm{Ivan} \sur{Oseledets}",
    )
    for author in authors:
        assert main.count(author) == 1
        assert supplement.count(author) == 1


def test_funding_is_part_of_acknowledgements() -> None:
    source = (ROOT / "paper" / "main.tex").read_text()
    acknowledgements = source.split(r"\bmhead{Acknowledgements}", 1)[1]
    acknowledgements = acknowledgements.split(
        r"\bmhead{Author contributions}", 1
    )[0]
    assert "funding_statement.tex" in acknowledgements
    assert r"\bmhead{Funding}" not in source


def test_yuri_maximov_has_confirmed_affiliation_in_both_documents() -> None:
    for name in ("main.tex", "supplementary.tex"):
        source = (ROOT / "paper" / name).read_text()
        assert r"\author[3]{\fnm{Yuri} \sur{Maximov}}" in source
        assert r"\affil[3]{\orgname{International Center for Corporate Data Analysis}," in source
        assert r"\city{Astana}, \country{Kazakhstan}" in source


def test_paper_states_complete_hres_seed_gate() -> None:
    methods = " ".join(
        (ROOT / "paper" / "main.tex").read_text().split(
            r"\subsection{Reproducibility and statistical analysis}", 1
        )[1].split(r"\subsection{Use of generative AI}", 1)[0].split()
    )
    for metric in ("hard-window RMSE", "ACC", "spectral metric", "Temporal curvature"):
        assert metric in methods
    assert "paired IFS HRES improvement in at least two of three seeds" in methods
    assert "non-positive mean change" in methods
    assert "15 million parameters" in methods
    assert "1.25 times the fastest candidate" in methods


def test_paper_defines_selector_latency_protocol() -> None:
    paper = " ".join((ROOT / "paper" / "main.tex").read_text().split())
    assert "batch size one" in paper
    assert "seven CUDA-event repeats" in paper
    assert "20 bfloat16 forward passes after five warm-ups" in paper
    assert "$t\\in\\{0.25,0.50,0.75\\}$" in paper
    assert "random-input seed 2027" in paper
    assert "fastest model measured in the same process" in paper


def test_paper_states_actual_spectral_start_time_index() -> None:
    paper = " ".join((ROOT / "paper" / "main.tex").read_text().split())
    assert "at the 00 and 12 UTC anchor starts" in paper
    assert "00/12 UTC start-time index" in paper
    assert "all valid anchor windows" not in paper


def test_paper_discloses_normalization_provenance_limit() -> None:
    paper = " ".join((ROOT / "paper" / "main.tex").read_text().split())
    assert "the original generator was not retained" in paper
    assert "was not bit-exact" in paper
    assert "SHA-256 hashes as the canonical scale" in paper
    assert "1990--2019" in paper
    assert "a factor 1.3 on climatological standard deviations" in paper


def test_paper_introduces_architecture_and_selector_candidates_precisely() -> None:
    source = (ROOT / "paper" / "main.tex").read_text()
    supplement = (ROOT / "paper" / "supplementary.tex").read_text()
    introduction = source.split("\\begin{deferredmethodsintro}", 1)[0]
    assert "Fig.~\\ref{fig:tikz_flow}" in introduction
    assert "against WeatherDCAE-14M" in supplement
    assert "Each transport candidate is compared" not in source

    assert "\\subsection{Detail-bypass configuration audit}" not in source
    assert "Detail-bypass ablation" not in source


def test_paper_describes_internal_geometry_without_overclaiming() -> None:
    main = " ".join((ROOT / "paper" / "main.tex").read_text().split())
    supplement = " ".join(
        (ROOT / "paper" / "supplementary.tex").read_text().split()
    )
    assert "latitude--longitude convolutional network" in main
    assert "the spectral operation in the SFNO branch uses a spherical-harmonic basis" in main
    assert "output need not be strictly band-limited" in main
    assert "spherical-harmonic basis" in main
    assert (
        "pole-aware trunk, such as a cubed-sphere convolutional network "
        r"\cite{weyn2020dlwp}, would require matched retraining"
    ) in main
    primary = main.split(r"\paragraph{WeatherBridge.}", 1)[1].split(
        r"\paragraph{Retrained architecture variants.}", 1
    )[0]
    variants = main.split(r"\paragraph{Retrained architecture variants.}", 1)[1].split(
        r"\end{deferredmethodsintro}", 1
    )[0]
    assert "16-token area-weighted latent transformer" not in primary
    assert "16-token area-weighted latent transformer" in variants
    assert "Refine" in variants and "defined in Methods" in supplement
    assert "$4t(1-t)$" in variants
    assert r"4\tau(1-\tau)" not in supplement
    assert "16-token spherical latent transformer" not in main


def test_abstract_identifies_held_query_and_hres_scope() -> None:
    source = (ROOT / "paper" / "main.tex").read_text()
    abstract = source.split("\\abstract{", 1)[1].split("\n\n\\keywords", 1)[0]
    normalized = " ".join(abstract.split())
    assert "query hours omitted from training" in normalized
    assert "matched six-hour WeatherBridge control" in normalized
    assert "lowest aggregate error at held-out query hours" in normalized
    assert "retrospective audit" in normalized
    assert "no model wins every field" in normalized
    assert "IFS HRES forecasts" in normalized


def test_hres_2022_result_is_not_presented_as_independent_confirmation() -> None:
    manifest = json.loads(
        (ROOT / "repro" / "hres_full_lead_finetune_v1.json").read_text()
    )
    selection = manifest["selection"]
    assert 2022 in selection["opened_years_not_eligible_for_confirmation"]
    assert selection["next_confirmation_year"] == 2023

    main = " ".join((ROOT / "paper" / "main.tex").read_text().split())
    hres = " ".join(
        (ROOT / "paper" / "hres_finetuning_results.tex").read_text().split()
    )
    table = " ".join(
        (ROOT / "paper" / "tab_hres_confirmation_2022.tex").read_text().split()
    )
    assert "24-initialisation 2022 audit" in main
    assert "confirmation requires a later operational archive" in main
    assert "retrospective rather than independent confirmation evidence" in hres
    assert "descriptive rather than independent confirmation" in table


def test_paper_treats_6h_as_primary_and_12h_as_interval_ablation() -> None:
    main = (ROOT / "paper" / "main.tex").read_text()
    normalized_main = " ".join(main.split())
    supplement = (ROOT / "paper" / "supplementary.tex").read_text()
    abstract = " ".join(
        main.split("\\abstract{", 1)[1].split("\n\n\\keywords", 1)[0].split()
    )

    assert "forecast archives provide six-hourly fields" in abstract
    assert "six- and twelve-hour horizons" in abstract
    assert "\\subsection{Twelve-hour interval-length ablation}" in main
    assert "for six-hour anchor gaps" in normalized_main
    assert "images/fig_hres_fields_12h.pdf" not in main
    assert "images/fig_hres_fields_12h.pdf" not in supplement
    assert "images/fig_channels_body_12h_phys.pdf" not in main
    assert "images/fig_channels_body_6h_phys.pdf" not in supplement
    assert "images/fig_channels_app_6h_phys.pdf" not in supplement
    assert supplement.index(r"\label{fig:supp_channels_12h_body}") < supplement.index(
        r"\label{fig:supp_channels_12h}"
    )


def test_nwp_adapter_is_low_cost_and_restricted_to_primary_6h_task() -> None:
    main = " ".join((ROOT / "paper" / "main.tex").read_text().split())
    supplement = (ROOT / "paper" / "supplementary.tex").read_text()
    assert "Low-cost forecast-anchor adaptation" in main
    assert "We keep WeatherBridge fixed" in main
    assert "144 remain active" in main
    assert "reducing the number of network query evaluations by 70\\%" in main
    assert "adapter belongs to the primary 6 h task" in main
    assert "12 h interval-length ablation excludes it" in main
    assert "images/fig_nwp_adapter_2022_6h.pdf" not in main
    assert "images/fig_nwp_adapter_2022_6h.pdf" in supplement
    assert (
        "WeatherDCAE-14M, raw WeatherBridge and the frozen coefficient adapter"
        in " ".join(supplement.split())
    )
    assert "adapter-versus-Linear test excludes both post-hoc curves" in main
    assert "70 improve and two regress; the remaining 48 cells are exact guarded ties" in main
    assert "The two regressions are T700" in main
    assert "improves alignment of small-scale coefficients" in main
    assert "does not fully calibrate their amplitude" in main
    assert "images/fig_hres_fields_6h.pdf" not in main
    assert "images/fig_hres_fields_6h.pdf" not in supplement


def test_supplementary_nwp_spectral_table_matches_frozen_2022_artifacts() -> None:
    main = (ROOT / "paper" / "main.tex").read_text()
    supplement = (ROOT / "paper" / "supplementary.tex").read_text()
    table = (ROOT / "paper" / "tab_hres_spectral_2022.tex").read_text()
    assert "\\input{tab_hres_spectral_2022.tex}" in supplement
    assert "tab_hres_spectral_2022.tex" not in main
    metrics_dir = ROOT / "metrics" / "nwp_blend_spectra_6h_2022_v2_endpoint_guard"
    for tau in (2, 3):
        payload = json.loads(
            (metrics_dir / f"flow_adapted_vs_linear_tau{tau}.json").read_text()
        )
        comparison = payload["comparisons"]["linear"]
        for channel in ("Q850", "U850"):
            values = comparison["per_channel"][channel]
            cells = []
            for metric in (
                "energy_log_error",
                "shape_log_error",
                "coherence",
                "signed_cospectrum",
            ):
                record = values[metric]
                suffix = "\\dagger" if record["p_holm"] < 0.05 else ""
                cells.append(
                    f"${record['delta_left_minus_right']:+.4f}{suffix}$"
                )
            row = f"{tau} & {channel} & " + " & ".join(cells) + " \\\\"
            assert row in table


def test_paper_reports_actual_adjusted_p_values_for_rejections() -> None:
    source = (ROOT / "paper" / "main.tex").read_text()
    supplement = (ROOT / "paper" / "supplementary.tex").read_text()
    normalized = " ".join(source.split())
    normalized_supplement = " ".join(supplement.split())
    assert "two regress" in normalized
    assert "two field--hour regressions" in normalized_supplement
    assert "temperature (T2m) regressions at $\\tau=1$ and $\\tau=5$" in normalized_supplement
    assert "Holm-adjusted $p=0.024$ each" in normalized_supplement
    assert "Holm--Bonferroni-adjusted values" in normalized
    assert "together with effect directions" in normalized


def test_reproducibility_tables_are_in_supplementary_inventory() -> None:
    main = (ROOT / "paper" / "main.tex").read_text()
    supplement = (ROOT / "paper" / "supplementary.tex").read_text()
    assert "\\input{tab_training_protocol.tex}" not in main
    assert "\\input{tab_architecture_settings.tex}" not in main
    assert "\\input{tab_training_protocol.tex}" in supplement
    assert "\\input{tab_architecture_settings.tex}" in supplement
    assert "Training and architecture inventory" in supplement


def test_2021_supplementary_captions_report_physical_units() -> None:
    supplement = " ".join(
        (ROOT / "paper" / "supplementary.tex").read_text().split()
    )
    assert (
        "Six-hour per-channel RMSE in physical units in 2021 for representative"
        in supplement
    )
    assert (
        "Six-hour RMSE in physical units in 2021 for the remaining pressure-level"
        in supplement
    )
    assert "Six-hour per-channel normalised RMSE in 2021" not in supplement
    assert "Six-hour normalised RMSE in 2021" not in supplement


def test_haishen_exports_use_canonical_transport_variants() -> None:
    base_path = (
        ROOT
        / "demo"
        / "precomputed"
        / "typhoon_haishen"
        / "East_Asia__mslp.npz"
    )
    with np.load(base_path, allow_pickle=True) as base:
        truth = base["panels"][list(base["methods"]).index("ERA5")]

    cases = (
        ("case_studies_weatherdcae_14m", "WeatherDCAE-14M", "dcae_14m"),
        ("case_studies_pixelattn_vfi", "PixelAttn-VFI", "atmvfi"),
        ("case_studies_weatherbridge", "WeatherBridge", "flow_pp3"),
    )
    for directory, label, arch in cases:
        path = ROOT / "metrics" / directory / "typhoon_haishen" / "mslp.npz"
        with np.load(path, allow_pickle=False) as case:
            assert case["methods"].tolist() == [label]
            assert case["model_arch"].item() == arch
            assert case["init_time"].item() == "2020-09-07T00:00:00"
            np.testing.assert_allclose(
                case["truth"],
                truth,
                rtol=0.0,
                atol=1e-4,
            )
    generator = (ROOT / "scripts" / "make_fig5_haishen_local.py").read_text()
    assert '"WeatherBridge": ROOT / "metrics" / "case_studies_weatherbridge"' in generator
    assert '"WeatherDCAE-14M": ROOT / "metrics" / "case_studies_weatherdcae_14m"' in generator
    assert '"PixelAttn-VFI": ROOT / "metrics" / "case_studies_pixelattn_vfi"' in generator


def test_laura_wind_exports_share_truth_linear_baseline_and_grid() -> None:
    cases = (
        ("case_studies_weatherdcae_14m", "WeatherDCAE-14M", "dcae_14m"),
        ("case_studies_pixelattn_vfi", "PixelAttn-VFI", "atmvfi"),
        ("case_studies_weatherbridge", "WeatherBridge", "flow_pp3"),
    )
    reference = None
    for directory, label, arch in cases:
        path = ROOT / "metrics" / directory / "hurricane_laura_2020" / "wind10.npz"
        with np.load(path, allow_pickle=False) as case:
            assert case["methods"].tolist() == [label]
            assert case["model_arch"].item() == arch
            assert case["init_time"].item() == "2020-08-27T03:00:00"
            provenance = json.loads(case["provenance_json"].item())
            assert provenance["event"] == "Hurricane Laura"
            shared = {
                key: np.array(case[key], copy=True)
                for key in ("taus", "lat", "lon", "truth_u", "truth_v", "linear_u", "linear_v")
            }
            if reference is None:
                reference = shared
            else:
                for key, value in shared.items():
                    np.testing.assert_allclose(value, reference[key], rtol=0.0, atol=1e-6)
