# Training Source Snapshots

This directory preserves byte-exact trainer sources referenced by paper
checkpoints after the live trainer advances. The filename suffix is the first
eight characters of the SHA-256 stored in
`hyper_parameters.training_code_sha256`.

- `train_capacity_matched_6h_33ddc981.py` is the source used by the
  seed-202707 UPR-Implicit-Global-14M and matched Flow ablation runs started on
  2026-07-26. Its full SHA-256 is
  `33ddc981230a9aaf93329a3055d932fb9a19e346cd1c586215dabefd16725fea`.
- `weatherbridge_flow_model_83f2fa23.py` is the Flow/PP3 model source used by
  the matched PP3-12h run started on 2026-07-27. Its full SHA-256 is
  `83f2fa236bc9b715afbede16c9224560243528d44b54b0987007cf6db32007c5`.
- `train_capacity_matched_6h_e8d3bd58.py` freezes the trainer for fresh
  three-seed post-selection confirmation. Its full SHA-256 is
  `e8d3bd5830ff7cc5a15dca986e94570c7e1b6a3707b8b7e3d5709dd17dcd6b6a`.
- `train_capacity_matched_6h_38ca4e17.py` freezes the two-width
  Compact-VP3 short-budget screen. Its full SHA-256 is
  `38ca4e176f84b217a641598f2a2d4cb7c4554ab4ae1d36f53630a8f9fc112ff0`.
- `train_capacity_matched_6h_08c8a559.py` freezes the under-10M widened
  Compact-VP3 and endpoint-Hermite screen. Its full SHA-256 is
  `08c8a559f84b363245c26607fe1a5047d37bad9897c4b040441acc5088bdce87`.
- `train_capacity_matched_6h_08e77586.py` freezes the field-balanced
  Compact-Hermite fine-tuning screen. Its full SHA-256 is
  `08e77586ae33e16c79582832561421da093e6b22a1eaae34d059d661f3f1b76b`.
- `train_capacity_matched_6h_cf4a8519.py` freezes the non-moisture
  Lagrange-knot residual expert. Its full SHA-256 is
  `cf4a85190bac7bcdf14e5b8d49de49218645884fe6316f77e2cc73b830c1ae82`.
- `train_capacity_matched_6h_7c4c9279.py` freezes head-only Lagrange
  adaptation from the four-epoch Hermite checkpoint. Its full SHA-256 is
  `7c4c9279f5adf323cd9ee2f5bc40605a7112d273584f1b08f5dfd8160829706b`.

New architecture development uses `tools/train/train_capacity_matched_6h.py`.
Hash-bound launchers may execute a snapshot directly so resumed jobs retain
the exact source recorded in their checkpoints.
