#!/usr/bin/env bash

ARCHITECTURE_BASE_TRAINER="legacy/training_snapshots/train_capacity_matched_6h_e8d3bd58.py"
ARCHITECTURE_BASE_TRAINER_SHA256="e8d3bd5830ff7cc5a15dca986e94570c7e1b6a3707b8b7e3d5709dd17dcd6b6a"
ARCHITECTURE_QUERY_TRAINER="legacy/training_snapshots/train_capacity_matched_6h_32f4ce54.py"
ARCHITECTURE_QUERY_TRAINER_SHA256="32f4ce54ddd3d033ea4ea55112d6e65658caf73961e568680aa627e8794e1f30"
ARCHITECTURE_LOCAL_CORR_TRAINER="legacy/training_snapshots/train_capacity_matched_6h_d7a6bb0d.py"
ARCHITECTURE_LOCAL_CORR_TRAINER_SHA256="d7a6bb0d0b801eadb3365bf7b84803c6f499f7ff7969dc5d8a8de9895353e733"
ARCHITECTURE_AMT_RESIDUAL_TRAINER="legacy/training_snapshots/train_capacity_matched_6h_62c9566f.py"
ARCHITECTURE_AMT_RESIDUAL_TRAINER_SHA256="62c9566ff7b69db0770afc510ade626b6155b30b5a5b48b26e23138e0694b140"

architecture_candidate_model_arch() {
  case "$1" in
    flow_pp3_hf)
      echo flow_pp3
      ;;
    upr_implicit_global_14m_nohf)
      echo upr_implicit_global_14m
      ;;
    *)
      echo "$1"
      ;;
  esac
}

architecture_candidate_layout() {
  case "$1" in
    upr_implicit_global_14m|upr_implicit_global_14m_nohf|\
    upr_endpoint_implicit_global_14m|upr_spherical_implicit_global_14m|\
    upr_query_match_14m|upr_local_corr_14m|flow_spherical_ep|\
    flow_pp3_hf|amt|amt_residual)
      echo eb16
      ;;
    *)
      echo b8
      ;;
  esac
}

architecture_candidate_batch_size() {
  if [[ "$(architecture_candidate_layout "$1")" == "eb16" ]]; then
    echo 4
  else
    echo 8
  fi
}

architecture_candidate_val_batch_size() {
  if [[ "$(architecture_candidate_layout "$1")" == "eb16" ]]; then
    echo 2
  else
    echo 4
  fi
}

architecture_candidate_accumulate() {
  if [[ "$(architecture_candidate_layout "$1")" == "eb16" ]]; then
    echo 4
  else
    echo 2
  fi
}

architecture_candidate_lambda_hf() {
  case "$1" in
    upr_lite|upr_implicit_global_14m_nohf)
      echo 0
      ;;
    *)
      echo 0.05
      ;;
  esac
}

architecture_candidate_highpass_boundary() {
  case "$1" in
    upr_spherical_implicit_global_14m|flow_spherical_ep|amt|amt_residual)
      echo antipodal_vector_parity
      ;;
    *)
      echo periodic_lon_replicate_lat
      ;;
  esac
}

architecture_candidate_trainer() {
  case "$1" in
    upr_query_match_14m)
      echo "$ARCHITECTURE_QUERY_TRAINER"
      ;;
    upr_local_corr_14m)
      echo "$ARCHITECTURE_LOCAL_CORR_TRAINER"
      ;;
    amt_residual)
      echo "$ARCHITECTURE_AMT_RESIDUAL_TRAINER"
      ;;
    *)
      echo "$ARCHITECTURE_BASE_TRAINER"
      ;;
  esac
}

architecture_candidate_trainer_sha256() {
  case "$1" in
    upr_query_match_14m)
      echo "$ARCHITECTURE_QUERY_TRAINER_SHA256"
      ;;
    upr_local_corr_14m)
      echo "$ARCHITECTURE_LOCAL_CORR_TRAINER_SHA256"
      ;;
    amt_residual)
      echo "$ARCHITECTURE_AMT_RESIDUAL_TRAINER_SHA256"
      ;;
    *)
      echo "$ARCHITECTURE_BASE_TRAINER_SHA256"
      ;;
  esac
}
