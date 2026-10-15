//! Discovering and fetching model weights.

pub mod curated;
pub mod download;
pub mod hf;
pub mod local;
pub mod pull;
pub mod repository;
pub mod search;

pub use curated::{find_curated, CuratedModel, CURATED_MODELS};
pub use download::{download_to_file, sha256_file, DownloadOutcome};
pub use hf::{
    choose_gguf, choose_gguf_exact, list_gguf_files, list_gguf_files_at_revision, parse_model_ref,
    quant_from_filename, resolve_download_url, resolve_pinned_download_url, resolve_repo_revision,
    GgufFile, ModelRef,
};
pub use local::{
    infer_filename_metadata, local_model_id, scan_model_folder, validate_model_folder_root,
    FolderScan, FolderScanError, LocalModelCandidate, ScanCounts, MAX_REPORTED_ERRORS, MAX_RESULTS,
    MAX_SCAN_DEPTH, MAX_VISITED_DIRECTORIES, MAX_VISITED_ENTRIES,
};
pub use pull::{
    exact_model_id, execute_pull, plan_pull, plan_pull_exact, prepare_pull, remove_model, run_pull,
    ExactModelSelection, PreparedPull, PullPlan,
};
