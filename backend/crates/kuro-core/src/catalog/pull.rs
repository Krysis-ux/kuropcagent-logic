//! Turning "I want this model" into weights on disk.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::db::{Db, DownloadStatus, ModelRecord, ModelSource, NewModel};
use crate::paths::Paths;
use crate::{KuroError, Result};

use super::curated::find_curated;
use super::download::download_to_file;
use super::hf::{
    choose_gguf, choose_gguf_exact, list_gguf_files_at_revision, parse_model_ref,
    quant_from_filename, resolve_pinned_download_url, resolve_repo_revision, GgufFile, ModelRef,
};

/// Everything needed to fetch one model, resolved against the source.
#[derive(Debug, Clone)]
pub struct PullPlan {
    pub model_id: String,
    pub display_name: String,
    pub source: ModelSource,
    pub repo: String,
    /// Immutable Hugging Face commit used for both metadata and bytes.
    pub revision: String,
    pub filename: String,
    pub url: String,
    pub size_bytes: u64,
    /// Published by Hugging Face for LFS files; used to verify the download.
    pub sha256: Option<String>,
    pub quant: Option<String>,
    pub param_count: Option<String>,
    pub family: Option<String>,
    pub capabilities: Vec<String>,
    pub context_length: Option<i64>,
}

/// Exact repository selection supplied by the model-detail flow.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ExactModelSelection {
    pub repo: String,
    pub revision: String,
    pub filename: String,
}

/// Resolve a user-supplied reference into a concrete download.
pub async fn plan_pull(client: &reqwest::Client, reference: &str) -> Result<PullPlan> {
    let parsed = parse_model_ref(reference)?;

    let (repo, requested_quant, curated) = match &parsed {
        ModelRef::Curated { slug, quant } => {
            let curated = find_curated(slug).ok_or_else(|| {
                KuroError::not_found(format!(
                    "`{slug}` is not in the recommended list. Pass a Hugging Face repository \
                     like `owner/repo` to pull anything else."
                ))
            })?;
            let quant = quant
                .clone()
                .unwrap_or_else(|| curated.default_quant.to_string());
            (curated.hf_repo.to_string(), Some(quant), Some(curated))
        }
        ModelRef::HuggingFace { repo, quant } => (repo.clone(), quant.clone(), None),
    };

    let revision = resolve_repo_revision(client, &repo).await?;
    let files = list_gguf_files_at_revision(client, &repo, &revision).await?;
    let chosen = choose_gguf(&files, requested_quant.as_deref())?;

    let quant = quant_from_filename(&chosen.filename).or(requested_quant);

    let (model_id, display_name, source) = match curated {
        Some(curated) => (
            curated.model_id(quant.as_deref().unwrap_or(curated.default_quant)),
            curated.display_name.to_string(),
            ModelSource::Curated,
        ),
        None => {
            let base = repo_slug(&repo);
            let id = match &quant {
                Some(quant) => format!("{base}:{}", quant.to_ascii_lowercase()),
                None => base.clone(),
            };
            (id, repo.clone(), ModelSource::Huggingface)
        }
    };

    Ok(PullPlan {
        model_id,
        display_name,
        source,
        repo: repo.clone(),
        revision: revision.clone(),
        url: resolve_pinned_download_url(&repo, &revision, &chosen.filename)?,
        filename: chosen.filename,
        size_bytes: chosen.size,
        sha256: chosen.sha256,
        quant,
        param_count: curated.map(|c| c.param_count.to_string()),
        family: curated.map(|c| c.family.to_string()),
        capabilities: curated
            .map(|c| c.capabilities.iter().map(|s| s.to_string()).collect())
            .unwrap_or_default(),
        context_length: curated.map(|c| i64::from(c.context_length)),
    })
}

/// Resolve one exact file from an immutable repository tree.
pub async fn plan_pull_exact(
    client: &reqwest::Client,
    selection: &ExactModelSelection,
) -> Result<PullPlan> {
    let files = list_gguf_files_at_revision(client, &selection.repo, &selection.revision).await?;
    plan_pull_exact_from_files(selection, &files)
}

fn plan_pull_exact_from_files(
    selection: &ExactModelSelection,
    files: &[GgufFile],
) -> Result<PullPlan> {
    let chosen = choose_gguf_exact(files, &selection.filename)?;
    let quant = quant_from_filename(&chosen.filename);
    let model_id = exact_model_id(&selection.repo, &chosen.filename, quant.as_deref());

    Ok(PullPlan {
        model_id,
        display_name: selection.repo.clone(),
        source: ModelSource::Huggingface,
        repo: selection.repo.clone(),
        revision: selection.revision.clone(),
        filename: chosen.filename.clone(),
        url: resolve_pinned_download_url(&selection.repo, &selection.revision, &chosen.filename)?,
        size_bytes: chosen.size,
        sha256: chosen.sha256,
        quant,
        param_count: None,
        family: None,
        capabilities: Vec::new(),
        context_length: None,
    })
}

/// Register the model and its download row without transferring anything yet.
pub fn prepare_pull(db: &Db, paths: &Paths, plan: &PullPlan) -> Result<PreparedPull> {
    // A resumed partial file belongs to one immutable commit. Reusing a path
    // across revisions could append new bytes to an older revision's prefix.
    let destination = paths.model_file(
        &plan.model_id,
        &format!("{}-{}", plan.revision, plan.filename),
    );

    let (record, previous_model) = db.prepare_model_download(
        &NewModel {
            id: plan.model_id.clone(),
            display_name: plan.display_name.clone(),
            source: plan.source,
            hf_repo: Some(plan.repo.clone()),
            hf_file: Some(plan.filename.clone()),
            quant: plan.quant.clone(),
            param_count: plan.param_count.clone(),
            family: plan.family.clone(),
            capabilities: plan.capabilities.clone(),
            context_length: plan.context_length,
            file_size_bytes: Some(plan.size_bytes as i64),
        },
        &plan.url,
        &destination.to_string_lossy(),
    )?;

    Ok(PreparedPull {
        download_id: record.id,
        destination: destination.to_string_lossy().to_string(),
        previous_model,
    })
}

#[derive(Debug, Clone)]
pub struct PreparedPull {
    pub download_id: String,
    pub destination: String,
    previous_model: Option<ModelRecord>,
}

/// Transfer the weights for an already-prepared pull.
pub async fn run_pull(
    client: &reqwest::Client,
    db: &Db,
    plan: &PullPlan,
    prepared: &PreparedPull,
    cancel: Arc<AtomicBool>,
) -> Result<()> {
    let destination = std::path::Path::new(&prepared.destination);

    let progress_db = db.clone();
    let progress_id = prepared.download_id.clone();
    let mut on_progress = move |downloaded: u64, total: Option<u64>| {
        let _ = progress_db.update_download_progress(
            &progress_id,
            downloaded as i64,
            total.map(|value| value as i64),
        );
    };

    let outcome = download_to_file(
        client,
        &plan.url,
        destination,
        plan.sha256.as_deref(),
        cancel.clone(),
        &mut on_progress,
    )
    .await;

    match outcome {
        Ok(outcome) => {
            db.complete_model_download(
                &prepared.download_id,
                &prepared.destination,
                outcome.bytes as i64,
                &outcome.sha256,
            )?;
            Ok(())
        }
        Err(error) if cancel.load(Ordering::Relaxed) || download_is_cancelled(db, prepared)? => {
            finish_cancelled_pull(db, plan, prepared)?;
            Err(error)
        }
        Err(error) => {
            let message = error.to_string();
            db.fail_model_download(&prepared.download_id, &message)?;
            Err(error)
        }
    }
}

fn download_is_cancelled(db: &Db, prepared: &PreparedPull) -> Result<bool> {
    Ok(db
        .get_download(&prepared.download_id)?
        .is_some_and(|record| record.status == DownloadStatus::Cancelled))
}

fn finish_cancelled_pull(db: &Db, _plan: &PullPlan, prepared: &PreparedPull) -> Result<()> {
    db.cancel_model_download(&prepared.download_id, prepared.previous_model.as_ref())
}

/// Prepare and run a pull in one call, for callers that block until it is done.
pub async fn execute_pull(
    client: &reqwest::Client,
    db: &Db,
    paths: &Paths,
    plan: &PullPlan,
    cancel: Arc<AtomicBool>,
) -> Result<()> {
    let prepared = prepare_pull(db, paths, plan)?;
    run_pull(client, db, plan, &prepared, cancel).await
}

/// Remove a model according to who owns its weights.
pub fn remove_model(db: &Db, paths: &Paths, model_id: &str) -> Result<()> {
    let model = db
        .get_model(model_id)?
        .ok_or_else(|| KuroError::not_found(format!("model `{model_id}`")))?;

    if model.source == ModelSource::Local {
        db.unregister_model(model_id)?;
        return Ok(());
    }

    if let Some(path) = &model.file_path {
        let _ = std::fs::remove_file(path);
    }

    // Also drop the per-model directory, which holds any leftover `.part` file
    // from an interrupted download.
    let model_dir = paths.model_file(model_id, "x");
    if let Some(parent) = model_dir.parent() {
        if parent.starts_with(paths.models_dir()) {
            let _ = std::fs::remove_dir_all(parent);
        }
    }

    db.delete_model(model_id)?;
    Ok(())
}

/// Last path segment of a repository, normalised for use as a model id.
fn repo_slug(repo: &str) -> String {
    let name = repo.rsplit('/').next().unwrap_or(repo).to_ascii_lowercase();
    // Nearly every GGUF repository ends in `-gguf`, which adds nothing to the id.
    name.strip_suffix("-gguf").unwrap_or(&name).to_string()
}

/// Collision-resistant id for the exact flow. The revision is deliberately
/// excluded so updating the same logical repository path reuses its row.
pub fn exact_model_id(repo: &str, filename: &str, quant: Option<&str>) -> String {
    let mut hasher = Sha256::new();
    hasher.update(repo.as_bytes());
    hasher.update([0]);
    hasher.update(filename.as_bytes());
    let digest = format!("{:x}", hasher.finalize());
    let quant = quant.unwrap_or("gguf").to_ascii_lowercase();
    format!("{}:{quant}:{}", repo_slug(repo), &digest[..8])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(model_id: &str, repo: &str, filename: &str, url: &str) -> PullPlan {
        PullPlan {
            model_id: model_id.to_string(),
            display_name: repo.to_string(),
            source: ModelSource::Huggingface,
            repo: repo.to_string(),
            revision: "0123456789abcdef0123456789abcdef01234567".to_string(),
            filename: filename.to_string(),
            url: url.to_string(),
            size_bytes: 4,
            sha256: None,
            quant: Some("Q4_K_M".to_string()),
            param_count: None,
            family: None,
            capabilities: Vec::new(),
            context_length: None,
        }
    }

    #[test]
    fn derives_a_readable_id_from_a_repository() {
        assert_eq!(
            repo_slug("unsloth/Qwen3-4B-Instruct-2507-GGUF"),
            "qwen3-4b-instruct-2507"
        );
        assert_eq!(repo_slug("bartowski/Phi-4-GGUF"), "phi-4");
        assert_eq!(repo_slug("owner/PlainModel"), "plainmodel");
    }

    #[test]
    fn removing_an_unknown_model_is_an_error_not_a_silent_success() {
        let db = Db::open_in_memory().expect("db");
        let paths =
            Paths::for_root(std::env::temp_dir().join(format!("kuro-rm-{}", uuid::Uuid::new_v4())));
        let error = remove_model(&db, &paths, "nope").unwrap_err();
        assert!(matches!(error, KuroError::NotFound(_)));
    }

    #[test]
    fn different_owners_same_basename_quant_do_not_collide() {
        let alice = exact_model_id("alice/Shared-GGUF", "Shared-Q4_K_M.gguf", Some("Q4_K_M"));
        let bob = exact_model_id("bob/Shared-GGUF", "Shared-Q4_K_M.gguf", Some("Q4_K_M"));
        let alternate = exact_model_id(
            "alice/Shared-GGUF",
            "alternate/Shared-Q4_K_M.gguf",
            Some("Q4_K_M"),
        );

        assert_ne!(alice, bob);
        assert_ne!(alice, alternate);
        assert!(alice.starts_with("shared:q4_k_m:"), "got {alice}");
        assert_eq!(
            alice,
            exact_model_id("alice/Shared-GGUF", "Shared-Q4_K_M.gguf", Some("Q4_K_M"))
        );
    }

    #[test]
    fn exact_plan_uses_pinned_revision_and_exact_filename() {
        let selection = ExactModelSelection {
            repo: "owner/Model-GGUF".to_string(),
            revision: "0123456789abcdef0123456789abcdef01234567".to_string(),
            filename: "nested/Model-Q4_K_M-v2.gguf".to_string(),
        };
        let files = vec![
            super::super::hf::GgufFile {
                filename: "Model-Q4_K_M-v1.gguf".to_string(),
                size: 100,
                sha256: None,
            },
            super::super::hf::GgufFile {
                filename: selection.filename.clone(),
                size: 200,
                sha256: Some(
                    "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789".to_string(),
                ),
            },
        ];

        let planned = plan_pull_exact_from_files(&selection, &files).expect("exact plan");

        assert_eq!(planned.filename, selection.filename);
        assert_eq!(planned.revision, selection.revision);
        assert_eq!(planned.size_bytes, 200);
        assert_eq!(
            planned.sha256.as_deref(),
            Some("abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789")
        );
        assert!(planned
            .url
            .contains("/resolve/0123456789abcdef0123456789abcdef01234567/"));
        assert!(!planned.url.contains("/main/"));
    }

    #[test]
    fn target_collision_is_rejected_without_retargeting_existing_model() {
        let db = Db::open_in_memory().expect("db");
        let paths = Paths::for_root(
            std::env::temp_dir().join(format!("kuro-collision-{}", uuid::Uuid::new_v4())),
        );
        let first = plan(
            "shared:q4_k_m:deadbeef",
            "alice/Shared-GGUF",
            "Shared-Q4_K_M.gguf",
            "https://example.invalid/alice.gguf",
        );
        let conflicting = plan(
            "shared:q4_k_m:deadbeef",
            "bob/Shared-GGUF",
            "Shared-Q4_K_M.gguf",
            "https://example.invalid/bob.gguf",
        );
        let prepared = prepare_pull(&db, &paths, &first).expect("first registration");
        db.set_download_status(&prepared.download_id, DownloadStatus::Completed, None)
            .expect("finish fixture download");

        let error = prepare_pull(&db, &paths, &conflicting).expect_err("collision");
        let stored = db
            .get_model("shared:q4_k_m:deadbeef")
            .expect("lookup")
            .expect("existing model");

        assert!(
            error.to_string().contains("target collision"),
            "got: {error}"
        );
        assert_eq!(stored.hf_repo.as_deref(), Some("alice/Shared-GGUF"));
        assert_eq!(stored.hf_file.as_deref(), Some("Shared-Q4_K_M.gguf"));
    }

    #[test]
    fn failed_download_registration_does_not_leave_an_orphan_model() {
        let db = Db::open_in_memory().unwrap();
        db.with(|conn| {
            conn.execute_batch("CREATE TRIGGER reject_download BEFORE INSERT ON downloads BEGIN SELECT RAISE(ABORT, 'disk failure fixture'); END;")?;
            Ok(())
        }).unwrap();
        let paths = Paths::for_root(std::env::temp_dir().join("kuro-atomic-prepare"));
        let pull = plan(
            "atomic",
            "owner/Model",
            "model.gguf",
            "https://example.invalid/model.gguf",
        );
        assert!(prepare_pull(&db, &paths, &pull).is_err());
        assert!(db.get_model("atomic").unwrap().is_none());
    }

    #[test]
    fn a_late_cancel_finalizer_cannot_delete_a_retry_registration() {
        let db = Db::open_in_memory().unwrap();
        let paths = Paths::for_root(std::env::temp_dir().join("kuro-cancel-owner"));
        let pull = plan(
            "retry",
            "owner/Model",
            "model.gguf",
            "https://example.invalid/model.gguf",
        );
        let old = prepare_pull(&db, &paths, &pull).unwrap();
        finish_cancelled_pull(&db, &pull, &old).unwrap();
        let retry = prepare_pull(&db, &paths, &pull).unwrap();
        finish_cancelled_pull(&db, &pull, &old).unwrap();
        assert!(db.get_model("retry").unwrap().is_some());
        assert_eq!(
            db.active_download_for_target("retry").unwrap().unwrap().id,
            retry.download_id
        );
    }

    #[test]
    fn concurrent_prepare_accepts_only_one_transfer_for_the_same_target() {
        let db = Db::open_in_memory().unwrap();
        let paths = Paths::for_root(std::env::temp_dir().join("kuro-concurrent-prepare"));
        let pull = plan(
            "concurrent",
            "owner/Model",
            "model.gguf",
            "https://example.invalid/model.gguf",
        );
        let barrier = Arc::new(std::sync::Barrier::new(16));
        let accepted = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..16)
                .map(|_| {
                    scope.spawn(|| {
                        barrier.wait();
                        prepare_pull(&db, &paths, &pull).is_ok()
                    })
                })
                .collect();
            workers
                .into_iter()
                .map(|worker| worker.join().unwrap())
                .filter(|accepted| *accepted)
                .count()
        });
        assert_eq!(accepted, 1);
        assert_eq!(db.list_downloads().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn cancelled_pull_stays_cancelled_and_can_be_retried() {
        use std::sync::atomic::Ordering;

        let db = Db::open_in_memory().expect("db");
        let paths = Paths::for_root(
            std::env::temp_dir().join(format!("kuro-cancel-{}", uuid::Uuid::new_v4())),
        );
        let pull = plan(
            "model:q4_k_m:12345678",
            "owner/Model-GGUF",
            "Model-Q4_K_M.gguf",
            "https://example.invalid/model.gguf",
        );
        let prepared = prepare_pull(&db, &paths, &pull).expect("prepare");
        let cancel = Arc::new(AtomicBool::new(true));
        assert!(cancel.load(Ordering::Relaxed));
        assert!(
            prepare_pull(&db, &paths, &pull).is_err(),
            "cancellation must keep its target until cleanup"
        );

        let error = run_pull(&reqwest::Client::new(), &db, &pull, &prepared, cancel)
            .await
            .expect_err("cancelled transfer");

        assert!(error.to_string().contains("cancelled"), "got: {error}");
        let download = db
            .get_download(&prepared.download_id)
            .expect("download lookup")
            .expect("download row");
        assert_eq!(download.status, DownloadStatus::Cancelled);
        assert!(download.error.is_none());
        assert!(db
            .get_model(&pull.model_id)
            .expect("model lookup")
            .is_none());
        prepare_pull(&db, &paths, &pull).expect("cancelled pull must be retryable");
    }

    #[tokio::test]
    async fn cancelling_an_update_preserves_the_previous_ready_file_and_separates_revisions() {
        let root =
            std::env::temp_dir().join(format!("kuro-update-cancel-{}", uuid::Uuid::new_v4()));
        let db = Db::open_in_memory().expect("db");
        let paths = Paths::for_root(root.clone());
        let first = plan(
            "same:q4_k_m:12345678",
            "owner/Model",
            "model.gguf",
            "https://example.invalid/model.gguf",
        );
        let prepared = prepare_pull(&db, &paths, &first).expect("first prepare");
        std::fs::create_dir_all(
            std::path::Path::new(&prepared.destination)
                .parent()
                .unwrap(),
        )
        .unwrap();
        std::fs::write(&prepared.destination, b"working model").unwrap();
        db.set_model_ready(&first.model_id, &prepared.destination, 13, "old-hash")
            .unwrap();
        db.set_download_status(&prepared.download_id, DownloadStatus::Completed, None)
            .unwrap();
        let mut next = first.clone();
        next.revision = "abcdef0123456789abcdef0123456789abcdef01".into();
        let update = prepare_pull(&db, &paths, &next).unwrap();
        assert_ne!(
            prepared.destination, update.destination,
            "partials must not cross revisions"
        );
        run_pull(
            &reqwest::Client::new(),
            &db,
            &next,
            &update,
            Arc::new(AtomicBool::new(true)),
        )
        .await
        .expect_err("cancelled");
        let model = db.get_model(&first.model_id).unwrap().unwrap();
        assert_eq!(model.status, crate::db::ModelStatus::Ready);
        assert_eq!(
            model.file_path.as_deref(),
            Some(prepared.destination.as_str())
        );
        assert_eq!(
            std::fs::read(&prepared.destination).unwrap(),
            b"working model"
        );
        std::fs::remove_dir_all(root).unwrap();
    }
}
