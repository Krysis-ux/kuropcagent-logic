use std::path::{Path as FsPath, PathBuf};

use axum::extract::{Path, Query, State};
use axum::Json;
use kuro_core::catalog::{
    self, search, FolderScan, FolderScanError, ScanCounts, CURATED_MODELS, MAX_REPORTED_ERRORS,
};
use kuro_core::free;
use kuro_core::hardware::estimate_fit;
use kuro_core::settings;
use kuro_core::{KuroError, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::error::AppResult;
use crate::state::SharedState;

#[derive(Debug, Deserialize)]
pub struct SearchQuery {
    #[serde(default)]
    pub q: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ModelFolder {
    pub path: String,
    pub available: bool,
}

#[derive(Debug, Deserialize)]
pub struct ModelFolderRequest {
    pub path: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct FolderMutationReport {
    pub folders: Vec<ModelFolder>,
    pub counts: ScanCounts,
    pub errors: Vec<FolderScanError>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FolderRescanReport {
    pub folders: Vec<FolderMutationReport>,
}

/// Selected external model roots in their persisted order.
pub async fn list_model_folders(State(state): State<SharedState>) -> AppResult<Json<Value>> {
    Ok(Json(json!({ "folders": selected_model_folders(&state)? })))
}

/// Validate, scan, and atomically select one external model root.
pub async fn add_model_folder(
    State(state): State<SharedState>,
    Json(request): Json<ModelFolderRequest>,
) -> AppResult<Json<FolderMutationReport>> {
    let scan = scan_in_background(PathBuf::from(request.path)).await?;
    let errors = scan.errors.clone();
    let counts = state.db.commit_model_folder_add(&scan)?;
    Ok(Json(FolderMutationReport {
        folders: selected_model_folders(&state)?,
        counts,
        errors,
    }))
}

/// Re-scan every selected root without treating an unavailable or partial
/// observation as proof that model files disappeared.
pub async fn rescan_model_folders(
    State(state): State<SharedState>,
) -> AppResult<Json<FolderRescanReport>> {
    let tickets = state.db.model_folder_scan_tickets()?;
    let mut reports = Vec::with_capacity(tickets.len());

    for ticket in tickets {
        let path = ticket.path().to_path_buf();
        let path_text = path.to_string_lossy().into_owned();
        match scan_in_background(path).await {
            Ok(scan) => {
                let available = scan.root_is_bound();
                let mut counts = scan.counts.clone();
                let mut errors = scan.errors.clone();

                if scan.is_complete() {
                    match state.db.reconcile_scanned_folder(&ticket, &scan) {
                        Ok(committed) => counts = committed,
                        Err(error) => push_report_error(
                            &mut errors,
                            &mut counts,
                            &path_text,
                            error.to_string(),
                        ),
                    }
                }

                reports.push(FolderMutationReport {
                    folders: vec![ModelFolder {
                        path: path_text,
                        available,
                    }],
                    counts,
                    errors,
                });
            }
            Err(error) => reports.push(FolderMutationReport {
                folders: vec![ModelFolder {
                    path: path_text.clone(),
                    available: false,
                }],
                counts: ScanCounts::default(),
                errors: vec![FolderScanError {
                    path: path_text,
                    message: error.to_string(),
                }],
            }),
        }
    }

    Ok(Json(FolderRescanReport { folders: reports }))
}

/// Stop discovering one exact selected root and unregister only rows it owns.
/// External weight files are never removed.
pub async fn remove_model_folder(
    State(state): State<SharedState>,
    Json(request): Json<ModelFolderRequest>,
) -> AppResult<Json<FolderMutationReport>> {
    let counts = state.db.commit_model_folder_remove(&request.path)?;
    Ok(Json(FolderMutationReport {
        folders: selected_model_folders(&state)?,
        counts,
        errors: Vec::new(),
    }))
}

fn selected_model_folders(state: &SharedState) -> Result<Vec<ModelFolder>> {
    settings::model_folders(&state.db).map(|folders| {
        folders
            .into_iter()
            .map(|path| ModelFolder {
                available: model_folder_available(FsPath::new(&path)),
                path,
            })
            .collect()
    })
}

fn model_folder_available(path: &FsPath) -> bool {
    catalog::validate_model_folder_root(path, &[]).is_ok()
}

async fn scan_in_background(path: PathBuf) -> Result<FolderScan> {
    tokio::task::spawn_blocking(move || catalog::scan_model_folder(&path))
        .await
        .map_err(|error| KuroError::other(format!("model folder scan task failed: {error}")))?
}

fn push_report_error(
    errors: &mut Vec<FolderScanError>,
    counts: &mut ScanCounts,
    path: &str,
    message: String,
) {
    if errors.len() < MAX_REPORTED_ERRORS {
        errors.push(FolderScanError {
            path: path.to_string(),
            message,
        });
    } else {
        counts.error_limit_reached = true;
    }
}

/// Search Hugging Face for GGUF repositories.
pub async fn search_hub(
    State(state): State<SharedState>,
    Query(query): Query<SearchQuery>,
) -> AppResult<Json<Value>> {
    let limit = query.limit.unwrap_or_else(search::default_limit);
    let term = query.q.as_deref().map(str::trim).unwrap_or("");

    let hits = if term.is_empty() {
        search::trending_gguf(&state.outbound, limit).await?
    } else {
        search::search_gguf(&state.outbound, term, limit).await?
    };

    let installed = state.db.list_models()?;

    let described: Vec<Value> = hits
        .into_iter()
        .map(|hit| {
            let fit = search::estimate_fit(&hit, &state.hardware);
            // Any installed model from this repository counts as installed, since
            // the quantization is chosen at download time.
            let is_installed = installed
                .iter()
                .any(|model| model.hf_repo.as_deref() == Some(hit.repo.as_str()));

            let mut encoded = serde_json::to_value(&hit).unwrap_or_else(|_| json!({}));
            encoded["fit"] = serde_json::to_value(fit).unwrap_or(Value::Null);
            encoded["installed"] = json!(is_installed);
            encoded
        })
        .collect();

    Ok(Json(json!({ "models": described })))
}

/// Models the user has pulled, each annotated with whether it is loaded and how
/// well it fits this machine.
pub async fn list_models(State(state): State<SharedState>) -> AppResult<Json<Value>> {
    let loaded = state.engines.loaded().await;
    let models = state.db.list_models()?;

    let described: Vec<Value> = models
        .into_iter()
        .map(|model| {
            let is_loaded = loaded.iter().any(|engine| engine.model_id == model.id);
            let fit = model
                .file_size_bytes
                .map(|bytes| estimate_fit(bytes as u64, &state.hardware));

            // The same classifier the provider catalogues go through.
            let classified = kuro_core::classify::classify(&model.id);

            json!({
                "model": model,
                "loaded": is_loaded,
                "fit": fit,
                "kind": classified.kind.as_str(),
                // False for embedding models, rerankers and the rest.
                "chat": classified.kind.is_chat(),
            })
        })
        .collect();

    // Provider models ride along in the same response so the composer needs one
    // request to render a picker containing everything the user can talk to.
    let mut remote = state.providers.remote_models()?;
    remote.extend(free_models(&state));

    Ok(Json(json!({
        "models": described,
        "remote": remote,
    })))
}

/// The free pool's entries for the model picker.
fn free_models(state: &SharedState) -> Vec<kuro_core::cloud::RemoteModel> {
    let keys = crate::routes::free::stored_keys(state);

    // Opening the picker is the natural moment to find out what these keys reach.
    crate::routes::free::refresh_catalogues_in_background(state, &keys);

    let mut out: Vec<kuro_core::cloud::RemoteModel> = free::FreeFlavour::ALL
        .iter()
        .filter(|flavour| state.free.choose(**flavour, &keys).is_some())
        .map(|flavour| kuro_core::cloud::RemoteModel {
            id: flavour.model_id(),
            name: flavour.label().to_string(),
            // One group in the picker, whichever provider happens to answer.
            connector_id: "kuro-free".to_string(),
            connector_label: "Kuro Free".to_string(),
            provider: "free".to_string(),
            // Every one of these routes across providers, so marking one as pooled and the rest
            // not would be a distinction without a difference.
            pooled: false,
            pool_size: 0,
            free: true,
            specialities: flavour
                .speciality()
                .map(|s| vec![s.label()])
                .unwrap_or_default(),
            params_b: None,
            // A pooled row is only listed when something can serve it, so by
            // construction it is available. That check is the `filter` above.
            unavailable: None,
            fix_url: None,
        })
        .collect();

    out.extend(named_free_models(state, &keys));
    out
}

/// Why a model cannot be picked, in words rather than a status code.
fn explain(trouble: free::Trouble, provider: &str) -> String {
    match trouble {
        free::Trouble::RateLimited => {
            format!("{provider} is out of allowance for now — it comes back")
        }
        free::Trouble::Rejected => {
            format!("{provider} rejected the key. Check it on the Free models screen")
        }
        // Not "this model was retired".
        free::Trouble::Gone => {
            format!("{provider} would not serve this model on your key")
        }
    }
}

/// Each free provider's own catalogue, named individually.
fn named_free_models(
    state: &SharedState,
    keys: &std::collections::HashMap<String, String>,
) -> Vec<kuro_core::cloud::RemoteModel> {
    let allow_keyless = state.free.allows_keyless();
    let mut out = Vec::new();

    for provider in free::FREE_PROVIDERS {
        if !provider.is_reachable(keys, allow_keyless) {
            continue;
        }
        let group = kuro_core::cloud::RemoteGroup {
            connector_id: &free::connector_id(provider.slug),
            connector_label: provider.name,
            provider: provider.slug,
        };

        // Why the whole provider is out, if it is. Applies to every one of its
        // models, so it is asked once.
        let provider_trouble = state.free.trouble(provider.slug);

        for model in state.free.advertised_chat_models(provider) {
            let mut row = kuro_core::cloud::RemoteModel::described(
                format!("{}{}/{model}", free::MODEL_PREFIX, provider.slug),
                group,
                &model,
                // Everything reachable here is inside the provider's free allowance by
                // construction.
                true,
            );

            // The model's own refusal wins over the provider's, because it is
            // the more specific fact and the one with a remedy attached.
            let trouble = state
                .free
                .model_trouble(provider.slug, &model)
                .or(provider_trouble);

            if let Some(trouble) = trouble {
                row.unavailable = Some(explain(trouble, provider.name));
                if trouble.stale_catalogue() {
                    row.fix_url = provider
                        .model_key_url
                        .map(|template| template.replace("{model}", &model));
                }
            }

            out.push(row);
        }
    }

    out
}

/// Kuro's built-in suggestions, with a fit estimate and whether each is already
/// installed.
pub async fn recommended_models(State(state): State<SharedState>) -> AppResult<Json<Value>> {
    let installed = state.db.list_models()?;

    let described: Vec<Value> = CURATED_MODELS
        .iter()
        .map(|curated| {
            let model_id = curated.model_id(curated.default_quant);
            let existing = installed.iter().find(|model| model.id == model_id);

            json!({
                "id": model_id,
                "slug": curated.slug,
                "displayName": curated.display_name,
                "repo": curated.hf_repo,
                "defaultQuant": curated.default_quant,
                "quants": curated.quants,
                "paramCount": curated.param_count,
                "family": curated.family,
                "capabilities": curated.capabilities,
                "purposes": curated.purposes,
                // The heading this model is filed under when it is shown once.
                "primaryPurpose": catalog::curated::primary_purpose(curated).as_str(),
                "contextLength": curated.context_length,
                "approxSizeBytes": curated.approx_size_bytes,
                "blurb": curated.blurb,
                "installed": existing.is_some(),
                "status": existing.map(|model| model.status),
                "fit": estimate_fit(curated.approx_size_bytes, &state.hardware),
            })
        })
        .collect();

    Ok(Json(json!({
        "models": described,
        // The headings travel with the list, so the screen does not keep its own
        // copy of what the purposes are or what order they belong in.
        "purposes": catalog::curated::Purpose::ALL
            .iter()
            .map(|purpose| json!({
                "id": purpose.as_str(),
                "label": purpose.label(),
                "blurb": purpose.blurb(),
            }))
            .collect::<Vec<_>>(),
    })))
}

pub async fn get_model(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> AppResult<Json<Value>> {
    let model = state
        .db
        .get_model(&id)?
        .ok_or_else(|| KuroError::not_found(format!("model `{id}`")))?;

    let fit = model
        .file_size_bytes
        .map(|bytes| estimate_fit(bytes as u64, &state.hardware));

    Ok(Json(json!({
        "model": model,
        "loaded": state.engines.is_loaded(&id).await,
        "fit": fit,
    })))
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum PullRequest {
    Exact(catalog::ExactModelSelection),
    Legacy(LegacyPullRequest),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LegacyPullRequest {
    /// A curated slug (`qwen3-4b`), a repository (`owner/repo`), or a pasted
    /// Hugging Face URL. An optional `:QUANT` suffix picks a quantization.
    pub model: String,
}

impl PullRequest {
    async fn plan(&self, client: &reqwest::Client) -> Result<catalog::PullPlan> {
        match self {
            Self::Exact(selection) => catalog::plan_pull_exact(client, selection).await,
            Self::Legacy(request) => catalog::plan_pull(client, &request.model).await,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RepositoryQuery {
    pub repo: String,
}

pub async fn repository_detail(
    State(state): State<SharedState>,
    Query(query): Query<RepositoryQuery>,
) -> AppResult<Json<Value>> {
    let installed = state.db.list_models()?;
    Ok(Json(
        catalog::repository::describe_repository(
            &state.outbound,
            &query.repo,
            &state.hardware,
            &installed,
        )
        .await?,
    ))
}

/// Start a download and return immediately with something to poll.
pub async fn pull_model(
    State(state): State<SharedState>,
    Json(request): Json<PullRequest>,
) -> AppResult<Json<Value>> {
    let plan = request.plan(&state.outbound).await?;
    let prepared = catalog::prepare_pull(&state.db, &state.paths, &plan)?;
    let cancel = state.register_download(&prepared.download_id).await;

    let background = state.clone();
    let plan_for_task = plan.clone();
    let prepared_for_task = prepared.clone();
    tokio::spawn(async move {
        let result = catalog::run_pull(
            &background.outbound,
            &background.db,
            &plan_for_task,
            &prepared_for_task,
            cancel,
        )
        .await;

        if let Err(error) = result {
            tracing::warn!(model = %plan_for_task.model_id, %error, "pull failed");
        } else {
            tracing::info!(model = %plan_for_task.model_id, "pull complete");
        }
        background
            .forget_download(&prepared_for_task.download_id)
            .await;
    });

    Ok(Json(json!({
        "downloadId": prepared.download_id,
        "model": {
            "id": plan.model_id,
            "displayName": plan.display_name,
            "repo": plan.repo,
            "revision": plan.revision,
            "file": plan.filename,
            "quant": plan.quant,
            "sizeBytes": plan.size_bytes,
        },
        "fit": estimate_fit(plan.size_bytes, &state.hardware),
    })))
}

/// Resolve a reference without downloading, so the UI can preview a pull.
pub async fn preview_pull(
    State(state): State<SharedState>,
    Json(request): Json<PullRequest>,
) -> AppResult<Json<Value>> {
    let plan = request.plan(&state.outbound).await?;

    Ok(Json(json!({
        "id": plan.model_id,
        "displayName": plan.display_name,
        "repo": plan.repo,
        "revision": plan.revision,
        "file": plan.filename,
        "quant": plan.quant,
        "sizeBytes": plan.size_bytes,
        "verifiable": plan.sha256.is_some(),
        "fit": estimate_fit(plan.size_bytes, &state.hardware),
    })))
}

pub async fn delete_model(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> AppResult<Json<Value>> {
    // Stop it first; deleting weights out from under a running engine would
    // leave a process serving a file that no longer exists.
    state.engines.unload(&id).await?;
    catalog::remove_model(&state.db, &state.paths, &id)?;

    Ok(Json(json!({ "deleted": true })))
}

pub async fn load_model(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> AppResult<Json<Value>> {
    let port = state.engines.ensure_loaded(&id).await?;
    Ok(Json(json!({ "loaded": true, "modelId": id, "port": port })))
}

pub async fn unload_model(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> AppResult<Json<Value>> {
    let was_loaded = state.engines.unload(&id).await?;
    Ok(Json(json!({ "unloaded": was_loaded })))
}

/// Models currently held in memory, for `kuro ps` and Settings → Server.
pub async fn loaded_models(State(state): State<SharedState>) -> AppResult<Json<Value>> {
    Ok(Json(json!({ "loaded": state.engines.loaded().await })))
}

#[cfg(test)]
mod contract_tests {
    use super::*;

    #[test]
    fn exact_and_legacy_pull_requests_are_unambiguous() {
        assert!(matches!(serde_json::from_value::<PullRequest>(json!({"model":"qwen3-4b"})).unwrap(), PullRequest::Legacy(_)));
        assert!(matches!(serde_json::from_value::<PullRequest>(json!({"repo":"owner/repo", "revision":"0123456789abcdef0123456789abcdef01234567", "filename":"model.gguf"})).unwrap(), PullRequest::Exact(_)));
        for value in [json!({"model":"qwen3-4b", "filename":"x.gguf"}), json!({"repo":"owner/repo", "filename":"x.gguf"}), json!({"repo":"owner/repo", "revision":"0123456789abcdef0123456789abcdef01234567", "filename":"x.gguf", "sizeBytes":42})] {
            assert!(serde_json::from_value::<PullRequest>(value).is_err());
        }
    }

    #[tokio::test]
    async fn a_moving_revision_is_rejected_before_network_or_database_work() {
        let request = PullRequest::Exact(catalog::ExactModelSelection { repo: "owner/repo".into(), revision: "main".into(), filename: "x.gguf".into() });
        assert!(matches!(request.plan(&reqwest::Client::new()).await, Err(KuroError::BadRequest(_))));
    }
}
