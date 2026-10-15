use axum::extract::{Path, State};
use axum::Json;
use kuro_core::KuroError;
use serde_json::{json, Value};

use crate::error::{AppError, AppResult};
use crate::state::SharedState;

pub async fn list_downloads(State(state): State<SharedState>) -> AppResult<Json<Value>> {
    Ok(Json(json!({ "downloads": state.db.list_downloads()? })))
}

pub async fn get_download(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> AppResult<Json<Value>> {
    let record = state
        .db
        .get_download(&id)?
        .ok_or_else(|| KuroError::not_found(format!("download `{id}`")))?;
    Ok(Json(json!(record)))
}

/// Stop an in-flight transfer.
pub async fn cancel_download(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> AppResult<Json<Value>> {
    let record = state
        .db
        .get_download(&id)?
        .ok_or_else(|| KuroError::not_found(format!("download `{id}`")))?;

    if !record.status.is_active() {
        return Err(AppError(KuroError::bad_request(format!(
            "download is already {}",
            record.status.as_str()
        ))));
    }

    if !state.cancel_download(&id).await {
        return Err(AppError(KuroError::bad_request(
            "This transfer has no cancellable worker in this Kuro server. Engine installation and downloads owned by another process cannot be cancelled here.",
        )));
    }

    // The worker owns the target until it closes the file and restores the previous model.

    Ok(Json(json!({ "cancelled": true, "pending": true })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::AppState;
    use kuro_core::{
        cloud::ProviderRegistry,
        db::{Db, DownloadKind},
        engine::EngineManager,
        free::FreePool,
        mcp::McpManager,
        workspace::ProcessRegistry,
        Paths, SecretStore,
    };
    use std::collections::HashMap;
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };
    use tokio::sync::Mutex;

    fn state() -> (tempfile::TempDir, SharedState) {
        let root = tempfile::tempdir().unwrap();
        let db = Db::open_in_memory().unwrap();
        let paths = Paths::for_root(root.path().join("app"));
        paths.create_all().unwrap();
        let hardware = kuro_core::hardware::detect();
        let secrets = SecretStore::new(paths.credentials_file());
        let outbound = reqwest::Client::new();
        let engines =
            Arc::new(EngineManager::new(db.clone(), paths.clone(), hardware.clone()).unwrap());
        let live_hardware = kuro_core::hardware::LiveHardwareMetrics::new(paths.models_dir());
        let state = Arc::new(AppState {
            db: db.clone(),
            paths,
            hardware,
            live_hardware,
            engines,
            outbound: outbound.clone(),
            secrets: secrets.clone(),
            mcp: Arc::new(McpManager::new(
                db.clone(),
                secrets.clone(),
                outbound.clone(),
            )),
            providers: Arc::new(ProviderRegistry::new(db, secrets, outbound)),
            free: FreePool::new(),
            processes: ProcessRegistry::new(),
            started_at: chrono::Utc::now(),
            port: 0,
            download_cancels: Mutex::new(HashMap::<String, Arc<AtomicBool>>::new()),
        });
        (root, state)
    }

    #[tokio::test]
    async fn cancellation_requests_keep_the_target_owned_until_worker_cleanup() {
        let (_root, state) = state();
        let download = state
            .db
            .create_download(
                DownloadKind::Model,
                "model",
                "Model",
                "https://example.invalid/model.gguf",
                "/tmp/unused.gguf",
                Some(100),
            )
            .unwrap();
        let cancel = state.register_download(&download.id).await;
        assert!(
            cancel_download(State(state.clone()), Path(download.id.clone()))
                .await
                .is_ok()
        );
        assert!(cancel.load(Ordering::Relaxed));
        assert!(state
            .db
            .get_download(&download.id)
            .unwrap()
            .unwrap()
            .status
            .is_active());
        assert!(state
            .db
            .active_download_for_target("model")
            .unwrap()
            .is_some());
    }

    #[tokio::test]
    async fn engine_bootstrap_without_a_cancel_handle_cannot_be_falsely_cancelled() {
        let (_root, state) = state();
        let download = state
            .db
            .create_download(
                DownloadKind::EngineBinary,
                "engine",
                "Engine",
                "https://example.invalid/engine",
                "/tmp/unused-engine",
                None,
            )
            .unwrap();
        assert!(
            cancel_download(State(state.clone()), Path(download.id.clone()))
                .await
                .is_err()
        );
        assert!(state
            .db
            .get_download(&download.id)
            .unwrap()
            .unwrap()
            .status
            .is_active());
    }
}
