//! Remote model endpoints: API providers, and hardware you rent.

use axum::extract::{Path, State};
use axum::Json;
use kuro_core::cloud::presets::surface_for;
use kuro_core::cloud::PRESETS;
use kuro_core::KuroError;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::error::AppResult;
use crate::state::SharedState;

/// Connected providers and the presets on offer.
pub async fn list_providers(State(state): State<SharedState>) -> AppResult<Json<Value>> {
    let connected: Vec<Value> = state
        .db
        .list_cloud_connectors()?
        .into_iter()
        .map(|connector| {
            let has_key = state.providers.has_key(&connector);
            let mut encoded = serde_json::to_value(&connector).unwrap_or_else(|_| json!({}));
            // The reference is an internal detail and looks like a secret to a
            // reader of the API, so it is not published.
            if let Some(object) = encoded.as_object_mut() {
                object.remove("keychain_ref");
            }
            encoded["hasKey"] = json!(has_key);
            // Which of the two screens this belongs on, resolved from the preset
            // it was created from so the client never has to know the mapping.
            encoded["surface"] = json!(surface_for(&connector.provider).as_str());
            encoded
        })
        .collect();

    // `surface` is derived from `kind` rather than stored on the preset, so that
    // there is one place deciding which screen an entry belongs on.
    let presets: Vec<Value> = PRESETS
        .iter()
        .map(|preset| {
            let mut encoded = serde_json::to_value(preset).unwrap_or_else(|_| json!({}));
            encoded["surface"] = json!(preset.kind.surface().as_str());
            encoded
        })
        .collect();

    Ok(Json(json!({
        "providers": connected,
        "presets": presets,
    })))
}

#[derive(Debug, Deserialize)]
pub struct AddProviderRequest {
    /// A preset slug, or `custom`.
    pub provider: String,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default, rename = "baseUrl")]
    pub base_url: Option<String>,
    #[serde(rename = "apiKey")]
    pub api_key: String,
}

/// Add a provider and immediately probe it.
pub async fn add_provider(
    State(state): State<SharedState>,
    Json(request): Json<AddProviderRequest>,
) -> AppResult<Json<Value>> {
    if request.api_key.trim().is_empty() {
        return Err(KuroError::bad_request("the API key is empty").into());
    }

    let connector = state
        .providers
        .add(
            &request.provider,
            request.label.as_deref(),
            request.base_url.as_deref(),
            &request.api_key,
        )
        .await?;

    // The probe already ran inside `add`, so the record carries its own verdict —
    // including a failure, which leaves the provider saved and fixable.
    Ok(Json(json!({
        "provider": {
            "id": connector.id,
            "provider": connector.provider,
            "label": connector.label,
            "base_url": connector.base_url,
            "status": connector.status,
            "last_error": connector.last_error,
            "models": connector.models,
            "enabled": connector.enabled,
            "hasKey": true,
        },
    })))
}

/// Re-probe a provider, refreshing its model list.
pub async fn test_provider(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> AppResult<Json<Value>> {
    match state.providers.test(&id).await {
        Ok(models) => Ok(Json(json!({ "ok": true, "models": models }))),
        Err(error @ KuroError::NotFound(_)) => Err(error.into()),
        Err(error) => Ok(Json(json!({ "ok": false, "error": error.to_string() }))),
    }
}

#[derive(Debug, Deserialize)]
pub struct KeyRequest {
    #[serde(rename = "apiKey")]
    pub api_key: String,
}

pub async fn replace_key(
    State(state): State<SharedState>,
    Path(id): Path<String>,
    Json(request): Json<KeyRequest>,
) -> AppResult<Json<Value>> {
    state.providers.replace_key(&id, &request.api_key).await?;

    let connector = state
        .db
        .get_cloud_connector(&id)?
        .ok_or_else(|| KuroError::not_found(format!("provider `{id}`")))?;

    Ok(Json(json!({
        "status": connector.status,
        "last_error": connector.last_error,
        "models": connector.models,
    })))
}

#[derive(Debug, Deserialize)]
pub struct EnabledRequest {
    pub enabled: bool,
}

pub async fn set_enabled(
    State(state): State<SharedState>,
    Path(id): Path<String>,
    Json(request): Json<EnabledRequest>,
) -> AppResult<Json<Value>> {
    state.db.set_cloud_enabled(&id, request.enabled)?;
    Ok(Json(json!({ "enabled": request.enabled })))
}

pub async fn delete_provider(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> AppResult<Json<Value>> {
    let removed = state.providers.remove(&id)?;
    if !removed {
        return Err(KuroError::not_found(format!("provider `{id}`")).into());
    }
    Ok(Json(json!({ "deleted": true })))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;

    use axum::extract::{Path, State};
    use kuro_core::cloud::ProviderRegistry;
    use kuro_core::db::Db;
    use kuro_core::engine::EngineManager;
    use kuro_core::free::FreePool;
    use kuro_core::mcp::McpManager;
    use kuro_core::workspace::ProcessRegistry;
    use kuro_core::{KuroError, Paths, SecretStore};
    use tempfile::TempDir;
    use tokio::sync::Mutex;

    use super::test_provider;
    use crate::error::AppError;
    use crate::state::{AppState, SharedState};

    fn state() -> (TempDir, SharedState) {
        let root = tempfile::tempdir().expect("temporary test root");
        let db = Db::open_in_memory().expect("database");
        let paths = Paths::for_root(root.path().join("app"));
        paths.create_all().expect("application directories");
        let hardware = kuro_core::hardware::detect();
        let secrets = SecretStore::new(paths.credentials_file());
        let outbound = reqwest::Client::new();
        let engines = Arc::new(
            EngineManager::new(db.clone(), paths.clone(), hardware.clone()).expect("engines"),
        );
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
    async fn testing_a_provider_deleted_in_another_window_returns_not_found() {
        let (_root, state) = state();

        let response = test_provider(State(state), Path("missing".to_string())).await;

        match response {
            Err(AppError(KuroError::NotFound(message))) => {
                assert!(message.contains("missing"), "got: {message}");
            }
            Err(AppError(other)) => panic!("expected not found, got {other}"),
            Ok(body) => panic!(
                "missing provider looked like a connection failure: {}",
                body.0
            ),
        }
    }
}
