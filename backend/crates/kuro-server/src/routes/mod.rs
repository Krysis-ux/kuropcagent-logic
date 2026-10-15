//! HTTP surface.

use axum::http::{header, HeaderValue, Method};
use axum::routing::{delete, get, patch, post, put};
use axum::Router;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use crate::state::SharedState;

pub mod anthropic;
pub mod browse;
pub mod chat;
pub mod common;
pub mod conversations;
pub mod downloads;
pub mod free;
pub mod incognito;
pub mod mcp;
pub mod models;
pub mod openai;
pub mod projects;
pub mod providers;
pub mod settings;
pub mod subagent;
pub mod system;
pub mod tools;
pub mod tools_runtime;
pub mod workspaces;

/// The port the Vite dev server runs on, and the only other origin allowed.
const DEV_SERVER_PORT: u16 = 5173;

pub fn router(state: SharedState) -> Router {
    let cors = cors_layer(state.port);

    Router::new()
        .merge(native_routes())
        .merge(openai_routes())
        .merge(crate::static_files::router())
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

/// Who is allowed to call this server from a browser.
fn cors_layer(port: u16) -> CorsLayer {
    // Both spellings of loopback, because a browser treats them as different
    // origins and people reach the UI by whichever the address bar holds.
    let origins: Vec<HeaderValue> = [
        format!("http://127.0.0.1:{port}"),
        format!("http://localhost:{port}"),
        format!("http://127.0.0.1:{DEV_SERVER_PORT}"),
        format!("http://localhost:{DEV_SERVER_PORT}"),
    ]
    .iter()
    .filter_map(|origin| origin.parse().ok())
    .collect();

    CorsLayer::new()
        .allow_origin(origins)
        .allow_methods([
            Method::GET,
            Method::POST,
            Method::PATCH,
            Method::DELETE,
            Method::OPTIONS,
        ])
        // Everything Kuro's own client sends. `Any` here would also have let a
        // page name whatever header it liked in a preflight.
        .allow_headers([header::CONTENT_TYPE, header::ACCEPT])
}

fn native_routes() -> Router<SharedState> {
    Router::new()
        .route("/api/health", get(system::health))
        .route("/api/status", get(system::status))
        .route("/api/hardware", get(system::hardware))
        .route("/api/shutdown", post(system::shutdown))
        .route("/api/restart", post(system::restart))
        // Literal segments are matched ahead of `{id}`, so `/models/loaded`
        // and `/models/recommended` are not captured as model ids.
        .route("/api/models", get(models::list_models))
        .route("/api/models/recommended", get(models::recommended_models))
        .route("/api/models/loaded", get(models::loaded_models))
        .route("/api/models/search", get(models::search_hub))
        .route("/api/models/repository", get(models::repository_detail))
        .route("/api/models/pull", post(models::pull_model))
        .route("/api/models/preview", post(models::preview_pull))
        .route(
            "/api/models/folders",
            get(models::list_model_folders)
                .post(models::add_model_folder)
                .delete(models::remove_model_folder),
        )
        .route(
            "/api/models/folders/rescan",
            post(models::rescan_model_folders),
        )
        .route("/api/models/{id}", get(models::get_model))
        .route("/api/models/{id}", delete(models::delete_model))
        .route("/api/models/{id}/load", post(models::load_model))
        .route("/api/models/{id}/unload", post(models::unload_model))
        .route("/api/downloads", get(downloads::list_downloads))
        .route("/api/downloads/{id}", get(downloads::get_download))
        .route(
            "/api/downloads/{id}/cancel",
            post(downloads::cancel_download),
        )
        .route("/api/conversations", get(conversations::list_conversations))
        .route(
            "/api/conversations",
            post(conversations::create_conversation),
        )
        .route(
            "/api/conversations/{id}",
            get(conversations::get_conversation),
        )
        .route(
            "/api/conversations/{id}",
            patch(conversations::update_conversation),
        )
        .route(
            "/api/conversations/{id}",
            delete(conversations::delete_conversation),
        )
        .route(
            "/api/conversations/{id}/messages",
            get(conversations::list_messages),
        )
        .route("/api/conversations/{id}/messages", post(chat::send_message))
        // Editing a message truncates the conversation there and answers again;
        // forking copies it up to a message and leaves the original alone.
        .route(
            "/api/conversations/{id}/messages/{message_id}",
            patch(chat::edit_message),
        )
        .route(
            "/api/conversations/{id}/fork",
            post(conversations::fork_conversation),
        )
        .route(
            "/api/conversations/{id}/project",
            post(projects::move_conversation),
        )
        // Projects: standing instructions plus a grouping of conversations.
        .route("/api/projects", get(projects::list_projects))
        .route("/api/projects", post(projects::create_project))
        .route("/api/projects/{id}", get(projects::get_project))
        .route("/api/projects/{id}", patch(projects::update_project))
        .route("/api/projects/{id}", delete(projects::delete_project))
        // Walking this machine's folders, so nothing in the interface has to ask
        // somebody to type a path.
        .route("/api/fs/browse", get(browse::browse))
        // The operating system's own dialog, which is the one people know.
        .route("/api/fs/choose", post(browse::native_picker))
        .route("/api/settings", get(settings::get_settings))
        .route("/api/settings", patch(settings::patch_settings))
        .route("/api/settings/engine", get(settings::engine_settings))
        .route("/api/settings/reset", post(settings::reset_settings))
        .route("/api/incognito/turn", post(incognito::turn))
        // MCP tool servers. `/registry` is the recommended list and is matched
        // ahead of `{id}` for the same reason `/models/loaded` is.
        .route("/api/mcp/servers", get(mcp::list_servers))
        .route("/api/mcp/servers", post(mcp::add_server))
        .route("/api/mcp/registry", get(mcp::list_registry))
        .route("/api/mcp/servers/{id}", delete(mcp::delete_server))
        .route("/api/mcp/servers/{id}/refresh", post(mcp::refresh_server))
        .route("/api/mcp/servers/{id}/enabled", post(mcp::set_enabled))
        .route("/api/mcp/servers/{id}/auth", post(mcp::set_auth))
        // Kuro's own tools, and what powers them.
        .route("/api/tools", get(tools::overview))
        .route("/api/tools/defaults", post(tools::configure_defaults))
        .route("/api/tools/skills", post(tools::set_skills))
        .route("/api/tools/skills/upload", post(tools::upload_skill))
        .route("/api/tools/skills/import", post(tools::import_skills))
        .route("/api/tools/skills/{slug}", delete(tools::delete_skill))
        .route("/api/tools/search", post(tools::configure_search))
        .route("/api/tools/search/test", post(tools::test_search))
        .route("/api/memories", get(tools::list_memories))
        .route("/api/memories", post(tools::create_memory))
        .route("/api/memories/{id}", delete(tools::delete_memory))
        // Coding workspaces: the only surface in Kuro with file access. Literal
        // segments come before `{id}` for the same reason `/models/loaded` does.
        .route("/api/workspaces", get(workspaces::list_workspaces))
        .route("/api/workspaces", post(workspaces::create_workspace))
        .route("/api/workspaces/{id}", get(workspaces::get_workspace))
        .route("/api/workspaces/{id}", patch(workspaces::update_workspace))
        .route("/api/workspaces/{id}", delete(workspaces::delete_workspace))
        .route("/api/workspaces/{id}/tree", get(workspaces::workspace_tree))
        .route(
            "/api/workspaces/{id}/file",
            get(workspaces::read_workspace_file),
        )
        .route(
            "/api/workspaces/{id}/changes",
            get(workspaces::list_changes),
        )
        .route(
            "/api/workspaces/{id}/changes/{change_id}/undo",
            post(workspaces::undo_change),
        )
        .route(
            "/api/workspaces/{id}/conversations",
            post(workspaces::create_conversation),
        )
        // Dev servers and other long-running commands, so the preview panel can
        // show what is running and the user can stop it without asking a model.
        .route(
            "/api/workspaces/{id}/changes/{change_id}",
            get(workspaces::change_diff),
        )
        .route(
            "/api/workspaces/{id}/processes",
            get(workspaces::list_processes),
        )
        .route(
            "/api/workspaces/{id}/processes",
            post(workspaces::start_process),
        )
        .route(
            "/api/workspaces/{id}/processes",
            delete(workspaces::clear_processes),
        )
        .route(
            "/api/workspaces/{id}/processes/{process_id}",
            delete(workspaces::forget_process),
        )
        .route(
            "/api/workspaces/{id}/processes/{process_id}/log",
            get(workspaces::process_log),
        )
        .route(
            "/api/workspaces/{id}/processes/{process_id}/stop",
            post(workspaces::stop_process),
        )
        // Providers with a free tier, pooled behind one model id.
        .route("/api/free", get(free::overview))
        .route("/api/free/keyless", post(free::set_keyless))
        .route("/api/free/usage", get(free::usage))
        .route("/api/free/{slug}/key", post(free::set_key))
        .route("/api/free/{slug}/key", delete(free::delete_key))
        .route("/api/free/{slug}/test", post(free::test_key))
        .route("/api/free/{slug}/allowance", put(free::set_allowance))
        .route("/api/free/{slug}/allowance", delete(free::delete_allowance))
        // Remote model providers.
        .route("/api/providers", get(providers::list_providers))
        .route("/api/providers", post(providers::add_provider))
        .route("/api/providers/{id}", delete(providers::delete_provider))
        .route("/api/providers/{id}/test", post(providers::test_provider))
        .route("/api/providers/{id}/key", post(providers::replace_key))
        .route("/api/providers/{id}/enabled", post(providers::set_enabled))
}

fn openai_routes() -> Router<SharedState> {
    Router::new()
        .route("/v1/models", get(openai::list_models))
        .route("/v1/messages", post(anthropic::messages))
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/v1/completions", post(openai::completions))
        .route("/v1/embeddings", post(openai::embeddings))
}

#[cfg(test)]
mod model_folder_route_tests {
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;

    use axum::body::{to_bytes, Body};
    use axum::http::{Method, Request, StatusCode};
    use kuro_core::cloud::ProviderRegistry;
    use kuro_core::db::{Db, ModelSource, NewModel};
    use kuro_core::engine::EngineManager;
    use kuro_core::free::FreePool;
    use kuro_core::mcp::McpManager;
    use kuro_core::workspace::ProcessRegistry;
    use kuro_core::{Paths, SecretStore};
    use serde_json::{json, Value};
    use tempfile::TempDir;
    use tokio::sync::Mutex;
    use tower::ServiceExt;

    use super::native_routes;
    use crate::state::{AppState, SharedState};

    struct Harness {
        root: TempDir,
        state: SharedState,
    }

    impl Harness {
        fn new() -> Self {
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
            Self { root, state }
        }

        fn external_folder(&self, name: &str) -> PathBuf {
            let path = self.root.path().join(name);
            std::fs::create_dir_all(&path).expect("external model folder");
            path.canonicalize()
                .expect("canonical external model folder")
        }

        fn write_model(&self, folder: &Path, name: &str) -> PathBuf {
            let path = folder.join(name);
            std::fs::write(&path, b"GGUF test fixture").expect("model fixture");
            path
        }

        async fn request(
            &self,
            method: Method,
            uri: &str,
            body: Option<Value>,
        ) -> (StatusCode, Value) {
            let body = body
                .map(|value| Body::from(serde_json::to_vec(&value).expect("JSON body")))
                .unwrap_or_else(Body::empty);
            let request = Request::builder()
                .method(method)
                .uri(uri)
                .header("content-type", "application/json")
                .body(body)
                .expect("request");
            let response = native_routes()
                .with_state(self.state.clone())
                .oneshot(request)
                .await
                .expect("route response");
            let status = response.status();
            let bytes = to_bytes(response.into_body(), usize::MAX)
                .await
                .expect("response body");
            let value = if bytes.is_empty() {
                Value::Null
            } else {
                serde_json::from_slice(&bytes)
                    .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()))
            };
            (status, value)
        }

        async fn add(&self, folder: &Path) -> (StatusCode, Value) {
            self.request(
                Method::POST,
                "/api/models/folders",
                Some(json!({ "path": folder.to_string_lossy() })),
            )
            .await
        }
    }

    #[tokio::test]
    async fn literal_folder_routes_are_not_captured_as_model_ids() {
        let harness = Harness::new();

        let (list_status, list) = harness
            .request(Method::GET, "/api/models/folders", None)
            .await;
        assert_eq!(list_status, StatusCode::OK);
        assert_eq!(list, json!({ "folders": [] }));

        let (rescan_status, rescan) = harness
            .request(Method::POST, "/api/models/folders/rescan", None)
            .await;
        assert_eq!(rescan_status, StatusCode::OK);
        assert_eq!(rescan, json!({ "folders": [] }));
    }

    #[tokio::test]
    async fn add_returns_canonical_ordered_folders_and_complete_counts() {
        let harness = Harness::new();
        let first = harness.external_folder("first");
        let second = harness.external_folder("second");
        harness.write_model(&first, "Qwen3-4B-Q4_K_M.gguf");
        harness.write_model(&second, "Phi-3-mini-Q5_K_M.GGUF");
        harness.write_model(&second, "notes.txt");

        let (first_status, first_body) = harness.add(&first).await;
        assert_eq!(first_status, StatusCode::OK);
        assert_eq!(first_body["counts"]["discovered"], 1);
        assert_eq!(first_body["counts"]["registered"], 1);
        assert_eq!(first_body["errors"], json!([]));

        let (second_status, second_body) = harness.add(&second).await;
        assert_eq!(second_status, StatusCode::OK);
        assert_eq!(
            second_body["folders"],
            json!([
                { "path": first.to_string_lossy(), "available": true },
                { "path": second.to_string_lossy(), "available": true }
            ])
        );
        assert_eq!(second_body["counts"]["discovered"], 1);
        assert_eq!(second_body["counts"]["registered"], 1);
        assert_eq!(second_body["counts"]["skipped_non_gguf"], 1);

        let count_keys = second_body["counts"]
            .as_object()
            .expect("count object")
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(
            count_keys.len(),
            15,
            "the route must expose every bounded scanner count"
        );
    }

    #[tokio::test]
    async fn failed_adds_do_not_persist_any_path() {
        let harness = Harness::new();
        let missing = harness.root.path().join("missing");
        let (missing_status, _) = harness.add(&missing).await;
        assert_eq!(missing_status, StatusCode::BAD_REQUEST);

        let file = harness.root.path().join("plain-file");
        std::fs::write(&file, b"not a directory").expect("plain file");
        let (file_status, _) = harness.add(&file).await;
        assert_eq!(file_status, StatusCode::BAD_REQUEST);

        let selected = harness.external_folder("selected");
        let nested = selected.join("nested");
        std::fs::create_dir(&nested).expect("nested directory");
        assert_eq!(harness.add(&selected).await.0, StatusCode::OK);
        assert_eq!(harness.add(&selected).await.0, StatusCode::BAD_REQUEST);
        assert_eq!(harness.add(&nested).await.0, StatusCode::BAD_REQUEST);

        #[cfg(unix)]
        {
            let target = harness.external_folder("symlink-target");
            let link = harness.root.path().join("symlink-root");
            std::os::unix::fs::symlink(target, &link).expect("symlink fixture");
            assert_eq!(harness.add(&link).await.0, StatusCode::BAD_REQUEST);
        }

        let (_, list) = harness
            .request(Method::GET, "/api/models/folders", None)
            .await;
        assert_eq!(list["folders"].as_array().expect("folder list").len(), 1);
        assert_eq!(
            list["folders"][0]["path"],
            selected.to_string_lossy().as_ref()
        );
    }

    #[tokio::test]
    async fn list_and_rescan_keep_an_offline_root_and_its_models_visible() {
        let harness = Harness::new();
        let folder = harness.external_folder("portable-drive");
        harness.write_model(&folder, "Llama-3-8B-Q4_K_M.gguf");
        assert_eq!(harness.add(&folder).await.0, StatusCode::OK);
        let model = harness
            .state
            .db
            .list_models()
            .expect("models")
            .into_iter()
            .find(|model| model.discovered_folder.as_deref() == folder.to_str())
            .expect("scanned model");

        let moved = harness.root.path().join("drive-unplugged");
        std::fs::rename(&folder, &moved).expect("simulate unavailable drive");

        let (list_status, list) = harness
            .request(Method::GET, "/api/models/folders", None)
            .await;
        assert_eq!(list_status, StatusCode::OK);
        assert_eq!(
            list["folders"],
            json!([{ "path": folder.to_string_lossy(), "available": false }])
        );

        let (rescan_status, rescan) = harness
            .request(Method::POST, "/api/models/folders/rescan", None)
            .await;
        assert_eq!(rescan_status, StatusCode::OK);
        assert_eq!(rescan["folders"][0]["folders"][0]["available"], false);
        assert_eq!(
            rescan["folders"][0]["errors"]
                .as_array()
                .expect("visible errors")
                .len(),
            1
        );
        assert!(harness
            .state
            .db
            .get_model(&model.id)
            .expect("model lookup")
            .is_some());
    }

    #[tokio::test]
    async fn a_result_limited_rescan_never_reconciles_a_partial_view() {
        let harness = Harness::new();
        let folder = harness.external_folder("wide-folder");
        for index in 0..502 {
            harness.write_model(&folder, &format!("model-{index:03}.gguf"));
        }
        let (add_status, add) = harness.add(&folder).await;
        assert_eq!(add_status, StatusCode::OK);
        assert_eq!(add["counts"]["result_limit_reached"], true);

        let registered = harness
            .state
            .db
            .list_models()
            .expect("models")
            .into_iter()
            .find(|model| model.discovered_folder.as_deref() == folder.to_str())
            .expect("registered candidate");
        std::fs::remove_file(registered.file_path.as_deref().expect("model path"))
            .expect("remove one external fixture");

        let (rescan_status, rescan) = harness
            .request(Method::POST, "/api/models/folders/rescan", None)
            .await;
        assert_eq!(rescan_status, StatusCode::OK);
        assert_eq!(rescan["folders"][0]["counts"]["result_limit_reached"], true);
        assert!(harness
            .state
            .db
            .get_model(&registered.id)
            .expect("model lookup")
            .is_some());
    }

    #[tokio::test]
    async fn removing_one_folder_unregisters_only_its_rows_and_never_its_bytes() {
        let harness = Harness::new();
        let first = harness.external_folder("first-root");
        let second = harness.external_folder("second-root");
        let first_file = harness.write_model(&first, "first-Q4_K_M.gguf");
        let second_file = harness.write_model(&second, "second-Q4_K_M.gguf");
        assert_eq!(harness.add(&first).await.0, StatusCode::OK);
        assert_eq!(harness.add(&second).await.0, StatusCode::OK);
        let models = harness.state.db.list_models().expect("models");
        let first_id = models
            .iter()
            .find(|model| model.discovered_folder.as_deref() == first.to_str())
            .expect("first model")
            .id
            .clone();
        let second_id = models
            .iter()
            .find(|model| model.discovered_folder.as_deref() == second.to_str())
            .expect("second model")
            .id
            .clone();

        let (status, body) = harness
            .request(
                Method::DELETE,
                "/api/models/folders",
                Some(json!({ "path": first.to_string_lossy() })),
            )
            .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["counts"]["unregistered"], 1);
        assert_eq!(
            body["folders"],
            json!([{ "path": second.to_string_lossy(), "available": true }])
        );
        assert!(first_file.exists());
        assert!(second_file.exists());
        assert!(harness
            .state
            .db
            .get_model(&first_id)
            .expect("first")
            .is_none());
        assert!(harness
            .state
            .db
            .get_model(&second_id)
            .expect("second")
            .is_some());
    }

    #[tokio::test]
    async fn deleting_one_local_registration_preserves_bytes_and_rescan_rediscovers_it() {
        let harness = Harness::new();
        let folder = harness.external_folder("local-delete");
        let file = harness.write_model(&folder, "Gemma-3-4B-Q4_K_M.gguf");
        assert_eq!(harness.add(&folder).await.0, StatusCode::OK);
        let model = harness
            .state
            .db
            .list_models()
            .expect("models")
            .into_iter()
            .find(|model| model.discovered_folder.as_deref() == folder.to_str())
            .expect("local model");

        let (delete_status, _) = harness
            .request(Method::DELETE, &format!("/api/models/{}", model.id), None)
            .await;
        assert_eq!(delete_status, StatusCode::OK);
        assert!(file.exists());
        assert!(harness
            .state
            .db
            .get_model(&model.id)
            .expect("deleted")
            .is_none());

        let (rescan_status, _) = harness
            .request(Method::POST, "/api/models/folders/rescan", None)
            .await;
        assert_eq!(rescan_status, StatusCode::OK);
        assert!(file.exists());
        assert!(harness
            .state
            .db
            .get_model(&model.id)
            .expect("rediscovered")
            .is_some());
    }

    #[tokio::test]
    async fn deleting_a_managed_model_still_removes_its_weight_file() {
        let harness = Harness::new();
        let id = "managed-test";
        let file = harness.state.paths.model_file(id, "weights.gguf");
        std::fs::create_dir_all(file.parent().expect("model directory")).expect("model directory");
        std::fs::write(&file, b"managed weights").expect("managed fixture");
        harness
            .state
            .db
            .upsert_model(&NewModel {
                id: id.to_string(),
                display_name: "Managed test".to_string(),
                source: ModelSource::Curated,
                hf_repo: Some("example/managed".to_string()),
                hf_file: Some("weights.gguf".to_string()),
                quant: None,
                param_count: None,
                family: None,
                capabilities: Vec::new(),
                context_length: None,
                file_size_bytes: Some(15),
            })
            .expect("managed model row");
        harness
            .state
            .db
            .set_model_ready(id, file.to_string_lossy().as_ref(), 15, "test")
            .expect("ready model");

        let (status, _) = harness
            .request(Method::DELETE, &format!("/api/models/{id}"), None)
            .await;
        assert_eq!(status, StatusCode::OK);
        assert!(!file.exists());
        assert!(harness.state.db.get_model(id).expect("deleted").is_none());
    }

    #[tokio::test]
    async fn hardware_and_engine_settings_routes_share_one_effective_profile_resolution() {
        let harness = Harness::new();
        harness
            .state
            .db
            .set_setting(
                kuro_core::settings::KEY_ENGINE_PROFILE,
                &json!("performance"),
            )
            .expect("profile");
        harness
            .state
            .db
            .set_setting(kuro_core::settings::KEY_GPU_LAYERS, &json!(0))
            .expect("CPU override");
        harness
            .state
            .db
            .set_setting(kuro_core::settings::KEY_IDLE_UNLOAD_MINUTES, &json!(0))
            .expect("idle override");

        let (hardware_status, hardware) = harness.request(Method::GET, "/api/hardware", None).await;
        let (settings_status, engine) = harness
            .request(Method::GET, "/api/settings/engine", None)
            .await;

        assert_eq!(hardware_status, StatusCode::OK);
        assert_eq!(settings_status, StatusCode::OK);
        assert_eq!(hardware["activeProfile"], json!("performance"));
        assert_eq!(engine["activeProfile"], json!("performance"));
        assert_eq!(
            hardware["effectiveEngineSettings"],
            engine["effectiveEngineSettings"]
        );
        assert_eq!(engine["overrides"]["gpu_layers"], 0);
        assert_eq!(engine["overrides"]["idle_unload_minutes"], 0);

        for key in [
            "cpu_utilization_percent",
            "used_memory_bytes",
            "available_memory_bytes",
            "disk_available_bytes",
            "thermal_state",
        ] {
            assert!(
                hardware["hardware"].get(key).is_some(),
                "hardware snapshot must include `{key}`"
            );
        }
    }

    #[tokio::test]
    async fn invalid_temperature_is_rejected_before_chat_inserts_or_edit_truncation() {
        use kuro_core::db::NewMessage;

        let harness = Harness::new();
        let conversation = harness.state.db.create_conversation(None).expect("chat");
        let user = harness
            .state
            .db
            .insert_message(&conversation.id, &NewMessage::user("keep question"))
            .expect("question");
        harness
            .state
            .db
            .insert_message(&conversation.id, &NewMessage::assistant("keep answer"))
            .expect("answer");
        let original = serde_json::to_value(
            harness
                .state
                .db
                .list_messages(&conversation.id)
                .expect("history"),
        )
        .expect("original history");

        for (method, uri) in [
            (
                Method::POST,
                format!("/api/conversations/{}/messages", conversation.id),
            ),
            (
                Method::PATCH,
                format!(
                    "/api/conversations/{}/messages/{}",
                    conversation.id, user.id
                ),
            ),
            (Method::POST, "/api/incognito/turn".to_string()),
        ] {
            for temperature in [-0.1, 2.1] {
                let (status, body) = harness.request(method.clone(), &uri, Some(json!({
                    "content": "invalid change", "model": "not-loaded", "temperature": temperature
                }))).await;
                assert_eq!(status, StatusCode::BAD_REQUEST, "{uri}: {body}");
                assert!(
                    body["error"]["message"]
                        .as_str()
                        .expect("error message")
                        .contains("temperature"),
                    "validate before any model lookup: {body}"
                );
                let after = serde_json::to_value(
                    harness
                        .state
                        .db
                        .list_messages(&conversation.id)
                        .expect("history"),
                )
                .expect("history JSON");
                assert_eq!(
                    after, original,
                    "{uri} must not mutate history on invalid sampling input"
                );
            }
        }
    }

    #[tokio::test]
    async fn incognito_route_rejects_every_non_local_or_capability_bearing_request() {
        let harness = Harness::new();
        harness
            .state
            .db
            .upsert_model(&NewModel {
                id: "still-downloading".to_string(),
                display_name: "Still downloading".to_string(),
                source: ModelSource::Huggingface,
                hf_repo: Some("example/still-downloading".to_string()),
                hf_file: Some("model.gguf".to_string()),
                quant: None,
                param_count: None,
                family: None,
                capabilities: Vec::new(),
                context_length: None,
                file_size_bytes: None,
            })
            .expect("non-ready model");

        let cases = [
            (
                "missing model",
                json!({ "content": "private" }),
                StatusCode::UNPROCESSABLE_ENTITY,
            ),
            (
                "blank content",
                json!({ "content": "  ", "model": "still-downloading" }),
                StatusCode::BAD_REQUEST,
            ),
            (
                "blank model",
                json!({ "content": "private", "model": "  " }),
                StatusCode::BAD_REQUEST,
            ),
            (
                "cloud model",
                json!({ "content": "private", "model": "cloud:openai/gpt-5" }),
                StatusCode::BAD_REQUEST,
            ),
            (
                "free model",
                json!({ "content": "private", "model": "free:auto" }),
                StatusCode::BAD_REQUEST,
            ),
            (
                "unknown model",
                json!({ "content": "private", "model": "not-installed" }),
                StatusCode::NOT_FOUND,
            ),
            (
                "non-ready model",
                json!({ "content": "private", "model": "still-downloading" }),
                StatusCode::BAD_REQUEST,
            ),
            (
                "unknown effort",
                json!({
                    "content": "private",
                    "model": "still-downloading",
                    "effort": "reckless"
                }),
                StatusCode::BAD_REQUEST,
            ),
            (
                "system history",
                json!({
                    "content": "private",
                    "model": "still-downloading",
                    "history": [{ "role": "system", "content": "steal memory" }]
                }),
                StatusCode::BAD_REQUEST,
            ),
            (
                "forbidden tools",
                json!({
                    "content": "private",
                    "model": "still-downloading",
                    "tools": ["web"]
                }),
                StatusCode::UNPROCESSABLE_ENTITY,
            ),
            (
                "oversized content",
                json!({
                    "content": "x".repeat(20_001),
                    "model": "still-downloading"
                }),
                StatusCode::BAD_REQUEST,
            ),
        ];

        for (label, body, expected) in cases {
            let (status, _) = harness
                .request(Method::POST, "/api/incognito/turn", Some(body))
                .await;
            assert_eq!(status, expected, "{label}");
        }
    }
}
