//! Coding workspaces: the Code page's API.

use std::path::PathBuf;

use axum::extract::{Path, Query, State};
use axum::Json;
use kuro_core::db::UndoPlan;
use kuro_core::tools::files;
use kuro_core::workspace::{self, Workspace, WorkspaceMode};
use kuro_core::KuroError;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::error::AppResult;
use crate::state::SharedState;

/// Changes kept in the panel. Enough to cover a long session, few enough that
/// the response stays small.
const CHANGE_HISTORY: usize = 200;

pub async fn list_workspaces(State(state): State<SharedState>) -> AppResult<Json<Value>> {
    Ok(Json(json!({
        "workspaces": state.db.list_workspaces()?,
        "modes": modes_json(),
        "tools": workspace::tools::describe_tools(),
    })))
}

#[derive(Debug, Deserialize)]
pub struct CreateRequest {
    pub name: String,
    #[serde(alias = "rootPath")]
    pub root_path: String,
}

/// Create a workspace over an existing folder.
pub async fn create_workspace(
    State(state): State<SharedState>,
    Json(request): Json<CreateRequest>,
) -> AppResult<Json<Value>> {
    let expanded = expand_home(request.root_path.trim());

    if !expanded.exists() {
        return Err(KuroError::bad_request(format!(
            "`{}` does not exist.",
            request.root_path.trim()
        ))
        .into());
    }
    if !expanded.is_dir() {
        return Err(KuroError::bad_request(format!(
            "`{}` is a file. Choose the folder that contains it.",
            request.root_path.trim()
        ))
        .into());
    }

    // Stored canonicalised so that a workspace cannot be pointed at one folder
    // and resolved against another through a symlink that changes later.
    let root = expanded.canonicalize().map_err(|error| {
        KuroError::bad_request(format!("that folder could not be read: {error}"))
    })?;

    let created = state
        .db
        .create_workspace(request.name.trim(), &root.to_string_lossy())?;

    // The stored default rather than the type's default.
    let preferred = kuro_core::settings::default_workspace_mode(&state.db)?;
    let created = if preferred == WorkspaceMode::parse(&created.mode).unwrap_or_default() {
        created
    } else {
        state
            .db
            .update_workspace(&created.id, None, Some(preferred), None)?
    };

    Ok(Json(json!(created)))
}

pub async fn get_workspace(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> AppResult<Json<Value>> {
    let workspace = load(&state, &id)?;
    Ok(Json(json!({
        "workspace": workspace,
        "conversations": state.db.list_workspace_conversations(&id)?,
    })))
}

#[derive(Debug, Deserialize)]
pub struct UpdateRequest {
    #[serde(default)]
    pub name: Option<String>,
    /// `ask` | `plan` | `agent`.
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default, alias = "modelId")]
    pub model_id: Option<Option<String>>,
}

pub async fn update_workspace(
    State(state): State<SharedState>,
    Path(id): Path<String>,
    Json(request): Json<UpdateRequest>,
) -> AppResult<Json<Value>> {
    let mode = match &request.mode {
        Some(raw) => Some(WorkspaceMode::parse(raw).ok_or_else(|| {
            KuroError::bad_request(format!("unknown mode `{raw}`. Use ask, plan or agent."))
        })?),
        None => None,
    };

    let updated = state.db.update_workspace(
        &id,
        request.name.as_deref(),
        mode,
        request.model_id.as_ref().map(|held| held.as_deref()),
    )?;
    Ok(Json(json!(updated)))
}

pub async fn delete_workspace(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> AppResult<Json<Value>> {
    let stopped_processes = state.processes.stop_all(&id);
    state.db.delete_workspace(&id)?;
    // Said plainly because "delete" on something describing a folder is a
    // reasonable thing to be nervous about.
    Ok(Json(json!({
        "deleted": true,
        "folderUntouched": true,
        "stoppedProcesses": stopped_processes,
    })))
}

/// Start a chat inside a workspace.
pub async fn create_conversation(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> AppResult<Json<Value>> {
    let workspace = load(&state, &id)?;
    let conversation = state
        .db
        .create_conversation(workspace.model_id.as_deref())?;
    state
        .db
        .set_conversation_workspace(&conversation.id, Some(&id))?;

    let attached = state
        .db
        .get_conversation(&conversation.id)?
        .ok_or_else(|| KuroError::other("conversation vanished after creation"))?;
    Ok(Json(json!(attached)))
}

pub async fn workspace_tree(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> AppResult<Json<Value>> {
    let record = load(&state, &id)?;
    let root = PathBuf::from(&record.root_path);

    if !root.is_dir() {
        return Err(KuroError::not_found(format!("the folder `{}`", record.root_path)).into());
    }

    Ok(Json(json!({ "entries": workspace::search::tree(&root)? })))
}

#[derive(Debug, Deserialize)]
pub struct FileQuery {
    pub path: String,
}

/// Read one file for the viewer.
pub async fn read_workspace_file(
    State(state): State<SharedState>,
    Path(id): Path<String>,
    Query(query): Query<FileQuery>,
) -> AppResult<Json<Value>> {
    let workspace = as_workspace(&load(&state, &id)?, WorkspaceMode::Plan);
    let resolved = workspace.permissions().resolve_path(&query.path, false)?;
    let content = files::read_file(&resolved)?;

    Ok(Json(json!({ "path": query.path, "content": content })))
}

pub async fn list_changes(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> AppResult<Json<Value>> {
    load(&state, &id)?;
    let changes = state.db.list_workspace_changes(&id, CHANGE_HISTORY)?;

    // `before`/`after` are whole file contents.
    let summarised: Vec<Value> = changes
        .iter()
        .map(|change| {
            json!({
                "id": change.id,
                "path": change.path,
                "kind": change.kind,
                "conversationId": change.conversation_id,
                "createdAt": change.created_at,
                "undone": change.undone,
                "undoable": change.is_undoable(),
                "created": change.before.is_none(),
                "beforeLines": change.before.as_deref().map(count_lines),
                "afterLines": change.after.as_deref().map(count_lines),
            })
        })
        .collect();

    Ok(Json(json!({ "changes": summarised })))
}

/// One change, with the contents on both sides of it.
pub async fn change_diff(
    State(state): State<SharedState>,
    Path((id, change_id)): Path<(String, String)>,
) -> AppResult<Json<Value>> {
    load(&state, &id)?;

    let change = state
        .db
        .list_workspace_changes(&id, CHANGE_HISTORY)?
        .into_iter()
        .find(|held| held.id == change_id)
        .ok_or_else(|| KuroError::not_found(format!("change `{change_id}`")))?;

    Ok(Json(json!({
        "id": change.id,
        "path": change.path,
        "kind": change.kind,
        "createdAt": change.created_at,
        "undone": change.undone,
        "created": change.before.is_none(),
        // Null rather than empty when the snapshot was skipped for size.
        "before": change.before,
        "after": change.after,
    })))
}

/// Put a change back.
pub async fn undo_change(
    State(state): State<SharedState>,
    Path((id, change_id)): Path<(String, String)>,
) -> AppResult<Json<Value>> {
    let record = load(&state, &id)?;
    let change = state
        .db
        .get_workspace_change(&change_id)?
        .ok_or_else(|| KuroError::not_found(format!("change `{change_id}`")))?;

    if change.workspace_id != id {
        return Err(KuroError::bad_request("that change belongs to another workspace").into());
    }

    // Agent access, because undoing is a write. The path still goes through the
    // same containment as everything else.
    let workspace = as_workspace(&record, WorkspaceMode::Agent);
    let resolved = workspace.permissions().resolve_path(&change.path, true)?;
    let current = files::read_file(&resolved).unwrap_or_default();

    match change.plan_undo(&current) {
        UndoPlan::Restore(before) => {
            files::write_file(&resolved, &before)?;
        }
        UndoPlan::Remove => {
            std::fs::remove_file(&resolved).map_err(|error| {
                KuroError::other(format!("could not remove `{}`: {error}", change.path))
            })?;
        }
        UndoPlan::Refused(reason) => return Err(KuroError::bad_request(reason).into()),
    }

    state.db.mark_change_undone(&change_id)?;
    Ok(Json(json!({
        "undone": true,
        "path": change.path,
        "removed": change.before.is_none(),
    })))
}

/* ---------- Background processes ---------- */

/// Lines of a process's output returned at once.
const LOG_LINES: usize = 400;

/// What is running in this workspace, and what each one is serving.
pub async fn list_processes(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> AppResult<Json<Value>> {
    load(&state, &id)?;
    Ok(Json(json!({ "processes": state.processes.list(&id) })))
}

#[derive(Debug, Deserialize)]
pub struct StartProcessRequest {
    pub command: String,
}

/// Start a command from the interface rather than from a model.
pub async fn start_process(
    State(state): State<SharedState>,
    Path(id): Path<String>,
    Json(request): Json<StartProcessRequest>,
) -> AppResult<Json<Value>> {
    let record = load(&state, &id)?;
    let mode = WorkspaceMode::parse(&record.mode).unwrap_or_default();

    if !mode.allows(kuro_core::workspace::ToolRisk::Execute) {
        return Err(KuroError::bad_request(format!(
            "{} mode does not run commands. Switch to Agent to run build and test commands.",
            mode.label()
        ))
        .into());
    }

    kuro_core::workspace::exec::vet(&request.command, mode.restricts_commands())
        .map_err(|refusal| KuroError::bad_request(refusal.to_string()))?;

    let root = PathBuf::from(&record.root_path);
    if !root.is_dir() {
        return Err(KuroError::not_found(format!("the folder `{}`", record.root_path)).into());
    }

    let started = state
        .processes
        .start(&id, &root, request.command.trim())
        .await
        .map_err(KuroError::bad_request)?;

    Ok(Json(json!({ "process": started })))
}

pub async fn process_log(
    State(state): State<SharedState>,
    Path((id, process_id)): Path<(String, String)>,
) -> AppResult<Json<Value>> {
    load(&state, &id)?;

    let found = state
        .processes
        .get(&process_id)
        .filter(|held| held.workspace_id == id)
        .ok_or_else(|| KuroError::not_found(format!("process `{process_id}`")))?;

    Ok(Json(json!({
        "process": found,
        "lines": state.processes.log(&process_id, LOG_LINES).unwrap_or_default(),
    })))
}

pub async fn stop_process(
    State(state): State<SharedState>,
    Path((id, process_id)): Path<(String, String)>,
) -> AppResult<Json<Value>> {
    load(&state, &id)?;

    let found = state
        .processes
        .get(&process_id)
        .filter(|held| held.workspace_id == id)
        .ok_or_else(|| KuroError::not_found(format!("process `{process_id}`")))?;

    let stopped = state.processes.stop(&process_id);
    Ok(Json(
        json!({ "stopped": stopped, "command": found.command }),
    ))
}

/// Take a finished process off the list.
pub async fn forget_process(
    State(state): State<SharedState>,
    Path((id, process_id)): Path<(String, String)>,
) -> AppResult<Json<Value>> {
    load(&state, &id)?;

    let found = state
        .processes
        .get(&process_id)
        .filter(|held| held.workspace_id == id)
        .ok_or_else(|| KuroError::not_found(format!("process `{process_id}`")))?;

    if found.running {
        return Err(KuroError::other(format!(
            "`{}` is still running. Stop it before clearing it.",
            found.command
        ))
        .into());
    }

    Ok(Json(
        json!({ "forgotten": state.processes.forget(&process_id) }),
    ))
}

/// Take every finished process off one workspace's list.
pub async fn clear_processes(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> AppResult<Json<Value>> {
    load(&state, &id)?;
    Ok(Json(
        json!({ "cleared": state.processes.forget_finished(&id) }),
    ))
}

fn load(state: &SharedState, id: &str) -> Result<kuro_core::db::WorkspaceRecord, KuroError> {
    state
        .db
        .get_workspace(id)?
        .ok_or_else(|| KuroError::not_found(format!("workspace `{id}`")))
}

/// The enforcement object for a stored workspace, at a chosen mode.
fn as_workspace(record: &kuro_core::db::WorkspaceRecord, mode: WorkspaceMode) -> Workspace {
    Workspace {
        id: record.id.clone(),
        root: PathBuf::from(&record.root_path),
        mode,
    }
}

fn modes_json() -> Vec<Value> {
    WorkspaceMode::ALL
        .iter()
        .map(|mode| {
            json!({
                "id": mode.as_str(),
                "label": mode.label(),
                "blurb": mode.blurb(),
                "tools": workspace::tools::tools_for_mode(*mode)
                    .iter()
                    .map(|tool| tool.name())
                    .collect::<Vec<_>>(),
            })
        })
        .collect()
}

fn count_lines(text: &str) -> usize {
    text.lines().count()
}

/// Expand a leading `~`, so a typed path matches what the tools will use.
fn expand_home(raw: &str) -> PathBuf {
    let Some(rest) = raw.strip_prefix('~') else {
        return PathBuf::from(raw);
    };
    let Some(home) = std::env::var_os("HOME").map(PathBuf::from) else {
        return PathBuf::from(raw);
    };
    let rest = rest.trim_start_matches(['/', '\\']);
    if rest.is_empty() {
        home
    } else {
        home.join(rest)
    }
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
    use kuro_core::{Paths, SecretStore};
    use tokio::sync::Mutex;

    use super::*;
    use crate::state::AppState;

    fn test_state(db: Db, processes: ProcessRegistry) -> SharedState {
        let root =
            std::env::temp_dir().join(format!("kuro-workspace-route-{}", uuid::Uuid::new_v4()));
        let paths = Paths::for_root(&root);
        let hardware = kuro_core::hardware::detect();
        let secrets = SecretStore::new(paths.credentials_file());
        let outbound = reqwest::Client::new();
        let engines = Arc::new(
            EngineManager::new(db.clone(), paths.clone(), hardware.clone()).expect("engines"),
        );
        let live_hardware = kuro_core::hardware::LiveHardwareMetrics::new(paths.models_dir());

        Arc::new(AppState {
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
            processes,
            started_at: chrono::Utc::now(),
            port: 0,
            download_cancels: Mutex::new(HashMap::<String, Arc<AtomicBool>>::new()),
        })
    }

    #[tokio::test]
    async fn deleting_a_workspace_stops_its_live_processes_and_reports_the_count() {
        let db = Db::open_in_memory().expect("database");
        let workspace = db.create_workspace("Test", "/tmp").expect("workspace");
        let processes = ProcessRegistry::new();

        for _ in 0..2 {
            processes
                .start(&workspace.id, &std::env::temp_dir(), "sleep 60")
                .await
                .expect("start process");
        }

        let state = test_state(db.clone(), processes.clone());
        let response = match delete_workspace(State(state), Path(workspace.id.clone())).await {
            Ok(Json(response)) => response,
            Err(_) => panic!("delete route failed"),
        };

        assert_eq!(response["stoppedProcesses"], json!(2));
        assert_eq!(response["folderUntouched"], json!(true));
        assert!(db.get_workspace(&workspace.id).expect("lookup").is_none());

        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        assert!(
            processes
                .list(&workspace.id)
                .iter()
                .all(|process| !process.running),
            "workspace deletion must ask every live process to stop"
        );
    }
}
