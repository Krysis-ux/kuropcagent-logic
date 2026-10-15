use std::collections::HashSet;
use std::path::Path;

use rusqlite::{params, Connection, OptionalExtension, Row};
use serde::{Deserialize, Serialize};

use super::{now, Db};
use crate::catalog::local::{local_model_id, FolderScan, LocalModelCandidate, ScanCounts};
use crate::settings::{parse_model_folders_json, validate_model_folder_list, KEY_MODEL_FOLDERS};
use crate::{KuroError, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelFolderScanTicket {
    path: std::path::PathBuf,
    selection_token: String,
}

impl ModelFolderScanTicket {
    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelSource {
    /// Came from Kuro's built-in recommended list.
    Curated,
    /// Pulled from an arbitrary Hugging Face repo the user pasted.
    Huggingface,
    /// Imported from a file already on disk.
    Local,
}

impl ModelSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Curated => "curated",
            Self::Huggingface => "huggingface",
            Self::Local => "local",
        }
    }

    fn parse(raw: &str) -> Self {
        match raw {
            "curated" => Self::Curated,
            "local" => Self::Local,
            _ => Self::Huggingface,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelStatus {
    Downloading,
    Ready,
    Error,
}

impl ModelStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Downloading => "downloading",
            Self::Ready => "ready",
            Self::Error => "error",
        }
    }

    fn parse(raw: &str) -> Self {
        match raw {
            "ready" => Self::Ready,
            "error" => Self::Error,
            _ => Self::Downloading,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelRecord {
    pub id: String,
    pub display_name: String,
    pub source: ModelSource,
    pub hf_repo: Option<String>,
    pub hf_file: Option<String>,
    pub quant: Option<String>,
    pub param_count: Option<String>,
    pub family: Option<String>,
    pub capabilities: Vec<String>,
    pub context_length: Option<i64>,
    pub file_path: Option<String>,
    pub file_size_bytes: Option<i64>,
    pub sha256: Option<String>,
    pub status: ModelStatus,
    pub error: Option<String>,
    pub added_at: String,
    pub last_used_at: Option<String>,
    pub discovered_folder: Option<String>,
}

/// Fields required to register a model before its weights exist on disk.
#[derive(Debug, Clone)]
pub struct NewModel {
    pub id: String,
    pub display_name: String,
    pub source: ModelSource,
    pub hf_repo: Option<String>,
    pub hf_file: Option<String>,
    pub quant: Option<String>,
    pub param_count: Option<String>,
    pub family: Option<String>,
    pub capabilities: Vec<String>,
    pub context_length: Option<i64>,
    pub file_size_bytes: Option<i64>,
}

pub(super) fn model_from_row(row: &Row<'_>) -> rusqlite::Result<ModelRecord> {
    let capabilities_raw: String = row.get("capabilities")?;
    let source_raw: String = row.get("source")?;
    let status_raw: String = row.get("status")?;

    Ok(ModelRecord {
        id: row.get("id")?,
        display_name: row.get("display_name")?,
        source: ModelSource::parse(&source_raw),
        hf_repo: row.get("hf_repo")?,
        hf_file: row.get("hf_file")?,
        quant: row.get("quant")?,
        param_count: row.get("param_count")?,
        family: row.get("family")?,
        capabilities: serde_json::from_str(&capabilities_raw).unwrap_or_default(),
        context_length: row.get("context_length")?,
        file_path: row.get("file_path")?,
        file_size_bytes: row.get("file_size_bytes")?,
        sha256: row.get("sha256")?,
        status: ModelStatus::parse(&status_raw),
        error: row.get("error")?,
        added_at: row.get("added_at")?,
        last_used_at: row.get("last_used_at")?,
        discovered_folder: row.get("discovered_folder")?,
    })
}

impl Db {
    /// Register a model, replacing any previous row with the same id.
    pub fn upsert_model(&self, model: &NewModel) -> Result<()> {
        self.with(|conn| upsert_model_on(conn, model))
    }

    pub fn get_model(&self, id: &str) -> Result<Option<ModelRecord>> {
        self.with(|conn| get_model_on(conn, id))
    }

    pub fn list_models(&self) -> Result<Vec<ModelRecord>> {
        self.with(|conn| {
            let mut stmt = conn.prepare("SELECT * FROM models ORDER BY added_at DESC")?;
            let rows = stmt
                .query_map([], model_from_row)?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            Ok(rows)
        })
    }

    /// Mark a model usable and record where its weights ended up.
    pub fn set_model_ready(
        &self,
        id: &str,
        file_path: &str,
        file_size_bytes: i64,
        sha256: &str,
    ) -> Result<()> {
        self.with(|conn| {
            conn.execute(
                "UPDATE models
                    SET status = 'ready', file_path = ?2, file_size_bytes = ?3,
                        sha256 = ?4, error = NULL
                  WHERE id = ?1",
                params![id, file_path, file_size_bytes, sha256],
            )?;
            Ok(())
        })
    }

    pub fn set_model_error(&self, id: &str, error: &str) -> Result<()> {
        self.with(|conn| {
            conn.execute(
                "UPDATE models SET status = 'error', error = ?2 WHERE id = ?1",
                params![id, error],
            )?;
            Ok(())
        })
    }

    pub fn touch_model_used(&self, id: &str) -> Result<()> {
        self.with(|conn| {
            conn.execute(
                "UPDATE models SET last_used_at = ?2 WHERE id = ?1",
                params![id, now()],
            )?;
            Ok(())
        })
    }

    pub fn delete_model(&self, id: &str) -> Result<()> {
        self.with(|conn| {
            conn.execute("DELETE FROM models WHERE id = ?1", params![id])?;
            Ok(())
        })
    }

    /// Atomically persist a selected folder and its first successful scan.
    pub fn commit_model_folder_add(&self, scan: &FolderScan) -> Result<ScanCounts> {
        let folder = validate_local_scan(scan)?;
        scan.verify_root_binding()?;

        self.with(|conn| {
            let transaction = conn.unchecked_transaction()?;
            let mut folders = model_folders_on(&transaction)?;
            validate_online_model_folders(&transaction, &folders)?;
            let validated =
                crate::catalog::local::validate_model_folder_root(Path::new(folder), &folders)?;
            if validated != Path::new(folder) {
                return Err(KuroError::bad_request(format!(
                    "model folder `{folder}` changed during scanning"
                )));
            }
            scan.verify_root_binding()?;
            insert_new_model_folder_lifetime(&transaction, folder)?;
            folders.push(folder.to_string());
            set_model_folders_on(&transaction, &folders)?;
            let counts = reconcile_scanned_folder_on(&transaction, scan)?;
            transaction.commit()?;
            Ok(counts)
        })
    }

    /// Atomically remove a selected folder and only the rows it owns.
    pub fn commit_model_folder_remove(&self, folder: &str) -> Result<ScanCounts> {
        self.with(|conn| {
            let transaction = conn.unchecked_transaction()?;
            let mut folders = model_folders_on(&transaction)?;
            let Some(index) = folders.iter().position(|selected| selected == folder) else {
                return Err(KuroError::bad_request(format!(
                    "model folder `{folder}` is not selected"
                )));
            };
            folders.remove(index);
            validate_online_model_folders(&transaction, &folders)?;
            set_model_folders_on(&transaction, &folders)?;
            let unregistered = unregister_scanned_folder_on(&transaction, folder)?;
            transaction.execute(
                "DELETE FROM model_folder_roots WHERE path = ?1",
                params![folder],
            )?;
            transaction.commit()?;
            Ok(ScanCounts {
                unregistered,
                ..ScanCounts::default()
            })
        })
    }

    /// Snapshot ordered selected-folder lifetimes before starting filesystem work.
    pub fn model_folder_scan_tickets(&self) -> Result<Vec<ModelFolderScanTicket>> {
        self.with(|conn| {
            let folders = model_folders_on(conn)?;
            let mut tickets = Vec::with_capacity(folders.len());
            for folder in folders {
                let selection_token = conn
                    .query_row(
                        "SELECT selection_token FROM model_folder_roots WHERE path = ?1",
                        params![folder],
                        |row| row.get::<_, String>(0),
                    )
                    .optional()?
                    .ok_or_else(|| {
                        KuroError::other(format!(
                            "selected model folder `{folder}` is missing its scan token"
                        ))
                    })?;
                tickets.push(ModelFolderScanTicket {
                    path: Path::new(&folder).to_path_buf(),
                    selection_token,
                });
            }
            Ok(tickets)
        })
    }

    /// Reconcile only against the exact selection lifetime captured before scanning.
    pub fn reconcile_scanned_folder(
        &self,
        ticket: &ModelFolderScanTicket,
        scan: &FolderScan,
    ) -> Result<ScanCounts> {
        let folder = validate_local_scan(scan)?;
        if Path::new(folder) != ticket.path() {
            return Err(KuroError::bad_request(
                "model folder scan does not match its selection ticket",
            ));
        }
        scan.verify_root_binding()?;
        self.with(|conn| {
            let transaction = conn.unchecked_transaction()?;
            let folders = model_folders_on(&transaction)?;
            if !folders.iter().any(|selected| selected == folder) {
                return Err(KuroError::bad_request(format!(
                    "model folder `{folder}` is no longer selected"
                )));
            }
            let current_token = transaction
                .query_row(
                    "SELECT selection_token FROM model_folder_roots WHERE path = ?1",
                    params![folder],
                    |row| row.get::<_, String>(0),
                )
                .optional()?;
            if current_token.as_deref() != Some(ticket.selection_token.as_str()) {
                return Err(KuroError::bad_request(format!(
                    "model folder `{folder}` selection changed while it was scanned"
                )));
            }
            record_validated_model_folder(&transaction, folder)?;
            let counts = reconcile_scanned_folder_on(&transaction, scan)?;
            transaction.commit()?;
            Ok(counts)
        })
    }

    /// Unregister scanner-owned rows for one selected root without touching files.
    pub fn unregister_scanned_folder(&self, folder: &str) -> Result<u32> {
        self.with(|conn| unregister_scanned_folder_on(conn, folder))
    }

    /// Remove one model registration without touching any file on disk.
    pub fn unregister_model(&self, id: &str) -> Result<u32> {
        self.with(|conn| {
            let removed = conn.execute("DELETE FROM models WHERE id = ?1", params![id])?;
            Ok(removed as u32)
        })
    }
}

pub(super) fn get_model_on(conn: &Connection, id: &str) -> Result<Option<ModelRecord>> {
    Ok(conn
        .query_row(
            "SELECT * FROM models WHERE id = ?1",
            params![id],
            model_from_row,
        )
        .optional()?)
}

pub(super) fn upsert_model_on(conn: &Connection, model: &NewModel) -> Result<()> {
    let capabilities = serde_json::to_string(&model.capabilities)?;
    conn.execute(
        "INSERT INTO models (
             id, display_name, source, hf_repo, hf_file, quant, param_count,
             family, capabilities, context_length, file_size_bytes, status, added_at
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, 'downloading', ?12)
         ON CONFLICT (id) DO UPDATE SET
             display_name = excluded.display_name, source = excluded.source,
             hf_repo = excluded.hf_repo, hf_file = excluded.hf_file,
             quant = excluded.quant, param_count = excluded.param_count,
             family = excluded.family, capabilities = excluded.capabilities,
             context_length = excluded.context_length,
             file_size_bytes = excluded.file_size_bytes,
             status = 'downloading', error = NULL",
        params![
            model.id,
            model.display_name,
            model.source.as_str(),
            model.hf_repo,
            model.hf_file,
            model.quant,
            model.param_count,
            model.family,
            capabilities,
            model.context_length,
            model.file_size_bytes,
            now()
        ],
    )?;
    Ok(())
}

/// Restore a cancelled replacement inside the download finalization transaction.
pub(super) fn restore_model_snapshot_on(conn: &Connection, model: &ModelRecord) -> Result<()> {
    let capabilities = serde_json::to_string(&model.capabilities)?;
    conn.execute(
        "UPDATE models SET
             display_name = ?2, source = ?3, hf_repo = ?4, hf_file = ?5,
             quant = ?6, param_count = ?7, family = ?8, capabilities = ?9,
             context_length = ?10, file_path = ?11, file_size_bytes = ?12,
             sha256 = ?13, status = ?14, error = ?15, added_at = ?16,
             last_used_at = ?17, discovered_folder = ?18
           WHERE id = ?1",
        params![
            model.id,
            model.display_name,
            model.source.as_str(),
            model.hf_repo,
            model.hf_file,
            model.quant,
            model.param_count,
            model.family,
            capabilities,
            model.context_length,
            model.file_path,
            model.file_size_bytes,
            model.sha256,
            model.status.as_str(),
            model.error,
            model.added_at,
            model.last_used_at,
            model.discovered_folder
        ],
    )?;
    Ok(())
}

fn validate_local_scan(scan: &FolderScan) -> Result<&str> {
    let folder = scan.committable_root()?.to_str().ok_or_else(|| {
        KuroError::bad_request("model folder path is not valid UTF-8 and cannot be stored")
    })?;
    validate_model_folder_list(&[folder.to_string()])?;
    validate_local_candidates(folder, &scan.candidates)?;
    Ok(folder)
}

fn validate_local_candidates(folder: &str, candidates: &[LocalModelCandidate]) -> Result<()> {
    let folder_path = Path::new(folder);
    if !folder_path.is_absolute() {
        return Err(KuroError::bad_request(format!(
            "model folder `{folder}` is not canonical"
        )));
    }
    let mut ids = HashSet::with_capacity(candidates.len());
    for candidate in candidates {
        let candidate_text = candidate.canonical_path.to_str().ok_or_else(|| {
            KuroError::bad_request("local model path is not valid UTF-8 and cannot be stored")
        })?;
        if candidate_text
            .split(['/', '\\'])
            .any(|component| matches!(component, "." | ".."))
        {
            return Err(KuroError::bad_request(format!(
                "local model path `{candidate_text}` is not canonical"
            )));
        }
        if !candidate.canonical_path.starts_with(folder_path) {
            return Err(KuroError::bad_request(format!(
                "local model `{}` is outside its discovered folder",
                candidate.canonical_path.display()
            )));
        }
        if candidate.id != local_model_id(&candidate.canonical_path) {
            return Err(KuroError::bad_request(format!(
                "local model id does not match `{}`",
                candidate.canonical_path.display()
            )));
        }
        if !ids.insert(candidate.id.as_str()) {
            return Err(KuroError::bad_request(format!(
                "local model `{}` appears more than once in one scan",
                candidate.id
            )));
        }
    }
    Ok(())
}

fn model_folders_on(conn: &Connection) -> Result<Vec<String>> {
    let raw = conn
        .query_row(
            "SELECT value FROM settings WHERE key = ?1",
            params![KEY_MODEL_FOLDERS],
            |row| row.get::<_, String>(0),
        )
        .optional()?;
    raw.map_or_else(|| Ok(Vec::new()), |value| parse_model_folders_json(&value))
}

fn validate_online_model_folders(conn: &Connection, folders: &[String]) -> Result<()> {
    for folder in folders {
        let path = Path::new(folder);
        let was_validated = conn
            .query_row(
                "SELECT 1 FROM model_folder_roots WHERE path = ?1",
                params![folder],
                |_| Ok(()),
            )
            .optional()?
            .is_some();
        let metadata = match std::fs::symlink_metadata(path) {
            Ok(metadata) => metadata,
            Err(_) if was_validated => continue,
            Err(error) => {
                return Err(KuroError::bad_request(format!(
                    "stored model folder `{folder}` is unavailable and has no validated provenance: {error}"
                )))
            }
        };
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err(KuroError::bad_request(format!(
                "stored model folder `{folder}` is not a canonical directory"
            )));
        }
        let canonical = path.canonicalize().map_err(|error| {
            KuroError::bad_request(format!(
                "stored model folder `{folder}` could not be canonicalized: {error}"
            ))
        })?;
        if canonical != path {
            return Err(KuroError::bad_request(format!(
                "stored model folder `{folder}` is not canonical"
            )));
        }
        if !was_validated {
            record_validated_model_folder(conn, folder)?;
        }
    }
    Ok(())
}

fn record_validated_model_folder(conn: &Connection, folder: &str) -> Result<()> {
    let selection_token = uuid::Uuid::new_v4().simple().to_string();
    conn.execute(
        "INSERT INTO model_folder_roots (path, selection_token, validated_at) VALUES (?1, ?2, ?3)
         ON CONFLICT (path) DO UPDATE SET validated_at = excluded.validated_at",
        params![folder, selection_token, now()],
    )?;
    Ok(())
}

fn insert_new_model_folder_lifetime(conn: &Connection, folder: &str) -> Result<()> {
    let selection_token = uuid::Uuid::new_v4().simple().to_string();
    conn.execute(
        "INSERT INTO model_folder_roots (path, selection_token, validated_at)
         VALUES (?1, ?2, ?3)",
        params![folder, selection_token, now()],
    )?;
    Ok(())
}

fn set_model_folders_on(conn: &Connection, folders: &[String]) -> Result<()> {
    let folder_json = serde_json::to_string(folders)?;
    conn.execute(
        "INSERT INTO settings (key, value, updated_at) VALUES (?1, ?2, ?3)
         ON CONFLICT (key) DO UPDATE SET value = excluded.value,
                                         updated_at = excluded.updated_at",
        params![KEY_MODEL_FOLDERS, folder_json, now()],
    )?;
    Ok(())
}

fn reconcile_scanned_folder_on(conn: &Connection, scan: &FolderScan) -> Result<ScanCounts> {
    let folder = scan
        .committable_root()?
        .to_str()
        .expect("scan roots were validated as UTF-8");
    let candidates = &scan.candidates;
    let candidate_ids: HashSet<&str> = candidates
        .iter()
        .map(|candidate| candidate.id.as_str())
        .collect();
    let owned_ids = {
        let mut stmt = conn
            .prepare("SELECT id FROM models WHERE source = 'local' AND discovered_folder = ?1")?;
        let ids = stmt
            .query_map(params![folder], |row| row.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        ids
    };

    let mut counts = scan.counts.clone();
    if scan.is_complete() {
        for id in owned_ids {
            if !candidate_ids.contains(id.as_str()) {
                counts.unregistered = counts.unregistered.saturating_add(
                    conn.execute("DELETE FROM models WHERE id = ?1", params![id])? as u32,
                );
            }
        }
    }

    for candidate in candidates {
        let existing = conn
            .query_row(
                "SELECT source, discovered_folder FROM models WHERE id = ?1",
                params![candidate.id],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?)),
            )
            .optional()?;
        if let Some((source, owner)) = &existing {
            if source != "local" || owner.as_deref() != Some(folder) {
                counts.conflicts = counts.conflicts.saturating_add(1);
                continue;
            }
        }
        let path = candidate
            .canonical_path
            .to_str()
            .expect("candidate paths were validated as UTF-8");
        if existing.is_some() {
            conn.execute(
                "UPDATE models SET
                     display_name = ?2, quant = ?3, param_count = ?4,
                     file_path = ?5, file_size_bytes = ?6, sha256 = NULL,
                     status = 'ready', error = NULL
                 WHERE id = ?1 AND source = 'local' AND discovered_folder = ?7",
                params![
                    candidate.id,
                    candidate.display_name,
                    candidate.quant,
                    candidate.param_count,
                    path,
                    candidate.file_size_bytes,
                    folder,
                ],
            )?;
            counts.updated = counts.updated.saturating_add(1);
        } else {
            conn.execute(
                "INSERT INTO models (
                 id, display_name, source, hf_repo, hf_file, quant, param_count,
                 family, capabilities, context_length, file_path, file_size_bytes,
                 sha256, status, error, added_at, discovered_folder
             ) VALUES (
                 ?1, ?2, 'local', NULL, NULL, ?3, ?4,
                 NULL, '[]', NULL, ?5, ?6,
                 NULL, 'ready', NULL, ?7, ?8
             )",
                params![
                    candidate.id,
                    candidate.display_name,
                    candidate.quant,
                    candidate.param_count,
                    path,
                    candidate.file_size_bytes,
                    now(),
                    folder,
                ],
            )?;
            counts.registered = counts.registered.saturating_add(1);
        }
    }
    Ok(counts)
}

fn unregister_scanned_folder_on(conn: &Connection, folder: &str) -> Result<u32> {
    let removed = conn.execute(
        "DELETE FROM models WHERE source = 'local' AND discovered_folder = ?1",
        params![folder],
    )?;
    Ok(removed as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::scan_model_folder;

    fn sample(id: &str) -> NewModel {
        NewModel {
            id: id.to_string(),
            display_name: "Qwen3 4B Instruct".to_string(),
            source: ModelSource::Curated,
            hf_repo: Some("unsloth/Qwen3-4B-Instruct-2507-GGUF".to_string()),
            hf_file: Some("Qwen3-4B-Instruct-2507-Q4_K_M.gguf".to_string()),
            quant: Some("Q4_K_M".to_string()),
            param_count: Some("4B".to_string()),
            family: Some("qwen3".to_string()),
            capabilities: vec!["tools".to_string()],
            context_length: Some(32768),
            file_size_bytes: Some(2_500_000_000),
        }
    }

    #[test]
    fn round_trips_a_model_through_its_lifecycle() {
        let db = Db::open_in_memory().expect("open");
        db.upsert_model(&sample("qwen3-4b:q4_k_m")).expect("insert");

        let stored = db.get_model("qwen3-4b:q4_k_m").expect("get").expect("some");
        assert_eq!(stored.status, ModelStatus::Downloading);
        assert_eq!(stored.capabilities, vec!["tools".to_string()]);
        assert_eq!(stored.source, ModelSource::Curated);

        db.set_model_ready(
            "qwen3-4b:q4_k_m",
            "/tmp/model.gguf",
            2_500_000_000,
            "abc123",
        )
        .expect("ready");
        let stored = db.get_model("qwen3-4b:q4_k_m").expect("get").expect("some");
        assert_eq!(stored.status, ModelStatus::Ready);
        assert_eq!(stored.file_path.as_deref(), Some("/tmp/model.gguf"));

        db.delete_model("qwen3-4b:q4_k_m").expect("delete");
        assert!(db.get_model("qwen3-4b:q4_k_m").expect("get").is_none());
    }

    #[test]
    fn re_pulling_a_model_resets_it_to_downloading() {
        let db = Db::open_in_memory().expect("open");
        db.upsert_model(&sample("qwen3-4b:q4_k_m")).expect("insert");
        db.set_model_error("qwen3-4b:q4_k_m", "disk full")
            .expect("error");

        db.upsert_model(&sample("qwen3-4b:q4_k_m"))
            .expect("re-insert");

        let stored = db.get_model("qwen3-4b:q4_k_m").expect("get").expect("some");
        assert_eq!(stored.status, ModelStatus::Downloading);
        assert!(stored.error.is_none(), "stale error must be cleared");
        assert_eq!(db.list_models().expect("list").len(), 1);
    }

    #[test]
    fn scanner_reconcile_does_not_seize_a_row_owned_by_another_folder() {
        let root =
            std::env::temp_dir().join(format!("kuro-model-owner-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).expect("create root");
        let root = root.canonicalize().expect("canonical root");
        std::fs::write(root.join("model.gguf"), b"model").expect("write model");
        let scan = scan_model_folder(&root).expect("scan");
        let id = scan.candidates[0].id.clone();
        let db = Db::open_in_memory().expect("db");
        db.with(|conn| reconcile_scanned_folder_on(conn, &scan))
            .expect("initial registration");
        db.with(|conn| {
            conn.execute(
                "UPDATE models SET discovered_folder = '/different/root' WHERE id = ?1",
                params![id],
            )?;
            Ok(())
        })
        .expect("change owner fixture");

        let counts = db
            .with(|conn| reconcile_scanned_folder_on(conn, &scan))
            .expect("conflict reconcile");
        let stored = db.get_model(&id).expect("lookup").expect("stored row");

        assert_eq!(counts.conflicts, 1);
        assert_eq!(stored.discovered_folder.as_deref(), Some("/different/root"));
        let _ = std::fs::remove_dir_all(root);
    }
}
