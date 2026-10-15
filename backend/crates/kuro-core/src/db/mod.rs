//! SQLite storage.

use std::path::Path;
use std::sync::{Arc, Mutex};

use rusqlite::Connection;

use crate::{KuroError, Result};

mod cloud;
mod conversations;
mod downloads;
mod kv;
mod mcp;
mod memories;
mod models;
mod projects;
mod runtimes;
mod skills;
mod usage;
mod workspaces;

pub use cloud::{CloudConnectorRecord, CloudStatus, NewCloudConnector};
pub use conversations::{Conversation, Message, MessageCompletion, NewMessage};
pub use downloads::{DownloadKind, DownloadRecord, DownloadStatus};
pub use mcp::{McpServerRecord, McpStatus, McpTransport, NewMcpServer};
pub use memories::MemoryRecord;
pub use models::{ModelFolderScanTicket, ModelRecord, ModelSource, ModelStatus, NewModel};
pub use projects::{NewProject, ProjectRecord, ProjectUpdate};
pub use runtimes::EngineRuntimeRecord;
pub use skills::{UserSkillRecord, MAX_INSTRUCTION_CHARS as MAX_USER_SKILL_CHARS};
pub use usage::ProviderUsage;
pub use workspaces::{UndoPlan, WorkspaceChange, WorkspaceRecord};

const SCHEMA: &str = include_str!("schema.sql");

/// Indexes over columns that `add_column_if_missing` may have only just created.
/// Run after that step; see the ordering note in [`Db::migrate`].
const LATE_INDEXES: &str = "
    CREATE INDEX IF NOT EXISTS idx_conversations_project
        ON conversations (project_id, updated_at DESC);
    CREATE INDEX IF NOT EXISTS idx_conversations_workspace
        ON conversations (workspace_id, updated_at DESC);
    -- Partial, so it covers only the turns that went to a provider. A machine
    -- full of local chats pays almost nothing for it.
    CREATE INDEX IF NOT EXISTS idx_messages_usage
        ON messages (provider_slug, created_at)
        WHERE provider_slug IS NOT NULL;
";
// Bumped to 10 when selected model-folder lifetimes gained opaque scan tokens.
const SCHEMA_VERSION: i32 = 10;

#[derive(Clone)]
pub struct Db {
    conn: Arc<Mutex<Connection>>,
}

impl Db {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path)?;
        Self::from_connection(conn)
    }

    /// In-memory database, used by tests.
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        Self::from_connection(conn)
    }

    fn from_connection(conn: Connection) -> Result<Self> {
        // WAL keeps reads from blocking the writer, which matters while a large
        // download is checkpointing progress during an active chat.
        let _mode: String = conn.query_row("PRAGMA journal_mode = WAL", [], |row| row.get(0))?;
        conn.execute_batch(
            "PRAGMA foreign_keys = ON;
             PRAGMA synchronous = NORMAL;
             PRAGMA busy_timeout = 5000;",
        )?;

        let db = Self {
            conn: Arc::new(Mutex::new(conn)),
        };
        db.migrate()?;
        Ok(db)
    }

    fn migrate(&self) -> Result<()> {
        self.with(|conn| {
            let current: i32 = conn.query_row("PRAGMA user_version", [], |row| row.get(0))?;
            if current >= SCHEMA_VERSION {
                return Ok(());
            }

            let transaction = conn.unchecked_transaction()?;

            // Four ordered phases. The order is load-bearing.
            transaction.execute_batch(SCHEMA)?;

            // 2.
            add_column_if_missing(&transaction, "mcp_servers", "slug", "TEXT")?;
            add_column_if_missing(&transaction, "mcp_servers", "tool_count", "INTEGER")?;
            add_column_if_missing(&transaction, "mcp_servers", "auth_ref", "TEXT")?;
            add_column_if_missing(
                &transaction,
                "cloud_connectors",
                "enabled",
                "INTEGER NOT NULL DEFAULT 1",
            )?;
            add_column_if_missing(
                &transaction,
                "cloud_connectors",
                "models",
                "TEXT NOT NULL DEFAULT '[]'",
            )?;
            add_column_if_missing(&transaction, "conversations", "project_id", "TEXT")?;
            add_column_if_missing(&transaction, "conversations", "forked_from_id", "TEXT")?;
            add_column_if_missing(&transaction, "conversations", "workspace_id", "TEXT")?;
            // Which provider's allowance a turn spent. Distinct from
            // `model_id`, which on a pooled turn records the pool.
            add_column_if_missing(&transaction, "messages", "provider_slug", "TEXT")?;
            // Prompt tokens summed across every tool round, rather than the
            // last round only. See `Aggregate::absorb`.
            add_column_if_missing(
                &transaction,
                "messages",
                "usage_prompt_tokens_total",
                "INTEGER",
            )?;
            add_column_if_missing(&transaction, "models", "discovered_folder", "TEXT")?;

            // 3. Rebuild version-nine folder provenance so the token invariant
            // is database-enforced for upgraded databases as well as fresh ones.
            migrate_model_folder_roots_v10(&transaction)?;

            // 4.
            transaction.execute_batch(LATE_INDEXES)?;

            transaction.pragma_update(None, "user_version", SCHEMA_VERSION)?;
            transaction.commit()?;
            Ok(())
        })
    }

    /// Run a closure with the connection held.
    pub(crate) fn with<T>(&self, f: impl FnOnce(&Connection) -> Result<T>) -> Result<T> {
        let guard = self
            .conn
            .lock()
            .map_err(|_| KuroError::other("database lock poisoned"))?;
        f(&guard)
    }
}

/// Current time as an RFC 3339 string, the storage format for every timestamp.
pub(crate) fn now() -> String {
    chrono::Utc::now().to_rfc3339()
}

/// Add a column only if the table does not already have it.
fn add_column_if_missing(
    conn: &Connection,
    table: &str,
    column: &str,
    declaration: &str,
) -> Result<()> {
    let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})"))?;
    let existing: Vec<String> = stmt
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    // An empty result means the table does not exist.
    if existing.is_empty() || existing.iter().any(|name| name == column) {
        return Ok(());
    }

    conn.execute_batch(&format!(
        "ALTER TABLE {table} ADD COLUMN {column} {declaration}"
    ))?;
    Ok(())
}

fn migrate_model_folder_roots_v10(conn: &Connection) -> Result<()> {
    let mut stmt = conn.prepare("PRAGMA table_info(model_folder_roots)")?;
    let columns = stmt
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    if columns.is_empty() || columns.iter().any(|column| column == "selection_token") {
        return Ok(());
    }

    conn.execute_batch(
        "ALTER TABLE model_folder_roots RENAME TO model_folder_roots_v9;
         CREATE TABLE model_folder_roots (
             path            TEXT PRIMARY KEY,
             selection_token TEXT NOT NULL UNIQUE,
             validated_at    TEXT NOT NULL
         );
         INSERT INTO model_folder_roots(path, selection_token, validated_at)
         SELECT path, lower(hex(randomblob(16))), validated_at
         FROM model_folder_roots_v9;
         DROP TABLE model_folder_roots_v9;",
    )?;
    Ok(())
}

/// Decode a JSON column, falling back to a default rather than failing a read.
pub(crate) fn json_or<T: serde::de::DeserializeOwned + Default>(raw: Option<&str>) -> T {
    raw.and_then(|text| serde_json::from_str(text).ok())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migrates_a_fresh_database_once() {
        let db = Db::open_in_memory().expect("open");
        let version: i32 = db
            .with(|conn| Ok(conn.query_row("PRAGMA user_version", [], |r| r.get(0))?))
            .expect("version");
        assert_eq!(version, SCHEMA_VERSION);

        // Re-running migration must be a no-op rather than an error.
        db.migrate().expect("idempotent migrate");
    }

    #[test]
    fn upgrades_version_nine_folder_provenance_with_stable_unique_tokens() {
        let conn = Connection::open_in_memory().expect("open v9 fixture");
        conn.execute_batch(SCHEMA).expect("create current tables");
        conn.execute_batch(
            "DROP TABLE model_folder_roots;
             CREATE TABLE model_folder_roots (
                 path TEXT PRIMARY KEY,
                 validated_at TEXT NOT NULL
             );
             INSERT INTO settings (key, value, updated_at)
                 VALUES ('models.folders', '[\"/models/a\",\"/models/b\"]', 'v9');
             INSERT INTO model_folder_roots (path, validated_at)
                 VALUES ('/models/a', 'a-time'), ('/models/b', 'b-time');
             INSERT INTO models (
                 id, display_name, source, capabilities, status, added_at, discovered_folder
             ) VALUES (
                 'existing-local', 'Existing local', 'local', '[]', 'ready', 'v9', '/models/a'
             );
             PRAGMA user_version = 9;",
        )
        .expect("seed v9 fixture");

        let db = Db::from_connection(conn).expect("migrate v9");
        let (version, first_tokens, timestamps, model_count) = db
            .with(|conn| {
                let version: i32 = conn.query_row("PRAGMA user_version", [], |row| row.get(0))?;
                let mut stmt = conn.prepare(
                    "SELECT selection_token, validated_at FROM model_folder_roots ORDER BY path",
                )?;
                let rows = stmt
                    .query_map([], |row| {
                        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                    })?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                let model_count = conn.query_row(
                    "SELECT COUNT(*) FROM models WHERE id = 'existing-local'",
                    [],
                    |row| row.get::<_, i64>(0),
                )?;
                Ok((
                    version,
                    rows.iter().map(|row| row.0.clone()).collect::<Vec<_>>(),
                    rows.iter().map(|row| row.1.clone()).collect::<Vec<_>>(),
                    model_count,
                ))
            })
            .expect("inspect migration");

        assert_eq!(version, 10);
        assert_eq!(first_tokens.len(), 2);
        assert!(first_tokens.iter().all(|token| token.len() == 32));
        assert_ne!(first_tokens[0], first_tokens[1]);
        assert_eq!(timestamps, vec!["a-time", "b-time"]);
        assert_eq!(model_count, 1);

        db.migrate().expect("idempotent reopen");
        let second_tokens = db
            .with(|conn| {
                let mut stmt =
                    conn.prepare("SELECT selection_token FROM model_folder_roots ORDER BY path")?;
                let tokens = stmt
                    .query_map([], |row| row.get::<_, String>(0))?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                Ok(tokens)
            })
            .expect("tokens after reopen");
        assert_eq!(second_tokens, first_tokens, "reopen must not rotate tokens");
    }

    /// The shape version 1 shipped with: no `project_id`, no `memories`, and
    /// `mcp_servers`/`cloud_connectors` without the columns added since.
    const V1_SCHEMA: &str = "
        CREATE TABLE conversations (
            id           TEXT PRIMARY KEY,
            title        TEXT NOT NULL DEFAULT 'New chat',
            title_mode   TEXT NOT NULL DEFAULT 'first_line',
            model_id     TEXT,
            pinned       INTEGER NOT NULL DEFAULT 0,
            archived     INTEGER NOT NULL DEFAULT 0,
            created_at   TEXT NOT NULL,
            updated_at   TEXT NOT NULL
        );
        CREATE TABLE messages (
            id              TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
            role            TEXT NOT NULL,
            content         TEXT NOT NULL DEFAULT '',
            reasoning_content TEXT, tool_calls TEXT, tool_call_id TEXT, attachments TEXT,
            used_web_search INTEGER NOT NULL DEFAULT 0, web_sources TEXT, model_id TEXT,
            usage_prompt_tokens INTEGER, usage_completion_tokens INTEGER,
            timing_ttft_ms INTEGER, timing_total_ms INTEGER, timing_tokens_per_sec REAL,
            finish_reason TEXT, created_at TEXT NOT NULL
        );
        CREATE TABLE mcp_servers (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, transport TEXT NOT NULL,
            command TEXT, args TEXT, env TEXT, url TEXT, headers TEXT,
            enabled INTEGER NOT NULL DEFAULT 1,
            status TEXT NOT NULL DEFAULT 'disconnected',
            last_error TEXT, created_at TEXT NOT NULL
        );
        CREATE TABLE cloud_connectors (
            id TEXT PRIMARY KEY, provider TEXT NOT NULL, label TEXT NOT NULL,
            keychain_ref TEXT NOT NULL, base_url TEXT,
            status TEXT NOT NULL DEFAULT 'untested',
            last_tested_at TEXT, last_error TEXT, created_at TEXT NOT NULL
        );
        CREATE TABLE settings (
            key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL
        );
    ";

    /// A version-1 database with a conversation and a message in it.
    fn version_one_database() -> Connection {
        let conn = Connection::open_in_memory().expect("open");
        conn.execute_batch(V1_SCHEMA).expect("v1 schema");
        conn.execute_batch(
            "INSERT INTO conversations (id, title, created_at, updated_at)
                 VALUES ('c1', 'What is a black hole?', '2026-01-01', '2026-01-01');
             INSERT INTO messages (id, conversation_id, role, content, created_at)
                 VALUES ('m1', 'c1', 'user', 'explain', '2026-01-01');",
        )
        .expect("seed");
        conn.pragma_update(None, "user_version", 1)
            .expect("version");
        conn
    }

    #[test]
    fn upgrades_a_version_one_database_without_losing_anything() {
        // This is the path every existing install takes.
        let db = Db::from_connection(version_one_database()).expect("migrate v1");

        let version: i32 = db
            .with(|conn| Ok(conn.query_row("PRAGMA user_version", [], |r| r.get(0))?))
            .expect("version");
        assert_eq!(version, SCHEMA_VERSION);

        let conversations = db.list_conversations(None).expect("conversations");
        assert_eq!(
            conversations.len(),
            1,
            "existing chats must survive an upgrade"
        );
        assert_eq!(conversations[0].title, "What is a black hole?");
        assert_eq!(db.list_messages("c1").expect("messages").len(), 1);
    }

    #[test]
    fn upgrades_a_version_seven_models_table_without_losing_rows() {
        let fixture_dir =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/kuro-migration-tests");
        std::fs::create_dir_all(&fixture_dir).expect("fixture directory");
        let fixture = fixture_dir.join(format!("v7-models-{}.sqlite", uuid::Uuid::new_v4()));

        {
            let conn = Connection::open(&fixture).expect("open version 7 fixture");
            conn.execute_batch(
                "CREATE TABLE models (
                     id TEXT PRIMARY KEY, display_name TEXT NOT NULL, source TEXT NOT NULL,
                     hf_repo TEXT, hf_file TEXT, quant TEXT, param_count TEXT, family TEXT,
                     capabilities TEXT NOT NULL DEFAULT '[]', context_length INTEGER,
                     file_path TEXT, file_size_bytes INTEGER, sha256 TEXT,
                     status TEXT NOT NULL DEFAULT 'downloading', error TEXT,
                     added_at TEXT NOT NULL, last_used_at TEXT
                 );
                 INSERT INTO models (
                     id, display_name, source, capabilities, status, added_at
                 ) VALUES (
                     'existing-model', 'Existing model', 'local', '[]', 'ready', '2026-01-01'
                 );",
            )
            .expect("version 7 schema and row");
            conn.pragma_update(None, "user_version", 7)
                .expect("version");
        }

        let db = Db::open(&fixture).expect("migrate version 7 fixture");
        let (version, columns, rows): (i32, Vec<String>, i64) = db
            .with(|conn| {
                let version = conn.query_row("PRAGMA user_version", [], |row| row.get(0))?;
                let mut stmt = conn.prepare("PRAGMA table_info(models)")?;
                let columns = stmt
                    .query_map([], |row| row.get::<_, String>(1))?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                let rows = conn.query_row("SELECT COUNT(*) FROM models", [], |row| row.get(0))?;
                Ok((version, columns, rows))
            })
            .expect("inspect migrated database");
        let existing = db
            .get_model("existing-model")
            .expect("read migrated model")
            .expect("existing row");
        drop(db);

        assert_eq!(version, SCHEMA_VERSION);
        assert!(
            columns.iter().any(|column| column == "discovered_folder"),
            "the ownership column must be added to an existing models table"
        );
        assert_eq!(rows, 1, "the pre-existing model registration must survive");
        assert_eq!(existing.display_name, "Existing model");
        assert!(existing.discovered_folder.is_none());

        std::fs::remove_file(&fixture).expect("remove migration fixture");
    }

    #[test]
    fn an_upgraded_database_has_everything_the_new_code_reads() {
        let db = Db::from_connection(version_one_database()).expect("migrate v1");

        // Tables added after version 1.
        assert!(db.list_projects().expect("projects").is_empty());
        assert_eq!(db.count_memories().expect("memories"), 0);

        // Columns added after version 1, exercised through the accessors that
        // read them rather than by inspecting the schema.
        assert!(db.list_mcp_servers().expect("mcp").is_empty());
        assert!(db.list_cloud_connectors().expect("providers").is_empty());

        // And the index that caused the original failure now exists.
        let indexes: Vec<String> = db
            .with(|conn| {
                let mut stmt = conn.prepare(
                    "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'conversations'",
                )?;
                let names = stmt
                    .query_map([], |row| row.get::<_, String>(0))?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                Ok(names)
            })
            .expect("indexes");
        assert!(
            indexes
                .iter()
                .any(|name| name == "idx_conversations_project"),
            "got: {indexes:?}"
        );
    }

    #[test]
    fn a_project_can_be_used_immediately_after_an_upgrade() {
        let db = Db::from_connection(version_one_database()).expect("migrate v1");

        let project = db
            .insert_project(&crate::db::NewProject {
                name: "Kuro".to_string(),
                ..Default::default()
            })
            .expect("insert");
        db.set_conversation_project("c1", Some(&project.id))
            .expect("move the pre-existing chat into a new project");

        assert_eq!(
            db.list_project_conversations(&project.id)
                .expect("list")
                .len(),
            1
        );
    }

    #[test]
    fn migrating_an_already_current_database_is_a_no_op() {
        let db = Db::open_in_memory().expect("open");
        db.create_conversation(None).expect("conversation");

        db.migrate().expect("second migrate");

        assert_eq!(db.list_conversations(None).expect("list").len(), 1);
    }

    #[test]
    fn adding_a_column_to_a_table_that_does_not_exist_is_not_an_error() {
        // Ordering inside `migrate` should not depend on which tables happen to
        // exist yet.
        let conn = Connection::open_in_memory().expect("open");
        add_column_if_missing(&conn, "not_a_table", "whatever", "TEXT").expect("no-op");
    }

    #[test]
    fn enforces_foreign_keys() {
        let db = Db::open_in_memory().expect("open");
        let result = db.with(|conn| {
            conn.execute(
                "INSERT INTO messages (id, conversation_id, role, content, created_at)
                 VALUES ('m1', 'missing-conversation', 'user', 'hi', '2026-01-01T00:00:00Z')",
                [],
            )?;
            Ok(())
        });
        assert!(
            result.is_err(),
            "a message referencing a missing conversation must be rejected"
        );
    }
}
