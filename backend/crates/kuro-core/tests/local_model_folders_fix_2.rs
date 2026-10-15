use std::path::{Path, PathBuf};

use kuro_core::catalog::scan_model_folder;
use kuro_core::settings::{model_folders, set_model_folders, KEY_MODEL_FOLDERS};
use kuro_core::{Db, KuroError};
use rusqlite::params;
use serde_json::json;

struct TestTree(PathBuf);

impl TestTree {
    fn new(label: &str) -> Self {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/kuro-local-model-fix-2-tests")
            .join(format!("{label}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).expect("create test tree");
        Self(root.canonicalize().expect("canonical test tree"))
    }

    fn dir(&self, relative: &str) -> PathBuf {
        let path = self.0.join(relative);
        std::fs::create_dir_all(&path).expect("create fixture directory");
        path
    }

    fn file(&self, relative: &str, bytes: &[u8]) -> PathBuf {
        let path = self.0.join(relative);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create fixture parent");
        }
        std::fs::write(&path, bytes).expect("write fixture");
        path
    }
}

impl Drop for TestTree {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn stale_scan_cannot_recreate_rows_after_its_folder_is_removed() {
    let tree = TestTree::new("stale-reconcile");
    let root = tree.dir("models").canonicalize().expect("canonical root");
    tree.file("models/stale.gguf", b"stale");
    let scan = scan_model_folder(&root).expect("scan");
    let model_id = scan.candidates[0].id.clone();
    let root_text = root.to_string_lossy().into_owned();
    let db = Db::open_in_memory().expect("db");
    db.commit_model_folder_add(&scan).expect("select folder");
    let ticket = db
        .model_folder_scan_tickets()
        .expect("scan ticket")
        .remove(0);
    db.commit_model_folder_remove(&root_text)
        .expect("remove folder");

    assert!(matches!(
        db.reconcile_scanned_folder(&ticket, &scan),
        Err(KuroError::BadRequest(_))
    ));
    assert!(model_folders(&db).expect("folders").is_empty());
    assert!(db.get_model(&model_id).expect("lookup").is_none());
}

#[test]
fn generic_and_typed_settings_writers_cannot_mutate_the_owned_folder_key() {
    let tree = TestTree::new("owned-key");
    let root = tree.dir("models").canonicalize().expect("canonical root");
    let root_text = root.to_string_lossy().into_owned();
    let db = Db::open_in_memory().expect("db");

    assert!(matches!(
        db.set_setting(KEY_MODEL_FOLDERS, &json!([root_text.clone()])),
        Err(KuroError::BadRequest(_))
    ));
    assert!(matches!(
        db.delete_setting(KEY_MODEL_FOLDERS),
        Err(KuroError::BadRequest(_))
    ));
    assert!(matches!(
        set_model_folders(&db, &[root_text]),
        Err(KuroError::BadRequest(_))
    ));
    assert!(model_folders(&db).expect("folders").is_empty());
}

#[test]
fn scanner_rejects_an_available_lexical_root_alias() {
    let tree = TestTree::new("available-alias");
    let root = tree.dir("models");

    assert!(matches!(
        scan_model_folder(&root.join(".")),
        Err(KuroError::BadRequest(_))
    ));
    assert!(matches!(
        scan_model_folder(&root.join("child").join("..").join(".")),
        Err(KuroError::BadRequest(_))
    ));
}

#[cfg(unix)]
#[test]
fn scanner_rejects_a_root_reached_through_a_symlinked_ancestor() {
    use std::os::unix::fs::symlink;

    let tree = TestTree::new("ancestor-alias");
    let target_parent = tree.dir("target");
    tree.dir("target/models");
    let alias_parent = tree.0.join("alias");
    symlink(target_parent, &alias_parent).expect("ancestor symlink");

    assert!(matches!(
        scan_model_folder(&alias_parent.join("models")),
        Err(KuroError::BadRequest(_))
    ));
}

#[cfg(unix)]
#[test]
fn typed_folder_writer_rejects_an_available_symlink() {
    use std::os::unix::fs::symlink;

    let tree = TestTree::new("typed-symlink");
    let target = tree.dir("target");
    let alias = tree.0.join("alias");
    symlink(target, &alias).expect("create alias");
    let db = Db::open_in_memory().expect("db");

    assert!(matches!(
        set_model_folders(&db, &[alias.to_string_lossy().into_owned()]),
        Err(KuroError::BadRequest(_))
    ));
    assert!(model_folders(&db).expect("folders").is_empty());
}

#[test]
fn an_unproven_offline_folder_from_a_version_eight_database_is_not_grandfathered() {
    let tree = TestTree::new("offline-provenance");
    let database = tree.0.join("legacy.sqlite3");
    let offline = tree.0.join("currently-offline");
    let connection = rusqlite::Connection::open(&database).expect("legacy database");
    connection
        .execute_batch(
            "CREATE TABLE settings (
                 key TEXT PRIMARY KEY,
                 value TEXT NOT NULL,
                 updated_at TEXT NOT NULL
             );
             PRAGMA user_version = 8;",
        )
        .expect("legacy schema");
    connection
        .execute(
            "INSERT INTO settings (key, value, updated_at) VALUES (?1, ?2, 'legacy')",
            params![
                KEY_MODEL_FOLDERS,
                json!([offline.to_string_lossy()]).to_string()
            ],
        )
        .expect("unchecked legacy setting");
    drop(connection);

    let added = tree.dir("added").canonicalize().expect("canonical added");
    tree.file("added/model.gguf", b"model");
    let scan = scan_model_folder(&added).expect("scan");
    let db = Db::open(&database).expect("upgrade database");

    assert!(matches!(
        db.commit_model_folder_add(&scan),
        Err(KuroError::BadRequest(_))
    ));
    assert_eq!(
        model_folders(&db).expect("unchanged folders"),
        vec![offline.to_string_lossy().into_owned()]
    );
    assert!(db
        .get_model(&scan.candidates[0].id)
        .expect("lookup")
        .is_none());
}

#[test]
fn a_folder_validated_by_add_can_remain_selected_while_offline() {
    let tree = TestTree::new("trusted-offline");
    let first = tree.dir("first").canonicalize().expect("first");
    let first_text = first.to_string_lossy().into_owned();
    tree.file("first/first.gguf", b"first");
    let first_scan = scan_model_folder(&first).expect("first scan");
    let db = Db::open_in_memory().expect("db");
    db.commit_model_folder_add(&first_scan)
        .expect("validated add");
    std::fs::remove_dir_all(&first).expect("take first offline");

    let second = tree.dir("second").canonicalize().expect("second");
    tree.file("second/second.gguf", b"second");
    let second_scan = scan_model_folder(&second).expect("second scan");
    db.commit_model_folder_add(&second_scan)
        .expect("add while validated root is offline");

    assert_eq!(model_folders(&db).expect("folders")[0], first_text);
}
