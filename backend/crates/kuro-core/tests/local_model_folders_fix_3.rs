use std::path::{Path, PathBuf};

use kuro_core::catalog::scan_model_folder;
use kuro_core::settings::model_folders;
use kuro_core::{Db, KuroError};

struct TestTree(PathBuf);

impl TestTree {
    fn new(label: &str) -> Self {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/kuro-local-model-fix-3-tests")
            .join(format!("{label}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).expect("create test tree");
        Self(root.canonicalize().expect("canonical test tree"))
    }

    fn dir(&self, relative: &str) -> PathBuf {
        let path = self.0.join(relative);
        std::fs::create_dir_all(&path).expect("create fixture directory");
        path.canonicalize().expect("canonical fixture directory")
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
fn remove_and_readd_rejects_a_scan_ticket_from_the_previous_selection_lifetime() {
    let tree = TestTree::new("aba");
    let root = tree.dir("models");
    let old_path = tree.file("models/old.gguf", b"old");
    let first_scan = scan_model_folder(&root).expect("first scan");
    let old_id = first_scan.candidates[0].id.clone();
    let root_text = root.to_string_lossy().into_owned();
    let db = Db::open_in_memory().expect("db");
    db.commit_model_folder_add(&first_scan).expect("first add");
    let old_ticket = db
        .model_folder_scan_tickets()
        .expect("old ticket")
        .remove(0);

    db.commit_model_folder_remove(&root_text).expect("remove");
    std::fs::remove_file(old_path).expect("remove old fixture");
    tree.file("models/new.gguf", b"new generation");
    let second_scan = scan_model_folder(&root).expect("second scan");
    let new_id = second_scan.candidates[0].id.clone();
    db.commit_model_folder_add(&second_scan)
        .expect("second add");
    let new_ticket = db
        .model_folder_scan_tickets()
        .expect("new ticket")
        .remove(0);

    assert_ne!(old_ticket, new_ticket);
    assert!(matches!(
        db.reconcile_scanned_folder(&old_ticket, &first_scan),
        Err(KuroError::BadRequest(_))
    ));
    assert!(db.get_model(&old_id).expect("old lookup").is_none());
    assert!(db.get_model(&new_id).expect("new lookup").is_some());
    assert_eq!(model_folders(&db).expect("folders"), vec![root_text]);
    assert_eq!(
        db.model_folder_scan_tickets().expect("current"),
        vec![new_ticket]
    );
}

#[test]
fn a_ticket_can_reconcile_multiple_scans_during_one_selection_lifetime() {
    let tree = TestTree::new("same-generation");
    let root = tree.dir("models");
    tree.file("models/model.gguf", b"first");
    let first_scan = scan_model_folder(&root).expect("first scan");
    let db = Db::open_in_memory().expect("db");
    db.commit_model_folder_add(&first_scan).expect("add");
    let ticket = db.model_folder_scan_tickets().expect("ticket").remove(0);

    tree.file("models/model.gguf", b"updated bytes");
    let second_scan = scan_model_folder(&root).expect("second scan");
    db.reconcile_scanned_folder(&ticket, &second_scan)
        .expect("same lifetime reconcile");
    db.reconcile_scanned_folder(&ticket, &second_scan)
        .expect("ticket remains valid");

    assert_eq!(
        db.get_model(&second_scan.candidates[0].id)
            .expect("lookup")
            .expect("model")
            .file_size_bytes,
        Some(b"updated bytes".len() as i64)
    );
}
