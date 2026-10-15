use std::path::{Path, PathBuf};

use kuro_core::catalog::{
    scan_model_folder, LocalModelCandidate, MAX_RESULTS, MAX_VISITED_DIRECTORIES,
    MAX_VISITED_ENTRIES,
};
use kuro_core::db::{ModelSource, NewModel};
use kuro_core::settings::model_folders;
use kuro_core::Db;

struct TestTree {
    root: PathBuf,
}

impl TestTree {
    fn new(label: &str) -> Self {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/kuro-local-model-fix-tests")
            .join(format!("{label}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).expect("create test tree");
        Self {
            root: root.canonicalize().expect("canonical test tree"),
        }
    }

    fn dir(&self, relative: &str) -> PathBuf {
        let path = self.root.join(relative);
        std::fs::create_dir_all(&path).expect("create fixture directory");
        path
    }

    fn file(&self, relative: &str, bytes: &[u8]) -> PathBuf {
        let path = self.root.join(relative);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create fixture parent");
        }
        std::fs::write(&path, bytes).expect("write fixture");
        path
    }
}

impl Drop for TestTree {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn candidate(scan_root: &Path, suffix: &str) -> LocalModelCandidate {
    scan_model_folder(scan_root)
        .expect("scan")
        .candidates
        .into_iter()
        .find(|candidate| candidate.canonical_path.ends_with(suffix))
        .expect("candidate")
}

fn existing_model(id: &str, source: ModelSource, display_name: &str) -> NewModel {
    NewModel {
        id: id.to_string(),
        display_name: display_name.to_string(),
        source,
        hf_repo: None,
        hf_file: None,
        quant: Some("Q5_K_M".to_string()),
        param_count: Some("9B".to_string()),
        family: Some("manual-family".to_string()),
        capabilities: vec!["manual-capability".to_string()],
        context_length: Some(8192),
        file_size_bytes: Some(17),
    }
}

#[cfg(unix)]
#[test]
fn subtree_errors_never_unregister_candidates_the_partial_scan_could_not_see() {
    use std::os::unix::fs::PermissionsExt;

    let tree = TestTree::new("partial-error");
    let root = tree.dir("models").canonicalize().expect("root");
    tree.file("models/visible.gguf", b"visible");
    tree.file("models/subtree/hidden.gguf", b"hidden");
    let complete = scan_model_folder(&root).expect("complete scan");
    let hidden_id = candidate(&root, "hidden.gguf").id;
    let db = Db::open_in_memory().expect("db");
    db.commit_model_folder_add(&complete)
        .expect("initial registration");
    let ticket = db
        .model_folder_scan_tickets()
        .expect("scan ticket")
        .remove(0);

    let subtree = root.join("subtree");
    let original = std::fs::metadata(&subtree).expect("metadata").permissions();
    std::fs::set_permissions(&subtree, std::fs::Permissions::from_mode(0o000))
        .expect("make unreadable");
    let partial = scan_model_folder(&root).expect("partial scan");
    std::fs::set_permissions(&subtree, original).expect("restore permissions");
    assert_eq!(partial.errors.len(), 1);

    db.reconcile_scanned_folder(&ticket, &partial)
        .expect("partial reconcile");

    assert!(
        db.get_model(&hidden_id).expect("lookup").is_some(),
        "absence from an errored scan is not proof that the model was removed"
    );
}

#[test]
fn result_limited_scans_never_unregister_candidates_beyond_the_limit() {
    let tree = TestTree::new("partial-results");
    let root = tree.dir("models").canonicalize().expect("root");
    tree.file("models/zz-owned/model.gguf", b"owned");
    let initial = scan_model_folder(&root).expect("initial scan");
    let owned_id = initial.candidates[0].id.clone();
    let db = Db::open_in_memory().expect("db");
    db.commit_model_folder_add(&initial)
        .expect("initial registration");
    let ticket = db
        .model_folder_scan_tickets()
        .expect("scan ticket")
        .remove(0);
    for index in 0..MAX_RESULTS {
        tree.file(&format!("models/aa-{index:04}.gguf"), b"x");
    }

    let mut partial = scan_model_folder(&root).expect("limited scan");
    assert!(partial.counts.result_limit_reached);
    assert!(!partial.candidates.iter().any(|held| held.id == owned_id));
    partial.counts.result_limit_reached = false;
    db.reconcile_scanned_folder(&ticket, &partial)
        .expect("partial reconcile");

    assert!(db.get_model(&owned_id).expect("lookup").is_some());
}

#[test]
fn directory_limited_scans_never_unregister_unvisited_owned_candidates() {
    let tree = TestTree::new("partial-directories");
    let root = tree.dir("models").canonicalize().expect("root");
    tree.file("models/zz-owned/model.gguf", b"owned");
    let initial = scan_model_folder(&root).expect("initial scan");
    let owned_id = initial.candidates[0].id.clone();
    let db = Db::open_in_memory().expect("db");
    db.commit_model_folder_add(&initial)
        .expect("initial registration");
    let ticket = db
        .model_folder_scan_tickets()
        .expect("scan ticket")
        .remove(0);
    for index in 0..MAX_VISITED_DIRECTORIES {
        tree.dir(&format!("models/aa-{index:04}"));
    }

    let partial = scan_model_folder(&root).expect("limited scan");
    assert!(partial.counts.directory_limit_reached);
    assert!(!partial.candidates.iter().any(|held| held.id == owned_id));
    db.reconcile_scanned_folder(&ticket, &partial)
        .expect("partial reconcile");

    assert!(db.get_model(&owned_id).expect("lookup").is_some());
}

#[test]
fn same_id_manual_local_rows_are_not_converted_to_scanner_ownership() {
    let tree = TestTree::new("manual-conflict");
    let root = tree.dir("models").canonicalize().expect("root");
    tree.file("models/collision.gguf", b"scanner bytes");
    let scan = scan_model_folder(&root).expect("scan");
    let candidate = &scan.candidates[0];
    let db = Db::open_in_memory().expect("db");
    db.upsert_model(&existing_model(
        &candidate.id,
        ModelSource::Local,
        "Manual identity",
    ))
    .expect("manual row");
    db.set_model_ready(&candidate.id, "/manual/location.gguf", 17, "manual-sha")
        .expect("manual ready");

    let counts = db
        .commit_model_folder_add(&scan)
        .expect("reconcile conflict");
    assert_eq!(counts.conflicts, 1);

    let stored = db
        .get_model(&candidate.id)
        .expect("lookup")
        .expect("manual survives");
    assert_eq!(stored.display_name, "Manual identity");
    assert_eq!(stored.file_path.as_deref(), Some("/manual/location.gguf"));
    assert_eq!(stored.sha256.as_deref(), Some("manual-sha"));
    assert!(stored.discovered_folder.is_none());
}

#[test]
fn same_id_managed_rows_are_not_converted_to_local_scanner_rows() {
    let tree = TestTree::new("managed-conflict");
    let root = tree.dir("models").canonicalize().expect("root");
    tree.file("models/collision.gguf", b"scanner bytes");
    let scan = scan_model_folder(&root).expect("scan");
    let candidate = &scan.candidates[0];
    let db = Db::open_in_memory().expect("db");
    db.upsert_model(&existing_model(
        &candidate.id,
        ModelSource::Curated,
        "Managed identity",
    ))
    .expect("managed row");

    let counts = db
        .commit_model_folder_add(&scan)
        .expect("reconcile conflict");
    assert_eq!(counts.conflicts, 1);

    let stored = db
        .get_model(&candidate.id)
        .expect("lookup")
        .expect("managed survives");
    assert_eq!(stored.source, ModelSource::Curated);
    assert_eq!(stored.display_name, "Managed identity");
    assert!(stored.discovered_folder.is_none());
}

#[test]
fn stale_whole_list_add_snapshots_cannot_drop_an_unrelated_folder() {
    let tree = TestTree::new("stale-add");
    let folder_a = tree.dir("a").canonicalize().expect("a");
    let folder_b = tree.dir("b").canonicalize().expect("b");
    tree.file("a/a.gguf", b"a");
    tree.file("b/b.gguf", b"b");
    let scan_a = scan_model_folder(&folder_a).expect("scan a");
    let scan_b = scan_model_folder(&folder_b).expect("scan b");
    let folder_a_text = folder_a.to_string_lossy().into_owned();
    let folder_b_text = folder_b.to_string_lossy().into_owned();
    let db = Db::open_in_memory().expect("db");

    db.commit_model_folder_add(&scan_a).expect("first add");
    db.commit_model_folder_add(&scan_b).expect("second add");

    assert_eq!(
        model_folders(&db).expect("folders"),
        vec![folder_a_text, folder_b_text]
    );
    assert_eq!(db.list_models().expect("models").len(), 2);
}

#[test]
fn entry_limited_scans_never_unregister_unvisited_owned_candidates() {
    let tree = TestTree::new("partial-entries");
    let root = tree.dir("models").canonicalize().expect("root");
    tree.file("models/zz-owned/model.gguf", b"owned");
    let initial = scan_model_folder(&root).expect("initial scan");
    let owned_id = initial.candidates[0].id.clone();
    let db = Db::open_in_memory().expect("db");
    db.commit_model_folder_add(&initial)
        .expect("initial registration");
    let ticket = db
        .model_folder_scan_tickets()
        .expect("scan ticket")
        .remove(0);
    for index in 0..MAX_VISITED_ENTRIES {
        tree.file(&format!("models/aa-{index:05}.txt"), b"x");
    }

    let partial = scan_model_folder(&root).expect("limited scan");
    assert!(partial.counts.entry_limit_reached);
    assert!(!partial.candidates.iter().any(|held| held.id == owned_id));
    db.reconcile_scanned_folder(&ticket, &partial)
        .expect("partial reconcile");

    assert!(db.get_model(&owned_id).expect("lookup").is_some());
}
