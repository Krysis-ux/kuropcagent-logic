use std::path::{Path, PathBuf};

use kuro_core::catalog::{
    infer_filename_metadata, local_model_id, remove_model, scan_model_folder,
    validate_model_folder_root, LocalModelCandidate, MAX_REPORTED_ERRORS, MAX_RESULTS,
    MAX_VISITED_DIRECTORIES,
};
use kuro_core::db::{ModelSource, ModelStatus, NewModel};
use kuro_core::settings::model_folders;
use kuro_core::{Db, KuroError, Paths};

struct TestTree {
    root: PathBuf,
}

impl TestTree {
    fn new(label: &str) -> Self {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/kuro-local-model-tests")
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

fn scanned_candidate(folder: &Path, relative: &str, bytes: &[u8]) -> LocalModelCandidate {
    let path = folder.join(relative);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("candidate parent");
    }
    std::fs::write(&path, bytes).expect("candidate fixture");
    scan_model_folder(folder)
        .expect("scan")
        .candidates
        .into_iter()
        .find(|candidate| candidate.canonical_path == path.canonicalize().expect("canonical"))
        .expect("candidate")
}

fn manual_model(id: &str) -> NewModel {
    NewModel {
        id: id.to_string(),
        display_name: "Manual local model".to_string(),
        source: ModelSource::Local,
        hf_repo: None,
        hf_file: None,
        quant: None,
        param_count: None,
        family: None,
        capabilities: Vec::new(),
        context_length: None,
        file_size_bytes: Some(3),
    }
}

#[test]
fn ordered_model_folders_round_trip_across_a_database_reopen() {
    let tree = TestTree::new("settings-round-trip");
    let first = tree.dir("z-first").canonicalize().expect("first canonical");
    let second = tree
        .dir("a-second")
        .canonicalize()
        .expect("second canonical");
    let expected = vec![
        first.to_string_lossy().into_owned(),
        second.to_string_lossy().into_owned(),
    ];
    let database = tree.root.join("settings.sqlite3");

    {
        let db = Db::open(&database).expect("open");
        assert!(model_folders(&db).expect("missing setting").is_empty());
        db.commit_model_folder_add(&scan_model_folder(&first).expect("scan first"))
            .expect("store first folder");
        db.commit_model_folder_add(&scan_model_folder(&second).expect("scan second"))
            .expect("store second folder");
    }

    let reopened = Db::open(&database).expect("reopen");
    assert_eq!(model_folders(&reopened).expect("read folders"), expected);
}

#[test]
fn folder_add_rejects_duplicate_and_overlapping_roots() {
    let tree = TestTree::new("add-overlap");
    let parent = tree.dir("models").canonicalize().expect("parent");
    let child = tree.dir("models/nested").canonicalize().expect("child");
    let db = Db::open_in_memory().expect("open");
    let parent_scan = scan_model_folder(&parent).expect("parent scan");
    let child_scan = scan_model_folder(&child).expect("child scan");
    db.commit_model_folder_add(&parent_scan).expect("first add");

    assert!(matches!(
        db.commit_model_folder_add(&parent_scan),
        Err(KuroError::BadRequest(_))
    ));
    assert!(matches!(
        db.commit_model_folder_add(&child_scan),
        Err(KuroError::BadRequest(_))
    ));
}

#[cfg(unix)]
#[test]
fn root_validation_rejects_duplicates_overlaps_symlinks_and_non_directories() {
    use std::os::unix::fs::symlink;

    let tree = TestTree::new("root-validation");
    let parent = tree.dir("parent");
    let child = tree.dir("parent/child");
    let file = tree.file("not-a-directory", b"x");
    let link = tree.root.join("linked-root");
    symlink(&parent, &link).expect("symlink root");

    let canonical_parent = parent.canonicalize().expect("canonical parent");
    let canonical_child = child.canonicalize().expect("canonical child");
    let parent_text = canonical_parent.to_string_lossy().into_owned();
    let child_text = canonical_child.to_string_lossy().into_owned();

    assert!(matches!(
        validate_model_folder_root(&parent, std::slice::from_ref(&parent_text)),
        Err(KuroError::BadRequest(_))
    ));
    assert!(matches!(
        validate_model_folder_root(&child, std::slice::from_ref(&parent_text)),
        Err(KuroError::BadRequest(_))
    ));
    assert!(matches!(
        validate_model_folder_root(&parent, std::slice::from_ref(&child_text)),
        Err(KuroError::BadRequest(_))
    ));
    assert!(matches!(
        validate_model_folder_root(&link, &[]),
        Err(KuroError::BadRequest(_))
    ));
    assert!(matches!(
        validate_model_folder_root(&file, &[]),
        Err(KuroError::BadRequest(_))
    ));
}

#[test]
fn scanner_accepts_case_insensitive_gguf_through_depth_four_in_sorted_order() {
    let tree = TestTree::new("scanner-basics");
    let root = tree.dir("models");
    tree.file("models/zeta.GGUF", b"z");
    tree.file("models/alpha.gguf", b"alpha");
    tree.file("models/ignore.txt", b"no");
    tree.file("models/d1/d2/d3/d4/depth-four.gguf", b"four");
    tree.file("models/d1/d2/d3/d4/d5/depth-five.gguf", b"five");

    let first = scan_model_folder(&root).expect("first scan");
    let second = scan_model_folder(&root).expect("second scan");
    let paths: Vec<_> = first
        .candidates
        .iter()
        .map(|candidate| candidate.canonical_path.clone())
        .collect();

    assert_eq!(
        first.candidates, second.candidates,
        "rescans must be deterministic"
    );
    assert!(paths.windows(2).all(|pair| pair[0] < pair[1]));
    assert_eq!(paths.len(), 3);
    assert!(paths.iter().any(|path| path.ends_with("zeta.GGUF")));
    assert!(paths.iter().any(|path| path.ends_with("depth-four.gguf")));
    assert!(!paths.iter().any(|path| path.ends_with("depth-five.gguf")));
    assert_eq!(first.counts.skipped_non_gguf, 1);
}

#[cfg(unix)]
#[test]
fn scanner_skips_file_and_directory_symlinks_and_never_escapes_the_root() {
    use std::os::unix::fs::symlink;

    let tree = TestTree::new("scanner-symlinks");
    let root = tree.dir("models");
    let outside_file = tree.file("outside/external.gguf", b"external");
    let outside_dir = tree.dir("outside/directory");
    tree.file("outside/directory/nested.gguf", b"nested");
    symlink(&outside_file, root.join("linked-file.gguf")).expect("file symlink");
    symlink(&outside_dir, root.join("linked-directory")).expect("directory symlink");
    tree.file("models/inside.gguf", b"inside");

    let scan = scan_model_folder(&root).expect("scan");
    let canonical_root = root.canonicalize().expect("canonical root");

    assert_eq!(scan.counts.skipped_symlinks, 2);
    assert_eq!(scan.candidates.len(), 1);
    assert!(scan
        .candidates
        .iter()
        .all(|candidate| candidate.canonical_path.starts_with(&canonical_root)));
    assert!(scan.candidates[0].canonical_path.ends_with("inside.gguf"));
}

#[cfg(unix)]
#[test]
fn scanner_skips_non_regular_entries_and_reports_unreadable_subtrees() {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::PermissionsExt;

    let tree = TestTree::new("scanner-errors");
    let root = tree.dir("models");
    let fifo = root.join("named-pipe.gguf");
    let fifo_name = CString::new(fifo.as_os_str().as_bytes()).expect("fifo path");
    let result = unsafe { libc::mkfifo(fifo_name.as_ptr(), 0o600) };
    assert_eq!(result, 0, "create FIFO fixture");

    let unreadable = tree.dir("models/unreadable");
    tree.file("models/unreadable/hidden.gguf", b"hidden");
    let original_permissions = std::fs::metadata(&unreadable)
        .expect("permissions")
        .permissions();
    std::fs::set_permissions(&unreadable, std::fs::Permissions::from_mode(0o000))
        .expect("make unreadable");

    let scan = scan_model_folder(&root).expect("scan remains representable");
    std::fs::set_permissions(&unreadable, original_permissions).expect("restore permissions");

    assert_eq!(scan.counts.skipped_non_regular, 1);
    assert_eq!(scan.errors.len(), 1);
    assert!(scan.errors[0].path.ends_with("unreadable"));
}

#[test]
fn scanner_enforces_the_result_cap() {
    let tree = TestTree::new("result-cap");
    let root = tree.dir("models");
    for index in 0..=MAX_RESULTS {
        tree.file(&format!("models/model-{index:04}.gguf"), b"x");
    }

    let scan = scan_model_folder(&root).expect("scan");

    assert_eq!(scan.candidates.len(), MAX_RESULTS);
    assert!(scan.counts.result_limit_reached);
}

#[test]
fn scanner_enforces_the_visited_directory_cap() {
    let tree = TestTree::new("directory-cap");
    let root = tree.dir("models");
    for index in 0..=MAX_VISITED_DIRECTORIES {
        tree.dir(&format!("models/directory-{index:04}"));
    }

    let scan = scan_model_folder(&root).expect("scan");

    assert!(scan.counts.directory_limit_reached);
}

#[cfg(unix)]
#[test]
fn scanner_bounds_reported_errors() {
    use std::os::unix::fs::PermissionsExt;

    struct RestorePermissions(Vec<PathBuf>);

    impl Drop for RestorePermissions {
        fn drop(&mut self) {
            for path in &self.0 {
                let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700));
            }
        }
    }

    let tree = TestTree::new("error-cap");
    let root = tree.dir("models");
    let mut unreadable = Vec::new();
    for index in 0..=MAX_REPORTED_ERRORS {
        let directory = tree.dir(&format!("models/unreadable-{index:03}"));
        std::fs::set_permissions(&directory, std::fs::Permissions::from_mode(0o000))
            .expect("make fixture unreadable");
        unreadable.push(directory);
    }
    let restore = RestorePermissions(unreadable);

    let scan = scan_model_folder(&root).expect("scan");

    assert_eq!(scan.errors.len(), MAX_REPORTED_ERRORS);
    assert!(scan.counts.error_limit_reached);
    drop(restore);
}

#[test]
fn path_id_is_full_sha256_and_does_not_depend_on_file_contents() {
    let tree = TestTree::new("path-id");
    let path = tree.file("models/Stable-7B-Q4_K_M.gguf", b"first contents");
    let canonical = path.canonicalize().expect("canonical");

    let first = local_model_id(&canonical);
    std::fs::write(&path, b"completely different contents").expect("rewrite fixture");
    let second = local_model_id(&canonical);

    assert_eq!(first, second);
    assert_eq!(first.len(), "local:".len() + 64);
    assert!(first.starts_with("local:"));
    #[cfg(unix)]
    assert_eq!(
        local_model_id(Path::new("/models/example.gguf")),
        "local:79fefee63d0266a35985c284b8c92a4fe31247137bbd985e41988d6454144365"
    );
    assert_eq!(
        local_model_id(&tree.root.join("does-not-exist.gguf")).len(),
        "local:".len() + 64,
        "deriving an id must not open or hash model contents"
    );
}

#[test]
fn filename_metadata_is_conservative_and_delimiter_bounded() {
    let (display_name, quant, params) =
        infer_filename_metadata(Path::new("Acme_Model-7B-Q4_K_M.gguf"));
    assert_eq!(display_name, "Acme Model 7B Q4 K M");
    assert_eq!(quant.as_deref(), Some("Q4_K_M"));
    assert_eq!(params.as_deref(), Some("7B"));

    let (_, quant, params) = infer_filename_metadata(Path::new("v2Bfoo-Q99.gguf"));
    assert!(quant.is_none());
    assert!(params.is_none());
}

#[test]
fn filename_metadata_handles_unicode_names_without_losing_obvious_suffixes() {
    let (display_name, quant, params) = infer_filename_metadata(Path::new("模型-7B-Q4_K_M.gguf"));

    assert_eq!(display_name, "模型 7B Q4 K M");
    assert_eq!(quant.as_deref(), Some("Q4_K_M"));
    assert_eq!(params.as_deref(), Some("7B"));
}

#[test]
fn ready_local_reconciliation_is_idempotent_and_ownership_isolated() {
    let tree = TestTree::new("reconcile");
    let folder_a = tree.dir("folder-a").canonicalize().expect("folder a");
    let folder_b = tree.dir("folder-b").canonicalize().expect("folder b");
    let candidate_a = scanned_candidate(&folder_a, "Model-A-7B-Q4_K_M.gguf", b"aaaa");
    let candidate_b = scanned_candidate(&folder_b, "Model-B-3B-Q8_0.gguf", b"bb");
    let folder_a_text = folder_a.to_string_lossy().into_owned();
    let db = Db::open_in_memory().expect("db");

    let scan_a = scan_model_folder(&folder_a).expect("scan a");
    let first_counts = db.commit_model_folder_add(&scan_a).expect("atomic add");
    let folder_a_ticket = db
        .model_folder_scan_tickets()
        .expect("folder a ticket")
        .remove(0);
    assert_eq!(first_counts.registered, 1);
    assert_eq!(
        model_folders(&db).expect("folders"),
        vec![folder_a_text.clone()]
    );

    let first = db
        .get_model(&candidate_a.id)
        .expect("get")
        .expect("registered");
    assert_eq!(first.source, ModelSource::Local);
    assert_eq!(first.status, ModelStatus::Ready);
    assert_eq!(
        first.file_path.as_deref(),
        candidate_a.canonical_path.to_str()
    );
    assert_eq!(first.file_size_bytes, Some(4));
    assert!(first.sha256.is_none());
    assert!(first.hf_repo.is_none());
    assert!(first.family.is_none());
    assert!(first.capabilities.is_empty());
    assert_eq!(
        first.discovered_folder.as_deref(),
        Some(folder_a_text.as_str())
    );

    let second_counts = db
        .reconcile_scanned_folder(&folder_a_ticket, &scan_a)
        .expect("idempotent rescan");
    let second = db
        .get_model(&candidate_a.id)
        .expect("get")
        .expect("registered");
    assert_eq!(second_counts.registered, 0);
    assert_eq!(second_counts.unregistered, 0);
    assert_eq!(second.added_at, first.added_at);
    assert_eq!(second.status, ModelStatus::Ready);

    let scan_b = scan_model_folder(&folder_b).expect("scan b");
    db.commit_model_folder_add(&scan_b).expect("folder b");
    db.upsert_model(&manual_model("manual-local"))
        .expect("manual local");
    let manual_path = tree.file("manual.gguf", b"man");
    db.set_model_ready(
        "manual-local",
        &manual_path.to_string_lossy(),
        3,
        "manual-content-hash",
    )
    .expect("manual ready");

    std::fs::remove_file(&candidate_a.canonical_path).expect("remove external fixture");
    let empty_a = scan_model_folder(&folder_a).expect("empty scan");
    let removed = db
        .reconcile_scanned_folder(&folder_a_ticket, &empty_a)
        .expect("completed empty scan");
    assert_eq!(removed.unregistered, 1);
    assert!(db.get_model(&candidate_a.id).expect("a").is_none());
    assert!(db.get_model(&candidate_b.id).expect("b").is_some());
    let manual = db
        .get_model("manual-local")
        .expect("manual")
        .expect("survives");
    assert!(manual.discovered_folder.is_none());
}

#[test]
fn an_offline_root_is_an_error_and_cannot_be_reconciled_as_empty() {
    let tree = TestTree::new("offline-root");
    let folder = tree.dir("external-models").canonicalize().expect("folder");
    let candidate = scanned_candidate(&folder, "offline.gguf", b"bytes");
    let db = Db::open_in_memory().expect("db");
    let scan = scan_model_folder(&folder).expect("scan");
    db.commit_model_folder_add(&scan)
        .expect("initial registration");

    std::fs::remove_dir_all(&folder).expect("simulate unplugged root");
    assert!(scan_model_folder(&folder).is_err());
    assert!(
        db.get_model(&candidate.id).expect("lookup").is_some(),
        "a failed scan supplies no successful candidate set to reconcile"
    );
}

#[test]
fn removing_a_scanner_owned_local_model_unregisters_without_touching_external_bytes() {
    let tree = TestTree::new("external-delete");
    let external = tree
        .dir("external-models")
        .canonicalize()
        .expect("external");
    let candidate = scanned_candidate(&external, "keep-me.gguf", b"external bytes");
    let db = Db::open_in_memory().expect("db");
    let scan = scan_model_folder(&external).expect("scan");
    db.commit_model_folder_add(&scan).expect("register");
    let paths = Paths::for_root(tree.root.join("kuro-home"));

    remove_model(&db, &paths, &candidate.id).expect("unregister local");

    assert!(
        candidate.canonical_path.is_file(),
        "external bytes must survive"
    );
    assert!(db.get_model(&candidate.id).expect("lookup").is_none());
}

#[test]
fn removing_a_folder_unregisters_only_its_rows_and_preserves_external_bytes() {
    let tree = TestTree::new("folder-delete");
    let folder_a = tree.dir("external-a").canonicalize().expect("a");
    let folder_b = tree.dir("external-b").canonicalize().expect("b");
    let candidate_a = scanned_candidate(&folder_a, "a.gguf", b"a");
    let candidate_b = scanned_candidate(&folder_b, "b.gguf", b"b");
    let folder_a_text = folder_a.to_string_lossy().into_owned();
    let folder_b_text = folder_b.to_string_lossy().into_owned();
    let db = Db::open_in_memory().expect("db");
    let scan_a = scan_model_folder(&folder_a).expect("scan a");
    let scan_b = scan_model_folder(&folder_b).expect("scan b");
    db.commit_model_folder_add(&scan_a).expect("a");
    db.commit_model_folder_add(&scan_b).expect("b");

    let counts = db
        .commit_model_folder_remove(&folder_a_text)
        .expect("atomic folder removal");

    assert_eq!(counts.unregistered, 1);
    assert_eq!(
        model_folders(&db).expect("remaining folders"),
        vec![folder_b_text]
    );
    assert!(db.get_model(&candidate_a.id).expect("a").is_none());
    assert!(db.get_model(&candidate_b.id).expect("b").is_some());
    assert!(
        candidate_a.canonical_path.is_file(),
        "folder removal is DB-only"
    );
    assert!(candidate_b.canonical_path.is_file());
}

#[test]
fn removing_a_managed_download_still_deletes_its_file_and_directory() {
    let tree = TestTree::new("managed-delete");
    let paths = Paths::for_root(tree.root.join("kuro-home"));
    let model_id = "managed-model:q4_k_m";
    let managed_file = paths.model_file(model_id, "weights.gguf");
    std::fs::create_dir_all(managed_file.parent().expect("parent")).expect("managed directory");
    std::fs::write(&managed_file, b"managed bytes").expect("managed fixture");
    let db = Db::open_in_memory().expect("db");
    let mut model = manual_model(model_id);
    model.source = ModelSource::Curated;
    db.upsert_model(&model).expect("managed row");
    db.set_model_ready(
        model_id,
        &managed_file.to_string_lossy(),
        13,
        "managed-hash",
    )
    .expect("managed ready");

    remove_model(&db, &paths, model_id).expect("remove managed model");

    assert!(!managed_file.exists());
    assert!(!managed_file.parent().expect("parent").exists());
    assert!(db.get_model(model_id).expect("lookup").is_none());
}
