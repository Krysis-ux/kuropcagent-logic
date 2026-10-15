use std::path::{Path, PathBuf};

use kuro_core::catalog::{
    scan_model_folder, MAX_RESULTS, MAX_VISITED_DIRECTORIES, MAX_VISITED_ENTRIES,
};

struct TestTree(PathBuf);

impl TestTree {
    fn new(label: &str) -> Self {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/kuro-local-model-limit-tests")
            .join(format!("{label}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).expect("create tree");
        Self(root.canonicalize().expect("canonical test tree"))
    }

    fn dir(&self, relative: &str) -> PathBuf {
        let path = self.0.join(relative);
        std::fs::create_dir_all(&path).expect("create directory");
        path
    }

    fn file(&self, relative: &str) {
        let path = self.0.join(relative);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create parent");
        }
        std::fs::write(path, b"x").expect("write fixture");
    }
}

impl Drop for TestTree {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn one_wide_directory_cannot_exceed_the_total_entry_work_budget() {
    let tree = TestTree::new("wide-directory");
    let root = tree.dir("models");
    for index in 0..=MAX_VISITED_ENTRIES {
        tree.file(&format!("models/not-a-model-{index:05}.txt"));
    }

    let scan = scan_model_folder(&root).expect("bounded scan");

    assert!(scan.counts.entry_limit_reached);
    assert_eq!(scan.counts.visited_entries as usize, MAX_VISITED_ENTRIES);
    assert!(scan.counts.skipped_non_gguf as usize <= MAX_VISITED_ENTRIES);
    assert!(scan.counts.visited_directories as usize <= MAX_VISITED_DIRECTORIES);
}

#[test]
fn over_budget_directories_discard_their_unordered_prefix() {
    let forward = TestTree::new("wide-forward");
    let reverse = TestTree::new("wide-reverse");
    let forward_root = forward.dir("models");
    let reverse_root = reverse.dir("models");
    for index in 0..=MAX_VISITED_ENTRIES {
        forward.file(&format!("models/model-{index:05}.gguf"));
    }
    for index in (0..=MAX_VISITED_ENTRIES).rev() {
        reverse.file(&format!("models/model-{index:05}.gguf"));
    }

    let forward_scan = scan_model_folder(&forward_root).expect("forward scan");
    let reverse_scan = scan_model_folder(&reverse_root).expect("reverse scan");

    assert!(forward_scan.counts.entry_limit_reached);
    assert!(reverse_scan.counts.entry_limit_reached);
    assert!(forward_scan.candidates.is_empty());
    assert!(reverse_scan.candidates.is_empty());
    assert_eq!(forward_scan.counts, reverse_scan.counts);
}

#[cfg(unix)]
#[test]
fn traversal_stops_before_later_subtrees_once_the_result_limit_is_exceeded() {
    use std::os::unix::fs::PermissionsExt;

    let tree = TestTree::new("result-stop");
    let root = tree.dir("models");
    for index in 0..=MAX_RESULTS {
        tree.file(&format!("models/aa-model-{index:04}.gguf"));
    }
    let unreadable = tree.dir("models/zz-after-limit");
    std::fs::set_permissions(&unreadable, std::fs::Permissions::from_mode(0o000))
        .expect("make unreadable");

    let scan = scan_model_folder(&root).expect("bounded scan");
    std::fs::set_permissions(&unreadable, std::fs::Permissions::from_mode(0o700))
        .expect("restore permissions");

    assert!(scan.counts.result_limit_reached);
    assert_eq!(scan.candidates.len(), MAX_RESULTS);
    assert!(
        scan.errors.is_empty(),
        "the scanner must not continue into later subtrees after proving truncation"
    );
}
