//! Bounded discovery of GGUF files stored outside Kuro's managed model directory.

#[cfg(any(unix, windows))]
use std::ffi::OsStr;
use std::fs::{File, Metadata};
#[cfg(any(unix, windows))]
use std::path::Component;
use std::path::{Path, PathBuf};

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::{KuroError, Result};

pub const MAX_SCAN_DEPTH: usize = 4;
pub const MAX_VISITED_DIRECTORIES: usize = 2_000;
pub const MAX_VISITED_ENTRIES: usize = 4_000;
pub const MAX_RESULTS: usize = 500;
pub const MAX_REPORTED_ERRORS: usize = 50;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct ScanCounts {
    pub discovered: u32,
    pub registered: u32,
    pub updated: u32,
    pub unregistered: u32,
    pub conflicts: u32,
    pub visited_directories: u32,
    pub visited_entries: u32,
    pub skipped_symlinks: u32,
    pub skipped_non_regular: u32,
    pub skipped_non_gguf: u32,
    pub skipped_outside_root: u32,
    pub directory_limit_reached: bool,
    pub entry_limit_reached: bool,
    pub result_limit_reached: bool,
    /// The public error vector stopped at [`MAX_REPORTED_ERRORS`].
    pub error_limit_reached: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FolderScanError {
    pub path: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalModelCandidate {
    pub id: String,
    pub display_name: String,
    pub quant: Option<String>,
    pub param_count: Option<String>,
    pub canonical_path: PathBuf,
    pub file_size_bytes: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FolderScan {
    requested_root: PathBuf,
    bound_root: Option<BoundScanRoot>,
    pub candidates: Vec<LocalModelCandidate>,
    pub counts: ScanCounts,
    pub errors: Vec<FolderScanError>,
    authoritative_candidate_ids: Option<Vec<String>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BoundScanRoot {
    canonical_path: PathBuf,
    identity: FileIdentity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FileIdentity {
    #[cfg(unix)]
    Unix { device: u64, inode: u64 },
    #[cfg(windows)]
    Windows { volume_serial: u32, file_index: u64 },
    #[cfg(not(any(unix, windows)))]
    Unsupported,
}

impl FolderScan {
    pub fn requested_root(&self) -> &Path {
        &self.requested_root
    }

    pub fn canonical_root(&self) -> Option<&Path> {
        self.bound_root
            .as_ref()
            .map(|bound| bound.canonical_path.as_path())
    }

    pub fn root_is_bound(&self) -> bool {
        self.bound_root.is_some()
    }

    pub(crate) fn committable_root(&self) -> Result<&Path> {
        let Some(bound) = &self.bound_root else {
            return Err(KuroError::bad_request(format!(
                "model folder `{}` changed during scanning",
                self.requested_root.display()
            )));
        };
        if self.requested_root != bound.canonical_path {
            return Err(KuroError::bad_request(format!(
                "model folder `{}` is not canonical",
                self.requested_root.display()
            )));
        }
        Ok(&bound.canonical_path)
    }

    pub(crate) fn verify_root_binding(&self) -> Result<()> {
        let bound = self.bound_root.as_ref().ok_or_else(|| {
            KuroError::bad_request(format!(
                "model folder `{}` changed during scanning",
                self.requested_root.display()
            ))
        })?;
        let reopened = open_requested_root(&self.requested_root, bound.identity)?;
        let reopened_identity = file_identity(&reopened.metadata().map_err(|error| {
            KuroError::bad_request(format!(
                "model folder `{}` changed before it could be selected: {error}",
                self.requested_root.display()
            ))
        })?)?;
        if reopened_identity != bound.identity {
            return Err(KuroError::bad_request(format!(
                "model folder `{}` changed before it could be selected",
                self.requested_root.display()
            )));
        }
        Ok(())
    }

    /// Only a complete observation can prove that an owned row disappeared.
    pub fn is_complete(&self) -> bool {
        let (Some(bound_root), Some(authoritative_ids)) =
            (&self.bound_root, &self.authoritative_candidate_ids)
        else {
            return false;
        };
        self.observation_is_complete()
            && bound_root.canonical_path == self.requested_root
            && authoritative_ids.len() == self.candidates.len()
            && authoritative_ids
                .iter()
                .zip(&self.candidates)
                .all(|(id, candidate)| id == &candidate.id)
    }

    fn observation_is_complete(&self) -> bool {
        self.errors.is_empty()
            && !self.counts.directory_limit_reached
            && !self.counts.entry_limit_reached
            && !self.counts.result_limit_reached
            && !self.counts.error_limit_reached
    }
}

/// Validate and canonicalize a prospective selected folder.
pub fn validate_model_folder_root(root: &Path, existing: &[String]) -> Result<PathBuf> {
    validate_root_lexical_form(root)?;
    validate_root_metadata(root)?;
    let canonical = canonicalize_root(root)?;
    validate_root_after_canonicalization(root, &canonical)?;
    validate_root_overlaps(&canonical, existing)?;
    Ok(canonical)
}

fn validate_root_lexical_form(root: &Path) -> Result<()> {
    let root_text = root.to_str().ok_or_else(|| {
        KuroError::bad_request("model folder path is not valid UTF-8 and cannot be stored")
    })?;
    if !root.is_absolute()
        || root_text
            .split(['/', '\\'])
            .any(|component| matches!(component, "." | ".."))
    {
        return Err(KuroError::bad_request(format!(
            "model folder `{}` is not canonical",
            root.display()
        )));
    }
    Ok(())
}

fn validate_root_metadata(root: &Path) -> Result<()> {
    let metadata = std::fs::symlink_metadata(root).map_err(|error| {
        KuroError::bad_request(format!(
            "model folder `{}` is unavailable: {error}",
            root.display()
        ))
    })?;
    if metadata.file_type().is_symlink() {
        return Err(KuroError::bad_request(format!(
            "model folder `{}` cannot be a symlink",
            root.display()
        )));
    }
    if !metadata.is_dir() {
        return Err(KuroError::bad_request(format!(
            "model folder `{}` is not a directory",
            root.display()
        )));
    }

    Ok(())
}

fn canonicalize_root(root: &Path) -> Result<PathBuf> {
    root.canonicalize().map_err(|error| {
        KuroError::bad_request(format!(
            "model folder `{}` could not be canonicalized: {error}",
            root.display()
        ))
    })
}

fn validate_root_after_canonicalization(root: &Path, canonical: &Path) -> Result<()> {
    validate_root_metadata(root)?;
    if canonical != root {
        return Err(KuroError::bad_request(format!(
            "model folder `{}` is not canonical",
            root.display()
        )));
    }
    let fresh = canonicalize_root(root)?;
    if fresh != canonical {
        return Err(KuroError::bad_request(format!(
            "model folder `{}` changed during validation",
            root.display()
        )));
    }
    Ok(())
}

fn validate_root_overlaps(canonical: &Path, existing: &[String]) -> Result<()> {
    for selected in existing {
        let selected = Path::new(selected);
        if canonical == selected {
            return Err(KuroError::bad_request(format!(
                "model folder `{}` is already selected",
                canonical.display()
            )));
        }
        if canonical.starts_with(selected) || selected.starts_with(canonical) {
            return Err(KuroError::bad_request(format!(
                "model folder `{}` overlaps selected folder `{}`",
                canonical.display(),
                selected.display()
            )));
        }
    }
    Ok(())
}

/// Scan a selected root without following links or reading model contents.
#[cfg(any(unix, windows))]
pub fn scan_model_folder(root: &Path) -> Result<FolderScan> {
    scan_model_folder_with_hook(root, |_, _| {})
}

/// Secure handle-relative traversal is currently implemented for Unix and Windows.
#[cfg(not(any(unix, windows)))]
pub fn scan_model_folder(_root: &Path) -> Result<FolderScan> {
    Err(KuroError::bad_request(
        "local model folders are not supported on this platform",
    ))
}

#[cfg(any(unix, windows))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScanTestEvent {
    AfterRootMetadataBeforeCanonicalization,
    BeforeRootHandleOpen,
    AfterRootHandleOpenAttempt,
    AfterRootHandleOpen,
    BeforeDirectoryRead,
    AfterDirectoryValidationBeforeOpen,
    BeforeDirectoryHandleOpen,
    AfterDirectoryHandleOpenAttempt,
    AfterDirectoryHandleOpen,
    BeforeFileAcceptance,
    BeforeFileHandleOpen,
    AfterFileHandleOpenAttempt,
    AfterFileHandleOpen,
}

#[cfg(all(test, any(unix, windows)))]
fn scan_model_folder_with_test_hook(
    root: &Path,
    hook: impl FnMut(ScanTestEvent, &Path),
) -> Result<FolderScan> {
    scan_model_folder_with_hook(root, hook)
}

#[cfg(any(unix, windows))]
fn scan_model_folder_with_hook(
    root: &Path,
    mut hook: impl FnMut(ScanTestEvent, &Path),
) -> Result<FolderScan> {
    validate_root_lexical_form(root)?;
    let initial_metadata = std::fs::symlink_metadata(root).map_err(|error| {
        KuroError::bad_request(format!(
            "model folder `{}` is unavailable: {error}",
            root.display()
        ))
    })?;
    if initial_metadata.file_type().is_symlink() || !initial_metadata.is_dir() {
        return Err(KuroError::bad_request(format!(
            "model folder `{}` is not a regular directory",
            root.display()
        )));
    }
    let initial_identity = file_identity(&initial_metadata)?;
    hook(ScanTestEvent::AfterRootMetadataBeforeCanonicalization, root);
    let mut scan = FolderScan {
        requested_root: root.to_path_buf(),
        bound_root: None,
        candidates: Vec::new(),
        counts: ScanCounts::default(),
        errors: Vec::new(),
        authoritative_candidate_ids: None,
    };

    let canonical_root = match root.canonicalize() {
        Ok(canonical) if canonical == root => canonical,
        Ok(canonical) => {
            let is_static_alias = std::fs::metadata(&canonical)
                .ok()
                .and_then(|metadata| file_identity(&metadata).ok())
                == Some(initial_identity);
            if is_static_alias {
                return Err(KuroError::bad_request(format!(
                    "model folder `{}` is not canonical",
                    root.display()
                )));
            }
            record_error(
                &mut scan,
                root,
                format!("model folder `{}` changed during scanning", root.display()),
            );
            return Ok(scan);
        }
        Err(error) => {
            record_error(&mut scan, root, error.to_string());
            return Ok(scan);
        }
    };

    hook(ScanTestEvent::BeforeRootHandleOpen, root);
    let root_handle = open_requested_root(root, initial_identity);
    hook(ScanTestEvent::AfterRootHandleOpenAttempt, root);
    let mut root_handle = match root_handle {
        Ok(handle) => handle,
        Err(error) => {
            record_error(&mut scan, root, error.to_string());
            return Ok(scan);
        }
    };
    hook(ScanTestEvent::AfterRootHandleOpen, root);

    let handle_metadata = match root_handle.metadata() {
        Ok(metadata) => metadata,
        Err(error) => {
            record_error(&mut scan, root, error.to_string());
            return Ok(scan);
        }
    };
    if !handle_metadata.is_dir() || file_identity(&handle_metadata)? != initial_identity {
        record_error(
            &mut scan,
            root,
            format!("model folder `{}` changed during scanning", root.display()),
        );
        return Ok(scan);
    }
    match root.canonicalize() {
        Ok(final_root) if final_root == root => {}
        Ok(_) => {
            record_error(
                &mut scan,
                root,
                format!("model folder `{}` changed during scanning", root.display()),
            );
            return Ok(scan);
        }
        Err(error) => {
            record_error(&mut scan, root, error.to_string());
            return Ok(scan);
        }
    }
    scan.bound_root = Some(BoundScanRoot {
        canonical_path: canonical_root.clone(),
        identity: initial_identity,
    });

    let mut enqueued_directories = 1usize;
    let mut stop_scan = false;
    scan_open_directory(
        &mut root_handle,
        &canonical_root,
        Path::new(""),
        0,
        &mut scan,
        &mut enqueued_directories,
        &mut stop_scan,
        &mut hook,
    );

    scan.candidates
        .sort_by(|left, right| left.canonical_path.cmp(&right.canonical_path));
    scan.counts.discovered = scan.candidates.len() as u32;
    if scan.observation_is_complete() && scan.bound_root.is_some() {
        scan.authoritative_candidate_ids = Some(
            scan.candidates
                .iter()
                .map(|candidate| candidate.id.clone())
                .collect(),
        );
    }
    Ok(scan)
}

#[cfg(any(unix, windows))]
#[allow(clippy::too_many_arguments)]
fn scan_open_directory(
    handle: &mut File,
    canonical_root: &Path,
    relative_directory: &Path,
    depth: usize,
    scan: &mut FolderScan,
    enqueued_directories: &mut usize,
    stop_scan: &mut bool,
    hook: &mut impl FnMut(ScanTestEvent, &Path),
) {
    if *stop_scan {
        return;
    }
    let directory_path = canonical_root.join(relative_directory);
    let mut entries = match fs_at::read_dir(handle) {
        Ok(entries) => entries,
        Err(error) => {
            record_error(scan, &directory_path, error.to_string());
            return;
        }
    };
    scan.counts.visited_directories = scan.counts.visited_directories.saturating_add(1);

    let remaining_entries =
        MAX_VISITED_ENTRIES.saturating_sub(scan.counts.visited_entries as usize);
    let mut names = Vec::new();
    let mut directory_entries = 0usize;
    while directory_entries < remaining_entries {
        let Some(entry) = entries.next() else {
            break;
        };
        if entry
            .as_ref()
            .is_ok_and(|entry| matches!(entry.name().to_str(), Some("." | "..")))
        {
            continue;
        }
        directory_entries += 1;
        scan.counts.visited_entries = scan.counts.visited_entries.saturating_add(1);
        match entry {
            Ok(entry) if is_one_normal_component(entry.name()) => {
                names.push(entry.name().to_os_string())
            }
            Ok(entry) => record_error(
                scan,
                &directory_path,
                format!(
                    "invalid directory entry `{}`",
                    entry.name().to_string_lossy()
                ),
            ),
            Err(error) => record_error(scan, &directory_path, error.to_string()),
        }
    }
    if directory_entries == remaining_entries {
        scan.counts.entry_limit_reached = true;
        *stop_scan = true;
        return;
    }
    drop(entries);
    names.sort();

    for name in names {
        if *stop_scan {
            break;
        }
        let relative_path = relative_directory.join(&name);
        let path = canonical_root.join(&relative_path);

        match precheck_entry_kind(handle, &name) {
            Ok(Some(PrecheckedEntryKind::Symlink)) => {
                scan.counts.skipped_symlinks = scan.counts.skipped_symlinks.saturating_add(1);
                continue;
            }
            Ok(Some(PrecheckedEntryKind::NonRegular)) => {
                scan.counts.skipped_non_regular = scan.counts.skipped_non_regular.saturating_add(1);
                continue;
            }
            Ok(None | Some(PrecheckedEntryKind::FileOrDirectory)) => {}
            Err(error) => {
                record_error(scan, &path, error.to_string());
                continue;
            }
        }

        hook(ScanTestEvent::BeforeDirectoryRead, &path);
        hook(ScanTestEvent::AfterDirectoryValidationBeforeOpen, &path);
        hook(ScanTestEvent::BeforeDirectoryHandleOpen, &path);
        hook(ScanTestEvent::BeforeFileAcceptance, &path);
        hook(ScanTestEvent::BeforeFileHandleOpen, &path);
        let child = open_child_nofollow(handle, &name);
        hook(ScanTestEvent::AfterDirectoryHandleOpenAttempt, &path);
        hook(ScanTestEvent::AfterFileHandleOpenAttempt, &path);
        let mut child = match child {
            Ok(child) => child,
            Err(error) if is_symlink_open_error(&error) => {
                scan.counts.skipped_symlinks = scan.counts.skipped_symlinks.saturating_add(1);
                record_error(
                    scan,
                    &path,
                    format!("entry changed during scanning: {error}"),
                );
                continue;
            }
            Err(error) => {
                record_error(scan, &path, error.to_string());
                continue;
            }
        };
        hook(ScanTestEvent::AfterDirectoryHandleOpen, &path);
        hook(ScanTestEvent::AfterFileHandleOpen, &path);

        let metadata = match child.metadata() {
            Ok(metadata) => metadata,
            Err(error) => {
                record_error(scan, &path, error.to_string());
                continue;
            }
        };
        if metadata.is_dir() {
            if depth < MAX_SCAN_DEPTH {
                if *enqueued_directories >= MAX_VISITED_DIRECTORIES {
                    scan.counts.directory_limit_reached = true;
                } else {
                    *enqueued_directories += 1;
                    scan_open_directory(
                        &mut child,
                        canonical_root,
                        &relative_path,
                        depth + 1,
                        scan,
                        enqueued_directories,
                        stop_scan,
                        hook,
                    );
                }
            }
            continue;
        }
        if !metadata.is_file() {
            scan.counts.skipped_non_regular = scan.counts.skipped_non_regular.saturating_add(1);
            continue;
        }
        if !has_gguf_extension(&path) {
            scan.counts.skipped_non_gguf = scan.counts.skipped_non_gguf.saturating_add(1);
            continue;
        }
        if path.file_stem().and_then(|stem| stem.to_str()).is_none() {
            record_error(scan, &path, "GGUF filename is not valid UTF-8");
            continue;
        }
        if scan.candidates.len() >= MAX_RESULTS {
            scan.counts.result_limit_reached = true;
            *stop_scan = true;
            break;
        }
        let file_size_bytes = match i64::try_from(metadata.len()) {
            Ok(size) => size,
            Err(_) => {
                record_error(scan, &path, "GGUF file is too large to represent");
                continue;
            }
        };
        let (display_name, quant, param_count) = infer_filename_metadata(&path);
        scan.candidates.push(LocalModelCandidate {
            id: local_model_id(&path),
            display_name,
            quant,
            param_count,
            canonical_path: path,
            file_size_bytes,
        });
    }
}

#[cfg(any(unix, windows))]
fn is_one_normal_component(name: &OsStr) -> bool {
    let mut components = Path::new(name).components();
    matches!(components.next(), Some(Component::Normal(component)) if component == name)
        && components.next().is_none()
}

#[cfg(any(unix, windows))]
fn open_child_nofollow(parent: &File, name: &OsStr) -> std::io::Result<File> {
    let mut options = fs_at::OpenOptions::default();
    options.read(true).follow(false);
    options.open_dir_at(parent, name)
}

#[cfg(any(unix, windows))]
fn open_requested_root(root: &Path, expected: FileIdentity) -> Result<File> {
    let mut current = open_namespace_anchor(root).map_err(|error| {
        KuroError::bad_request(format!(
            "model folder `{}` could not open its filesystem root: {error}",
            root.display()
        ))
    })?;
    for component in root.components() {
        match component {
            Component::Prefix(_) | Component::RootDir => {}
            Component::Normal(name) => {
                current = open_child_nofollow(&current, name).map_err(|error| {
                    KuroError::bad_request(format!(
                        "model folder `{}` changed during scanning: {error}",
                        root.display()
                    ))
                })?;
                let metadata = current.metadata().map_err(|error| {
                    KuroError::bad_request(format!(
                        "model folder `{}` changed during scanning: {error}",
                        root.display()
                    ))
                })?;
                if !metadata.is_dir() {
                    return Err(KuroError::bad_request(format!(
                        "model folder `{}` is not a regular directory",
                        root.display()
                    )));
                }
            }
            Component::CurDir | Component::ParentDir => {
                return Err(KuroError::bad_request(format!(
                    "model folder `{}` is not canonical",
                    root.display()
                )))
            }
        }
    }
    let metadata = current.metadata().map_err(|error| {
        KuroError::bad_request(format!(
            "model folder `{}` changed during scanning: {error}",
            root.display()
        ))
    })?;
    if !metadata.is_dir() || file_identity(&metadata)? != expected {
        return Err(KuroError::bad_request(format!(
            "model folder `{}` changed during scanning",
            root.display()
        )));
    }
    Ok(current)
}

#[cfg(not(any(unix, windows)))]
fn open_requested_root(_root: &Path, _expected: FileIdentity) -> Result<File> {
    Err(KuroError::bad_request(
        "local model folders are not supported on this platform",
    ))
}

#[cfg(unix)]
fn open_namespace_anchor(_root: &Path) -> std::io::Result<File> {
    File::open(Path::new("/"))
}

#[cfg(windows)]
fn open_namespace_anchor(root: &Path) -> std::io::Result<File> {
    use std::os::windows::fs::OpenOptionsExt;
    use windows_sys::Win32::Storage::FileSystem::FILE_FLAG_BACKUP_SEMANTICS;

    let mut components = root.components();
    let Some(Component::Prefix(prefix)) = components.next() else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "model folder is missing a Windows path prefix",
        ));
    };
    let mut anchor = PathBuf::from(prefix.as_os_str());
    anchor.push(Path::new(r"\"));
    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
        .open(anchor)
}

#[cfg(unix)]
fn file_identity(metadata: &Metadata) -> Result<FileIdentity> {
    use std::os::unix::fs::MetadataExt;
    Ok(FileIdentity::Unix {
        device: metadata.dev(),
        inode: metadata.ino(),
    })
}

#[cfg(windows)]
fn file_identity(metadata: &Metadata) -> Result<FileIdentity> {
    use std::os::windows::fs::MetadataExt;
    let volume_serial = metadata
        .volume_serial_number()
        .ok_or_else(|| KuroError::bad_request("model folder filesystem identity is unavailable"))?;
    let file_index = metadata
        .file_index()
        .ok_or_else(|| KuroError::bad_request("model folder filesystem identity is unavailable"))?;
    Ok(FileIdentity::Windows {
        volume_serial,
        file_index,
    })
}

#[cfg(not(any(unix, windows)))]
fn file_identity(_metadata: &Metadata) -> Result<FileIdentity> {
    let _ = FileIdentity::Unsupported;
    Err(KuroError::bad_request(
        "local model folders are not supported on this platform",
    ))
}

#[cfg(any(unix, windows))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrecheckedEntryKind {
    Symlink,
    NonRegular,
    FileOrDirectory,
}

#[cfg(unix)]
fn precheck_entry_kind(
    parent: &File,
    name: &OsStr,
) -> std::io::Result<Option<PrecheckedEntryKind>> {
    use std::os::fd::AsRawFd;
    use std::os::unix::ffi::OsStrExt;

    let name = std::ffi::CString::new(name.as_bytes())
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidInput, error))?;
    let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
    let result = unsafe {
        libc::fstatat(
            parent.as_raw_fd(),
            name.as_ptr(),
            stat.as_mut_ptr(),
            libc::AT_SYMLINK_NOFOLLOW,
        )
    };
    if result != 0 {
        return Err(std::io::Error::last_os_error());
    }
    let mode = unsafe { stat.assume_init() }.st_mode;
    let kind = mode & libc::S_IFMT;
    if kind == libc::S_IFLNK {
        Ok(Some(PrecheckedEntryKind::Symlink))
    } else if kind == libc::S_IFREG || kind == libc::S_IFDIR {
        Ok(Some(PrecheckedEntryKind::FileOrDirectory))
    } else {
        Ok(Some(PrecheckedEntryKind::NonRegular))
    }
}

#[cfg(windows)]
fn precheck_entry_kind(
    _parent: &File,
    _name: &OsStr,
) -> std::io::Result<Option<PrecheckedEntryKind>> {
    Ok(None)
}

#[cfg(unix)]
fn is_symlink_open_error(error: &std::io::Error) -> bool {
    error.raw_os_error() == Some(libc::ELOOP)
}

#[cfg(windows)]
fn is_symlink_open_error(error: &std::io::Error) -> bool {
    error.kind() == std::io::ErrorKind::FilesystemLoop
}

/// Stable id derived only from the canonical path bytes, never weight bytes.
pub fn local_model_id(canonical_path: &Path) -> String {
    let mut digest = Sha256::new();
    update_digest_with_path(&mut digest, canonical_path);
    format!("local:{:x}", digest.finalize())
}

#[cfg(unix)]
fn update_digest_with_path(digest: &mut Sha256, path: &Path) {
    use std::os::unix::ffi::OsStrExt;
    digest.update(path.as_os_str().as_bytes());
}

#[cfg(windows)]
fn update_digest_with_path(digest: &mut Sha256, path: &Path) {
    use std::os::windows::ffi::OsStrExt;
    for unit in path.as_os_str().encode_wide() {
        digest.update(unit.to_le_bytes());
    }
}

#[cfg(not(any(unix, windows)))]
fn update_digest_with_path(digest: &mut Sha256, path: &Path) {
    digest.update(path.as_os_str().to_string_lossy().as_bytes());
}

/// Infer only metadata the filename states unambiguously.
pub fn infer_filename_metadata(path: &Path) -> (String, Option<String>, Option<String>) {
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("Local model");
    let mut display_name = String::with_capacity(stem.len());
    let mut previous_space = false;
    for character in stem.chars() {
        if character.is_alphanumeric() {
            display_name.push(character);
            previous_space = false;
        } else if !previous_space && !display_name.is_empty() {
            display_name.push(' ');
            previous_space = true;
        }
    }
    let display_name = display_name.trim().to_string();
    let display_name = if display_name.is_empty() {
        "Local model".to_string()
    } else {
        display_name
    };

    (display_name, infer_quant(stem), infer_parameter_count(stem))
}

fn infer_quant(stem: &str) -> Option<String> {
    let upper = stem.to_ascii_uppercase();
    let bytes = upper.as_bytes();
    for (start, _) in upper.char_indices() {
        if start > 0 && bytes[start - 1].is_ascii_alphanumeric() {
            continue;
        }
        let rest = &upper[start..];
        let end = rest.find(['-', '.']).unwrap_or(rest.len());
        let token = &rest[..end];
        if is_obvious_quant(token) {
            return Some(token.to_string());
        }
    }
    None
}

fn is_obvious_quant(token: &str) -> bool {
    let suffix = if let Some(rest) = token.strip_prefix("IQ") {
        let Some(digit) = rest.as_bytes().first().copied() else {
            return false;
        };
        if !(b'1'..=b'4').contains(&digit) {
            return false;
        }
        &rest[1..]
    } else if let Some(rest) = token.strip_prefix('Q') {
        let Some(digit) = rest.as_bytes().first().copied() else {
            return false;
        };
        if !(b'2'..=b'8').contains(&digit) {
            return false;
        }
        &rest[1..]
    } else {
        return false;
    };
    suffix.is_empty()
        || suffix.strip_prefix('_').is_some_and(|parts| {
            !parts.is_empty()
                && parts
                    .split('_')
                    .all(|part| !part.is_empty() && part.chars().all(|c| c.is_ascii_alphanumeric()))
        })
}

fn infer_parameter_count(stem: &str) -> Option<String> {
    let upper = stem.to_ascii_uppercase();
    let bytes = upper.as_bytes();
    for end in 1..bytes.len() {
        if !matches!(bytes[end], b'B' | b'M') {
            continue;
        }
        if end + 1 < bytes.len() && bytes[end + 1].is_ascii_alphanumeric() {
            continue;
        }
        let mut start = end;
        while start > 0 && (bytes[start - 1].is_ascii_digit() || bytes[start - 1] == b'.') {
            start -= 1;
        }
        if start > 0 && bytes[start - 1].is_ascii_alphanumeric() {
            continue;
        }
        let number = &upper[start..end];
        if valid_decimal(number) {
            return Some(upper[start..=end].to_string());
        }
    }
    None
}

fn valid_decimal(raw: &str) -> bool {
    let mut dot_seen = false;
    let mut digit_seen = false;
    for character in raw.chars() {
        if character.is_ascii_digit() {
            digit_seen = true;
        } else if character == '.' && !dot_seen {
            dot_seen = true;
        } else {
            return false;
        }
    }
    digit_seen && !raw.starts_with('.') && !raw.ends_with('.')
}

#[cfg(any(unix, windows))]
fn has_gguf_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
}

#[cfg(any(unix, windows))]
fn record_error(scan: &mut FolderScan, path: &Path, message: impl Into<String>) {
    if scan.errors.len() < MAX_REPORTED_ERRORS {
        scan.errors.push(FolderScanError {
            path: path.to_string_lossy().into_owned(),
            message: message.into(),
        });
    } else {
        scan.counts.error_limit_reached = true;
    }
}

#[cfg(all(test, any(unix, windows)))]
mod replacement_race_tests {
    use super::*;

    struct TestTree(PathBuf);

    impl TestTree {
        fn new(label: &str) -> Self {
            let root = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../target/kuro-local-model-race-tests")
                .join(format!("{label}-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir_all(&root).expect("create tree");
            Self(root.canonicalize().expect("canonical test tree"))
        }

        fn dir(&self, relative: &str) -> PathBuf {
            let path = self.0.join(relative);
            std::fs::create_dir_all(&path).expect("create directory");
            path
        }

        fn file(&self, relative: &str, bytes: &[u8]) -> PathBuf {
            let path = self.0.join(relative);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).expect("create parent");
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

    #[cfg(unix)]
    #[test]
    fn root_replaced_after_metadata_before_canonicalization_is_a_partial_error() {
        use std::os::unix::fs::symlink;

        let tree = TestTree::new("root-race");
        let root = tree.dir("models");
        tree.file("models/original.gguf", b"original");
        let outside = tree.dir("outside");
        tree.file("outside/escaped.gguf", b"escaped");
        let mut replaced = false;

        let scan = scan_model_folder_with_test_hook(&root, |event, path| {
            if !replaced
                && event == ScanTestEvent::AfterRootMetadataBeforeCanonicalization
                && path == root
            {
                std::fs::remove_dir_all(path).expect("remove selected root");
                symlink(&outside, path).expect("replace root with symlink");
                replaced = true;
            }
        })
        .expect("partial scan");

        assert!(replaced);
        assert!(scan.candidates.is_empty());
        assert_eq!(scan.errors.len(), 1);
        assert!(!scan.is_complete());
        assert_eq!(scan.requested_root(), root);
        assert!(scan.canonical_root().is_none());

        let db = crate::Db::open_in_memory().expect("db");
        assert!(matches!(
            db.commit_model_folder_add(&scan),
            Err(KuroError::BadRequest(_))
        ));
        assert!(crate::settings::model_folders(&db)
            .expect("folders")
            .is_empty());
        db.with(|conn| {
            assert_eq!(
                conn.query_row("SELECT COUNT(*) FROM model_folder_roots", [], |row| {
                    row.get::<_, i64>(0)
                })?,
                0
            );
            assert_eq!(
                conn.query_row("SELECT COUNT(*) FROM models", [], |row| {
                    row.get::<_, i64>(0)
                })?,
                0
            );
            Ok(())
        })
        .expect("unchanged database");
    }

    #[cfg(unix)]
    #[test]
    fn root_swap_open_restore_cannot_redirect_enumeration() {
        use std::os::unix::fs::symlink;

        let tree = TestTree::new("root-swap-open-restore");
        let root = tree.dir("models");
        tree.file("models/original.gguf", b"original");
        let outside = tree.dir("outside");
        tree.file("outside/escaped.gguf", b"escaped");
        let parked = tree.0.join("parked-root");
        let mut swapped = false;

        let scan = scan_model_folder_with_test_hook(&root, |event, path| {
            if event == ScanTestEvent::BeforeRootHandleOpen && path == root {
                std::fs::rename(path, &parked).expect("park root");
                symlink(&outside, path).expect("redirect root");
                swapped = true;
            } else if event == ScanTestEvent::AfterRootHandleOpenAttempt && swapped {
                std::fs::remove_file(&root).expect("remove redirect");
                std::fs::rename(&parked, &root).expect("restore root");
            }
        })
        .expect("partial scan");

        assert!(swapped);
        assert!(scan.candidates.is_empty());
        assert!(!scan.root_is_bound());
        assert!(!scan.is_complete());
    }

    #[cfg(unix)]
    #[test]
    fn child_swap_open_restore_cannot_redirect_enumeration_or_spend_outside_budget() {
        use std::os::unix::fs::symlink;

        let tree = TestTree::new("child-swap-open-restore");
        let root = tree.dir("models");
        let child = tree.dir("models/child");
        tree.file("models/child/original.gguf", b"original");
        let outside = tree.dir("outside");
        for index in 0..100 {
            tree.file(&format!("outside/escaped-{index:03}.gguf"), b"escaped");
        }
        let parked = tree.0.join("parked-child");
        let mut swapped = false;

        let scan = scan_model_folder_with_test_hook(&root, |event, path| {
            if event == ScanTestEvent::BeforeDirectoryHandleOpen && path == child {
                std::fs::rename(path, &parked).expect("park child");
                symlink(&outside, path).expect("redirect child");
                swapped = true;
            } else if event == ScanTestEvent::AfterDirectoryHandleOpenAttempt
                && path == child
                && swapped
            {
                std::fs::remove_file(&child).expect("remove redirect");
                std::fs::rename(&parked, &child).expect("restore child");
            }
        })
        .expect("partial scan");

        assert!(swapped);
        assert!(scan.candidates.is_empty());
        assert!(scan.counts.visited_entries < 100);
        assert!(!scan.is_complete());
    }

    #[cfg(unix)]
    #[test]
    fn queued_directory_replaced_after_validation_is_not_iterated() {
        use std::os::unix::fs::symlink;

        let tree = TestTree::new("directory-open-race");
        let root = tree.dir("models");
        let queued = tree
            .dir("models/queued")
            .canonicalize()
            .expect("canonical queued directory");
        tree.file("models/queued/original.gguf", b"original");
        let outside = tree.dir("outside");
        tree.file("outside/escaped.gguf", b"escaped");
        let mut replaced = false;

        let scan = scan_model_folder_with_test_hook(&root, |event, path| {
            if !replaced
                && event == ScanTestEvent::AfterDirectoryValidationBeforeOpen
                && path == queued
            {
                std::fs::remove_dir_all(path).expect("remove queued directory");
                symlink(&outside, path).expect("replace with symlink");
                replaced = true;
            }
        })
        .expect("partial scan");

        assert!(replaced);
        assert!(scan.candidates.is_empty());
        assert_eq!(scan.errors.len(), 1);
        assert!(!scan.is_complete());
    }

    #[cfg(unix)]
    #[test]
    fn a_queued_directory_replaced_by_a_symlink_is_not_opened() {
        use std::os::unix::fs::symlink;

        let tree = TestTree::new("directory-race");
        let root = tree.dir("models");
        let queued = tree
            .dir("models/queued")
            .canonicalize()
            .expect("canonical queued directory");
        tree.file("models/queued/original.gguf", b"original");
        let outside = tree.dir("outside");
        tree.file("outside/escaped.gguf", b"escaped");
        let mut replaced = false;

        let scan = scan_model_folder_with_test_hook(&root, |event, path| {
            if !replaced && event == ScanTestEvent::BeforeDirectoryRead && path == queued {
                std::fs::remove_dir_all(path).expect("remove queued directory");
                symlink(&outside, path).expect("replace with symlink");
                replaced = true;
            }
        })
        .expect("scan");

        assert!(replaced);
        assert!(scan.candidates.is_empty());
        assert_eq!(scan.errors.len(), 1);
    }

    #[cfg(unix)]
    #[test]
    fn a_regular_file_replaced_by_a_symlink_is_rejected_with_fresh_metadata() {
        use std::os::unix::fs::symlink;

        let tree = TestTree::new("file-race");
        let root = tree.dir("models");
        let entry = tree
            .file("models/raced.gguf", b"old")
            .canonicalize()
            .expect("canonical entry");
        let replacement = tree.file("replacement.gguf", b"replacement is larger");
        let mut replaced = false;

        let scan = scan_model_folder_with_test_hook(&root, |event, path| {
            if !replaced && event == ScanTestEvent::BeforeFileAcceptance && path == entry {
                std::fs::remove_file(path).expect("remove original file");
                symlink(&replacement, path).expect("replace with symlink");
                replaced = true;
            }
        })
        .expect("scan");

        assert!(replaced);
        assert!(scan.candidates.is_empty());
        assert_eq!(scan.errors.len(), 1);
    }

    #[cfg(unix)]
    #[test]
    fn file_swap_open_restore_cannot_redirect_metadata_reads() {
        use std::os::unix::fs::symlink;

        let tree = TestTree::new("file-swap-open-restore");
        let root = tree.dir("models");
        let entry = tree.file("models/raced.gguf", b"old");
        let replacement = tree.file("replacement.gguf", b"replacement is much larger");
        let parked = tree.0.join("parked-file");
        let mut swapped = false;

        let scan = scan_model_folder_with_test_hook(&root, |event, path| {
            if event == ScanTestEvent::BeforeFileHandleOpen && path == entry {
                std::fs::rename(path, &parked).expect("park file");
                symlink(&replacement, path).expect("redirect file");
                swapped = true;
            } else if event == ScanTestEvent::AfterFileHandleOpenAttempt && path == entry && swapped
            {
                std::fs::remove_file(&entry).expect("remove redirect");
                std::fs::rename(&parked, &entry).expect("restore file");
            }
        })
        .expect("partial scan");

        assert!(swapped);
        assert!(scan.candidates.is_empty());
        assert!(!scan.is_complete());
    }

    #[cfg(unix)]
    #[test]
    fn replacing_a_file_after_handle_open_does_not_change_reported_size() {
        let tree = TestTree::new("file-after-open");
        let root = tree.dir("models");
        let entry = tree.file("models/stable.gguf", b"opened bytes");
        let parked = tree.0.join("opened-file");
        let mut replaced = false;

        let scan = scan_model_folder_with_test_hook(&root, |event, path| {
            if !replaced && event == ScanTestEvent::AfterFileHandleOpen && path == entry {
                std::fs::rename(path, &parked).expect("park opened file");
                std::fs::write(path, b"new pathname bytes are much longer")
                    .expect("replace pathname");
                replaced = true;
            }
        })
        .expect("scan");

        assert!(replaced);
        assert_eq!(scan.candidates.len(), 1);
        assert_eq!(
            scan.candidates[0].file_size_bytes,
            b"opened bytes".len() as i64
        );
    }
}
