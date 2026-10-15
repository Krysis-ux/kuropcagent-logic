//! Resolving what to download from a Hugging Face repository.

use serde::Deserialize;
use std::time::Duration;

use crate::{KuroError, Result};

const HF_HOST: &str = "https://huggingface.co";

/// What the user asked for, before it is resolved to a file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelRef {
    /// A slug from the curated list, e.g. `qwen3-4b`.
    Curated { slug: String, quant: Option<String> },
    /// An arbitrary repository, e.g. `unsloth/Qwen3-4B-Instruct-2507-GGUF`.
    HuggingFace { repo: String, quant: Option<String> },
}

/// Parse `qwen3-4b`, `qwen3-4b:Q5_K_M`, `owner/repo`, `owner/repo:Q4_K_M` or a
/// full `huggingface.co` URL.
pub fn parse_model_ref(raw: &str) -> Result<ModelRef> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(KuroError::bad_request("model name is empty"));
    }

    // Accept a pasted URL by reducing it to `owner/repo`.
    let without_host = trimmed
        .trim_start_matches("https://")
        .trim_start_matches("http://")
        .trim_start_matches("huggingface.co/")
        .trim_start_matches("hf.co/")
        .trim_end_matches('/');

    // Strip a trailing `/tree/main` and similar from a copied browser URL.
    let without_host = match without_host.find("/tree/") {
        Some(index) => &without_host[..index],
        None => without_host,
    };
    let without_host = match without_host.find("/blob/") {
        Some(index) => &without_host[..index],
        None => without_host,
    };

    // A quantization suffix is the part after the last colon, and never
    // contains a slash — that distinguishes it from anything URL-shaped.
    let (name, quant) = match without_host.rsplit_once(':') {
        Some((head, tail)) if !tail.contains('/') && !tail.is_empty() && !head.is_empty() => {
            (head, Some(tail.to_string()))
        }
        _ => (without_host, None),
    };

    if name.is_empty() {
        return Err(KuroError::bad_request("model name is empty"));
    }

    if name.contains('/') {
        let segments: Vec<&str> = name.split('/').filter(|s| !s.is_empty()).collect();
        if segments.len() != 2 {
            return Err(KuroError::bad_request(format!(
                "expected a Hugging Face repository like `owner/repo`, got `{name}`"
            )));
        }
        Ok(ModelRef::HuggingFace {
            repo: segments.join("/"),
            quant,
        })
    } else {
        Ok(ModelRef::Curated {
            slug: name.to_string(),
            quant,
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
struct TreeEntry {
    #[serde(rename = "type")]
    entry_type: String,
    path: String,
    #[serde(default)]
    size: u64,
    #[serde(default)]
    lfs: Option<LfsInfo>,
}

#[derive(Debug, Clone, Deserialize)]
struct LfsInfo {
    #[serde(default)]
    size: u64,
    /// Hugging Face exposes the content hash as `lfs.oid`. Older fixtures and
    /// compatible mirrors may still use `sha256`, so accept that spelling too.
    #[serde(default, alias = "sha256")]
    oid: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ModelInfo {
    #[serde(default)]
    sha: Option<String>,
}

/// A downloadable weights file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufFile {
    pub filename: String,
    pub size: u64,
    /// Present for LFS-tracked files, which is how large weights are stored.
    /// When present it is used to verify the download.
    pub sha256: Option<String>,
}

/// List every `.gguf` file in a repository.
pub async fn list_gguf_files(client: &reqwest::Client, repo: &str) -> Result<Vec<GgufFile>> {
    let revision = resolve_repo_revision(client, repo).await?;
    list_gguf_files_at_revision(client, repo, &revision).await
}

/// Resolve the moving default branch once so every subsequent operation is
/// tied to the same immutable repository state.
pub async fn resolve_repo_revision(client: &reqwest::Client, repo: &str) -> Result<String> {
    let (owner, name) = validate_repo_id(repo)?;
    let url = build_hf_url(&["api", "models", owner, name])?;
    let (body, _) = metadata_page(client, url, repo).await?;
    repo_revision_from_json(&body)
}

/// List GGUF files at one immutable commit instead of the moving `main`
/// branch. Exact selection must use this function.
pub async fn list_gguf_files_at_revision(
    client: &reqwest::Client,
    repo: &str,
    revision: &str,
) -> Result<Vec<GgufFile>> {
    let (owner, name) = validate_repo_id(repo)?;
    validate_commit_revision(revision)?;
    let mut url = build_hf_url(&["api", "models", owner, name, "tree", revision])?;
    url.query_pairs_mut()
        .append_pair("recursive", "true")
        .append_pair("expand", "true");
    let expected_path = url.path().to_string();
    let mut files = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut entries = 0;
    for _ in 0..10 {
        if !seen.insert(url.to_string()) {
            return Err(invalid_metadata("Hugging Face returned a pagination loop"));
        }
        let (body, next) = metadata_page(client, url, repo).await?;
        let page: Vec<TreeEntry> = serde_json::from_slice(&body)
            .map_err(|_| invalid_metadata("Hugging Face returned invalid file metadata"))?;
        entries += page.len();
        if entries > 5_000 {
            return Err(invalid_metadata(
                "This repository exceeds the 5,000-file browsing limit",
            ));
        }
        files.extend(gguf_files_from_json(&body)?);
        let Some(next) = next else {
            files.sort_by(|a, b| a.size.cmp(&b.size).then(a.filename.cmp(&b.filename)));
            let mut filenames = std::collections::HashSet::new();
            if files.iter().any(|file| !filenames.insert(&file.filename)) {
                return Err(invalid_metadata(
                    "Hugging Face returned duplicate file metadata",
                ));
            }
            return Ok(files);
        };
        url = reqwest::Url::parse(&next)
            .map_err(|_| invalid_metadata("Hugging Face returned an invalid pagination link"))?;
        if url.scheme() != "https"
            || url.host_str() != Some("huggingface.co")
            || url.port_or_known_default() != Some(443)
            || !url.username().is_empty()
            || url.password().is_some()
            || url.path() != expected_path
        {
            return Err(invalid_metadata(
                "Hugging Face returned an unexpected pagination destination",
            ));
        }
    }
    Err(invalid_metadata(
        "This repository exceeds the 10-page browsing limit",
    ))
}

pub(crate) fn invalid_metadata(message: &str) -> KuroError {
    KuroError::Hub {
        status: 502,
        code: "hugging_face_invalid_response",
        message: message.into(),
        retry_after_seconds: None,
    }
}

/// Metadata has a bounded body and request deadline; model downloads intentionally
/// use a different, streaming lifetime.
pub(crate) async fn metadata_page(
    client: &reqwest::Client,
    url: reqwest::Url,
    repo: &str,
) -> Result<(Vec<u8>, Option<String>)> {
    tokio::time::timeout(Duration::from_secs(20), async {
        let mut response = client.get(url).send().await?;
        validate_hub_response(&response, repo)?;
        let next = response
            .headers()
            .get(reqwest::header::LINK)
            .and_then(|header| header.to_str().ok())
            .and_then(next_page_link);
        let mut body = Vec::new();
        while let Some(chunk) = response.chunk().await? {
            if body.len() + chunk.len() > 8 * 1024 * 1024 {
                return Err(invalid_metadata(
                    "Hugging Face metadata exceeded the response limit",
                ));
            }
            body.extend_from_slice(&chunk);
        }
        Ok((body, next))
    })
    .await
    .map_err(|_| KuroError::Hub {
        status: 503,
        code: "hugging_face_unavailable",
        message: "Hugging Face took too long to respond. Try again.".into(),
        retry_after_seconds: None,
    })?
}

fn next_page_link(header: &str) -> Option<String> {
    header.split(',').find_map(|part| {
        let (url, attributes) = part.trim().split_once('>')?;
        attributes
            .split(';')
            .any(|attribute| matches!(attribute.trim(), "rel=\"next\"" | "rel=next"))
            .then(|| url.trim_start_matches('<').to_string())
    })
}

fn validate_hub_response(response: &reqwest::Response, repo: &str) -> Result<()> {
    if response.status() == reqwest::StatusCode::NOT_FOUND {
        return Err(KuroError::not_found(format!(
            "Hugging Face repository or revision `{repo}`"
        )));
    }
    if response.status() == reqwest::StatusCode::UNAUTHORIZED
        || response.status() == reqwest::StatusCode::FORBIDDEN
    {
        return Err(KuroError::Hub {
            status: 403,
            code: "hugging_face_access_required",
            message: format!(
                "`{repo}` is gated or private. Kuro currently downloads public repositories only."
            ),
            retry_after_seconds: None,
        });
    }
    if !response.status().is_success() {
        let limited = response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS;
        return Err(KuroError::Hub {
            status: if limited { 429 } else { 503 },
            code: if limited {
                "hugging_face_rate_limited"
            } else {
                "hugging_face_unavailable"
            },
            message: if limited {
                "Hugging Face is rate limited. Try again later.".into()
            } else {
                format!("Hugging Face returned {} for `{repo}`", response.status())
            },
            retry_after_seconds: response
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|h| h.to_str().ok())
                .and_then(|h| h.parse().ok()),
        });
    }
    Ok(())
}

pub(crate) fn repo_revision_from_json(body: &[u8]) -> Result<String> {
    let info: ModelInfo = serde_json::from_slice(body)?;
    let revision = info
        .sha
        .ok_or_else(|| KuroError::model("Hugging Face model metadata omitted its commit SHA"))?;
    validate_commit_revision(&revision)?;
    Ok(revision.to_ascii_lowercase())
}

fn gguf_files_from_json(body: &[u8]) -> Result<Vec<GgufFile>> {
    let entries: Vec<TreeEntry> = serde_json::from_slice(body)?;
    Ok(entries
        .into_iter()
        .filter(|entry| entry.entry_type == "file")
        .filter(|entry| entry.path.to_ascii_lowercase().ends_with(".gguf"))
        .map(|entry| {
            let lfs_size = entry.lfs.as_ref().map(|l| l.size).unwrap_or(0);
            GgufFile {
                size: entry.size.max(lfs_size),
                sha256: entry
                    .lfs
                    .and_then(|l| l.oid)
                    .and_then(|oid| normalize_lfs_sha256(&oid)),
                filename: entry.path,
            }
        })
        .collect())
}

fn normalize_lfs_sha256(raw: &str) -> Option<String> {
    let digest = raw.strip_prefix("sha256:").unwrap_or(raw);
    if digest.len() == 64
        && digest
            .chars()
            .all(|character| character.is_ascii_hexdigit())
    {
        Some(digest.to_ascii_lowercase())
    } else {
        None
    }
}

/// Quantizations to try, best-value first, when the caller has no preference.
const QUANT_PREFERENCE: &[&str] = &["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q6_K", "Q8_0"];

/// Pick the file to download.
pub fn choose_gguf(files: &[GgufFile], preferred_quant: Option<&str>) -> Result<GgufFile> {
    if files.is_empty() {
        return Err(KuroError::model(
            "this repository contains no .gguf files. Kuro can only run GGUF weights.",
        ));
    }

    // Weights above ~50 GB are published as numbered shards.
    let single_part: Vec<&GgufFile> = files
        .iter()
        .filter(|f| !is_split_part(&f.filename) && !is_auxiliary_gguf(&f.filename))
        .collect();
    if single_part.is_empty() {
        return Err(KuroError::model(
            "this repository contains only split GGUF shards or auxiliary projector files. \
             Select a complete language-model GGUF; Kuro cannot load these files alone.",
        ));
    }

    if let Some(quant) = preferred_quant {
        if let Some(found) = find_by_quant(&single_part, quant) {
            return Ok(found.clone());
        }
        let available = available_quants(&single_part);
        return Err(KuroError::model(format!(
            "no `{quant}` file in this repository. Available: {}",
            if available.is_empty() {
                "none detected".to_string()
            } else {
                available.join(", ")
            }
        )));
    }

    for quant in QUANT_PREFERENCE {
        if let Some(found) = find_by_quant(&single_part, quant) {
            return Ok(found.clone());
        }
    }

    // Nothing recognisable: take the smallest file, which is the safest default
    // on a machine whose memory we do not want to overrun.
    let smallest = single_part
        .iter()
        .min_by_key(|file| file.size)
        .expect("single_part is non-empty");
    Ok((*smallest).clone())
}

/// Select one exact case-sensitive file from a pinned repository tree.
pub fn choose_gguf_exact(files: &[GgufFile], filename: &str) -> Result<GgufFile> {
    validate_gguf_filename(filename)?;

    let mut matches = files.iter().filter(|file| file.filename == filename);
    let selected = matches
        .next()
        .ok_or_else(|| KuroError::not_found(format!("GGUF file `{filename}`")))?;
    if matches.next().is_some() {
        return Err(KuroError::model(format!(
            "repository metadata contains more than one exact `{filename}` entry"
        )));
    }
    if is_split_part(&selected.filename) {
        return Err(KuroError::model(
            "this file is one split GGUF shard, which Kuro cannot load yet",
        ));
    }
    if is_auxiliary_gguf(&selected.filename) {
        return Err(KuroError::model(
            "this file is an auxiliary multimodal projector, not a standalone language model",
        ));
    }

    Ok(selected.clone())
}

fn find_by_quant<'a>(files: &[&'a GgufFile], quant: &str) -> Option<&'a GgufFile> {
    let needle = quant.to_ascii_lowercase();
    files
        .iter()
        .find(|file| {
            let name = file.filename.to_ascii_lowercase();
            // Match on a delimited token so `Q4_K_M` never matches `Q4_K_M_XL`.
            name.split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                .any(|token| token == needle)
        })
        .copied()
}

/// Quantization labels detectable from the filenames present.
pub fn available_quants(files: &[&GgufFile]) -> Vec<String> {
    let mut found: Vec<String> = Vec::new();
    for file in files {
        if let Some(quant) = quant_from_filename(&file.filename) {
            if !found.iter().any(|q| q.eq_ignore_ascii_case(&quant)) {
                found.push(quant);
            }
        }
    }
    found
}

/// Extract a quantization label such as `Q4_K_M` or `IQ3_XXS` from a filename.
pub fn quant_from_filename(filename: &str) -> Option<String> {
    let stem = filename.rsplit('/').next().unwrap_or(filename);
    let stem = stem.strip_suffix(".gguf").unwrap_or(stem);

    stem.split(['.', '-'])
        .rev()
        .find(|token| {
            let upper = token.to_ascii_uppercase();
            (upper.starts_with('Q')
                || upper.starts_with("IQ")
                || upper.starts_with("BF")
                || upper.starts_with('F'))
                && upper.chars().any(|c| c.is_ascii_digit())
        })
        .map(|token| token.to_ascii_uppercase())
}

/// Known projector naming conventions. Inspect only complete delimited tokens
/// so an unrelated model name containing the same letters is not excluded.
pub fn is_auxiliary_gguf(filename: &str) -> bool {
    filename
        .to_ascii_lowercase()
        .split(['/', '.', '-', '_'])
        .any(|token| matches!(token, "mmproj" | "projector"))
}

/// True for one shard of a multi-part GGUF, e.g. `model-00002-of-00003.gguf`.
pub fn is_split_part(filename: &str) -> bool {
    let lower = filename.to_ascii_lowercase();
    let Some(stem) = lower.strip_suffix(".gguf") else {
        return false;
    };
    // Look for the `-<digits>-of-<digits>` tail that split GGUF weights use.
    let mut parts = stem.rsplit('-');
    let Some(total) = parts.next() else {
        return false;
    };
    let Some(of) = parts.next() else {
        return false;
    };
    let Some(index) = parts.next() else {
        return false;
    };
    of == "of"
        && !total.is_empty()
        && !index.is_empty()
        && total.chars().all(|c| c.is_ascii_digit())
        && index.chars().all(|c| c.is_ascii_digit())
}

pub fn resolve_download_url(repo: &str, filename: &str) -> String {
    format!("{HF_HOST}/{repo}/resolve/main/{filename}")
}

/// Build an immutable Hugging Face download URL without allowing path content
/// to become URL syntax.
pub fn resolve_pinned_download_url(repo: &str, revision: &str, filename: &str) -> Result<String> {
    let (owner, name) = validate_repo_id(repo)?;
    validate_commit_revision(revision)?;
    validate_gguf_filename(filename)?;

    let mut url = reqwest::Url::parse(HF_HOST)
        .map_err(|error| KuroError::other(format!("invalid Hugging Face base URL: {error}")))?;
    {
        let mut segments = url
            .path_segments_mut()
            .map_err(|_| KuroError::other("Hugging Face base URL cannot contain path segments"))?;
        segments.pop_if_empty();
        segments.push(owner);
        segments.push(name);
        segments.push("resolve");
        segments.push(revision);
        for segment in filename.split('/') {
            segments.push(segment);
        }
    }
    Ok(url.into())
}

pub(crate) fn build_hf_url(segments: &[&str]) -> Result<reqwest::Url> {
    let mut url = reqwest::Url::parse(HF_HOST)
        .map_err(|error| KuroError::other(format!("invalid Hugging Face base URL: {error}")))?;
    {
        let mut path = url
            .path_segments_mut()
            .map_err(|_| KuroError::other("Hugging Face base URL cannot contain path segments"))?;
        path.pop_if_empty();
        path.extend(segments.iter().copied());
    }
    Ok(url)
}

pub fn validate_repo_id(repo: &str) -> Result<(&str, &str)> {
    let mut segments = repo.split('/');
    let owner = segments.next().unwrap_or_default();
    let name = segments.next().unwrap_or_default();
    let invalid_segment = |segment: &str| {
        segment.is_empty()
            || matches!(segment, "." | "..")
            || segment.contains(['\\', '#', '@', ':'])
            || segment.chars().any(char::is_control)
            || !segment
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.'))
    };
    if invalid_segment(owner) || invalid_segment(name) || segments.next().is_some() {
        return Err(KuroError::bad_request(format!(
            "expected a Hugging Face repository like `owner/repo`, got `{repo}`"
        )));
    }
    Ok((owner, name))
}

pub fn validate_commit_revision(revision: &str) -> Result<()> {
    if revision.len() != 40
        || !revision
            .chars()
            .all(|character| character.is_ascii_hexdigit())
    {
        return Err(KuroError::bad_request(
            "revision must be a full 40-character Hugging Face commit SHA",
        ));
    }
    Ok(())
}

fn validate_gguf_filename(filename: &str) -> Result<()> {
    if filename.is_empty()
        || filename.contains('\\')
        || filename.chars().any(char::is_control)
        || !filename.to_ascii_lowercase().ends_with(".gguf")
        || filename
            .split('/')
            .any(|segment| segment.is_empty() || matches!(segment, "." | ".."))
    {
        return Err(KuroError::bad_request(format!(
            "`{filename}` is not a safe relative GGUF filename"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(name: &str, size: u64) -> GgufFile {
        GgufFile {
            filename: name.to_string(),
            size,
            sha256: None,
        }
    }

    #[test]
    fn parses_a_curated_slug_with_and_without_quant() {
        assert_eq!(
            parse_model_ref("qwen3-4b").unwrap(),
            ModelRef::Curated {
                slug: "qwen3-4b".to_string(),
                quant: None
            }
        );
        assert_eq!(
            parse_model_ref("qwen3-4b:Q5_K_M").unwrap(),
            ModelRef::Curated {
                slug: "qwen3-4b".to_string(),
                quant: Some("Q5_K_M".to_string())
            }
        );
    }

    #[test]
    fn parses_repositories_and_pasted_urls() {
        let expected = ModelRef::HuggingFace {
            repo: "unsloth/Qwen3-4B-Instruct-2507-GGUF".to_string(),
            quant: None,
        };
        for input in [
            "unsloth/Qwen3-4B-Instruct-2507-GGUF",
            "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF",
            "huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/",
            "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/tree/main",
        ] {
            assert_eq!(parse_model_ref(input).unwrap(), expected, "input: {input}");
        }
    }

    #[test]
    fn keeps_the_quantization_from_a_repository_reference() {
        assert_eq!(
            parse_model_ref("https://huggingface.co/owner/repo:Q8_0").unwrap(),
            ModelRef::HuggingFace {
                repo: "owner/repo".to_string(),
                quant: Some("Q8_0".to_string())
            }
        );
    }

    #[test]
    fn rejects_empty_and_malformed_references() {
        assert!(parse_model_ref("").is_err());
        assert!(parse_model_ref("   ").is_err());
        assert!(parse_model_ref("a/b/c").is_err());
    }

    #[test]
    fn prefers_the_requested_quantization() {
        let files = vec![file("model-Q4_K_M.gguf", 100), file("model-Q8_0.gguf", 200)];
        let chosen = choose_gguf(&files, Some("Q8_0")).unwrap();
        assert_eq!(chosen.filename, "model-Q8_0.gguf");
    }

    #[test]
    fn falls_back_to_a_sensible_default_quantization() {
        let files = vec![
            file("model-Q8_0.gguf", 900),
            file("model-Q4_K_M.gguf", 400),
            file("model-F16.gguf", 1600),
        ];
        let chosen = choose_gguf(&files, None).unwrap();
        assert_eq!(
            chosen.filename, "model-Q4_K_M.gguf",
            "Q4_K_M is the best default"
        );
    }

    #[test]
    fn picks_the_smallest_when_no_quantization_is_recognisable() {
        let files = vec![
            file("weights-big.gguf", 900),
            file("weights-small.gguf", 100),
        ];
        let chosen = choose_gguf(&files, None).unwrap();
        assert_eq!(chosen.filename, "weights-small.gguf");
    }

    #[test]
    fn auxiliary_projectors_are_never_selected_as_language_models() {
        for auxiliary in [
            "mmproj-F16.gguf",
            "nested/MMProj-Q4_K_M.gguf",
            "model-projector.gguf",
            "vision/mm_projector-f16.gguf",
        ] {
            let files = vec![file(auxiliary, 10), file("language-F16.gguf", 1000)];
            assert_eq!(
                choose_gguf(&files, None).unwrap().filename,
                "language-F16.gguf"
            );
            assert!(
                choose_gguf_exact(&files, auxiliary).is_err(),
                "accepted {auxiliary}"
            );
            assert!(
                choose_gguf(&files[..1], None).is_err(),
                "accepted auxiliary-only repository"
            );
        }
    }

    #[test]
    fn reports_available_quantizations_when_the_request_cannot_be_met() {
        let files = vec![file("model-Q4_K_M.gguf", 100)];
        let error = choose_gguf(&files, Some("Q2_K")).unwrap_err().to_string();
        assert!(
            error.contains("Q4_K_M"),
            "error should list what is available: {error}"
        );
    }

    #[test]
    fn refuses_split_shards_rather_than_downloading_a_broken_model() {
        let files = vec![
            file("model-Q4_K_M-00001-of-00003.gguf", 100),
            file("model-Q4_K_M-00002-of-00003.gguf", 100),
        ];
        let error = choose_gguf(&files, None).unwrap_err().to_string();
        assert!(error.contains("split"), "got: {error}");
    }

    #[test]
    fn ignores_shards_but_still_uses_single_files_in_the_same_repo() {
        let files = vec![
            file("model-Q8_0-00001-of-00002.gguf", 500),
            file("model-Q4_K_M.gguf", 100),
        ];
        let chosen = choose_gguf(&files, None).unwrap();
        assert_eq!(chosen.filename, "model-Q4_K_M.gguf");
    }

    #[test]
    fn detects_split_parts_precisely() {
        assert!(is_split_part("model-00001-of-00002.gguf"));
        assert!(!is_split_part("model-Q4_K_M.gguf"));
        assert!(!is_split_part("llama-3-of-legends.gguf"));
    }

    #[test]
    fn extracts_quantization_labels_from_filenames() {
        assert_eq!(
            quant_from_filename("Qwen3-4B-Instruct-2507-Q4_K_M.gguf").as_deref(),
            Some("Q4_K_M")
        );
        assert_eq!(
            quant_from_filename("some/dir/model.IQ3_XXS.gguf").as_deref(),
            Some("IQ3_XXS")
        );
        assert_eq!(quant_from_filename("plain.gguf"), None);
    }

    #[test]
    fn empty_repository_is_an_understandable_error() {
        let error = choose_gguf(&[], None).unwrap_err().to_string();
        assert!(error.contains("GGUF"));
    }

    #[test]
    fn builds_a_resolve_url() {
        assert_eq!(
            resolve_download_url("owner/repo", "model.gguf"),
            "https://huggingface.co/owner/repo/resolve/main/model.gguf"
        );
    }

    #[test]
    fn tree_lfs_oid_becomes_verified_sha256() {
        let checksum = "ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789";
        let payload = format!(
            r#"[{{"type":"file","path":"model.gguf","size":128,"lfs":{{"size":4096,"oid":"sha256:{checksum}"}}}}]"#
        );

        let files = gguf_files_from_json(payload.as_bytes()).expect("tree fixture");

        assert_eq!(files.len(), 1);
        assert_eq!(
            files[0].size, 4096,
            "published LFS bytes must win over pointer bytes"
        );
        assert_eq!(
            files[0].sha256.as_deref(),
            Some("abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789")
        );
    }

    #[test]
    fn malformed_lfs_oid_is_not_treated_as_verifiable() {
        for oid in ["abc123", "sha256:not-hex", &"f".repeat(63)] {
            let payload = format!(
                r#"[{{"type":"file","path":"model.gguf","size":4096,"lfs":{{"size":4096,"oid":"{oid}"}}}}]"#
            );
            let files = gguf_files_from_json(payload.as_bytes()).expect("tree fixture");
            assert_eq!(
                files[0].sha256, None,
                "oid {oid:?} must not reach verification"
            );
        }
    }

    #[test]
    fn same_quant_files_require_exact_filename() {
        let files = vec![
            file("alpha/model-Q4_K_M.gguf", 100),
            file("beta/model-Q4_K_M.gguf", 200),
        ];

        let chosen = choose_gguf_exact(&files, "beta/model-Q4_K_M.gguf").expect("exact file");

        assert_eq!(chosen.filename, "beta/model-Q4_K_M.gguf");
        assert_eq!(chosen.size, 200);
    }

    #[test]
    fn exact_shard_selection_is_rejected() {
        let files = vec![
            file("model-Q4_K_M-00001-of-00002.gguf", 100),
            file("model-Q4_K_M-00002-of-00002.gguf", 100),
        ];

        let error = choose_gguf_exact(&files, "model-Q4_K_M-00001-of-00002.gguf")
            .expect_err("one shard is not runnable")
            .to_string();

        assert!(error.contains("split"), "got: {error}");
    }

    #[test]
    fn download_url_is_pinned_and_segment_encoded() {
        let revision = "0123456789abcdef0123456789abcdef01234567";
        let url = resolve_pinned_download_url(
            "owner/repo-name",
            revision,
            "nested folder/model #1?.gguf",
        )
        .expect("pinned URL");

        assert_eq!(
            url,
            "https://huggingface.co/owner/repo-name/resolve/0123456789abcdef0123456789abcdef01234567/nested%20folder/model%20%231%3F.gguf"
        );
        assert!(!url.contains("/main/"));
    }

    #[test]
    fn model_info_revision_is_a_full_commit_sha() {
        let payload = br#"{"sha":"ABCDEF0123456789ABCDEF0123456789ABCDEF01"}"#;

        assert_eq!(
            repo_revision_from_json(payload).expect("revision"),
            "abcdef0123456789abcdef0123456789abcdef01"
        );
        assert!(repo_revision_from_json(br#"{"sha":"main"}"#).is_err());
        assert!(repo_revision_from_json(br#"{}"#).is_err());
    }

    #[test]
    fn extracts_only_next_pagination_link() {
        assert_eq!(next_page_link("<https://huggingface.co/api/models/a/b/tree/sha?cursor=2>; rel=\"next\", <https://huggingface.co/last>; rel=\"last\""), Some("https://huggingface.co/api/models/a/b/tree/sha?cursor=2".into()));
        assert_eq!(
            next_page_link("<https://huggingface.co/last>; rel=\"last\""),
            None
        );
    }

    #[test]
    fn repository_ids_cannot_supply_url_syntax() {
        for repo in [
            "a/b?x=y",
            "a/b#fragment",
            "a/../b",
            "https://evil.test",
            "a/b c",
            "a/b%2fc",
            "a/b\\c",
            "a/b\n",
        ] {
            assert!(validate_repo_id(repo).is_err(), "accepted {repo:?}");
        }
        assert!(validate_repo_id("meta-llama/Llama-3.2_1B").is_ok());
    }
}
