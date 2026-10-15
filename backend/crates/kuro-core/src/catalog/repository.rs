//! Public Hub repository details. Publisher facts stay separate from Kuro's
//! machine-fit estimate; every install action identifies an exact pinned file.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use super::hf::{self, GgufFile};
use crate::db::ModelRecord;
use crate::hardware::{estimate_fit, HardwareInfo};
use crate::Result;

#[derive(Clone)]
struct Snapshot {
    info: Value,
    revision: String,
    files: Vec<GgufFile>,
    fetched_at: String,
    fetched: Instant,
}

// Only public, unauthenticated metadata is cached. Fit and installed state are
// derived for each request, so changing the machine or downloads cannot stale them.
type Cache = Mutex<HashMap<String, Snapshot>>;
static CACHE: OnceLock<Cache> = OnceLock::new();

pub async fn describe_repository(
    client: &reqwest::Client,
    repo: &str,
    hardware: &HardwareInfo,
    installed: &[ModelRecord],
) -> Result<Value> {
    let (owner, name) = hf::validate_repo_id(repo)?;
    let cached = CACHE
        .get_or_init(Cache::default)
        .lock()
        .ok()
        .and_then(|cache| cache.get(repo).cloned())
        .filter(|snapshot| snapshot.fetched.elapsed() < Duration::from_secs(300));
    let snapshot = match cached {
        Some(snapshot) => snapshot,
        None => {
            let url = hf::build_hf_url(&["api", "models", owner, name])?;
            let (body, _) = hf::metadata_page(client, url, repo).await?;
            let revision = hf::repo_revision_from_json(&body)?;
            let info = serde_json::from_slice(&body).map_err(|_| {
                hf::invalid_metadata("Hugging Face returned invalid model metadata")
            })?;
            let files = hf::list_gguf_files_at_revision(client, repo, &revision).await?;
            let snapshot = Snapshot {
                info,
                revision,
                files,
                fetched_at: chrono::Utc::now().to_rfc3339(),
                fetched: Instant::now(),
            };
            if let Ok(mut cache) = CACHE.get_or_init(Cache::default).lock() {
                if cache.len() >= 32 {
                    if let Some(oldest) = cache
                        .iter()
                        .min_by_key(|(_, value)| value.fetched)
                        .map(|(key, _)| key.clone())
                    {
                        cache.remove(&oldest);
                    }
                }
                cache.insert(repo.into(), snapshot.clone());
            }
            snapshot
        }
    };
    Ok(describe_snapshot(repo, &snapshot, hardware, installed))
}

fn byte_fact(size: u64) -> Value {
    json!({ "bytes": size, "precision": "actual", "source": "hugging_face_file_metadata" })
}

fn numeric_fact(value: Option<u64>, precision: &str, source: &str, repo: &str) -> Value {
    match value {
        Some(value) => {
            json!({ "value": value, "precision": precision, "source": source, "sourceRepo": repo })
        }
        None => {
            json!({ "value": null, "precision": "unknown", "source": null, "sourceRepo": null })
        }
    }
}

fn bounded_string(value: &Value) -> Option<String> {
    value
        .as_str()
        .map(|text| text.chars().take(4_000).collect())
        .filter(|text: &String| !text.trim().is_empty())
}

fn strings(value: &Value) -> Vec<String> {
    match value {
        Value::String(_) => bounded_string(value).into_iter().collect(),
        Value::Array(values) => values.iter().take(100).filter_map(bounded_string).collect(),
        _ => Vec::new(),
    }
}

fn describe_snapshot(
    repo: &str,
    snapshot: &Snapshot,
    hardware: &HardwareInfo,
    installed: &[ModelRecord],
) -> Value {
    let info = &snapshot.info;
    let card = &info["cardData"];
    let owner = repo.split('/').next().unwrap_or(repo);
    let publisher = info["author"]
        .as_str()
        .filter(|author| !author.is_empty())
        .unwrap_or(owner);
    let publisher_source = if info["author"].as_str().is_some() {
        "hub_author"
    } else {
        "repository_owner"
    };
    let base_models: Vec<Value> = strings(&card["base_model"]).into_iter()
        .filter(|base| hf::validate_repo_id(base).is_ok())
        .map(|base| json!({ "repositoryUrl": format!("https://huggingface.co/{base}"), "repo": base, "relation": bounded_string(&card["base_model_relation"]) })).collect();
    let mut files = snapshot.files.clone();
    files.sort_by(|a, b| a.size.cmp(&b.size).then(a.filename.cmp(&b.filename)));
    let mut variants = Vec::new();
    let mut unsupported = Vec::new();
    for file in files {
        if hf::is_auxiliary_gguf(&file.filename) {
            unsupported.push(json!({ "filename": file.filename, "size": byte_fact(file.size), "reason": "auxiliary_gguf_not_supported" }));
            continue;
        }
        if hf::is_split_part(&file.filename) {
            unsupported.push(json!({ "filename": file.filename, "size": byte_fact(file.size), "reason": "split_gguf_not_supported" }));
            continue;
        }
        if hf::choose_gguf_exact(std::slice::from_ref(&file), &file.filename).is_err() {
            continue;
        }
        let mut digest = Sha256::new();
        for item in [
            repo,
            &snapshot.revision,
            &file.filename,
            file.sha256.as_deref().unwrap_or(""),
        ] {
            digest.update(item.as_bytes());
            digest.update([0]);
        }
        digest.update(file.size.to_be_bytes());
        let variant_id = format!("{:x}", digest.finalize());
        let mut fit =
            serde_json::to_value(estimate_fit(file.size, hardware)).unwrap_or(Value::Null);
        fit["basis"] = json!("actual_file_size");
        fit["assumedContextTokens"] = json!(4096);
        let existing = installed.iter().find(|model| {
            model.hf_repo.as_deref() == Some(repo)
                && model.hf_file.as_deref() == Some(&file.filename)
        });
        variants.push(json!({
            "variantId": variant_id, "filename": file.filename,
            "quantization": hf::quant_from_filename(&file.filename),
            "size": byte_fact(file.size), "sha256": file.sha256,
            "supported": true, "unsupportedReason": null, "fit": fit,
            "installed": existing.map(|model| json!({ "id": model.id, "status": model.status })),
        }));
    }
    let recommendation = recommend(&variants);
    let description = bounded_string(&card["description"]);
    let gated = match &info["gated"] {
        Value::Bool(value) => *value,
        Value::String(value) => value != "false",
        _ => false,
    };
    json!({
        "repository": {
            "repo": repo, "name": repo.rsplit('/').next().unwrap_or(repo), "revision": snapshot.revision,
            "repositoryUrl": format!("https://huggingface.co/{repo}"),
            "modelCardUrl": format!("https://huggingface.co/{repo}/blob/{}/README.md", snapshot.revision),
            "publisher": { "hubId": publisher, "displayName": publisher, "profileUrl": format!("https://huggingface.co/{publisher}"), "source": publisher_source },
            "description": { "source": description.as_ref().map(|_| "publisher_model_card"), "text": description },
            "access": { "kind": if gated { "gated" } else { "public" }, "gatedMode": info["gated"].as_str() },
            "downloads": info["downloads"].as_u64().unwrap_or(0), "likes": info["likes"].as_u64().unwrap_or(0),
            "lastModified": bounded_string(&info["lastModified"]), "license": bounded_string(&card["license"]),
            "pipelineTag": bounded_string(&info["pipeline_tag"]), "libraryName": bounded_string(&info["library_name"]),
            "tags": strings(&info["tags"]), "baseModels": base_models,
            "parameterCount": numeric_fact(info["gguf"]["total"].as_u64(), "exact", "hugging_face_gguf_metadata", repo),
            "contextLength": numeric_fact(info["gguf"]["context_length"].as_u64().or_else(|| card["context_length"].as_u64()), "declared", if info["gguf"]["context_length"].as_u64().is_some() { "hugging_face_gguf_metadata" } else { "publisher_model_card" }, repo),
            "variants": variants, "unsupportedFiles": unsupported,
        },
        "recommendation": recommendation,
        "cache": { "stale": false, "fetchedAt": snapshot.fetched_at },
    })
}

fn recommend(variants: &[Value]) -> Value {
    let comfortable: Vec<&Value> = variants
        .iter()
        .filter(|variant| matches!(variant["fit"]["verdict"].as_str(), Some("great" | "fits")))
        .collect();
    let selected = ["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q6_K", "Q8_0"]
        .iter()
        .find_map(|quant| {
            comfortable
                .iter()
                .copied()
                .find(|variant| variant["quantization"].as_str() == Some(quant))
        });
    let (selected, confidence, reason, summary) = if let Some(selected) = selected {
        (
            Some(selected),
            "comfortable",
            "balanced_default_fits",
            "Balanced default with room for the system at a 4K context.",
        )
    } else if let Some(selected) = comfortable.first() {
        (
            Some(*selected),
            "comfortable",
            "smallest_comfortable_variant",
            "Smallest variant with comfortable estimated memory headroom.",
        )
    } else if let Some(selected) = variants
        .iter()
        .find(|variant| variant["fit"]["verdict"] == "tight")
    {
        (
            Some(selected),
            "caution",
            "only_tight_variant",
            "Memory will be tight. Close other applications or choose a smaller model.",
        )
    } else {
        (
            None,
            "none",
            if variants.is_empty() {
                "no_supported_single_file"
            } else {
                "no_comfortable_fit"
            },
            "No variant is recommended for the memory available on this machine.",
        )
    };
    json!({ "variantId": selected.map(|variant| &variant["variantId"]), "confidence": confidence, "reasonCode": reason, "summary": summary })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn snapshot(files: Vec<GgufFile>) -> Snapshot {
        Snapshot {
            info: json!({ "author": "publisher", "gguf": { "total": 4_022_468_096u64 }, "cardData": { "base_model": "Qwen/Qwen3-4B", "license": "apache-2.0" } }),
            revision: "0123456789abcdef0123456789abcdef01234567".into(),
            files,
            fetched_at: "2026-09-10T00:00:00Z".into(),
            fetched: Instant::now(),
        }
    }
    fn file(filename: &str, size: u64) -> GgufFile {
        GgufFile {
            filename: filename.into(),
            size,
            sha256: None,
        }
    }
    #[test]
    fn exact_variants_keep_duplicates_quantized_and_split_files_are_unavailable() {
        let source = snapshot(vec![
            file("a-Q4_K_M.gguf", 200),
            file("b-Q4_K_M.gguf", 100),
            file("c-Q8_0-00001-of-00002.gguf", 50),
        ]);
        let result = describe_snapshot("publisher/Model", &source, &crate::hardware::detect(), &[]);
        let variants = result["repository"]["variants"].as_array().unwrap();
        assert_eq!(variants.len(), 2);
        assert_eq!(variants[0]["filename"], "b-Q4_K_M.gguf");
        assert_ne!(variants[0]["variantId"], variants[1]["variantId"]);
        assert_eq!(variants[0]["size"]["bytes"], 100);
        assert_eq!(
            result["repository"]["unsupportedFiles"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
        assert!(result["repository"]["description"]["text"].is_null());
        assert_eq!(result["repository"]["parameterCount"]["precision"], "exact");
    }
    #[test]
    fn recommendation_uses_fit_before_quant_preference() {
        let variants = vec![
            json!({"variantId":"small","quantization":"Q3_K_M","fit":{"verdict":"fits"}}),
            json!({"variantId":"large","quantization":"Q4_K_M","fit":{"verdict":"wont_fit"}}),
        ];
        assert_eq!(recommend(&variants)["variantId"], "small");
        assert_eq!(recommend(&variants[1..])["confidence"], "none");
        assert!(recommend(&[])["variantId"].is_null());
    }

    #[test]
    fn auxiliary_projectors_are_explained_but_not_recommended() {
        let source = snapshot(vec![file("mmproj-F16.gguf", 100)]);
        let result = describe_snapshot("publisher/Model", &source, &crate::hardware::detect(), &[]);
        assert!(result["repository"]["variants"]
            .as_array()
            .unwrap()
            .is_empty());
        assert_eq!(
            result["repository"]["unsupportedFiles"][0]["reason"],
            "auxiliary_gguf_not_supported"
        );
        assert!(result["recommendation"]["variantId"].is_null());
    }

    #[tokio::test]
    #[ignore = "requires public Hugging Face access; transfers metadata only"]
    async fn live_repository_selection_matches_exact_preview() {
        let client = reqwest::Client::new();
        let response = describe_repository(
            &client,
            "unsloth/Qwen3-4B-Instruct-2507-GGUF",
            &crate::hardware::detect(),
            &[],
        )
        .await
        .expect("live repository");
        let repo = &response["repository"];
        let variants = repo["variants"].as_array().expect("variants");
        assert!(!variants.is_empty());
        let selected = &variants[0];
        let selection = super::super::ExactModelSelection {
            repo: repo["repo"].as_str().unwrap().into(),
            revision: repo["revision"].as_str().unwrap().into(),
            filename: selected["filename"].as_str().unwrap().into(),
        };
        let plan = super::super::plan_pull_exact(&client, &selection)
            .await
            .expect("exact preview");
        assert_eq!(plan.filename, selected["filename"].as_str().unwrap());
        assert_eq!(plan.size_bytes, selected["size"]["bytes"].as_u64().unwrap());
        assert_eq!(plan.sha256.as_deref(), selected["sha256"].as_str());
        assert!(plan.sha256.is_some());
        assert!(plan.url.contains(&selection.revision));
    }
}
