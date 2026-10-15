use std::collections::HashMap;

use rusqlite::OptionalExtension;
use serde::{Deserialize, Serialize};

use crate::db::Db;
use crate::hardware::HardwareInfo;
use crate::tools::web_search::SearchProvider;
use crate::tools::ToolGroup;
use crate::{KuroError, Result};

pub const KEY_CONTEXT_SIZE: &str = "engine.contextSize";
pub const KEY_GPU_LAYERS: &str = "engine.gpuLayers";
pub const KEY_THREADS: &str = "engine.threads";
pub const KEY_IDLE_UNLOAD_MINUTES: &str = "engine.idleUnloadMinutes";
pub const KEY_ENGINE_PROFILE: &str = "engine.profile";
pub const KEY_ENGINE_RELEASE_TAG: &str = "engine.releaseTag";
pub const KEY_DEFAULT_MODEL: &str = "chat.defaultModel";
pub const KEY_SEARCH_PROVIDER: &str = "search.provider";
pub const KEY_SEARCH_BASE_URL: &str = "search.baseUrl";
pub const KEY_DEFAULT_TOOL_GROUPS: &str = "tools.defaultGroups";
pub const KEY_MEMORY_PRELOAD: &str = "tools.memoryPreload";
pub const KEY_MEMORY_ABOUT_YOU: &str = "memory.aboutYou";
pub const KEY_MODEL_FOLDERS: &str = "models.folders";

/// Ordered canonical folders selected for external-model discovery.
pub fn model_folders(db: &Db) -> Result<Vec<String>> {
    let raw = db.with(|conn| {
        let raw = conn
            .query_row(
                "SELECT value FROM settings WHERE key = ?1",
                rusqlite::params![KEY_MODEL_FOLDERS],
                |row| row.get::<_, String>(0),
            )
            .optional()?;
        Ok(raw)
    })?;
    match raw {
        Some(value) => parse_model_folders_json(&value),
        None => Ok(Vec::new()),
    }
}

/// Selected folders are an owned setting changed only by atomic add/remove.
pub fn set_model_folders(_db: &Db, _folders: &[String]) -> Result<()> {
    Err(KuroError::bad_request(format!(
        "setting `{KEY_MODEL_FOLDERS}` can only be changed through model-folder add/remove"
    )))
}

pub(crate) fn parse_model_folders_json(raw: &str) -> Result<Vec<String>> {
    let value: serde_json::Value = serde_json::from_str(raw).map_err(|error| {
        KuroError::bad_request(format!("{KEY_MODEL_FOLDERS} is malformed: {error}"))
    })?;
    parse_model_folders_value(&value)
}

fn parse_model_folders_value(value: &serde_json::Value) -> Result<Vec<String>> {
    let values = value.as_array().ok_or_else(|| {
        KuroError::bad_request(format!("{KEY_MODEL_FOLDERS} must be an array of paths"))
    })?;
    let mut folders = Vec::with_capacity(values.len());
    for value in values {
        let folder = value.as_str().ok_or_else(|| {
            KuroError::bad_request(format!(
                "{KEY_MODEL_FOLDERS} must contain only string paths"
            ))
        })?;
        folders.push(folder.to_string());
    }
    validate_model_folder_list(&folders)?;
    Ok(folders)
}

pub(crate) fn validate_model_folder_list(folders: &[String]) -> Result<()> {
    for (index, folder) in folders.iter().enumerate() {
        let path = std::path::Path::new(folder);
        let has_alias_component = folder
            .split(['/', '\\'])
            .any(|component| matches!(component, "." | ".."));
        if folder.is_empty() || !path.is_absolute() || has_alias_component {
            return Err(KuroError::bad_request(format!(
                "{KEY_MODEL_FOLDERS} contains a non-canonical path `{folder}`"
            )));
        }
        for previous in &folders[..index] {
            let previous_path = std::path::Path::new(previous);
            if path == previous_path {
                return Err(KuroError::bad_request(format!(
                    "model folder `{folder}` is duplicated"
                )));
            }
            if path.starts_with(previous_path) || previous_path.starts_with(path) {
                return Err(KuroError::bad_request(format!(
                    "model folders `{}` and `{folder}` overlap",
                    previous_path.display()
                )));
            }
        }
    }
    Ok(())
}

pub fn about_you(db: &Db) -> Result<Option<String>> {
    Ok(db
        .get_setting(KEY_MEMORY_ABOUT_YOU)?
        .and_then(|value| value.as_str().map(str::to_string))
        .map(|text| text.trim().to_string())
        .filter(|text| !text.is_empty()))
}

pub const KEY_CHAT_AUTO_ORCHESTRATE: &str = "chat.autoOrchestrate";
pub const KEY_CODE_AUTO_ORCHESTRATE: &str = "code.autoOrchestrate";
pub const KEY_CHAT_DEFAULT_EFFORT: &str = "chat.defaultEffort";
pub const KEY_CODE_DEFAULT_EFFORT: &str = "code.defaultEffort";
pub const KEY_CODE_DEFAULT_MODE: &str = "code.defaultMode";
pub const KEY_MODELS_DIRECTORY: &str = "models.directory";
pub const KEY_FREE_ALLOW_KEYLESS: &str = "free.allowKeyless";
pub const KEY_FREE_ALLOWANCES: &str = "free.allowances";

pub fn allow_keyless(db: &Db) -> Result<bool> {
    Ok(db
        .get_setting(KEY_FREE_ALLOW_KEYLESS)?
        .and_then(|value| value.as_bool())
        .unwrap_or(true))
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Allowance {
    #[serde(rename = "tokensPerDay", skip_serializing_if = "Option::is_none")]
    pub tokens_per_day: Option<i64>,
    #[serde(rename = "tokensPerMonth", skip_serializing_if = "Option::is_none")]
    pub tokens_per_month: Option<i64>,
}

impl Allowance {
    pub fn is_empty(self) -> bool {
        self.tokens_per_day.is_none() && self.tokens_per_month.is_none()
    }
}

pub fn free_allowances(db: &Db) -> Result<HashMap<String, Allowance>> {
    let Some(value) = db.get_setting(KEY_FREE_ALLOWANCES)? else {
        return Ok(HashMap::new());
    };
    Ok(serde_json::from_value(value).unwrap_or_default())
}

pub fn set_free_allowance(db: &Db, slug: &str, allowance: Option<Allowance>) -> Result<()> {
    let mut held = free_allowances(db)?;
    match allowance.filter(|entry| !entry.is_empty()) {
        Some(entry) => held.insert(slug.to_string(), entry),
        None => held.remove(slug),
    };
    db.set_setting(KEY_FREE_ALLOWANCES, &serde_json::to_value(held)?)
}

pub fn auto_orchestrate(db: &Db, surface: crate::orchestrate::Surface) -> Result<bool> {
    let key = match surface {
        crate::orchestrate::Surface::Chat => KEY_CHAT_AUTO_ORCHESTRATE,
        crate::orchestrate::Surface::Code => KEY_CODE_AUTO_ORCHESTRATE,
    };
    Ok(db
        .get_setting(key)?
        .and_then(|value| value.as_bool())
        .unwrap_or(true))
}

pub fn default_effort(db: &Db, surface: crate::orchestrate::Surface) -> Result<Effort> {
    let (key, fallback) = match surface {
        crate::orchestrate::Surface::Chat => (KEY_CHAT_DEFAULT_EFFORT, Effort::Balanced),
        crate::orchestrate::Surface::Code => (KEY_CODE_DEFAULT_EFFORT, Effort::High),
    };

    Ok(db
        .get_setting(key)?
        .and_then(|value| value.as_str().and_then(Effort::parse))
        .unwrap_or(fallback))
}

pub fn default_workspace_mode(db: &Db) -> Result<crate::workspace::WorkspaceMode> {
    Ok(db
        .get_setting(KEY_CODE_DEFAULT_MODE)?
        .and_then(|value| {
            value
                .as_str()
                .and_then(crate::workspace::WorkspaceMode::parse)
        })
        .unwrap_or(crate::workspace::WorkspaceMode::Agent))
}

const DEFAULT_IDLE_UNLOAD_MINUTES: u32 = 30;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EngineProfile {
    #[default]
    Balanced,
    Performance,
    MemorySaver,
    Quiet,
}

impl EngineProfile {
    pub const ALL: [Self; 4] = [
        Self::Balanced,
        Self::Performance,
        Self::MemorySaver,
        Self::Quiet,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Balanced => "balanced",
            Self::Performance => "performance",
            Self::MemorySaver => "memory_saver",
            Self::Quiet => "quiet",
        }
    }

    pub fn parse(raw: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|profile| profile.as_str().eq_ignore_ascii_case(raw.trim()))
    }

    fn base(self, hardware: &HardwareInfo) -> EngineSettings {
        let physical = u32::try_from(hardware.physical_cores).unwrap_or(u32::MAX);
        let logical = u32::try_from(hardware.logical_cores).unwrap_or(u32::MAX);
        match self {
            Self::Balanced => EngineSettings {
                profile: self,
                context_size: hardware.recommended.context_size,
                gpu_layers: hardware.recommended.gpu_layers,
                threads: hardware.recommended.threads.max(1),
                idle_unload_minutes: DEFAULT_IDLE_UNLOAD_MINUTES,
            },
            Self::Performance => EngineSettings {
                profile: self,
                context_size: 8192,
                gpu_layers: if hardware.gpu_available { 999 } else { 0 },
                threads: logical.max(1),
                idle_unload_minutes: 60,
            },
            Self::MemorySaver => EngineSettings {
                profile: self,
                context_size: 2048,
                gpu_layers: 0,
                threads: physical.clamp(1, 4),
                idle_unload_minutes: 5,
            },
            Self::Quiet => EngineSettings {
                profile: self,
                context_size: 4096,
                gpu_layers: hardware.recommended.gpu_layers,
                threads: (physical / 2).max(1),
                idle_unload_minutes: 10,
            },
        }
    }
}

pub fn active_engine_profile(db: &Db) -> Result<EngineProfile> {
    Ok(db
        .get_setting(KEY_ENGINE_PROFILE)?
        .and_then(|value| value.as_str().and_then(EngineProfile::parse))
        .unwrap_or_default())
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineSettingOverrides {
    pub context_size: Option<u32>,
    pub gpu_layers: Option<i32>,
    pub threads: Option<u32>,
    pub idle_unload_minutes: Option<u32>,
}

impl EngineSettingOverrides {
    pub fn read(db: &Db) -> Result<Self> {
        Ok(Self {
            context_size: read_u32(db, KEY_CONTEXT_SIZE)?,
            gpu_layers: read_i64(db, KEY_GPU_LAYERS)?.and_then(|value| i32::try_from(value).ok()),
            threads: read_u32(db, KEY_THREADS)?,
            idle_unload_minutes: read_u32(db, KEY_IDLE_UNLOAD_MINUTES)?,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EngineSettings {
    #[serde(skip)]
    pub profile: EngineProfile,
    pub context_size: u32,
    pub gpu_layers: i32,
    pub threads: u32,
    pub idle_unload_minutes: u32,
}

impl EngineSettings {
    pub fn resolve(db: &Db, hardware: &HardwareInfo) -> Result<Self> {
        let profile = active_engine_profile(db)?;
        let base = profile.base(hardware);
        let overrides = EngineSettingOverrides::read(db)?;

        let context_size = overrides
            .context_size
            .filter(|value| *value > 0)
            .unwrap_or(base.context_size);

        let gpu_layers = overrides
            .gpu_layers
            .filter(|value| *value >= 0)
            .unwrap_or(base.gpu_layers);

        let threads = overrides
            .threads
            .filter(|value| *value > 0)
            .unwrap_or(base.threads);

        let idle_unload_minutes = overrides
            .idle_unload_minutes
            .unwrap_or(base.idle_unload_minutes);

        Ok(Self {
            profile,
            context_size,
            gpu_layers,
            threads,
            idle_unload_minutes,
        })
    }
}

#[derive(Debug, Clone)]
pub struct SearchSettings {
    pub provider: SearchProvider,
    pub base_url: Option<String>,
}

impl SearchSettings {
    pub const KEY_REFERENCE: &'static str = "search:api-key";

    pub fn resolve(db: &Db) -> Result<Self> {
        let provider = db
            .get_setting(KEY_SEARCH_PROVIDER)?
            .and_then(|value| value.as_str().and_then(SearchProvider::parse))
            .unwrap_or(SearchProvider::Duckduckgo);

        let base_url = db
            .get_setting(KEY_SEARCH_BASE_URL)?
            .and_then(|value| value.as_str().map(str::to_string))
            .filter(|url| !url.trim().is_empty());

        Ok(Self { provider, base_url })
    }
}

pub fn default_tool_groups(db: &Db) -> Result<Vec<ToolGroup>> {
    let fallback = || vec![ToolGroup::Memory, ToolGroup::Projects];

    let Some(stored) = db.get_setting(KEY_DEFAULT_TOOL_GROUPS)? else {
        return Ok(fallback());
    };

    Ok(stored
        .as_array()
        .map(|values| {
            values
                .iter()
                .filter_map(|value| value.as_str())
                .filter_map(ToolGroup::parse)
                .collect()
        })
        .unwrap_or_else(fallback))
}

pub fn memory_preload_enabled(db: &Db) -> Result<bool> {
    Ok(db
        .get_setting(KEY_MEMORY_PRELOAD)?
        .and_then(|value| value.as_bool())
        .unwrap_or(true))
}

fn read_u32(db: &Db, key: &str) -> Result<Option<u32>> {
    Ok(db
        .get_setting(key)?
        .and_then(|value| value.as_u64())
        .and_then(|value| u32::try_from(value).ok()))
}

fn read_i64(db: &Db, key: &str) -> Result<Option<i64>> {
    Ok(db.get_setting(key)?.and_then(|value| value.as_i64()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Effort {
    Low,
    #[default]
    Balanced,
    High,
    Max,
    Ultra,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct EffortParams {
    pub temperature: f32,
    pub top_p: f32,
    pub max_tokens: u32,
    pub reasoning_effort: Option<&'static str>,
}

impl Effort {
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.to_ascii_lowercase().as_str() {
            "low" | "instant" => Some(Self::Low),
            "balanced" | "medium" => Some(Self::Balanced),
            "high" | "thinking" => Some(Self::High),
            "max" | "extended" => Some(Self::Max),
            "ultra" | "ultracode" => Some(Self::Ultra),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Balanced => "balanced",
            Self::High => "high",
            Self::Max => "max",
            Self::Ultra => "ultra",
        }
    }

    pub fn coding_only(self) -> bool {
        self == Self::Ultra
    }

    pub fn ceiling(coding: bool) -> Self {
        if coding {
            Self::Ultra
        } else {
            Self::Max
        }
    }

    pub fn params(self) -> EffortParams {
        match self {
            Self::Low => EffortParams {
                temperature: 0.5,
                top_p: 0.9,
                max_tokens: 512,
                reasoning_effort: Some("low"),
            },
            Self::Balanced => EffortParams {
                temperature: 0.7,
                top_p: 0.95,
                max_tokens: 2048,
                reasoning_effort: Some("medium"),
            },
            Self::High => EffortParams {
                temperature: 0.7,
                top_p: 0.95,
                max_tokens: 4096,
                reasoning_effort: Some("high"),
            },
            Self::Max => EffortParams {
                temperature: 0.8,
                top_p: 0.97,
                max_tokens: 8192,
                reasoning_effort: Some("high"),
            },
            Self::Ultra => EffortParams {
                temperature: 0.7,
                top_p: 0.95,
                max_tokens: 16384,
                reasoning_effort: Some("high"),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::{self, HardwareInfo, RecommendedEngineDefaults};
    use serde_json::json;

    fn fixture_hardware() -> HardwareInfo {
        HardwareInfo {
            os: "test".to_string(),
            arch: "fixture".to_string(),
            chip: Some("Fixture CPU".to_string()),
            total_memory_bytes: 32 * 1024 * 1024 * 1024,
            physical_cores: 8,
            logical_cores: 12,
            gpu_available: true,
            gpu_backend: "fixture",
            recommended: RecommendedEngineDefaults {
                context_size: 6144,
                gpu_layers: 42,
                threads: 6,
            },
        }
    }

    #[test]
    fn absent_and_invalid_profiles_resolve_to_balanced() {
        let db = Db::open_in_memory().expect("open");
        assert_eq!(
            active_engine_profile(&db).expect("profile"),
            EngineProfile::Balanced
        );

        db.set_setting(KEY_ENGINE_PROFILE, &json!("turbo"))
            .expect("invalid profile fixture");
        assert_eq!(
            active_engine_profile(&db).expect("profile"),
            EngineProfile::Balanced
        );
    }

    #[test]
    fn every_profile_has_concrete_supported_settings() {
        let db = Db::open_in_memory().expect("open");
        let hardware = fixture_hardware();
        let cases = [
            (EngineProfile::Balanced, 6144, 42, 6, 30),
            (EngineProfile::Performance, 8192, 999, 12, 60),
            (EngineProfile::MemorySaver, 2048, 0, 4, 5),
            (EngineProfile::Quiet, 4096, 42, 4, 10),
        ];

        for (profile, context_size, gpu_layers, threads, idle_unload_minutes) in cases {
            db.set_setting(KEY_ENGINE_PROFILE, &json!(profile.as_str()))
                .expect("profile");
            let settings = EngineSettings::resolve(&db, &hardware).expect("settings");
            assert_eq!(settings.profile, profile);
            assert_eq!(settings.context_size, context_size);
            assert_eq!(settings.gpu_layers, gpu_layers);
            assert_eq!(settings.threads, threads);
            assert_eq!(settings.idle_unload_minutes, idle_unload_minutes);
        }
    }

    #[test]
    fn valid_overrides_win_independently_and_zero_semantics_are_preserved() {
        let db = Db::open_in_memory().expect("open");
        let hardware = fixture_hardware();
        db.set_setting(KEY_ENGINE_PROFILE, &json!("quiet"))
            .expect("profile");
        db.set_setting(KEY_CONTEXT_SIZE, &json!(0))
            .expect("context auto");
        db.set_setting(KEY_GPU_LAYERS, &json!(0)).expect("CPU only");
        db.set_setting(KEY_THREADS, &json!(0))
            .expect("threads auto");
        db.set_setting(KEY_IDLE_UNLOAD_MINUTES, &json!(0))
            .expect("never unload");

        let settings = EngineSettings::resolve(&db, &hardware).expect("settings");
        assert_eq!(settings.context_size, 4096);
        assert_eq!(settings.gpu_layers, 0);
        assert_eq!(settings.threads, 4);
        assert_eq!(settings.idle_unload_minutes, 0);
        assert_eq!(
            EngineSettingOverrides::read(&db).expect("overrides"),
            EngineSettingOverrides {
                context_size: Some(0),
                gpu_layers: Some(0),
                threads: Some(0),
                idle_unload_minutes: Some(0),
            }
        );

        db.set_setting(KEY_CONTEXT_SIZE, &json!(12_345))
            .expect("context");
        db.set_setting(KEY_THREADS, &json!(7)).expect("threads");
        let settings = EngineSettings::resolve(&db, &hardware).expect("settings");
        assert_eq!(settings.context_size, 12_345);
        assert_eq!(settings.gpu_layers, 0);
        assert_eq!(settings.threads, 7);
        assert_eq!(settings.idle_unload_minutes, 0);
    }

    #[test]
    fn falls_back_to_hardware_defaults_when_unset() {
        let db = Db::open_in_memory().expect("open");
        let hw = hardware::detect();

        let settings = EngineSettings::resolve(&db, &hw).expect("resolve");

        assert_eq!(settings.context_size, hw.recommended.context_size);
        assert_eq!(settings.threads, hw.recommended.threads);
        assert_eq!(settings.idle_unload_minutes, DEFAULT_IDLE_UNLOAD_MINUTES);
    }

    #[test]
    fn user_overrides_win_over_defaults() {
        let db = Db::open_in_memory().expect("open");
        let hw = hardware::detect();
        db.set_setting(KEY_CONTEXT_SIZE, &json!(16384))
            .expect("set");
        db.set_setting(KEY_THREADS, &json!(4)).expect("set");

        let settings = EngineSettings::resolve(&db, &hw).expect("resolve");

        assert_eq!(settings.context_size, 16384);
        assert_eq!(settings.threads, 4);
    }

    #[test]
    fn zero_means_auto_not_zero() {
        let db = Db::open_in_memory().expect("open");
        let hw = hardware::detect();
        db.set_setting(KEY_CONTEXT_SIZE, &json!(0)).expect("set");
        db.set_setting(KEY_THREADS, &json!(0)).expect("set");

        let settings = EngineSettings::resolve(&db, &hw).expect("resolve");

        assert_eq!(settings.context_size, hw.recommended.context_size);
        assert_eq!(settings.threads, hw.recommended.threads);
    }

    #[test]
    fn idle_unload_can_be_disabled() {
        let db = Db::open_in_memory().expect("open");
        let hw = hardware::detect();
        db.set_setting(KEY_IDLE_UNLOAD_MINUTES, &json!(0))
            .expect("set");

        let settings = EngineSettings::resolve(&db, &hw).expect("resolve");

        assert_eq!(settings.idle_unload_minutes, 0, "0 must mean never unload");
    }

    #[test]
    fn effort_increases_the_token_budget_monotonically() {
        let budgets: Vec<u32> = [
            Effort::Low,
            Effort::Balanced,
            Effort::High,
            Effort::Max,
            Effort::Ultra,
        ]
        .iter()
        .map(|effort| effort.params().max_tokens)
        .collect();

        assert!(budgets.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(Effort::default(), Effort::Balanced);
        assert_eq!(Effort::parse("HIGH"), Some(Effort::High));
        assert_eq!(Effort::parse("nonsense"), None);
    }

    #[test]
    fn ultra_is_offered_when_coding_and_nowhere_else() {
        assert_eq!(Effort::ceiling(true), Effort::Ultra);
        assert_eq!(Effort::ceiling(false), Effort::Max);
        assert!(Effort::Ultra.coding_only());
        assert!(!Effort::Max.coding_only());
    }

    #[test]
    fn ultra_buys_persistence_rather_than_randomness() {
        let max = Effort::Max.params();
        let ultra = Effort::Ultra.params();

        assert!(ultra.max_tokens > max.max_tokens);
        assert!(ultra.temperature <= max.temperature);
    }

    #[test]
    fn the_names_other_coding_tools_use_are_accepted() {
        assert_eq!(Effort::parse("ultracode"), Some(Effort::Ultra));
        assert_eq!(Effort::parse("medium"), Some(Effort::Balanced));
        assert_eq!(Effort::parse("instant"), Some(Effort::Low));
    }

    #[test]
    fn reading_and_writing_your_own_description_round_trips() {
        let db = Db::open_in_memory().expect("open");
        assert!(about_you(&db).expect("read").is_none());

        db.set_setting(KEY_MEMORY_ABOUT_YOU, &json!("  I work in Rust.  "))
            .expect("set");
        assert_eq!(
            about_you(&db).expect("read").as_deref(),
            Some("I work in Rust.")
        );

        db.set_setting(KEY_MEMORY_ABOUT_YOU, &json!("   "))
            .expect("set");
        assert!(about_you(&db).expect("read").is_none());
    }

    #[test]
    fn malformed_raw_model_folder_json_is_not_mistaken_for_an_empty_setting() {
        let db = Db::open_in_memory().expect("open");
        db.with(|conn| {
            conn.execute(
                "INSERT INTO settings (key, value, updated_at) VALUES (?1, ?2, ?3)",
                rusqlite::params![KEY_MODEL_FOLDERS, "[not-json", "2026-08-16"],
            )?;
            Ok(())
        })
        .expect("seed malformed setting");

        assert!(matches!(model_folders(&db), Err(KuroError::BadRequest(_))));
    }

    #[test]
    fn a_fresh_install_reads_memory_and_projects_but_does_not_search() {
        let db = Db::open_in_memory().expect("open");
        let groups = default_tool_groups(&db).expect("groups");

        assert!(groups.contains(&ToolGroup::Memory));
        assert!(groups.contains(&ToolGroup::Projects));
        assert!(
            !groups.contains(&ToolGroup::Web),
            "searching is the moment a question leaves the machine, so it is opted into"
        );
    }
}
