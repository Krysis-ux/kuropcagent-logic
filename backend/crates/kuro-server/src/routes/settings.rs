use axum::extract::State;
use axum::Json;
use serde_json::{json, Value};

use crate::error::AppResult;
use crate::state::SharedState;

/// Every stored setting.
pub async fn get_settings(State(state): State<SharedState>) -> AppResult<Json<Value>> {
    Ok(Json(Value::Object(state.db.all_settings()?)))
}

/// The effective engine policy after the selected profile and explicit
/// per-setting overrides have been combined.
pub async fn engine_settings(State(state): State<SharedState>) -> AppResult<Json<Value>> {
    let effective = kuro_core::settings::EngineSettings::resolve(&state.db, &state.hardware)?;
    let overrides = kuro_core::settings::EngineSettingOverrides::read(&state.db)?;

    Ok(Json(json!({
        "activeProfile": effective.profile.as_str(),
        "effectiveEngineSettings": effective,
        "overrides": overrides,
    })))
}

/// Merge a partial settings object. Sending `null` for a key removes it.
pub async fn patch_settings(
    State(state): State<SharedState>,
    Json(payload): Json<Value>,
) -> AppResult<Json<Value>> {
    Ok(Json(patch_settings_on(&state.db, payload)?))
}

fn patch_settings_on(db: &kuro_core::Db, payload: Value) -> kuro_core::Result<Value> {
    let Value::Object(updates) = payload else {
        return Err(kuro_core::KuroError::bad_request(
            "settings must be a JSON object",
        ));
    };
    if updates.contains_key(kuro_core::settings::KEY_MODEL_FOLDERS) {
        return Err(kuro_core::KuroError::bad_request(format!(
            "setting `{}` can only be changed through model-folder add/remove",
            kuro_core::settings::KEY_MODEL_FOLDERS
        )));
    }
    Ok(Value::Object(db.apply_settings_patch(&updates)?))
}

/// Restore defaults by clearing every stored override.
pub async fn reset_settings(State(state): State<SharedState>) -> AppResult<Json<Value>> {
    Ok(Json(reset_settings_on(&state.db)?))
}

fn reset_settings_on(db: &kuro_core::Db) -> kuro_core::Result<Value> {
    Ok(Value::Object(db.reset_unowned_settings()?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kuro_core::settings::KEY_MODEL_FOLDERS;
    use kuro_core::Db;
    use serde_json::json;

    #[test]
    fn route_patch_rejects_owned_key_without_applying_an_ordinary_sibling() {
        let db = Db::open_in_memory().expect("db");

        assert!(patch_settings_on(&db, json!({"ordinary": true, KEY_MODEL_FOLDERS: []}),).is_err());
        assert!(db.all_settings().expect("unchanged").is_empty());
    }

    #[test]
    fn route_patch_rejects_a_non_object_without_writes() {
        let db = Db::open_in_memory().expect("db");

        assert!(patch_settings_on(&db, json!(["not", "an", "object"])).is_err());
        assert!(db.all_settings().expect("unchanged").is_empty());
    }

    #[test]
    fn route_reset_returns_the_preserved_owned_setting() {
        let db = Db::open_in_memory().expect("db");
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/kuro-settings-route-tests")
            .join(uuid::Uuid::new_v4().to_string());
        std::fs::create_dir_all(&root).expect("fixture root");
        let root = root.canonicalize().expect("canonical root");
        db.commit_model_folder_add(&kuro_core::catalog::scan_model_folder(&root).expect("scan"))
            .expect("select folder");
        db.set_setting("ordinary", &json!(true)).expect("ordinary");

        let reset = reset_settings_on(&db).expect("reset");

        assert!(reset
            .as_object()
            .expect("object")
            .contains_key(KEY_MODEL_FOLDERS));
        assert!(db.get_setting("ordinary").expect("ordinary").is_none());
        assert_eq!(db.model_folder_scan_tickets().expect("ticket").len(), 1);
        std::fs::remove_dir_all(root).expect("cleanup fixture");
    }
}
