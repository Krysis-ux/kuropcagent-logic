//! Raw settings key/value storage.

use rusqlite::{params, Connection, OptionalExtension};
use serde_json::{Map, Value};

use super::{now, Db};
use crate::settings::KEY_MODEL_FOLDERS;
use crate::{KuroError, Result};

impl Db {
    pub fn get_setting(&self, key: &str) -> Result<Option<Value>> {
        self.with(|conn| {
            let raw: Option<String> = conn
                .query_row(
                    "SELECT value FROM settings WHERE key = ?1",
                    params![key],
                    |row| row.get(0),
                )
                .optional()?;

            match raw {
                Some(text) => Ok(serde_json::from_str(&text).ok()),
                None => Ok(None),
            }
        })
    }

    pub fn set_setting(&self, key: &str, value: &Value) -> Result<()> {
        reject_owned_setting_key(key)?;
        let encoded = serde_json::to_string(value)?;
        self.with(|conn| {
            conn.execute(
                "INSERT INTO settings (key, value, updated_at) VALUES (?1, ?2, ?3)
                 ON CONFLICT (key) DO UPDATE SET value = excluded.value,
                                                 updated_at = excluded.updated_at",
                params![key, encoded, now()],
            )?;
            Ok(())
        })
    }

    /// Every stored setting. Keys absent here fall back to code defaults.
    pub fn all_settings(&self) -> Result<Map<String, Value>> {
        self.with(all_settings_on)
    }

    pub fn delete_setting(&self, key: &str) -> Result<()> {
        reject_owned_setting_key(key)?;
        self.with(|conn| {
            conn.execute("DELETE FROM settings WHERE key = ?1", params![key])?;
            Ok(())
        })
    }

    pub fn apply_settings_patch(&self, updates: &Map<String, Value>) -> Result<Map<String, Value>> {
        for key in updates.keys() {
            reject_owned_setting_key(key)?;
        }
        self.with(|conn| {
            let transaction = conn.unchecked_transaction()?;
            for (key, value) in updates {
                if value.is_null() {
                    transaction.execute("DELETE FROM settings WHERE key = ?1", params![key])?;
                } else {
                    let encoded = serde_json::to_string(value)?;
                    transaction.execute(
                        "INSERT INTO settings (key, value, updated_at) VALUES (?1, ?2, ?3)
                         ON CONFLICT (key) DO UPDATE SET value = excluded.value,
                                                         updated_at = excluded.updated_at",
                        params![key, encoded, now()],
                    )?;
                }
            }
            let committed = all_settings_on(&transaction)?;
            transaction.commit()?;
            Ok(committed)
        })
    }

    pub fn reset_unowned_settings(&self) -> Result<Map<String, Value>> {
        self.with(|conn| {
            let transaction = conn.unchecked_transaction()?;
            transaction.execute(
                "DELETE FROM settings WHERE key <> ?1",
                params![KEY_MODEL_FOLDERS],
            )?;
            let committed = all_settings_on(&transaction)?;
            transaction.commit()?;
            Ok(committed)
        })
    }
}

fn all_settings_on(conn: &Connection) -> Result<Map<String, Value>> {
    let mut stmt = conn.prepare("SELECT key, value FROM settings")?;
    let rows = stmt
        .query_map([], |row| {
            let key: String = row.get(0)?;
            let value: String = row.get(1)?;
            Ok((key, value))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    let mut map = Map::new();
    for (key, value) in rows {
        if let Ok(parsed) = serde_json::from_str(&value) {
            map.insert(key, parsed);
        }
    }
    Ok(map)
}

fn reject_owned_setting_key(key: &str) -> Result<()> {
    if key == KEY_MODEL_FOLDERS {
        return Err(KuroError::bad_request(format!(
            "setting `{KEY_MODEL_FOLDERS}` can only be changed through model-folder add/remove"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn stores_and_overwrites_typed_values() {
        let db = Db::open_in_memory().expect("open");
        assert_eq!(db.get_setting("theme").expect("get"), None);

        db.set_setting("theme", &json!("dark")).expect("set");
        assert_eq!(db.get_setting("theme").expect("get"), Some(json!("dark")));

        db.set_setting("theme", &json!("light")).expect("overwrite");
        assert_eq!(db.get_setting("theme").expect("get"), Some(json!("light")));

        db.set_setting("server", &json!({"port": 8420}))
            .expect("object");
        let all = db.all_settings().expect("all");
        assert_eq!(all["server"]["port"], json!(8420));
        assert_eq!(all.len(), 2);

        db.delete_setting("theme").expect("delete");
        assert_eq!(db.get_setting("theme").expect("get"), None);
    }

    #[test]
    fn settings_patch_commits_mixed_updates_and_deletes_atomically() {
        let db = Db::open_in_memory().expect("open");
        db.set_setting("remove", &json!(true)).expect("seed");
        let updates = serde_json::Map::from_iter([
            ("first".to_string(), json!(1)),
            ("remove".to_string(), Value::Null),
            ("second".to_string(), json!({"nested": true})),
        ]);

        let committed = db.apply_settings_patch(&updates).expect("atomic patch");

        assert_eq!(committed.get("first"), Some(&json!(1)));
        assert_eq!(committed.get("second"), Some(&json!({"nested": true})));
        assert!(!committed.contains_key("remove"));
        assert_eq!(db.all_settings().expect("stored"), committed);
    }

    #[test]
    fn settings_patch_rejects_owned_key_in_any_position_without_partial_writes() {
        for updates in [
            serde_json::Map::from_iter([
                (KEY_MODEL_FOLDERS.to_string(), json!([])),
                ("ordinary".to_string(), json!(1)),
            ]),
            serde_json::Map::from_iter([
                ("ordinary".to_string(), json!(1)),
                (KEY_MODEL_FOLDERS.to_string(), json!([])),
            ]),
        ] {
            let db = Db::open_in_memory().expect("open");

            assert!(matches!(
                db.apply_settings_patch(&updates),
                Err(KuroError::BadRequest(_))
            ));
            assert!(db.all_settings().expect("unchanged").is_empty());
        }
    }

    #[test]
    fn reset_preserves_owned_model_folders_and_removes_only_unowned_keys() {
        let db = Db::open_in_memory().expect("open");
        db.with(|conn| {
            conn.execute(
                "INSERT INTO settings (key, value, updated_at) VALUES (?1, ?2, ?3)",
                params![KEY_MODEL_FOLDERS, json!(["/selected"]).to_string(), now()],
            )?;
            Ok(())
        })
        .expect("seed owned setting");
        db.set_setting("ordinary", &json!(true)).expect("ordinary");

        let committed = db.reset_unowned_settings().expect("atomic reset");

        assert_eq!(
            committed,
            serde_json::Map::from_iter([(KEY_MODEL_FOLDERS.to_string(), json!(["/selected"]),)])
        );
        assert_eq!(db.all_settings().expect("stored"), committed);
    }

    #[test]
    fn reset_without_a_selected_folder_returns_an_empty_map() {
        let db = Db::open_in_memory().expect("open");
        db.set_setting("ordinary", &json!(true)).expect("ordinary");

        assert!(db.reset_unowned_settings().expect("reset").is_empty());
        assert!(db.all_settings().expect("stored").is_empty());
    }
}
