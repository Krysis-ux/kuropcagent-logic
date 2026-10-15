use std::convert::Infallible;
use std::time::Instant;

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::StreamExt;
use kuro_core::settings::Effort;
use kuro_core::sse::{drain_events, is_done};
use kuro_core::KuroError;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::error::AppResult;
use crate::routes::common::{generation_request_body, resolve_model_id, validate_temperature};
use crate::state::SharedState;

const MAX_CONTENT_CHARS: usize = 20_000;
const MAX_HISTORY_MESSAGES: usize = 32;
const MAX_HISTORY_MESSAGE_CHARS: usize = 8_000;
const MAX_AGGREGATE_CHARS: usize = 64_000;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IncognitoTurnRequest {
    pub content: String,
    #[serde(default)]
    pub temperature: Option<f32>,
    pub model: String,
    #[serde(default)]
    pub effort: Option<String>,
    #[serde(default)]
    pub history: Vec<HistoryMessage>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HistoryMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug)]
struct ValidatedTurn {
    model_id: String,
    effort: Effort,
    temperature: Option<f32>,
    messages: Vec<Value>,
}

#[derive(Debug, Clone)]
struct NativeEvent {
    name: &'static str,
    data: Value,
}

impl NativeEvent {
    fn new(name: &'static str, data: Value) -> Self {
        Self { name, data }
    }

    fn into_sse(self) -> Event {
        Event::default()
            .event(self.name)
            .data(self.data.to_string())
    }
}

/// Stream one temporary, local-only turn.
pub async fn turn(
    State(state): State<SharedState>,
    Json(request): Json<IncognitoTurnRequest>,
) -> AppResult<Response> {
    let mut validated = validate_shape(request)?;
    validated.model_id = resolve_model_id(&state, Some(&validated.model_id)).await?;

    let base_url = state.engines.ensure_base_url(&validated.model_id).await?;
    let client = state.engines.loopback_client().clone();
    let model_id = validated.model_id.clone();
    let task_state = state.clone();
    let (sender, receiver) = mpsc::channel::<NativeEvent>(64);

    tokio::spawn(async move {
        if let Err(error) = stream_from_engine(&client, &base_url, &validated, &sender).await {
            let _ = sender
                .send(NativeEvent::new(
                    "error",
                    json!({ "message": bounded_error(&error.to_string()) }),
                ))
                .await;
        }
        task_state.engines.touch(&model_id).await;
    });

    let stream =
        ReceiverStream::new(receiver).map(|event| Ok::<Event, Infallible>(event.into_sse()));
    Ok(Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response())
}

fn validate_shape(request: IncognitoTurnRequest) -> Result<ValidatedTurn, KuroError> {
    validate_temperature(request.temperature)?;
    if request.content.trim().is_empty() {
        return Err(KuroError::bad_request("message content is empty"));
    }
    let content_chars = request.content.chars().count();
    if content_chars > MAX_CONTENT_CHARS {
        return Err(KuroError::bad_request(format!(
            "message content exceeds the {MAX_CONTENT_CHARS}-character incognito limit"
        )));
    }

    let model_id = request.model.trim();
    if model_id.is_empty() {
        return Err(KuroError::bad_request(
            "incognito requires an explicit ready local model",
        ));
    }
    if model_id.starts_with(kuro_core::cloud::MODEL_PREFIX)
        || model_id.starts_with(kuro_core::free::MODEL_PREFIX)
        || kuro_core::cloud::is_remote_model(model_id)
        || kuro_core::free::is_free_model(model_id)
    {
        return Err(KuroError::bad_request(
            "incognito accepts ready local models only; provider and Free models stay off",
        ));
    }

    let effort = match request.effort.as_deref() {
        Some(raw) => Effort::parse(raw)
            .ok_or_else(|| KuroError::bad_request(format!("unknown effort `{}`", raw.trim())))?,
        None => Effort::default(),
    };

    if request.history.len() > MAX_HISTORY_MESSAGES {
        return Err(KuroError::bad_request(format!(
            "incognito history exceeds the {MAX_HISTORY_MESSAGES}-message limit"
        )));
    }

    let mut aggregate_chars = content_chars;
    let mut messages = Vec::with_capacity(request.history.len() + 1);
    for (index, message) in request.history.into_iter().enumerate() {
        if !matches!(message.role.as_str(), "user" | "assistant") {
            return Err(KuroError::bad_request(format!(
                "incognito history item {} has role `{}`; only user and assistant text is allowed",
                index + 1,
                message.role
            )));
        }
        if message.content.trim().is_empty() {
            return Err(KuroError::bad_request(format!(
                "incognito history item {} has empty content",
                index + 1
            )));
        }
        let chars = message.content.chars().count();
        if chars > MAX_HISTORY_MESSAGE_CHARS {
            return Err(KuroError::bad_request(format!(
                "incognito history item {} exceeds the {MAX_HISTORY_MESSAGE_CHARS}-character limit",
                index + 1
            )));
        }
        aggregate_chars = aggregate_chars.saturating_add(chars);
        if aggregate_chars > MAX_AGGREGATE_CHARS {
            return Err(KuroError::bad_request(format!(
                "incognito history exceeds the {MAX_AGGREGATE_CHARS}-character session limit"
            )));
        }
        messages.push(json!({
            "role": message.role,
            "content": message.content,
        }));
    }
    messages.push(json!({ "role": "user", "content": request.content }));

    Ok(ValidatedTurn {
        model_id: model_id.to_string(),
        effort,
        temperature: request.temperature,
        messages,
    })
}

async fn stream_from_engine(
    client: &reqwest::Client,
    base_url: &str,
    turn: &ValidatedTurn,
    sender: &mpsc::Sender<NativeEvent>,
) -> Result<bool, KuroError> {
    let body = generation_request_body(
        &turn.model_id,
        &turn.messages,
        turn.effort,
        turn.temperature,
    );
    let response = client
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&body)
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let detail = bounded_error(&response.text().await.unwrap_or_default());
        return Err(KuroError::engine(format!(
            "the local engine rejected the private turn ({status}): {detail}"
        )));
    }

    let started = Instant::now();
    let mut first_token_at = None;
    let mut last_token_at = None;
    let mut finish_reason = None;
    let mut prompt_tokens = None;
    let mut completion_tokens = None;
    let mut client_listening = true;
    let mut buffer = Vec::new();
    let mut stream = response.bytes_stream();

    'stream: while let Some(chunk) = stream.next().await {
        buffer.extend_from_slice(&chunk?);
        for incoming in drain_events(&mut buffer) {
            if is_done(&incoming.data) {
                continue;
            }
            let Ok(event) = serde_json::from_str::<Value>(&incoming.data) else {
                continue;
            };

            if let Some(usage) = event.get("usage").filter(|value| !value.is_null()) {
                prompt_tokens = usage.get("prompt_tokens").and_then(Value::as_i64);
                completion_tokens = usage.get("completion_tokens").and_then(Value::as_i64);
            }
            let Some(choice) = event.get("choices").and_then(|choices| choices.get(0)) else {
                continue;
            };
            if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
                finish_reason = Some(reason.to_string());
            }
            let Some(delta) = choice.get("delta") else {
                continue;
            };

            if let Some(text) = delta.get("reasoning_content").and_then(Value::as_str) {
                if !text.is_empty()
                    && sender
                        .send(NativeEvent::new("reasoning", json!({ "content": text })))
                        .await
                        .is_err()
                {
                    client_listening = false;
                    break 'stream;
                }
            }
            if let Some(text) = delta.get("content").and_then(Value::as_str) {
                if !text.is_empty() {
                    first_token_at.get_or_insert_with(Instant::now);
                    last_token_at = Some(Instant::now());
                    if sender
                        .send(NativeEvent::new("token", json!({ "content": text })))
                        .await
                        .is_err()
                    {
                        client_listening = false;
                        break 'stream;
                    }
                }
            }
        }
    }

    let total_ms = started.elapsed().as_millis() as i64;
    let ttft_ms = first_token_at.map(|at| at.duration_since(started).as_millis() as i64);
    let generation_ms = match (first_token_at, last_token_at) {
        (Some(first), Some(last)) => last.duration_since(first).as_millis() as i64,
        _ => 0,
    };
    let tokens_per_second = tokens_per_second(completion_tokens, generation_ms);

    if client_listening {
        let delivered = sender
            .send(NativeEvent::new(
                "done",
                json!({
                    "modelId": turn.model_id,
                    "finishReason": finish_reason.unwrap_or_else(|| "stop".to_string()),
                    "usage": {
                        "promptTokens": prompt_tokens,
                        "completionTokens": completion_tokens,
                    },
                    "timings": {
                        "ttftMs": ttft_ms,
                        "totalMs": total_ms,
                        "tokensPerSecond": tokens_per_second,
                    },
                }),
            ))
            .await
            .is_ok();
        client_listening = delivered;
    }

    Ok(client_listening)
}

fn tokens_per_second(completion_tokens: Option<i64>, generation_ms: i64) -> Option<f64> {
    let tokens = completion_tokens? as f64;
    if tokens <= 0.0 || generation_ms <= 0 {
        return None;
    }
    Some(tokens / (generation_ms as f64 / 1000.0))
}

fn bounded_error(text: &str) -> String {
    const LIMIT: usize = 300;
    let one_line = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if one_line.chars().count() <= LIMIT {
        return one_line;
    }
    let kept: String = one_line.chars().take(LIMIT).collect();
    format!("{}…", kept.trim_end())
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::Arc;

    use axum::body::Body;
    use axum::extract::State;
    use axum::http::header::CONTENT_TYPE;
    use axum::response::Response;
    use axum::routing::post;
    use axum::{Json, Router};
    use kuro_core::db::{Db, NewMessage};
    use serde_json::{json, Value};
    use tokio::sync::{mpsc, Mutex};

    use super::*;

    #[derive(Clone)]
    struct RecordingEngine {
        requests: Arc<Mutex<Vec<Value>>>,
        stream: &'static str,
    }

    async fn record_engine_request(
        State(engine): State<RecordingEngine>,
        Json(request): Json<Value>,
    ) -> Response {
        engine.requests.lock().await.push(request);
        Response::builder()
            .header(CONTENT_TYPE, "text/event-stream")
            .body(Body::from(engine.stream))
            .expect("mock response")
    }

    async fn recording_engine() -> (String, Arc<Mutex<Vec<Value>>>) {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let state = RecordingEngine {
            requests: requests.clone(),
            stream: concat!(
                "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"think\"}}]}\n\n",
                "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n",
                "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],",
                "\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2}}\n\n",
                "data: [DONE]\n\n",
            ),
        };
        let app = Router::new()
            .route("/v1/chat/completions", post(record_engine_request))
            .with_state(state);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("mock listener");
        let address = listener.local_addr().expect("mock address");
        tokio::spawn(async move {
            axum::serve(listener, app).await.expect("mock engine");
        });
        (format!("http://{address}"), requests)
    }

    fn private_turn() -> ValidatedTurn {
        validate_shape(IncognitoTurnRequest {
            temperature: Some(0.25),
            content: "latest question".to_string(),
            model: "local-model".to_string(),
            effort: Some("high".to_string()),
            history: vec![
                HistoryMessage {
                    role: "user".to_string(),
                    content: "earlier question".to_string(),
                },
                HistoryMessage {
                    role: "assistant".to_string(),
                    content: "earlier answer".to_string(),
                },
            ],
        })
        .expect("valid private turn")
    }

    #[test]
    fn shape_validation_appends_the_current_user_message_and_enforces_bounds() {
        let turn = private_turn();
        assert_eq!(turn.model_id, "local-model");
        assert_eq!(turn.effort, kuro_core::settings::Effort::High);
        assert_eq!(
            turn.messages,
            vec![
                json!({ "role": "user", "content": "earlier question" }),
                json!({ "role": "assistant", "content": "earlier answer" }),
                json!({ "role": "user", "content": "latest question" }),
            ]
        );

        let invalid = [
            IncognitoTurnRequest {
                temperature: None,
                content: "current".to_string(),
                model: "local".to_string(),
                effort: None,
                history: vec![HistoryMessage {
                    role: "tool".to_string(),
                    content: "not allowed".to_string(),
                }],
            },
            IncognitoTurnRequest {
                temperature: None,
                content: "current".to_string(),
                model: "local".to_string(),
                effort: None,
                history: (0..33)
                    .map(|_| HistoryMessage {
                        role: "user".to_string(),
                        content: "bounded".to_string(),
                    })
                    .collect(),
            },
            IncognitoTurnRequest {
                temperature: None,
                content: "current".to_string(),
                model: "local".to_string(),
                effort: None,
                history: vec![HistoryMessage {
                    role: "user".to_string(),
                    content: "x".repeat(8_001),
                }],
            },
            IncognitoTurnRequest {
                temperature: None,
                content: "current".to_string(),
                model: "local".to_string(),
                effort: None,
                history: (0..9)
                    .map(|_| HistoryMessage {
                        role: "user".to_string(),
                        content: "x".repeat(8_000),
                    })
                    .collect(),
            },
        ];

        for request in invalid {
            assert!(validate_shape(request).is_err());
        }
    }

    #[tokio::test]
    async fn local_stream_sends_only_plain_messages_and_native_private_events() {
        let (base_url, requests) = recording_engine().await;
        let (sender, mut receiver) = mpsc::channel(16);

        let summary =
            stream_from_engine(&reqwest::Client::new(), &base_url, &private_turn(), &sender)
                .await
                .expect("private stream");
        drop(sender);

        let mut events = Vec::new();
        while let Some(event) = receiver.recv().await {
            events.push(event);
        }

        assert!(summary);
        assert_eq!(
            events.iter().map(|event| event.name).collect::<Vec<_>>(),
            vec!["reasoning", "token", "done"]
        );
        assert_eq!(events[0].data, json!({ "content": "think" }));
        assert_eq!(events[1].data, json!({ "content": "hello" }));
        assert_eq!(events[2].data["modelId"], "local-model");
        assert_eq!(events[2].data["finishReason"], "stop");
        assert_eq!(events[2].data["usage"]["promptTokens"], 3);
        assert_eq!(events[2].data["usage"]["completionTokens"], 2);
        assert!(events[2].data.get("messageId").is_none());
        assert!(events[2].data.get("sources").is_none());
        assert!(events[2].data.get("toolRounds").is_none());

        let requests = requests.lock().await;
        assert_eq!(requests.len(), 1);
        let request = &requests[0];
        assert_eq!(request["model"], "local-model");
        assert_eq!(request["temperature"], 0.25);
        assert_eq!(request["messages"], json!(private_turn().messages));
        for forbidden in [
            "tools",
            "tool_choice",
            "web_search",
            "skills",
            "workspace",
            "project",
            "authorization",
            "provider",
        ] {
            assert!(request.get(forbidden).is_none(), "unexpected `{forbidden}`");
        }
    }

    #[derive(Debug, PartialEq, Eq)]
    struct RowCounts(Vec<i64>);

    impl RowCounts {
        fn read(path: &Path) -> Self {
            let connection = rusqlite::Connection::open(path).expect("count connection");
            let tables = [
                "conversations",
                "messages",
                "memories",
                "projects",
                "workspaces",
                "workspace_changes",
                "mcp_servers",
                "conversation_mcp_servers",
                "user_skills",
            ];
            Self(
                tables
                    .iter()
                    .map(|table| {
                        connection
                            .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
                                row.get(0)
                            })
                            .expect("row count")
                    })
                    .collect(),
            )
        }
    }

    #[tokio::test]
    async fn successful_and_aborted_private_streams_leave_persisted_rows_untouched() {
        let root = tempfile::tempdir().expect("temporary database");
        let database_path = root.path().join("kuro.db");
        let db = Db::open(&database_path).expect("database");
        let before = RowCounts::read(&database_path);
        let (base_url, _) = recording_engine().await;

        let (sender, mut receiver) = mpsc::channel(16);
        stream_from_engine(&reqwest::Client::new(), &base_url, &private_turn(), &sender)
            .await
            .expect("successful private stream");
        while receiver.try_recv().is_ok() {}
        assert_eq!(RowCounts::read(&database_path), before);

        let (sender, receiver) = mpsc::channel(1);
        drop(receiver);
        let aborted =
            stream_from_engine(&reqwest::Client::new(), &base_url, &private_turn(), &sender)
                .await
                .expect("aborted private stream");
        assert!(!aborted);
        assert_eq!(RowCounts::read(&database_path), before);

        let conversation = db.create_conversation(None).expect("normal conversation");
        db.insert_message(&conversation.id, &NewMessage::user("normal user"))
            .expect("normal user row");
        db.insert_message(
            &conversation.id,
            &NewMessage {
                role: "assistant".to_string(),
                content: String::new(),
                ..Default::default()
            },
        )
        .expect("normal assistant row");

        let after_control = RowCounts::read(&database_path);
        assert_eq!(after_control.0[0], 1, "normal chat persists a conversation");
        assert_eq!(after_control.0[1], 2, "normal chat persists both messages");
    }
}
