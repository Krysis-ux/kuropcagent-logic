use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use kuro_core::KuroError;
use serde_json::json;

/// HTTP wrapper around [`KuroError`].
pub struct AppError(pub KuroError);

impl From<KuroError> for AppError {
    fn from(error: KuroError) -> Self {
        Self(error)
    }
}

impl From<reqwest::Error> for AppError {
    fn from(error: reqwest::Error) -> Self {
        Self(KuroError::Http(error))
    }
}

impl From<serde_json::Error> for AppError {
    fn from(error: serde_json::Error) -> Self {
        Self(KuroError::Json(error))
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, kind) = match &self.0 {
            KuroError::NotFound(_) => (StatusCode::NOT_FOUND, "not_found"),
            KuroError::BadRequest(_) => (StatusCode::BAD_REQUEST, "invalid_request_error"),
            KuroError::Model(_) => (StatusCode::BAD_REQUEST, "model_error"),
            KuroError::Engine(_) => (StatusCode::SERVICE_UNAVAILABLE, "engine_error"),
            KuroError::Hub { status, .. } => (
                StatusCode::from_u16(*status).unwrap_or(StatusCode::BAD_GATEWAY),
                "upstream_error",
            ),
            _ => (StatusCode::INTERNAL_SERVER_ERROR, "internal_error"),
        };

        // Server-side faults are logged in full; the client gets the message.
        if status == StatusCode::INTERNAL_SERVER_ERROR {
            tracing::error!(error = %self.0, "request failed");
        }

        let (code, retry_after) = match &self.0 {
            KuroError::Hub {
                code,
                retry_after_seconds,
                ..
            } => (Some(*code), *retry_after_seconds),
            _ => (None, None),
        };
        let body = Json(json!({
            "error": {
                "message": self.0.to_string(),
                "type": kind,
                "code": code,
                "retry_after_seconds": retry_after,
            }
        }));

        let mut response = (status, body).into_response();
        if let Some(seconds) = retry_after {
            if let Ok(value) = seconds.to_string().parse() {
                response
                    .headers_mut()
                    .insert(axum::http::header::RETRY_AFTER, value);
            }
        }
        response
    }
}

pub type AppResult<T> = std::result::Result<T, AppError>;
