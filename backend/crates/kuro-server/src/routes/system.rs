use axum::extract::State;
use axum::Json;
use serde_json::{json, Value};

use kuro_core::KuroError;

use crate::error::AppResult;
use crate::state::SharedState;

/// Liveness probe used by the CLI to find a running daemon.
pub async fn health() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}

/// What the Settings → Server page shows.
pub async fn status(State(state): State<SharedState>) -> AppResult<Json<Value>> {
    let loaded = state.engines.loaded().await;

    Ok(Json(json!({
        "name": "Kuro LLM",
        "version": env!("CARGO_PKG_VERSION"),
        "status": "running",
        "host": "127.0.0.1",
        "port": state.port,
        "address": format!("http://127.0.0.1:{}", state.port),
        "uptimeSeconds": state.uptime_seconds(),
        "startedAt": state.started_at.to_rfc3339(),
        "dataDirectory": state.paths.root.to_string_lossy(),
        "loadedModels": loaded,
    })))
}

/// Stop the daemon.
pub async fn shutdown(State(state): State<SharedState>) -> AppResult<Json<Value>> {
    let loaded = state.engines.loaded().await.len();
    tracing::info!(engines = loaded, "shutdown requested from the interface");

    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
        state.engines.unload_all().await;
        tracing::info!("stopping");
        // The graceful-shutdown future in `main` is watching for a signal, so raising one here
        // takes the same path as Ctrl-C rather than a second, separately-maintained exit route.
        #[cfg(unix)]
        // SAFETY: raising SIGTERM in our own process, which `main` handles.
        unsafe {
            libc::raise(libc::SIGTERM);
        }
        #[cfg(not(unix))]
        std::process::exit(0);
    });

    Ok(Json(json!({
        "stopping": true,
        "unloadingEngines": loaded,
    })))
}

/// Relaunch the daemon.
pub async fn restart(State(state): State<SharedState>) -> AppResult<Json<Value>> {
    let executable = std::env::current_exe()
        .map_err(|error| KuroError::other(format!("cannot find Kuro's own binary: {error}")))?;

    // Arguments are carried over so a daemon started with flags restarts with the
    // same ones.
    let arguments: Vec<String> = std::env::args().skip(1).collect();

    let mut command = std::process::Command::new(&executable);
    command
        .args(&arguments)
        .env(crate::RESTART_MARKER, "1")
        // Detached from this process's group, so the signal that stops us does not
        // also stop the successor.
        .stdin(std::process::Stdio::null());

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }

    let child = command
        .spawn()
        .map_err(|error| KuroError::other(format!("could not start the new process: {error}")))?;

    tracing::info!(pid = child.id(), "restarting; successor started");

    let port = state.port;
    tokio::spawn(async move {
        // Long enough for this response to reach the browser, which is what lets
        // the page start polling for the successor instead of showing an error.
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
        state.engines.unload_all().await;
        tracing::info!("handing over");

        #[cfg(unix)]
        // SAFETY: raising SIGTERM in our own process, which `main` handles.
        unsafe {
            libc::raise(libc::SIGTERM);
        }
        #[cfg(not(unix))]
        std::process::exit(0);
    });

    Ok(Json(json!({
        "restarting": true,
        "port": port,
    })))
}

/// Detected hardware plus the engine defaults derived from it.
pub async fn hardware(State(state): State<SharedState>) -> AppResult<Json<Value>> {
    let settings = kuro_core::settings::EngineSettings::resolve(&state.db, &state.hardware)?;
    let snapshot =
        kuro_core::hardware::HardwareSnapshot::new(&state.hardware, state.live_hardware.snapshot());

    Ok(Json(json!({
        "hardware": snapshot,
        "activeProfile": settings.profile.as_str(),
        "effectiveEngineSettings": settings,
    })))
}
