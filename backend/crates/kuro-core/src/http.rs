//! Shared outbound HTTP client.

use std::time::Duration;

use crate::Result;

pub const USER_AGENT: &str = concat!("kuro-llm/", env!("CARGO_PKG_VERSION"));

/// Client for talking to Hugging Face, GitHub and cloud provider APIs.
pub fn client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .user_agent(USER_AGENT)
        .connect_timeout(Duration::from_secs(20))
        .pool_idle_timeout(Duration::from_secs(60))
        .build()
        .map_err(Into::into)
}

/// Client for talking to a local engine process.
pub fn loopback_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .user_agent(USER_AGENT)
        .connect_timeout(Duration::from_secs(5))
        .no_proxy()
        .build()
        .map_err(Into::into)
}
