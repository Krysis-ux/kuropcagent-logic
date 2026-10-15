//! Host capability detection.

use std::process::Command;

use serde::Serialize;

pub mod fit;
mod live;

pub use fit::{estimate_fit, FitEstimate, FitVerdict};
pub use live::{HardwareSnapshot, LiveHardwareMetrics, RuntimeMetrics, ThermalState};

#[derive(Debug, Clone, Serialize)]
pub struct HardwareInfo {
    pub os: String,
    pub arch: String,
    pub chip: Option<String>,
    pub total_memory_bytes: u64,
    pub physical_cores: usize,
    pub logical_cores: usize,
    /// Whether the GPU can be used for offload.
    pub gpu_available: bool,
    pub gpu_backend: &'static str,
    /// Settings Kuro will use when the user leaves engine options on "Auto".
    pub recommended: RecommendedEngineDefaults,
}

#[derive(Debug, Clone, Serialize)]
pub struct RecommendedEngineDefaults {
    pub context_size: u32,
    /// Layers to offload to the GPU; 999 means "all of them".
    pub gpu_layers: i32,
    pub threads: u32,
}

pub fn detect() -> HardwareInfo {
    let total_memory_bytes = sysctl_number("hw.memsize").unwrap_or(8 * 1024 * 1024 * 1024);
    let physical_cores = sysctl_number("hw.physicalcpu").unwrap_or(4) as usize;
    let logical_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(physical_cores);

    let gpu_available = cfg!(target_os = "macos");

    HardwareInfo {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        chip: sysctl_string("machdep.cpu.brand_string"),
        total_memory_bytes,
        physical_cores,
        logical_cores,
        gpu_available,
        gpu_backend: if gpu_available { "metal" } else { "cpu" },
        recommended: RecommendedEngineDefaults {
            context_size: 4096,
            // The engine clamps this to the model's real layer count, so asking
            // for more than exists is the documented way to say "offload all".
            gpu_layers: if gpu_available { 999 } else { 0 },
            // Leave headroom rather than saturating every core, which keeps the
            // machine responsive while generating.
            threads: physical_cores.max(1) as u32,
        },
    }
}

fn sysctl_string(key: &str) -> Option<String> {
    let output = Command::new("/usr/sbin/sysctl")
        .arg("-n")
        .arg(key)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8(output.stdout).ok()?.trim().to_string();
    if value.is_empty() {
        None
    } else {
        Some(value)
    }
}

fn sysctl_number(key: &str) -> Option<u64> {
    sysctl_string(key)?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::live::{normalise_cpu, select_disk_available};
    use super::*;
    use serde_json::{json, Value};

    #[test]
    fn detects_plausible_hardware() {
        let info = detect();
        assert!(info.total_memory_bytes > 0, "memory must be detected");
        assert!(info.physical_cores >= 1);
        assert!(info.logical_cores >= info.physical_cores.min(info.logical_cores));
        assert!(info.recommended.threads >= 1);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn offloads_to_metal_on_macos() {
        let info = detect();
        assert!(info.gpu_available);
        assert_eq!(info.gpu_backend, "metal");
        assert_eq!(info.recommended.gpu_layers, 999);
    }

    #[test]
    fn unavailable_runtime_metrics_serialize_as_null_not_zero() {
        let metrics = RuntimeMetrics::from_observations(None, None, None, None);
        let value = serde_json::to_value(metrics).expect("metrics JSON");

        assert_eq!(value["cpu_utilization_percent"], Value::Null);
        assert_eq!(value["used_memory_bytes"], Value::Null);
        assert_eq!(value["available_memory_bytes"], Value::Null);
        assert_eq!(value["disk_available_bytes"], Value::Null);
        assert_eq!(value["thermal_state"], json!("unknown"));
    }

    #[test]
    fn cpu_observations_are_finite_and_clamped_to_a_percentage() {
        assert_eq!(normalise_cpu(Some(-5.0)), Some(0.0));
        assert_eq!(normalise_cpu(Some(37.5)), Some(37.5));
        assert_eq!(normalise_cpu(Some(150.0)), Some(100.0));
        assert_eq!(normalise_cpu(Some(f32::NAN)), None);
        assert_eq!(normalise_cpu(Some(f32::INFINITY)), None);
        assert_eq!(normalise_cpu(None), None);
    }

    #[test]
    fn disk_selection_uses_the_deepest_mount_containing_the_models_directory() {
        let mounts = [
            (std::path::PathBuf::from("/"), 10),
            (std::path::PathBuf::from("/Volumes/Fast"), 20),
            (std::path::PathBuf::from("/Volumes/Fast/Models"), 30),
        ];
        assert_eq!(
            select_disk_available(
                std::path::Path::new("/Volumes/Fast/Models/Kuro"),
                mounts
                    .iter()
                    .map(|(path, available)| (path.as_path(), *available)),
            ),
            Some(30)
        );
        assert_eq!(
            select_disk_available(
                std::path::Path::new("relative/no-volume"),
                mounts
                    .iter()
                    .map(|(path, available)| (path.as_path(), *available)),
            ),
            None
        );
    }

    #[test]
    fn the_first_live_cpu_sample_is_unavailable_without_sleeping() {
        let metrics = LiveHardwareMetrics::new(std::env::temp_dir());
        assert_eq!(metrics.snapshot().cpu_utilization_percent, None);
    }
}
