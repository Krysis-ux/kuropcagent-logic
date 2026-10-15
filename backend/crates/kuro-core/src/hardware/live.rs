use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde::Serialize;
use sysinfo::{Disks, System, MINIMUM_CPU_UPDATE_INTERVAL};

use super::HardwareInfo;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ThermalState {
    #[default]
    Unknown,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct RuntimeMetrics {
    pub cpu_utilization_percent: Option<f32>,
    pub used_memory_bytes: Option<u64>,
    pub available_memory_bytes: Option<u64>,
    pub disk_available_bytes: Option<u64>,
    pub thermal_state: ThermalState,
}

impl RuntimeMetrics {
    pub(super) fn from_observations(
        cpu_utilization_percent: Option<f32>,
        used_memory_bytes: Option<u64>,
        available_memory_bytes: Option<u64>,
        disk_available_bytes: Option<u64>,
    ) -> Self {
        Self {
            cpu_utilization_percent: normalise_cpu(cpu_utilization_percent),
            used_memory_bytes,
            available_memory_bytes,
            disk_available_bytes,
            thermal_state: ThermalState::Unknown,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct HardwareSnapshot {
    #[serde(flatten)]
    pub hardware: HardwareInfo,
    #[serde(flatten)]
    pub runtime: RuntimeMetrics,
}

impl HardwareSnapshot {
    pub fn new(hardware: &HardwareInfo, runtime: RuntimeMetrics) -> Self {
        Self {
            hardware: hardware.clone(),
            runtime,
        }
    }
}

#[derive(Clone)]
pub struct LiveHardwareMetrics {
    inner: Arc<Mutex<Collector>>,
}

struct Collector {
    system: System,
    disks: Disks,
    models_directory: PathBuf,
    last_cpu_refresh: Option<Instant>,
    cpu_ready: bool,
}

impl LiveHardwareMetrics {
    pub fn new(models_directory: impl Into<PathBuf>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Collector {
                system: System::new(),
                disks: Disks::new_with_refreshed_list(),
                models_directory: models_directory.into(),
                last_cpu_refresh: None,
                cpu_ready: false,
            })),
        }
    }

    /// Refresh what the platform can measure without blocking for a CPU sample.
    pub fn snapshot(&self) -> RuntimeMetrics {
        let mut collector = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        collector.system.refresh_memory();
        let memory_known = collector.system.total_memory() > 0;
        let used_memory_bytes = memory_known.then(|| collector.system.used_memory());
        let available_memory_bytes = memory_known.then(|| collector.system.available_memory());

        let now = Instant::now();
        let cpu_utilization_percent = match collector.last_cpu_refresh {
            None => {
                collector.system.refresh_cpu_usage();
                collector.last_cpu_refresh = Some(now);
                None
            }
            Some(previous) if now.duration_since(previous) >= MINIMUM_CPU_UPDATE_INTERVAL => {
                collector.system.refresh_cpu_usage();
                collector.last_cpu_refresh = Some(now);
                collector.cpu_ready = true;
                Some(collector.system.global_cpu_usage())
            }
            Some(_) if collector.cpu_ready => Some(collector.system.global_cpu_usage()),
            Some(_) => None,
        };

        collector.disks.refresh(true);
        let models_directory = collector
            .models_directory
            .canonicalize()
            .unwrap_or_else(|_| collector.models_directory.clone());
        let disk_available_bytes = select_disk_available(
            &models_directory,
            collector
                .disks
                .list()
                .iter()
                .map(|disk| (disk.mount_point(), disk.available_space())),
        );

        RuntimeMetrics::from_observations(
            cpu_utilization_percent,
            used_memory_bytes,
            available_memory_bytes,
            disk_available_bytes,
        )
    }
}

pub(super) fn normalise_cpu(value: Option<f32>) -> Option<f32> {
    value.and_then(|value| value.is_finite().then(|| value.clamp(0.0, 100.0)))
}

pub(super) fn select_disk_available<'a>(
    models_directory: &Path,
    mounts: impl Iterator<Item = (&'a Path, u64)>,
) -> Option<u64> {
    mounts
        .filter(|(mount, _)| models_directory.starts_with(mount))
        .max_by_key(|(mount, _)| mount.components().count())
        .map(|(_, available)| available)
}
