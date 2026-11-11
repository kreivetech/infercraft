use crate::error::InferError;
use std::process::Command;
use std::thread;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuKind {
    Discrete,
    Integrated,
}

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub detected: bool,
    pub name: Option<String>,
    pub kind: Option<GpuKind>,
    pub is_integrated: bool,
    pub vram_total: Option<u32>, // in MB (dedicated or shared UMA)
    pub vram_free: Option<u32>,  // in MB (dedicated or available shared UMA)
    pub use_gpu: bool,
    pub gpu_layers: u32,
}

#[derive(Debug, Clone)]
pub struct HardwareInfo {
    pub gpu: GpuInfo,
    pub cpu_cores: usize,
    pub ram_total_mb: u64,
    pub is_weak_system: bool,
    pub recommendation: String,
    pub recommended_model_id: String,
}

pub struct HardwareDetector;

impl Default for HardwareDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl HardwareDetector {
    pub fn new() -> Self {
        HardwareDetector
    }

    pub fn detect_system(&self) -> Result<HardwareInfo, InferError> {
        let ram_total_mb = self.detect_ram();
        let gpu = self.detect_gpu();
        let cpu_cores = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        // System is weak if RAM < 4500MB or less than 2 cores.
        // If it has >= 7000MB and an integrated GPU, it is capable of local acceleration.
        let is_weak_system = if gpu.detected && gpu.is_integrated {
            ram_total_mb < 5500 || cpu_cores < 2
        } else if gpu.detected {
            ram_total_mb < 6000 || cpu_cores < 2
        } else {
            ram_total_mb < 7500 || cpu_cores < 4
        };

        let mut recommended_model_id = "llama-3.2-1b-q4".to_string();

        if gpu.detected {
            if let Some(vram_free) = gpu.vram_free {
                if vram_free > 4000 {
                    recommended_model_id = "qwen-2.5-3b-q4".to_string();
                } else if vram_free > 2000 {
                    recommended_model_id = "deepseek-r1-1.5b-q8".to_string();
                } else if vram_free > 1000 {
                    recommended_model_id = "qwen-2.5-1.5b-q8".to_string();
                } else {
                    recommended_model_id = "smollm2-135m-q4".to_string();
                }
            }
        } else {
            // CPU-only logic
            if ram_total_mb > 12000 {
                recommended_model_id = "qwen-2.5-3b-q4".to_string();
            } else if ram_total_mb > 7500 {
                recommended_model_id = "deepseek-r1-1.5b-q8".to_string();
            } else if ram_total_mb > 4500 {
                recommended_model_id = "qwen-2.5-0.5b-q4".to_string();
            } else {
                recommended_model_id = "smollm2-135m-q4".to_string();
            }
        }

        let recommendation = if gpu.detected && gpu.is_integrated {
            format!(
                "Integrated GPU detected ({}) with shared UMA memory. Acceleration active ({} layers).",
                gpu.name.as_deref().unwrap_or("iGPU"),
                gpu.gpu_layers
            )
        } else if is_weak_system {
            "Low hardware detected. Local LLM might be slow. Consider Cloud providers.".to_string()
        } else {
            "Your system is well-equipped for local AI inference.".to_string()
        };

        Ok(HardwareInfo {
            gpu,
            cpu_cores,
            ram_total_mb,
            is_weak_system,
            recommendation,
            recommended_model_id,
        })
    }

    pub fn detect_gpu(&self) -> GpuInfo {
        let ram_total_mb = self.detect_ram();

        // 1. Attempt using nvidia-smi (Discrete NVIDIA GPU)
        if let Ok(output) = Command::new("nvidia-smi")
            .args([
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ])
            .output()
            && output.status.success()
        {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let line = stdout.trim();
            if !line.is_empty() {
                let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                if parts.len() >= 3 {
                    let name = parts[0].to_string();
                    let total = parts[1].parse::<u32>().unwrap_or(0);
                    let free = parts[2].parse::<u32>().unwrap_or(0);

                    tracing::info!(
                        "HardwareDetector: Discrete NVIDIA GPU detected: {} ({}MB VRAM)",
                        name,
                        total
                    );

                    let use_gpu = free > 1024;
                    let gpu_layers = if free > 4096 {
                        35
                    } else if free > 2048 {
                        32
                    } else if free > 1024 {
                        16
                    } else {
                        0
                    };

                    return GpuInfo {
                        detected: true,
                        name: Some(name),
                        kind: Some(GpuKind::Discrete),
                        is_integrated: false,
                        vram_total: Some(total),
                        vram_free: Some(free),
                        use_gpu,
                        gpu_layers,
                    };
                }
            }
        }

        // 2. Detect Integrated GPU (Intel UHD/Iris Xe/Arc, AMD Radeon/Vega, Apple Silicon)
        if let Some(igpu) = self.detect_integrated_gpu(ram_total_mb) {
            return igpu;
        }

        tracing::debug!("HardwareDetector: No compatible discrete or integrated GPU detected.");
        GpuInfo {
            detected: false,
            name: None,
            kind: None,
            is_integrated: false,
            vram_total: None,
            vram_free: None,
            use_gpu: false,
            gpu_layers: 0,
        }
    }

    fn detect_integrated_gpu(&self, ram_total_mb: u64) -> Option<GpuInfo> {
        let os = std::env::consts::OS;
        let mut igpu_name: Option<String> = None;

        if os == "linux" {
            // 1. Try lspci
            if let Ok(output) = Command::new("sh")
                .arg("-c")
                .arg("lspci 2>/dev/null | grep -i -E 'vga|3d|display' || true")
                .output()
            {
                let stdout = String::from_utf8_lossy(&output.stdout);
                for raw_line in stdout.lines() {
                    let line = raw_line.trim();
                    let lower = line.to_lowercase();
                    if lower.contains("intel") || lower.contains("amd") || lower.contains("radeon") {
                        let cleaned = if let Some(pos) = line.find(':') {
                            let rest = &line[pos + 1..];
                            if let Some(pos2) = rest.find(':') {
                                rest[pos2 + 1..].trim()
                            } else {
                                rest.trim()
                            }
                        } else {
                            line
                        };
                        let clean_name = if let Some(rev_pos) = cleaned.find("(rev") {
                            cleaned[..rev_pos].trim().to_string()
                        } else {
                            cleaned.to_string()
                        };
                        igpu_name = Some(clean_name);
                        break;
                    }
                }
            }

            // 2. Fallback: inspect /sys/class/drm/ card devices
            if igpu_name.is_none()
                && let Ok(entries) = std::fs::read_dir("/sys/class/drm")
            {
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if name.starts_with("card") && !name.contains('-') {
                        let vendor_path = entry.path().join("device/vendor");
                        if let Ok(vendor_hex) = std::fs::read_to_string(vendor_path) {
                            let v = vendor_hex.trim().to_lowercase();
                            if v == "0x8086" {
                                igpu_name = Some("Intel Integrated Graphics".to_string());
                                break;
                            } else if v == "0x1002" {
                                igpu_name = Some("AMD Radeon Integrated Graphics".to_string());
                                break;
                            }
                        }
                    }
                }
            }
        } else if os == "windows" {
            if let Ok(output) = Command::new("wmic")
                .args(["path", "win32_VideoController", "get", "name"])
                .output()
            {
                let stdout = String::from_utf8_lossy(&output.stdout);
                for line in stdout.lines() {
                    let trimmed = line.trim();
                    let lower = trimmed.to_lowercase();
                    if (lower.contains("intel") || lower.contains("amd") || lower.contains("radeon"))
                        && !lower.contains("name")
                    {
                        igpu_name = Some(trimmed.to_string());
                        break;
                    }
                }
            }
        } else if os == "macos" {
            let is_arm = std::env::consts::ARCH == "aarch64";
            if is_arm {
                igpu_name = Some("Apple Silicon GPU (Unified Memory)".to_string());
            } else if let Ok(output) = Command::new("sysctl")
                .args(["-n", "machdep.cpu.brand_string"])
                .output()
            {
                let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if stdout.to_lowercase().contains("intel") {
                    igpu_name = Some("Intel Integrated Graphics (macOS)".to_string());
                }
            }
        }

        if let Some(name) = igpu_name {
            tracing::info!(
                "HardwareDetector: Integrated GPU (iGPU) detected: {} (Unified Memory architecture)",
                name
            );

            // Integrated GPUs share system RAM dynamically (UMA)
            // Total VRAM allowed by modern drivers: up to ~50% of system RAM
            let vram_total = ((ram_total_mb as f64) * 0.50) as u32;
            // Conservative available VRAM for inference: ~35% of system RAM
            let vram_free = ((ram_total_mb as f64) * 0.35) as u32;

            let use_gpu = ram_total_mb >= 3500;
            let gpu_layers = if vram_free > 4000 {
                24
            } else if vram_free > 2000 {
                16
            } else if vram_free > 1000 {
                8
            } else {
                4
            };

            Some(GpuInfo {
                detected: true,
                name: Some(name),
                kind: Some(GpuKind::Integrated),
                is_integrated: true,
                vram_total: Some(vram_total),
                vram_free: Some(vram_free),
                use_gpu,
                gpu_layers,
            })
        } else {
            None
        }
    }

    pub fn detect_ram(&self) -> u64 {
        let os = std::env::consts::OS;
        if os == "linux" {
            if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
                for line in content.lines() {
                    if line.starts_with("MemTotal:") {
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        if parts.len() >= 2
                            && let Ok(kb) = parts[1].parse::<u64>() {
                                return kb / 1024;
                            }
                    }
                }
            }
        } else if os == "macos" {
            if let Ok(output) = Command::new("sysctl")
                .args(["-n", "hw.memsize"])
                .output()
                && output.status.success() {
                    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
                    if let Ok(bytes) = stdout.parse::<u64>() {
                        return bytes / (1024 * 1024);
                    }
                }
        } else if os == "windows"
            && let Ok(output) = Command::new("wmic")
                .args(["ComputerSystem", "get", "TotalPhysicalMemory"])
                .output()
                && output.status.success() {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    for line in stdout.lines() {
                        let val = line.trim();
                        if !val.is_empty() && val.chars().all(|c| c.is_ascii_digit())
                            && let Ok(bytes) = val.parse::<u64>() {
                                return bytes / (1024 * 1024);
                            }
                    }
                }

        // Fallback: 8 GB
        8192
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_detector_lifecycle() {
        let detector = HardwareDetector::new();
        let info = detector.detect_system().unwrap();
        assert!(info.cpu_cores > 0);
        assert!(info.ram_total_mb > 0);
        if info.gpu.detected {
            assert!(info.gpu.name.is_some());
            assert!(info.gpu.vram_total.is_some());
            assert!(info.gpu.vram_free.is_some());
            assert!(info.gpu.kind.is_some());
            if info.gpu.is_integrated {
                assert_eq!(info.gpu.kind, Some(GpuKind::Integrated));
            }
        }
    }

    #[test]
    fn test_integrated_gpu_calculation() {
        let detector = HardwareDetector::new();
        if let Some(igpu) = detector.detect_integrated_gpu(16384) {
            assert!(igpu.detected);
            assert!(igpu.is_integrated);
            assert_eq!(igpu.kind, Some(GpuKind::Integrated));
            assert!(igpu.vram_total.unwrap() > 4000);
            assert!(igpu.gpu_layers > 0);
        }
    }
}
