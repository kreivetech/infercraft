use candle_core::Device;

/// Priority: CUDA → Metal → CPU.
pub fn select_device() -> Device {
    #[cfg(feature = "cuda")]
    {
        match Device::new_cuda(0) {
            Ok(d) => {
                tracing::info!("Inference: selected CUDA device 0");
                return d;
            }
            Err(e) => tracing::warn!("Inference: CUDA init failed ({e}), trying next"),
        }
    }

    #[cfg(feature = "metal")]
    {
        match Device::new_metal(0) {
            Ok(d) => {
                tracing::info!("Inference: selected Metal device 0");
                return d;
            }
            Err(e) => tracing::warn!("Inference: Metal init failed ({e}), trying next"),
        }
    }

    tracing::info!("Inference: selected CPU");
    Device::Cpu
}

pub fn device_label(d: &Device) -> String {
    match d {
        Device::Cpu => "CPU".to_string(),
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => "CUDA:0".to_string(),
        #[cfg(feature = "metal")]
        Device::Metal(_) => "Metal:0".to_string(),
        #[allow(unreachable_patterns)]
        _ => "GPU".to_string(),
    }
}

pub fn is_oom(e: &anyhow::Error) -> bool {
    let s = e.to_string().to_lowercase();
    s.contains("out of memory")
        || s.contains("insufficient memory")
        || s.contains("cuda error")
        || s.contains("metal error")
        || s.contains("oom")
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::anyhow;

    #[test]
    fn test_device_label_cpu() {
        assert_eq!(device_label(&Device::Cpu), "CPU");
    }

    #[test]
    fn test_is_oom_detection() {
        assert!(is_oom(&anyhow!("CUDA out of memory")));
        assert!(is_oom(&anyhow!("Insufficient memory on device")));
        assert!(is_oom(&anyhow!("OOM occurred")));
        assert!(!is_oom(&anyhow!("File not found")));
    }
}
