use thiserror::Error;

/// Error types for `infercraft`.
#[derive(Debug, Error)]
pub enum InferError {
    /// LLM inference or execution error
    #[error("Inference error: {0}")]
    LlmError(String),

    /// Requested model is not loaded or not found
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    /// System or GPU ran out of memory
    #[error("Out of memory: {0}")]
    OutOfMemory(String),

    /// Inference operation was interrupted
    #[error("Operation interrupted")]
    Interrupted,

    /// Hardware detection or setup error
    #[error("Hardware error: {0}")]
    HardwareError(String),

    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// General internal error
    #[error("Error: {0}")]
    Other(String),
}

/// Backward-compatible type alias
pub type UnevenError = InferError;
