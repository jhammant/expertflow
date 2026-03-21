//! MLX backend via Python subprocess communication.
//!
//! This module spawns the MLX Python bridge as a subprocess and communicates
//! via JSON-over-stdio. This avoids PyO3/FFI complexity while leveraging MLX's
//! native Apple Silicon optimizations.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MlxError {
    #[error("Failed to spawn MLX bridge: {0}")]
    SpawnFailed(String),

    #[error("Failed to load model: {0}")]
    ModelLoadFailed(String),

    #[error("Communication error: {0}")]
    CommunicationError(String),

    #[error("Bridge returned error: {0}")]
    BridgeError(String),

    #[error("Invalid response from bridge")]
    InvalidResponse,

    #[error("Bridge process died")]
    ProcessDied,
}

pub type Result<T> = std::result::Result<T, MlxError>;

// JSON protocol structures

#[derive(Serialize, Debug)]
#[serde(tag = "cmd")]
enum BridgeCommand {
    #[serde(rename = "load_model")]
    LoadModel {
        path: String,
        model_type: String,
    },
    #[serde(rename = "get_expert_state")]
    GetExpertState {
        expert_id: u32,
    },
    #[serde(rename = "prefetch_expert")]
    PrefetchExpert {
        expert_id: u32,
    },
    #[serde(rename = "evict_expert")]
    EvictExpert {
        expert_id: u32,
    },
    #[serde(rename = "router_predict")]
    RouterPredict {
        layer: u32,
        hidden_states: Vec<f32>,
    },
    #[serde(rename = "generate_token")]
    GenerateToken {
        input_ids: Vec<u32>,
    },
    #[serde(rename = "shutdown")]
    Shutdown,
}

#[derive(Deserialize, Debug)]
struct BridgeResponse {
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    num_experts: Option<u32>,
    #[serde(default)]
    num_layers: Option<u32>,
    #[serde(default)]
    model_type: Option<String>,
    #[serde(default)]
    expert_id: Option<u32>,
    #[serde(default)]
    memory_bytes: Option<usize>,
    #[serde(default)]
    expert_ids: Option<Vec<u32>>,
    #[serde(default)]
    weights: Option<Vec<f32>>,
    #[serde(default)]
    token_id: Option<u32>,
    #[serde(default)]
    text: Option<String>,
}

/// Subprocess handle for MLX Python bridge
struct MlxBridgeProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl MlxBridgeProcess {
    /// Spawn the MLX bridge Python process
    fn spawn() -> Result<Self> {
        let bridge_path = std::env::current_exe()
            .map_err(|e| MlxError::SpawnFailed(format!("Failed to get current exe: {}", e)))?
            .parent()
            .ok_or_else(|| MlxError::SpawnFailed("No parent directory".to_string()))?
            .join("../src/bridge/mlx_bridge.py");

        let mut child = Command::new("python3")
            .arg(&bridge_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit()) // Let Python stderr go to our stderr
            .spawn()
            .map_err(|e| MlxError::SpawnFailed(format!("Failed to spawn python3: {}", e)))?;

        let stdin = child.stdin.take()
            .ok_or_else(|| MlxError::SpawnFailed("Failed to capture stdin".to_string()))?;

        let stdout = BufReader::new(
            child.stdout.take()
                .ok_or_else(|| MlxError::SpawnFailed("Failed to capture stdout".to_string()))?
        );

        Ok(Self { child, stdin, stdout })
    }

    /// Send a command and receive a response
    fn send_command(&mut self, cmd: &BridgeCommand) -> Result<BridgeResponse> {
        // Serialize and send command
        let cmd_json = serde_json::to_string(cmd)
            .map_err(|e| MlxError::CommunicationError(format!("JSON encode error: {}", e)))?;

        writeln!(self.stdin, "{}", cmd_json)
            .map_err(|e| MlxError::CommunicationError(format!("Write error: {}", e)))?;

        self.stdin.flush()
            .map_err(|e| MlxError::CommunicationError(format!("Flush error: {}", e)))?;

        // Read response
        let mut response_line = String::new();
        self.stdout.read_line(&mut response_line)
            .map_err(|e| MlxError::CommunicationError(format!("Read error: {}", e)))?;

        if response_line.is_empty() {
            return Err(MlxError::ProcessDied);
        }

        // Parse response
        let response: BridgeResponse = serde_json::from_str(&response_line)
            .map_err(|e| MlxError::CommunicationError(format!("JSON decode error: {}", e)))?;

        // Check for errors
        if let Some(error) = response.error {
            return Err(MlxError::BridgeError(error));
        }

        Ok(response)
    }

    /// Shutdown the bridge gracefully
    fn shutdown(&mut self) -> Result<()> {
        let _ = self.send_command(&BridgeCommand::Shutdown);
        let _ = self.child.wait();
        Ok(())
    }
}

impl Drop for MlxBridgeProcess {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

/// Safe wrapper around MLX model loaded via Python bridge
pub struct MlxModel {
    process: Arc<Mutex<MlxBridgeProcess>>,
    num_experts: u32,
    num_layers: u32,
    model_type: String,
}

impl MlxModel {
    /// Load an MLX model from HuggingFace or local path
    ///
    /// # Arguments
    /// * `path` - HuggingFace model ID (e.g., "mlx-community/DeepSeek-V3-Base-4bit")
    ///           or local path to MLX model directory
    ///
    /// # Example
    /// ```no_run
    /// use expertflow::bridge::mlx::MlxModel;
    ///
    /// let model = MlxModel::load("mlx-community/Qwen3-8B-4bit").unwrap();
    /// ```
    pub fn load(path: &str) -> Result<Self> {
        let mut process = MlxBridgeProcess::spawn()?;

        let response = process.send_command(&BridgeCommand::LoadModel {
            path: path.to_string(),
            model_type: "auto".to_string(),
        })?;

        let num_experts = response.num_experts
            .ok_or(MlxError::InvalidResponse)?;
        let num_layers = response.num_layers
            .ok_or(MlxError::InvalidResponse)?;
        let model_type = response.model_type
            .unwrap_or_else(|| "unknown".to_string());

        Ok(Self {
            process: Arc::new(Mutex::new(process)),
            num_experts,
            num_layers,
            model_type,
        })
    }

    /// Get the number of experts per layer
    pub fn num_experts(&self) -> u32 {
        self.num_experts
    }

    /// Get the number of layers
    pub fn num_layers(&self) -> u32 {
        self.num_layers
    }

    /// Get the model type
    pub fn model_type(&self) -> &str {
        &self.model_type
    }

    /// Get expert state (loaded vs evicted)
    pub fn get_expert_state(&self, expert_id: u32) -> Result<ExpertState> {
        let mut process = self.process.lock()
            .map_err(|_| MlxError::CommunicationError("Lock poisoned".to_string()))?;

        let response = process.send_command(&BridgeCommand::GetExpertState { expert_id })?;

        let state_str = response.status
            .ok_or(MlxError::InvalidResponse)?;

        let state = match state_str.as_str() {
            "loaded" => ExpertState::Loaded,
            "evicted" => ExpertState::Evicted,
            _ => ExpertState::Unknown,
        };

        Ok(state)
    }

    /// Prefetch expert weights (hint to MLX to load into memory)
    pub fn prefetch_expert(&self, expert_id: u32) -> Result<()> {
        let mut process = self.process.lock()
            .map_err(|_| MlxError::CommunicationError("Lock poisoned".to_string()))?;

        process.send_command(&BridgeCommand::PrefetchExpert { expert_id })?;
        Ok(())
    }

    /// Evict expert weights from memory
    pub fn evict_expert(&self, expert_id: u32) -> Result<()> {
        let mut process = self.process.lock()
            .map_err(|_| MlxError::CommunicationError("Lock poisoned".to_string()))?;

        process.send_command(&BridgeCommand::EvictExpert { expert_id })?;
        Ok(())
    }

    /// Run router network to predict expert selection
    ///
    /// Returns (expert_ids, weights) for the top-k experts
    pub fn router_predict(&self, layer: u32, hidden_states: &[f32]) -> Result<(Vec<u32>, Vec<f32>)> {
        let mut process = self.process.lock()
            .map_err(|_| MlxError::CommunicationError("Lock poisoned".to_string()))?;

        let response = process.send_command(&BridgeCommand::RouterPredict {
            layer,
            hidden_states: hidden_states.to_vec(),
        })?;

        let expert_ids = response.expert_ids
            .ok_or(MlxError::InvalidResponse)?;
        let weights = response.weights
            .ok_or(MlxError::InvalidResponse)?;

        Ok((expert_ids, weights))
    }

    /// Generate next token
    pub fn generate_token(&self, input_ids: &[u32]) -> Result<(u32, String)> {
        let mut process = self.process.lock()
            .map_err(|_| MlxError::CommunicationError("Lock poisoned".to_string()))?;

        let response = process.send_command(&BridgeCommand::GenerateToken {
            input_ids: input_ids.to_vec(),
        })?;

        let token_id = response.token_id
            .ok_or(MlxError::InvalidResponse)?;
        let text = response.text
            .unwrap_or_default();

        Ok((token_id, text))
    }
}

unsafe impl Send for MlxModel {}
unsafe impl Sync for MlxModel {}

/// Expert state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertState {
    Loaded,
    Evicted,
    Unknown,
}

/// Context for MLX inference (currently a lightweight wrapper)
pub struct MlxContext {
    model: Arc<MlxModel>,
}

impl MlxContext {
    /// Create a new context for the given model
    pub fn new(model: Arc<MlxModel>) -> Self {
        Self { model }
    }

    /// Generate tokens (simplified interface)
    pub fn generate(&self, input_ids: &[u32]) -> Result<(u32, String)> {
        self.model.generate_token(input_ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires MLX installed and model available
    fn test_mlx_model_load() {
        let result = MlxModel::load("mlx-community/Qwen3-1.7B-4bit");
        assert!(result.is_ok());

        let model = result.unwrap();
        assert!(model.num_experts() > 0 || model.num_layers() > 0);
    }

    #[test]
    #[ignore]
    fn test_expert_state() {
        let model = MlxModel::load("mlx-community/Qwen3-1.7B-4bit").unwrap();
        let state = model.get_expert_state(0).unwrap();
        // State should be Evicted or Loaded
        assert!(matches!(state, ExpertState::Evicted | ExpertState::Loaded));
    }
}
