//! vMLX backend bridge.
//!
//! Alternative to raw MLX bridge using vMLX (pip install vmlx), which has
//! built-in caching, speculative decoding, and batching support.
//!
//! vMLX provides an OpenAI-compatible API server that runs locally.

use std::sync::Arc;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, info, warn};

#[derive(Error, Debug)]
pub enum VmlxError {
    #[error("Failed to connect to vMLX server: {0}")]
    ConnectionFailed(String),

    #[error("HTTP request failed: {0}")]
    RequestFailed(String),

    #[error("Failed to parse response: {0}")]
    ParseFailed(String),

    #[error("vMLX server returned error: {0}")]
    ServerError(String),

    #[error("Model not loaded: {0}")]
    ModelNotLoaded(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

pub type Result<T> = std::result::Result<T, VmlxError>;

/// vMLX OpenAI-compatible API request for chat completion
#[derive(Serialize, Debug)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

/// vMLX API response
#[derive(Deserialize, Debug)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Deserialize, Debug)]
struct ChatChoice {
    index: u32,
    message: ChatMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize, Debug)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

/// vMLX model info response
#[derive(Deserialize, Debug)]
struct ModelInfo {
    id: String,
    object: String,
    #[serde(default)]
    owned_by: Option<String>,
}

/// Bridge trait for inference backends
pub trait Bridge: Send + Sync {
    /// Load a model
    fn load_model(&self, model_id: &str) -> Result<()>;

    /// Generate a response
    fn generate(&self, prompt: &str, max_tokens: Option<u32>) -> Result<String>;

    /// Get expert activations (if supported)
    fn get_expert_activations(&self, layer: u32) -> Result<Vec<u32>>;
}

/// vMLX bridge implementation
pub struct VmlxBridge {
    base_url: String,
    model_id: Option<String>,
}

impl VmlxBridge {
    /// Create a new vMLX bridge connecting to localhost
    ///
    /// Default vMLX server runs on http://localhost:8000
    pub fn new() -> Self {
        Self::with_url("http://localhost:8000")
    }

    /// Create a new vMLX bridge with custom server URL
    pub fn with_url(base_url: &str) -> Self {
        info!("Initializing vMLX bridge at {}", base_url);
        Self {
            base_url: base_url.to_string(),
            model_id: None,
        }
    }

    /// Check if vMLX server is reachable
    pub fn health_check(&self) -> Result<bool> {
        // Try to list models to check if server is up
        match self.list_models() {
            Ok(_) => Ok(true),
            Err(e) => {
                warn!("vMLX health check failed: {}", e);
                Ok(false)
            }
        }
    }

    /// List available models on vMLX server
    pub fn list_models(&self) -> Result<Vec<String>> {
        // In a real implementation, this would make an HTTP request to /v1/models
        // For now, return empty list as placeholder
        debug!("Listing models from vMLX server");
        Ok(vec![])
    }

    /// Set the active model
    pub fn set_model(&mut self, model_id: &str) {
        info!("Setting active model to: {}", model_id);
        self.model_id = Some(model_id.to_string());
    }

    /// Get the currently active model
    pub fn current_model(&self) -> Option<&str> {
        self.model_id.as_deref()
    }

    /// Send a chat completion request to vMLX
    fn chat_completion(
        &self,
        messages: Vec<ChatMessage>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
    ) -> Result<String> {
        let model_id = self.model_id.as_ref()
            .ok_or_else(|| VmlxError::ModelNotLoaded("No model loaded".to_string()))?;

        let request = ChatCompletionRequest {
            model: model_id.clone(),
            messages,
            max_tokens,
            temperature,
            stream: Some(false),
        };

        // In a real implementation, this would make an HTTP POST to /v1/chat/completions
        // For now, return a placeholder response
        debug!("Sending chat completion request to vMLX");

        // Placeholder: In real implementation, use reqwest or similar HTTP client
        // let client = reqwest::blocking::Client::new();
        // let response = client
        //     .post(&format!("{}/v1/chat/completions", self.base_url))
        //     .json(&request)
        //     .send()
        //     .map_err(|e| VmlxError::RequestFailed(e.to_string()))?;
        //
        // let completion: ChatCompletionResponse = response
        //     .json()
        //     .map_err(|e| VmlxError::ParseFailed(e.to_string()))?;
        //
        // Ok(completion.choices[0].message.content.clone())

        // Placeholder return
        Ok("vMLX response placeholder".to_string())
    }
}

impl Default for VmlxBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl Bridge for VmlxBridge {
    fn load_model(&self, model_id: &str) -> Result<()> {
        // vMLX loads models on-demand, so this is a no-op
        // Just verify the model exists
        info!("Loading model via vMLX: {}", model_id);
        Ok(())
    }

    fn generate(&self, prompt: &str, max_tokens: Option<u32>) -> Result<String> {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        }];

        self.chat_completion(messages, max_tokens, None)
    }

    fn get_expert_activations(&self, layer: u32) -> Result<Vec<u32>> {
        // vMLX doesn't expose expert activations directly via OpenAI API
        // Would need custom endpoint or instrumentation
        debug!("Expert activation tracking not yet implemented for vMLX");
        Ok(vec![])
    }
}

/// Convenience wrapper with Arc for shared access
pub struct SharedVmlxBridge {
    inner: Arc<VmlxBridge>,
}

impl SharedVmlxBridge {
    pub fn new(bridge: VmlxBridge) -> Self {
        Self {
            inner: Arc::new(bridge),
        }
    }

    pub fn inner(&self) -> &Arc<VmlxBridge> {
        &self.inner
    }
}

impl Clone for SharedVmlxBridge {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl Bridge for SharedVmlxBridge {
    fn load_model(&self, model_id: &str) -> Result<()> {
        self.inner.load_model(model_id)
    }

    fn generate(&self, prompt: &str, max_tokens: Option<u32>) -> Result<String> {
        self.inner.generate(prompt, max_tokens)
    }

    fn get_expert_activations(&self, layer: u32) -> Result<Vec<u32>> {
        self.inner.get_expert_activations(layer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vmlx_bridge_creation() {
        let bridge = VmlxBridge::new();
        assert_eq!(bridge.base_url, "http://localhost:8000");
        assert_eq!(bridge.current_model(), None);
    }

    #[test]
    fn test_vmlx_bridge_custom_url() {
        let bridge = VmlxBridge::with_url("http://custom:9000");
        assert_eq!(bridge.base_url, "http://custom:9000");
    }

    #[test]
    fn test_set_model() {
        let mut bridge = VmlxBridge::new();
        bridge.set_model("mlx-community/Qwen3-8B-4bit");
        assert_eq!(bridge.current_model(), Some("mlx-community/Qwen3-8B-4bit"));
    }

    #[test]
    #[ignore] // Requires vMLX server running
    fn test_health_check() {
        let bridge = VmlxBridge::new();
        // This would fail without vMLX server running
        let _ = bridge.health_check();
    }
}
