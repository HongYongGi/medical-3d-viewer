//! JSON progress output to stdout for Python subprocess integration.

use serde::Serialize;

#[derive(Serialize)]
struct ProgressMessage {
    progress: f64,
    message: String,
}

pub struct ProgressReporter {
    enabled: bool,
}

impl ProgressReporter {
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    pub fn report(&mut self, progress: f64, message: &str) {
        if !self.enabled {
            return;
        }
        let msg = ProgressMessage {
            progress: (progress * 100.0).round() / 100.0,
            message: message.to_string(),
        };
        if let Ok(json) = serde_json::to_string(&msg) {
            println!("{json}");
        }
    }
}
