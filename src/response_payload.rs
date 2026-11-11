use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsePayload {
    pub session_id: Option<String>,
    pub cmd: Option<String>,
    pub message: Option<String>,
    pub need_file: Option<String>,
    pub web: Option<WebObject>,
    pub agents: Option<AgentObject>,
    pub file: Option<FileObject>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WebObject {
    pub url: Option<String>,
    pub search: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTask {
    pub id: Option<u64>,
    pub task: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AgentObject {
    pub list: Option<Vec<AgentTask>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FileObject {
    pub summary: Option<String>,
    pub patch: Option<Vec<PatchObject>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchObject {
    pub path: Option<String>,
    pub mode: Option<String>,
    pub anchor: Option<String>,
    pub search: Option<String>,
    pub content: Option<String>,
    pub rename_to: Option<String>,
}

/// Extracts a balanced JSON string from the given text and parses it into a [`ResponsePayload`].
pub fn get_response_object(response: &str) -> Result<ResponsePayload> {
    if let Ok(payload) = serde_json::from_str::<ResponsePayload>(response) {
        return Ok(payload);
    }
    let json_str = extract_balanced_json(response)
        .ok_or_else(|| anyhow::anyhow!("no JSON object found in response"))?;
    let payload = serde_json::from_str::<ResponsePayload>(&json_str)?;
    Ok(payload)
}

/// Extracts a balanced JSON string from the given text.
pub fn extract_balanced_json(text: &str) -> Option<String> {
    let start = text.find('{')?;
    let bytes = text.as_bytes();

    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;

    for (i, &b) in bytes[start..].iter().enumerate() {
        let c = b as char;

        if in_string {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }

        match c {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(text[start..start + i + 1].to_string());
                }
            }
            _ => {}
        }
    }

    None
}

/// Extracts a [`ResponsePayload`] from the given text, optionally setting the session ID.
pub fn payload_object(response: &str) -> Option<ResponsePayload> {
    get_response_object(response).ok()
}

/// Extracts the web object from the given [`ResponsePayload`].
pub fn get_web_object(payload: &ResponsePayload) -> WebObject {
    let mut web_obj: WebObject = WebObject::default();
    if let Some(web) = &payload.web {
        web_obj.url = web.url.clone();
        web_obj.search = web.search.clone();
    }
    web_obj
}

/// Extracts the agent object from the given [`ResponsePayload`].
pub fn get_agent_obj(payload: &ResponsePayload) -> AgentObject {
    let mut agent_obj: AgentObject = AgentObject::default();
    if let Some(agents) = &payload.agents
        && let Some(list) = &agents.list {
            agent_obj.list = Some(list.clone());
        }
    agent_obj
}

/// Extracts the file object from the given [`ResponsePayload`].
pub fn get_file_obj(payload: &ResponsePayload) -> FileObject {
    let mut file_obj: FileObject = FileObject::default();
    if let Some(file) = &payload.file {
        if let Some(summary) = &file.summary {
            file_obj.summary = Some(summary.clone());
        }
        if let Some(patch) = &file.patch {
            file_obj.patch = Some(patch.clone());
        }
    }
    file_obj
}
