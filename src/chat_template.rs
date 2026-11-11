use regex::Regex;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplate {
    Llama3,
    Chatml,
    DeepseekR1,
    Gemma3,
    Phi3,
}

pub fn detect_template(model_id: &str) -> ChatTemplate {
    let id = model_id.to_lowercase();
    if id.starts_with("deepseek") {
        ChatTemplate::DeepseekR1
    } else if id.starts_with("llama") {
        ChatTemplate::Llama3
    } else if id.starts_with("smollm") || id.starts_with("qwen") {
        ChatTemplate::Chatml
    } else if id.starts_with("gemma") {
        ChatTemplate::Gemma3
    } else if id.starts_with("phi") {
        ChatTemplate::Phi3
    } else {
        ChatTemplate::Llama3
    }
}

#[derive(Debug, Clone)]
pub struct ChatTurn {
    pub role: String, // "user" or "assistant"
    pub content: String,
}

pub fn apply_template(
    template: &ChatTemplate,
    system: &str,
    history: &[ChatTurn],
    user: &str,
) -> String {
    match template {
        ChatTemplate::Llama3 => {
            let mut prompt = format!(
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>",
                system
            );
            for turn in history {
                prompt.push_str(&format!(
                    "<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>",
                    role = turn.role,
                    content = turn.content
                ));
            }
            prompt.push_str(&format!(
                "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                user
            ));
            prompt
        }

        ChatTemplate::Chatml => {
            let mut prompt = format!("<|im_start|>system\n{}<|im_end|>\n", system);
            for turn in history {
                prompt.push_str(&format!(
                    "<|im_start|>{role}\n{content}<|im_end|>\n",
                    role = turn.role,
                    content = turn.content
                ));
            }
            prompt.push_str(&format!(
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                user
            ));
            prompt
        }

        ChatTemplate::DeepseekR1 => {
            // Note: DeepSeek R1 GGUF templates can be picky, using full-width characters in prompts
            let mut prompt = format!("<｜begin▁of▁sentence｜>{}\n", system);
            for turn in history {
                if turn.role == "user" {
                    prompt.push_str(&format!("<｜User｜>{}\n", turn.content));
                } else {
                    prompt.push_str(&format!("<｜Assistant｜>{}<｜end▁of▁sentence｜>\n", turn.content));
                }
            }
            prompt.push_str(&format!("<｜User｜>{}\n<｜Assistant｜>", user));
            prompt
        }

        ChatTemplate::Gemma3 => {
            let mut prompt = format!("<start_of_turn>user\n{}\n\n", system);
            if !history.is_empty() {
                prompt.push_str("<end_of_turn>\n");
                for turn in history {
                    let role = if turn.role == "user" { "user" } else { "model" };
                    prompt.push_str(&format!(
                        "<start_of_turn>{role}\n{content}<end_of_turn>\n",
                        role = role,
                        content = turn.content
                    ));
                }
                prompt.push_str(&format!(
                    "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n",
                    user
                ));
            } else {
                prompt.push_str(&format!("{}<end_of_turn>\n<start_of_turn>model\n", user));
            }
            prompt
        }

        ChatTemplate::Phi3 => {
            let mut prompt = format!("<|system|>\n{}<|end|>\n", system);
            for turn in history {
                if turn.role == "user" {
                    prompt.push_str(&format!("<|user|>\n{}<|end|>\n", turn.content));
                } else {
                    prompt.push_str(&format!("<|assistant|>\n{}<|end|>\n", turn.content));
                }
            }
            prompt.push_str(&format!("<|user|>\n{}<|end|>\n<|assistant|>\n", user));
            prompt
        }
    }
}

pub fn strip_thinking_tokens(text: &str) -> String {
    let re = Regex::new(r"(?s)<think>.*?</think>\s*").expect("valid regex");
    re.replace_all(text, "").trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_template() {
        assert_eq!(detect_template("llama-3-8b"), ChatTemplate::Llama3);
        assert_eq!(detect_template("qwen-2.5"), ChatTemplate::Chatml);
        assert_eq!(detect_template("smollm2-1.7b-q8"), ChatTemplate::Chatml);
        assert_eq!(detect_template("deepseek-r1-7b"), ChatTemplate::DeepseekR1);
    }

    #[test]
    fn test_strip_thinking_tokens() {
        let input = "hello <think>let me think\nabout this</think>world";
        assert_eq!(strip_thinking_tokens(input), "hello world");
    }
}
