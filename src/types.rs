/// Parameters controlling a single generation pass.
pub struct GenerateParams {
    pub max_tokens:     usize,
    pub temperature:    f64,
    pub top_p:          Option<f64>,
    pub repeat_penalty: f32,
    pub max_context:    usize,
    pub grammar:        Option<String>,
}
