# LLM Benchmark Parameters Explained

- **temperature**
  - Controls randomness and creativity of responses.
  - Lower (e.g., 0.3): More focused, deterministic.
  - Higher (e.g., 0.7): More creative and varied.

- **top_p**
  - Nucleus sampling: only most likely tokens up to top_p are considered.
  - Typical range: 0.85 to 0.95.

- **top_k**
  - Only the top_k most likely tokens are considered in each generation step.
  - Common values: 40 to 60.

- **min_p**
  - Minimum probability threshold; tokens below this probability are ignored.

- **repetition_penalty**
  - Penalizes repeated tokens in output.
  - Value >1.0 discourages repetitions.

- **frequency_penalty**
  - Reduces the likelihood of frequently used tokens.
  - Higher value encourages more diverse output.

- **presence_penalty**
  - Increases likelihood of new topics being introduced in the output.

- **context_window**
  - How many input tokens the model can use at a time.
  - Larger window allows referencing longer context.

- **max_tokens**
  - Maximum number of tokens (words/parts of words) in the model output.
  - Larger number allows longer outputs.