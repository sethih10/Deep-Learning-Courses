def apply_chat_template_llama3(messages: list[dict], add_bot: bool = False) -> str:
    prompt = "<|begin_of_text|>" if add_bot else ""
    for msg in messages:
        if msg["role"] not in ["system", "user", "assistant"]:
            raise ValueError(f"Role {msg['role']} not recognized")
        prompt += f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt
