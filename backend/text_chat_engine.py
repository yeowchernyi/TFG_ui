from backend.qwen_engine import qwen

def reply_text(user_text: str, system_prompt: str = None, max_new_tokens: int = 256):
    user_text = (user_text or "").strip()
    if not user_text:
        return {"assistant_text": "请先输入内容。"}

    assistant_text = qwen.chat(
        user_text=user_text,
        system_prompt=system_prompt or "你是一个逻辑严密的助手，请确保计算准确后再简洁回答。",
        max_new_tokens=max_new_tokens,
    )
    return {"assistant_text": assistant_text}
