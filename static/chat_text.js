(() => {
  const messagesEl = document.getElementById("chat-messages");
  const inputEl = document.getElementById("chat-input");
  const sendBtn = document.getElementById("chat-send");

  if (!messagesEl || !inputEl || !sendBtn) return;

  const history = [];

  function appendBubble(role, text) {
    const row = document.createElement("div");
    row.className = `msg msg-${role}`;

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;

    row.appendChild(bubble);
    messagesEl.appendChild(row);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  async function sendMessage() {
    const text = (inputEl.value || "").trim();
    if (!text) return;

    appendBubble("user", text);
    history.push({ role: "user", content: text });

    inputEl.value = "";
    sendBtn.disabled = true;

    const thinkingRow = document.createElement("div");
    thinkingRow.className = "msg msg-assistant";
    const thinkingBubble = document.createElement("div");
    thinkingBubble.className = "bubble bubble-thinking";
    thinkingBubble.textContent = "思考中...（随后生成数字人视频）";
    thinkingRow.appendChild(thinkingBubble);
    messagesEl.appendChild(thinkingRow);
    messagesEl.scrollTop = messagesEl.scrollHeight;

    try {
      const resp = await fetch("/api/chat/avatar", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          history,
          max_new_tokens: 256
        })
      });

      const data = await resp.json();
      thinkingRow.remove();

      if (!data.success) {
        appendBubble("assistant", `出错：${data.error || "unknown error"}`);
        return;
      }

      appendBubble("assistant", data.assistant_text);
      const videoEl = document.getElementById("chatVideo");
      if (videoEl && data.video_url) {
      // 防止缓存：加时间戳
        videoEl.src = data.video_url + "?t=" + Date.now();
        videoEl.load();
        videoEl.play().catch(() => {});
      }

      history.push({ role: "assistant", content: data.assistant_text });
    } catch (e) {
      thinkingRow.remove();
      appendBubble("assistant", `请求失败：${String(e)}`);
    } finally {
      sendBtn.disabled = false;
      inputEl.focus();
    }
  }

  sendBtn.addEventListener("click", sendMessage);
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  appendBubble("assistant", "文字聊天已就绪。");
})();
