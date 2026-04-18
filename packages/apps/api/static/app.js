const state = {
  file: null,
  durationSec: null,
  lastScore: null,
  sessionId: "demo-session",
  userId: "demo-user",
};

const els = {
  videoInput: document.getElementById("videoInput"),
  videoPreview: document.getElementById("videoPreview"),
  videoMeta: document.getElementById("videoMeta"),
  scoreButton: document.getElementById("scoreButton"),
  scoreEmpty: document.getElementById("scoreEmpty"),
  scoreResult: document.getElementById("scoreResult"),
  scoreValue: document.getElementById("scoreValue"),
  gradeValue: document.getElementById("gradeValue"),
  scoreSummary: document.getElementById("scoreSummary"),
  strengthList: document.getElementById("strengthList"),
  improvementList: document.getElementById("improvementList"),
  chatLog: document.getElementById("chatLog"),
  chatInput: document.getElementById("chatInput"),
  chatButton: document.getElementById("chatButton"),
  statusBadge: document.getElementById("statusBadge"),
  kbTitle: document.getElementById("kbTitle"),
  kbFilename: document.getElementById("kbFilename"),
  kbContent: document.getElementById("kbContent"),
  kbCreateButton: document.getElementById("kbCreateButton"),
  kbRefreshButton: document.getElementById("kbRefreshButton"),
  reindexButton: document.getElementById("reindexButton"),
  kbSummary: document.getElementById("kbSummary"),
  kbList: document.getElementById("kbList"),
};

function setStatus(text) {
  els.statusBadge.textContent = text;
}

function appendChat(role, text) {
  const item = document.createElement("div");
  item.className = `chat-item ${role}`;
  item.textContent = text;
  els.chatLog.appendChild(item);
  els.chatLog.scrollTop = els.chatLog.scrollHeight;
}

function renderList(target, items) {
  target.innerHTML = "";
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    target.appendChild(li);
  });
}

function bytesToMb(size) {
  return (size / (1024 * 1024)).toFixed(2);
}

function setButtonBusy(button, busy, busyText, idleText) {
  button.disabled = busy;
  button.textContent = busy ? busyText : idleText;
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || "请求失败");
  }
  return data;
}

function renderKbList(items) {
  els.kbList.innerHTML = "";
  els.kbSummary.textContent = `当前共有 ${items.length} 份文档`;

  if (!items.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "知识库里还没有文档，可以先在左侧添加一份。";
    els.kbList.appendChild(empty);
    return;
  }

  items.forEach((item) => {
    const card = document.createElement("div");
    card.className = "kb-item";

    const titleRow = document.createElement("div");
    titleRow.className = "kb-item-head";

    const titleBox = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = item.title;
    const meta = document.createElement("div");
    meta.className = "kb-item-meta";
    meta.textContent = `${item.source_path} | ${item.chunk_count} chunks`;
    titleBox.appendChild(title);
    titleBox.appendChild(meta);

    const deleteButton = document.createElement("button");
    deleteButton.className = "ghost danger";
    deleteButton.textContent = "删除";
    deleteButton.addEventListener("click", async () => {
      const confirmed = window.confirm(`确认删除文档《${item.title}》吗？`);
      if (!confirmed) {
        return;
      }
      try {
        setStatus(`正在删除 ${item.title}`);
        deleteButton.disabled = true;
        await requestJson(`/v1/kb/documents/${item.doc_id}`, { method: "DELETE" });
        await loadKbDocuments();
        appendChat("bot", `知识库文档《${item.title}》已删除。`);
        setStatus("文档已删除");
      } catch (error) {
        appendChat("bot", `删除文档失败：${error.message}`);
        setStatus("删除文档失败");
      } finally {
        deleteButton.disabled = false;
      }
    });

    titleRow.appendChild(titleBox);
    titleRow.appendChild(deleteButton);
    card.appendChild(titleRow);
    els.kbList.appendChild(card);
  });
}

async function loadKbDocuments() {
  try {
    els.kbSummary.textContent = "正在刷新知识库列表";
    const items = await requestJson("/v1/kb/documents");
    renderKbList(items);
  } catch (error) {
    els.kbSummary.textContent = `加载失败：${error.message}`;
  }
}

els.videoInput.addEventListener("change", () => {
  const [file] = els.videoInput.files || [];
  state.file = file || null;
  state.durationSec = null;
  state.lastScore = null;

  if (!file) {
    els.scoreButton.disabled = true;
    els.videoMeta.textContent = "尚未选择视频";
    els.videoPreview.removeAttribute("src");
    setStatus("等待上传视频");
    return;
  }

  const url = URL.createObjectURL(file);
  els.videoPreview.src = url;
  els.videoMeta.textContent = `文件：${file.name} | 大小：${bytesToMb(file.size)} MB | 正在读取时长`;
  els.scoreButton.disabled = false;
  setStatus("视频已选择，可开始评分");
});

els.videoPreview.addEventListener("loadedmetadata", () => {
  if (!state.file) {
    return;
  }
  state.durationSec = Number.isFinite(els.videoPreview.duration)
    ? Number(els.videoPreview.duration.toFixed(2))
    : null;
  els.videoMeta.textContent = `文件：${state.file.name} | 大小：${bytesToMb(state.file.size)} MB | 时长：${state.durationSec ?? "未知"} 秒`;
});

els.scoreButton.addEventListener("click", async () => {
  if (!state.file) {
    return;
  }

  setStatus("正在上传并评分");
  els.scoreButton.disabled = true;

  try {
    const buffer = await state.file.arrayBuffer();
    const headers = {
      "Content-Type": state.file.type || "application/octet-stream",
      "X-Filename": encodeURIComponent(state.file.name),
    };
    if (state.durationSec != null) {
      headers["X-Video-Duration"] = String(state.durationSec);
    }

    const data = await requestJson("/v1/demo/video_score", {
      method: "POST",
      headers,
      body: buffer,
    });

    state.lastScore = data;
    els.scoreEmpty.classList.add("hidden");
    els.scoreResult.classList.remove("hidden");
    els.scoreValue.textContent = data.score;
    els.gradeValue.textContent = data.grade_label;
    els.scoreSummary.textContent = data.summary;
    renderList(els.strengthList, data.strengths);
    renderList(els.improvementList, data.improvements);
    appendChat("bot", `视频《${decodeURIComponent(data.filename)}》评分完成，当前得分 ${data.score}/100，等级 ${data.grade_label}。你可以继续问我如何改进。`);
    setStatus("评分完成，可继续问答");
  } catch (error) {
    appendChat("bot", `评分失败：${error.message}`);
    setStatus("评分失败，请重试");
  } finally {
    els.scoreButton.disabled = false;
  }
});

els.chatButton.addEventListener("click", sendChat);
els.chatInput.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    sendChat();
  }
});

els.kbRefreshButton.addEventListener("click", loadKbDocuments);

els.reindexButton.addEventListener("click", async () => {
  try {
    setButtonBusy(els.reindexButton, true, "重建中...", "重建索引");
    setStatus("正在重建知识库索引");
    const result = await requestJson("/v1/kb/documents/reindex", { method: "POST" });
    await loadKbDocuments();
    appendChat("bot", `知识库索引已重建，共处理 ${result.documents} 份文档、${result.chunks} 个 chunks。`);
    setStatus("知识库索引已重建");
  } catch (error) {
    appendChat("bot", `重建索引失败：${error.message}`);
    setStatus("重建索引失败");
  } finally {
    setButtonBusy(els.reindexButton, false, "重建中...", "重建索引");
  }
});

els.kbCreateButton.addEventListener("click", async () => {
  const title = els.kbTitle.value.trim();
  const filename = els.kbFilename.value.trim();
  const content = els.kbContent.value.trim();

  if (!title || !content) {
    appendChat("bot", "新增知识文档前，请至少填写标题和内容。");
    return;
  }

  try {
    setButtonBusy(els.kbCreateButton, true, "保存中...", "保存到知识库");
    setStatus(`正在保存文档 ${title}`);
    const item = await requestJson("/v1/kb/documents", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title,
        content,
        filename: filename || null,
      }),
    });
    els.kbTitle.value = "";
    els.kbFilename.value = "";
    els.kbContent.value = "";
    await loadKbDocuments();
    appendChat("bot", `知识库文档《${item.title}》已保存，当前切成 ${item.chunk_count} 个 chunks。`);
    setStatus("文档已保存到知识库");
  } catch (error) {
    appendChat("bot", `保存知识文档失败：${error.message}`);
    setStatus("保存知识文档失败");
  } finally {
    setButtonBusy(els.kbCreateButton, false, "保存中...", "保存到知识库");
  }
});

async function sendChat() {
  const content = els.chatInput.value.trim();
  if (!content) {
    return;
  }

  appendChat("user", content);
  els.chatInput.value = "";
  setStatus("正在生成回答");

  const score = state.lastScore?.score ?? null;
  const prefix = state.lastScore
    ? `当前视频评分 ${state.lastScore.score}/100，等级 ${state.lastScore.grade_label}。${state.lastScore.summary}\n\n`
    : "当前还没有视频评分结果，请先上传视频。如果用户直接提问，也请先给通用训练建议。\n\n";

  try {
    const data = await requestJson("/v1/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        session_id: state.sessionId,
        user_id: state.userId,
        content: `${prefix}${content}`,
        mock_score_value: score,
      }),
    });
    appendChat("bot", data.reply);
    setStatus("回答已生成");
  } catch (error) {
    appendChat("bot", `问答失败：${error.message}`);
    setStatus("问答失败，请检查服务状态");
  }
}

loadKbDocuments();
