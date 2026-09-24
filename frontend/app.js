// 前端单页应用：流式对话 + 内联引用 + 检索链路可视化 + 可拖拽侧栏 + 历史对话。零构建，原生 JS。
(() => {
  const $ = (s) => document.querySelector(s);
  const messagesEl = $("#messages");
  const inputEl = $("#input");
  const sendBtn = $("#send");
  const statusEl = $("#status");
  const modeBadge = $("#mode-badge");
  const tooltip = $("#tooltip");
  const appEl = $("#app");

  // ---------------- 本地存储 ----------------
  const LS = {
    chats: "pm_chats",
    cur: "pm_cur_session",
    sidebarW: "pm_sidebar_w",
    showChain: "pm_show_chain",
  };
  const read = (k, d) => {
    try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch { return d; }
  };
  const write = (k, v) => {
    try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* 容量超限时静默忽略 */ }
  };

  // ---------------- 侧栏宽度（可拖拽） ----------------
  const MIN_W = 220, MAX_W = 520;
  let sidebarW = read(LS.sidebarW, 284);
  function applySidebarWidth(w) {
    sidebarW = Math.max(MIN_W, Math.min(MAX_W, w));
    document.documentElement.style.setProperty("--sidebar-w", sidebarW + "px");
  }
  applySidebarWidth(sidebarW);

  const resizer = $("#resizer");
  let dragging = false;
  resizer.addEventListener("mousedown", (e) => {
    if (window.innerWidth <= 820) return;
    dragging = true;
    resizer.classList.add("dragging");
    document.body.classList.add("resizing");
    e.preventDefault();
  });
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    applySidebarWidth(e.clientX);
  });
  window.addEventListener("mouseup", () => {
    if (!dragging) return;
    dragging = false;
    resizer.classList.remove("dragging");
    document.body.classList.remove("resizing");
    write(LS.sidebarW, sidebarW);
  });

  // ---------------- 侧栏折叠 ----------------
  $("#sidebar-toggle").addEventListener("click", () => {
    if (window.innerWidth <= 820) appEl.classList.toggle("sidebar-open");
    else appEl.classList.toggle("sidebar-collapsed");
  });
  $("#sidebar-close").addEventListener("click", () => {
    if (window.innerWidth <= 820) appEl.classList.remove("sidebar-open");
    else appEl.classList.add("sidebar-collapsed");
  });

  // ---------------- 检索链路开关 ----------------
  let showChain = read(LS.showChain, true);
  const chainSwitch = $("#chain-switch");
  function applyChainSwitch() {
    chainSwitch.classList.toggle("on", showChain);
    chainSwitch.setAttribute("aria-checked", String(showChain));
  }
  applyChainSwitch();
  chainSwitch.addEventListener("click", () => {
    showChain = !showChain;
    applyChainSwitch();
    write(LS.showChain, showChain);
  });

  // ---------------- 历史对话 ----------------
  let chats = read(LS.chats, {});           // { id: { id, title, ts, msgs: [] } }
  let curId = read(LS.cur, null);
  if (!curId || !chats[curId]) {
    curId = "sid-" + Math.random().toString(36).slice(2, 10);
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }

  function fmtTime(ts) {
    const d = new Date(ts);
    const now = new Date();
    const sameDay = d.toDateString() === now.toDateString();
    if (sameDay) return String(d.getHours()).padStart(2, "0") + ":" + String(d.getMinutes()).padStart(2, "0");
    return (d.getMonth() + 1) + "/" + d.getDate();
  }

  function renderHistory() {
    const box = $("#history-list");
    box.innerHTML = "";
    const list = Object.values(chats).sort((a, b) => (b.ts || 0) - (a.ts || 0));
    if (!list.length) {
      const empty = document.createElement("div");
      empty.className = "hist-empty";
      empty.textContent = "暂无对话记录";
      box.appendChild(empty);
      return;
    }
    list.forEach((c) => {
      const item = document.createElement("div");
      item.className = "hist-item" + (c.id === curId ? " active" : "");
      item.innerHTML =
        `<span class="t">${escapeHtml(c.title || "新对话")}</span>` +
        `<span class="n">${fmtTime(c.ts || Date.now())}</span>` +
        `<button class="hist-del" title="删除">×</button>`;

      item.addEventListener("click", (e) => {
        if (e.target.classList.contains("hist-del")) return;
        if (c.id === curId) return;
        loadChat(c.id);
      });
      item.querySelector(".hist-del").addEventListener("click", (e) => {
        e.stopPropagation();
        deleteChat(c.id);
      });
      box.appendChild(item);
    });
  }

  function ensureChat() {
    if (!chats[curId]) {
      chats[curId] = { id: curId, title: "新对话", ts: Date.now(), msgs: [] };
    }
    return chats[curId];
  }

  // 把当前 DOM 里的对话快照下来（保留气泡 / 链路面板 / 引用 / 复制按钮）
  function snapshot() {
    const msgs = [];
    messagesEl.querySelectorAll(".msg").forEach((wrap) => {
      const col = wrap.querySelector(".col");
      if (!col) return;
      const isUser = wrap.classList.contains("user");
      const bubble = col.querySelector(".bubble");
      if (isUser) {
        msgs.push({ role: "user", text: bubble ? bubble.textContent : "" });
      } else {
        msgs.push({ role: "bot", html: col.innerHTML });
      }
    });
    return msgs;
  }

  function persist() {
    const c = ensureChat();
    c.msgs = snapshot();
    c.ts = Date.now();
    if (!c.title || c.title === "新对话") {
      const first = c.msgs.find((m) => m.role === "user" && m.text && m.text.trim());
      if (first) c.title = first.text.trim().slice(0, 20);
    }
    write(LS.chats, chats);
    write(LS.cur, curId);
    renderHistory();
  }

  function renderStoredMessages(msgs) {
    messagesEl.innerHTML = "";
    msgs.forEach((m) => {
      const wrap = document.createElement("div");
      wrap.className = "msg " + (m.role === "user" ? "user" : "bot");
      const who = document.createElement("div");
      who.className = "who";
      who.textContent = m.role === "user" ? "面" : "AI";
      const col = document.createElement("div");
      col.className = "col";
      if (m.role === "user") {
        const b = document.createElement("div");
        b.className = "bubble";
        b.textContent = m.text || "";
        col.appendChild(b);
      } else {
        col.innerHTML = m.html || "";
        // 恢复链路面板的折叠交互
        col.querySelectorAll(".chain-head").forEach((head) => {
          head.addEventListener("click", () => head.parentElement.classList.toggle("collapsed"));
        });
        bindCopyButtons(col);
      }
      wrap.appendChild(who);
      wrap.appendChild(col);
      messagesEl.appendChild(wrap);
    });
    scrollBottom();
  }

  function loadChat(id) {
    persist();                       // 先存好当前会话
    curId = id;
    const c = ensureChat();
    renderStoredMessages(c.msgs || []);
    $("#suggests").style.display = (c.msgs && c.msgs.length) ? "none" : "flex";
    write(LS.cur, curId);
    renderHistory();
  }

  function newChat() {
    persist();
    curId = "sid-" + Math.random().toString(36).slice(2, 10);
    chats[curId] = { id: curId, title: "新对话", ts: Date.now(), msgs: [] };
    messagesEl.innerHTML = "";
    $("#suggests").style.display = "flex";
    write(LS.cur, curId);
    renderHistory();
    inputEl.focus();
  }

  function deleteChat(id) {
    delete chats[id];
    if (curId === id) {
      curId = "sid-" + Math.random().toString(36).slice(2, 10);
      chats[curId] = { id: curId, title: "新对话", ts: Date.now(), msgs: [] };
      messagesEl.innerHTML = "";
      $("#suggests").style.display = "flex";
      write(LS.cur, curId);
    }
    write(LS.chats, chats);
    renderHistory();
  }

  // ---------------- 复制按钮 ----------------
  function bindCopyButtons(scope) {
    scope.querySelectorAll(".copy-btn").forEach((btn) => {
      if (btn.dataset.bound) return;
      btn.dataset.bound = "1";
      btn.addEventListener("click", () => {
        const plain = btn.dataset.answer || "";
        navigator.clipboard.writeText(plain).then(() => {
          btn.textContent = "已复制";
          btn.classList.add("done");
          setTimeout(() => { btn.textContent = "复制回答"; btn.classList.remove("done"); }, 1500);
        });
      });
    });
  }

  // ---------------- 后端状态 ----------------
  fetch("/api/health")
    .then((r) => r.json())
    .then((d) => {
      if (d.llm === "configured") {
        modeBadge.textContent = "真实模型 · " + (d.embedding || "已接入");
        modeBadge.style.background = "var(--accent-soft)";
        modeBadge.style.color = "var(--accent)";
      } else {
        modeBadge.textContent = "演示模式（规则生成）";
      }
    })
    .catch(() => (modeBadge.textContent = "后端未连接"));

  function scrollBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  // 轻量渲染（流式过程中使用，避免半截 Markdown 被错误解析）
  function renderAnswer(text) {
    const esc = escapeHtml(text);
    return esc.replace(/\[(\d+)\]/g, '<sup class="cite-ref" data-id="$1">$1</sup>');
  }

  // 最终渲染：把 LLM 的 Markdown 转为 HTML（标题/列表/粗体/代码/引用角标）
  function renderMarkdown(src) {
    const lines = escapeHtml(src).split("\n");
    const out = [];
    let inList = false;
    let inPre = false;
    let preBuf = [];
    const closeList = () => { if (inList) { out.push("</ul>"); inList = false; } };
    for (const raw of lines) {
      if (/^```/.test(raw)) {
        if (inPre) { out.push("<pre class='md-pre'><code>" + preBuf.join("\n") + "</code></pre>"); preBuf = []; inPre = false; }
        else { closeList(); inPre = true; }
        continue;
      }
      if (inPre) { preBuf.push(raw); continue; }
      let m;
      if ((m = raw.match(/^(#{1,4})\s+(.*)$/))) { closeList(); out.push("<h" + m[1].length + ">" + inline(m[2]) + "</h" + m[1].length + ">"); continue; }
      if ((m = raw.match(/^\s*[-*]\s+(.*)$/))) { if (!inList) { out.push("<ul>"); inList = true; } out.push("<li>" + inline(m[1]) + "</li>"); continue; }
      if (raw.trim() === "") { closeList(); continue; }
      closeList();
      out.push("<p>" + inline(raw) + "</p>");
    }
    closeList();
    if (inPre) out.push("<pre class='md-pre'><code>" + preBuf.join("\n") + "</code></pre>");
    return out.join("");
  }

  // 行内 Markdown：代码、`*斜体*`、**粗体**、[n] 引用角标
  function inline(t) {
    return t
      .replace(/`([^`]+)`/g, "<code class='md-code'>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/\*([^*]+)\*/g, "<em>$1</em>")
      .replace(/\[(\d+)\]/g, '<sup class="cite-ref" data-id="$1">$1</sup>');
  }

  function addMessage(role, text = "") {
    const wrap = document.createElement("div");
    wrap.className = "msg " + role;
    const who = document.createElement("div");
    who.className = "who";
    who.textContent = role === "user" ? "面" : "AI";
    const col = document.createElement("div");
    col.className = "col";
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;
    col.appendChild(bubble);
    wrap.appendChild(who);
    wrap.appendChild(col);
    messagesEl.appendChild(wrap);
    scrollBottom();
    return { wrap, col, bubble };
  }

  // ---------- 检索链路面板（随 step 事件增量构建） ----------
  function ensureChain(col) {
    let chain = col.querySelector(".chain");
    if (!chain) {
      chain = document.createElement("div");
      // 默认折叠：只留一行「检索链路 + 状态标签」，点开看细节
      chain.className = "chain collapsed";
      const head = document.createElement("div");
      head.className = "chain-head";
      head.innerHTML = '<span class="ico">🧠</span><span>检索链路</span><span class="chev">▾</span><span class="tag">RAG</span>';
      const body = document.createElement("div");
      body.className = "chain-body";
      chain.appendChild(head);
      chain.appendChild(body);
      head.addEventListener("click", () => chain.classList.toggle("collapsed"));
      col.appendChild(chain);
    }
    return chain;
  }

  function addChainStep(col, html, kind) {
    if (!showChain) return;                 // 开关关闭时不渲染链路
    const chain = ensureChain(col);
    const body = chain.querySelector(".chain-body");
    const row = document.createElement("div");
    row.className = "chain-step" + (kind ? " " + kind : "");
    row.innerHTML = html;
    body.appendChild(row);
  }

  function setChainTag(col, text, kind) {
    if (!showChain) return;
    const chain = ensureChain(col);
    const tag = chain.querySelector(".tag");
    if (tag) {
      tag.textContent = text;
      tag.classList.toggle("warn", kind === "warn");
    }
  }

  function renderSources(col, items) {
    if (!items || !items.length) return;
    const box = document.createElement("div");
    box.className = "cites";
    items.forEach((c) => {
      const tag = document.createElement("span");
      tag.className = "cite";
      tag.dataset.id = c.id || "";
      const label = c.label || c.heading || c.title || c.source || "来源";
      tag.textContent = `[${c.id || ""}] ${label}`;
      tag.title = (c.text || "").slice(0, 260);
      box.appendChild(tag);
    });
    col.appendChild(box);

    // 把引用数据随消息一起持久化，刷新后悬浮预览依然可用
    const dataEl = document.createElement("script");
    dataEl.type = "application/json";
    dataEl.className = "cites-data";
    dataEl.textContent = JSON.stringify(items).replace(/</g, "\\u003c");
    col.appendChild(dataEl);
  }

  // ---------- 引用悬浮预览 ----------
  function citesOf(node) {
    const msg = node.closest(".msg");
    const dataEl = msg && msg.querySelector(".cites-data");
    if (dataEl) {
      try { return JSON.parse(dataEl.textContent || "[]"); } catch { /* fallthrough */ }
    }
    return citationsData;
  }

  messagesEl.addEventListener("mouseover", (e) => {
    const t = e.target.closest(".cite-ref");
    if (!t) return;
    const item = citesOf(t)[Number(t.dataset.id) - 1];
    if (!item) return;
    const label = item.label || item.heading || item.title || item.source || "来源";
    tooltip.innerHTML =
      `<div class="tt-title">[${item.id || ""}] ${escapeHtml(label)}</div>` +
      `<div class="tt-text">${escapeHtml((item.text || "").slice(0, 220))}</div>`;
    tooltip.style.display = "block";
    const r = t.getBoundingClientRect();
    const tw = tooltip.offsetWidth, th = tooltip.offsetHeight;
    let x = r.left, y = r.bottom + 8;
    if (x + tw > window.innerWidth - 12) x = window.innerWidth - tw - 12;
    if (y + th > window.innerHeight - 12) y = r.top - th - 8;
    tooltip.style.left = x + "px";
    tooltip.style.top = y + "px";
  });
  messagesEl.addEventListener("mouseout", (e) => {
    if (e.target.closest(".cite-ref")) tooltip.style.display = "none";
  });

  let citationsData = [];
  let busy = false;

  async function send(query) {
    if (busy || !query.trim()) return;
    busy = true;
    sendBtn.disabled = true;
    $("#suggests").style.display = "none";
    setStatus("生成中…");

    addMessage("user", query);
    const { col, bubble } = addMessage("bot", "");
    bubble.classList.add("cursor");

    let acc = "";
    let reliable = true;
    citationsData = [];

    try {
      const resp = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: curId, query }),
      });
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const parts = buf.split("\n\n");
        buf = parts.pop();
        for (const p of parts) {
          if (!p.startsWith("data:")) continue;
          let evt;
          try { evt = JSON.parse(p.slice(5).trim()); } catch { continue; }
          if (evt.type === "step") {
            if (evt.stage === "rewrite") {
              addChainStep(col, `<span class="ico">🔄</span><span class="body">查询改写：<b>${escapeHtml(evt.detail || "")}</b></span>`);
            } else if (evt.stage === "retrieve") {
              const n = evt.count || 0;
              let chips = "";
              if (evt.chunks && evt.chunks.length) {
                chips = '<div class="chips">' + evt.chunks.map((c) =>
                  `<span class="chip">${escapeHtml(c.label || c.heading || c.title || "来源")}</span>`
                ).join("") + "</div>";
              }
              addChainStep(col, `<span class="ico">🔍</span><span class="body">混合检索：召回 <b>${n}</b> 个片段${chips}</span>`);
              setChainTag(col, "召回 " + n);
            } else if (evt.stage === "judge") {
              reliable = !!evt.reliable;
              if (reliable) {
                addChainStep(col, `<span class="ico">✅</span><span class="body">判定：有可靠依据，进入生成</span>`, "ok");
                setChainTag(col, "判定通过");
              } else {
                addChainStep(col, `<span class="ico">⚠️</span><span class="body">判定：知识库无可靠依据，触发拒答</span>`, "warn");
                setChainTag(col, "已拒答", "warn");
              }
            }
          } else if (evt.type === "error") {
            // 链路某环节出错：写进链路面板，并在气泡下给出可见告警
            const msg = `${evt.stage || "链路"} 环节出错：${evt.message || "未知错误"}`;
            addChainStep(col, `<span class="ico">⚠️</span><span class="body">${escapeHtml(msg)}</span>`, "warn");
            setChainTag(col, "异常", "warn");
            const warn = document.createElement("div");
            warn.className = "refuse-badge";
            warn.textContent = "⚠️ " + msg;
            col.insertBefore(warn, bubble.nextSibling);
          } else if (evt.type === "token") {
            acc += evt.delta;
            bubble.innerHTML = renderAnswer(acc);
            scrollBottom();
          } else if (evt.type === "citations") {
            citationsData = evt.items || [];
          } else if (evt.type === "answer_replace") {
            // 后端剔除了越界的引用角标，用校验后的全文替换已渲染内容
            acc = evt.answer || acc;
            bubble.innerHTML = renderAnswer(acc);
            const dropped = evt.dropped || [];
            if (dropped.length) {
              addChainStep(
                col,
                `<span class="ico">🧹</span><span class="body">引用校验：剔除 ${dropped.length} 处越界角标（<b>${escapeHtml(dropped.join(", "))}</b>）</span>`,
                "warn"
              );
            }
            scrollBottom();
          } else if (evt.type === "done") {
            // 流式结束：用 Markdown 渲染最终全文（标题/列表/粗体/代码）
            bubble.innerHTML = renderMarkdown(acc);
            scrollBottom();
          }
        }
      }
    } catch (e) {
      bubble.innerHTML = "请求失败：" + escapeHtml(e.message);
    } finally {
      bubble.classList.remove("cursor");
      if (!reliable) {
        bubble.classList.add("refuse");
        const badge = document.createElement("div");
        badge.className = "refuse-badge";
        badge.textContent = "⚠️ 知识库无可靠依据，已拒答";
        col.insertBefore(badge, bubble.nextSibling);
      }
      renderSources(col, citationsData);

      // 复制按钮
      const toolbar = document.createElement("div");
      toolbar.className = "toolbar";
      const copyBtn = document.createElement("button");
      copyBtn.className = "copy-btn";
      copyBtn.textContent = "复制回答";
      copyBtn.dataset.answer = acc.replace(/\[\d+\]/g, "").trim();
      toolbar.appendChild(copyBtn);
      const chainEl = col.querySelector(".chain");
      if (chainEl) col.insertBefore(toolbar, chainEl);
      else col.appendChild(toolbar);
      bindCopyButtons(toolbar);

      persist();

      busy = false;
      sendBtn.disabled = false;
      setStatus("就绪");
      inputEl.focus();
    }
  }

  function setStatus(t) { statusEl.textContent = t; }

  // ---------------- 事件绑定 ----------------
  sendBtn.addEventListener("click", () => {
    const v = inputEl.value;
    inputEl.value = "";
    autoGrow();
    send(v);
  });
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendBtn.click(); }
  });
  inputEl.addEventListener("input", autoGrow);
  function autoGrow() {
    inputEl.style.height = "auto";
    inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + "px";
  }
  $("#suggests").addEventListener("click", (e) => {
    if (e.target.classList.contains("suggest")) send(e.target.textContent);
  });
  $("#new-chat").addEventListener("click", newChat);

  // ---------------- 更换资料：填写 / 上传 -> 自动切片入库 ----------------
  const LS_BOOT_TOKEN = "pm_boot_token";
  const bootModal = $("#boot-modal");
  const bootName = $("#boot-name");
  const bootProfile = $("#boot-profile");
  const bootProjects = $("#boot-projects");
  const bootQa = $("#boot-qa");
  const bootAnchors = $("#boot-anchors");
  const bootKeywords = $("#boot-keywords");
  const bootToken = $("#boot-token");
  const bootResult = $("#boot-result");
  const bootMsg = $("#boot-msg");
  const bootSubmit = $("#boot-submit");

  // 回显内容含用户可控的文件名，必须转义后再进 innerHTML
  const esc = (s) =>
    String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  bootToken.value = read(LS_BOOT_TOKEN, "");

  function openBoot() {
    bootModal.hidden = false;
    bootResult.hidden = true;
    setBootMsg("");
    bootName.focus();
  }
  function closeBoot() { bootModal.hidden = true; }

  function setBootMsg(t, isErr) {
    bootMsg.textContent = t || "";
    bootMsg.classList.toggle("err", !!isErr);
  }
  function showBootResult(html, isErr) {
    bootResult.hidden = false;
    bootResult.classList.toggle("err", !!isErr);
    bootResult.innerHTML = html;
  }

  $("#open-bootstrap").addEventListener("click", openBoot);
  $("#boot-cancel").addEventListener("click", closeBoot);
  bootModal.querySelectorAll("[data-close]").forEach((el) => el.addEventListener("click", closeBoot));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !bootModal.hidden) closeBoot();
  });

  // 每个项目单独成一个文档，切片边界更干净，不会把两个项目揉进同一段
  $("#boot-add-project").addEventListener("click", () => {
    const ta = document.createElement("textarea");
    ta.className = "boot-project";
    ta.rows = 4;
    ta.placeholder = "# 项目名称\n背景 / 我的职责 / 技术要点 / 结果";
    bootProjects.appendChild(ta);
    ta.focus();
  });

  bootSubmit.addEventListener("click", async () => {
    const name = bootName.value.trim();
    if (!name) { setBootMsg("请先填写姓名", true); bootName.focus(); return; }

    const profile = bootProfile.value.trim();
    const qa = bootQa.value.trim();
    const projects = [...bootProjects.querySelectorAll(".boot-project")]
      .map((t) => t.value.trim())
      .filter(Boolean);

    const fd = new FormData();
    fd.append("name", name);
    if (profile) fd.append("profile_text", profile);
    if (qa) fd.append("qa_text", qa);
    // 同一个字段 append 多次，后端用 list[str] 接收
    projects.forEach((t) => fd.append("project_text", t));

    const anchors = bootAnchors.value.trim();
    const keywords = bootKeywords.value.trim();
    if (anchors) fd.append("anchors", anchors);
    if (keywords) fd.append("keywords", keywords);

    const addFiles = (sel, field) => {
      for (const f of $(sel).files || []) fd.append(field, f, f.name);
    };
    addFiles("#boot-profile-files", "profile_files");
    addFiles("#boot-project-files", "project_files");
    addFiles("#boot-qa-files", "qa_files");

    const hasFile = ["profile_files", "project_files", "qa_files"].some((k) => fd.getAll(k).length);
    if (!profile && !qa && !projects.length && !hasFile) {
      setBootMsg("请至少填写一段资料或上传一个文件", true);
      return;
    }

    const token = bootToken.value.trim();
    write(LS_BOOT_TOKEN, token);

    bootSubmit.disabled = true;
    const oldLabel = bootSubmit.textContent;
    bootSubmit.textContent = "入库中…";
    bootResult.hidden = true;
    setBootMsg("正在切片并写入向量库，首次会下载模型，可能需要一两分钟…");

    try {
      const resp = await fetch("/api/bootstrap", {
        method: "POST",
        headers: token ? { "X-Reindex-Token": token } : {},
        body: fd,  // 不要手写 Content-Type，浏览器需要自动补 multipart boundary
      });
      const data = await resp.json().catch(() => ({}));

      if (!resp.ok) {
        // 401 单独说清楚，否则用户只看到干巴巴的 unauthorized
        const msg = resp.status === 401
          ? "鉴权失败：请填写服务端配置的 REINDEX_TOKEN"
          : data.error || "提交失败";
        setBootMsg(msg, true);
        showBootResult(esc(msg), true);
        return;
      }

      const files = (data.written || []).map(esc).join("\n");
      showBootResult(
        `已保存 <b>${(data.written || []).length}</b> 个文件，切片 <b>${data.chunks ?? 0}</b> 段，` +
        `库内共 <b>${data.stored ?? 0}</b> 段${data.elapsed ? `（耗时 ${data.elapsed}s）` : ""}` +
        (files ? `\n\n${files}` : "")
      );
      setBootMsg("完成，可以开始提问了");

      // 换人了，旧会话的多轮上下文属于旧人，直接开新对话
      const avatarEl = $(".avatar");
      if (avatarEl) avatarEl.textContent = name.slice(0, 1);
      newChat();
    } catch (e) {
      setBootMsg("请求失败：" + e.message, true);
    } finally {
      bootSubmit.disabled = false;
      bootSubmit.textContent = oldLabel;
    }
  });

  // ---------------- 启动：恢复上次的会话 ----------------
  ensureChat();
  const saved = chats[curId];
  if (saved && saved.msgs && saved.msgs.length) {
    renderStoredMessages(saved.msgs);
    $("#suggests").style.display = "none";
  }
  renderHistory();
  inputEl.focus();
})();
