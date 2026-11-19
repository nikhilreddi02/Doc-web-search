const state = {
  docs: [],
  currentId: null,
  chats: new Map(), // doc_id -> array of rendered entries
};

const $ = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const res = await fetch(path, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || res.statusText);
  return body;
}

// ---------- Document list ----------

async function loadDocs() {
  state.docs = await api("/api/documents");
  renderDocList();
}

function renderDocList() {
  const ul = $("docList");
  ul.innerHTML = "";
  for (const d of state.docs) {
    const li = document.createElement("li");
    if (d.id === state.currentId) li.classList.add("active");

    const name = document.createElement("span");
    name.textContent = `${d.name} · ${d.chunk_count} chunks`;
    name.title = d.name;

    const del = document.createElement("button");
    del.className = "del";
    del.textContent = "✕";
    del.title = "Delete";
    del.onclick = (e) => { e.stopPropagation(); removeDoc(d.id); };

    li.append(name, del);
    li.onclick = () => selectDoc(d.id);
    ul.appendChild(li);
  }
}

async function removeDoc(id) {
  if (!confirm("Delete this document?")) return;
  await api(`/api/documents/${id}`, { method: "DELETE" });
  state.chats.delete(id);
  if (state.currentId === id) {
    state.currentId = null;
    $("workspace").hidden = true;
    $("placeholder").hidden = false;
  }
  await loadDocs();
}

// ---------- Viewing a document ----------

async function selectDoc(id) {
  const doc = await api(`/api/documents/${id}`);
  state.currentId = id;

  $("placeholder").hidden = true;
  $("workspace").hidden = false;
  $("docTitle").textContent = doc.name;
  $("docMeta").textContent =
    `${doc.chunk_count} chunks · ${doc.chars.toLocaleString()} characters · uploaded ${doc.uploaded_at}`;
  $("docText").textContent = doc.text;

  $("askForm").hidden = false;
  renderChat();
  renderDocList();
  showTab("viewer");
}

function showTab(name) {
  document.querySelectorAll(".tabs button").forEach((b) =>
    b.classList.toggle("active", b.dataset.tab === name)
  );
  $("viewer").hidden = name !== "viewer";
  $("chat").hidden = name !== "chat";
  $("askForm").hidden = name !== "chat";
}

document.querySelectorAll(".tabs button").forEach((b) =>
  b.addEventListener("click", () => showTab(b.dataset.tab))
);

// ---------- Upload ----------

const fileInput = $("fileInput");
const uploadBtn = $("uploadBtn");

fileInput.addEventListener("change", () => {
  uploadBtn.disabled = !fileInput.files.length;
});

const dropZone = $("dropZone");
dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("over"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("over"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("over");
  if (e.dataTransfer.files.length) {
    fileInput.files = e.dataTransfer.files;
    uploadBtn.disabled = false;
  }
});

uploadBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) return;

  const form = new FormData();
  form.append("file", file);

  uploadBtn.disabled = true;
  $("uploadStatus").textContent = "Indexing… (embedding may take a moment)";
  try {
    const doc = await api("/api/documents", { method: "POST", body: form });
    $("uploadStatus").textContent = `Indexed ${doc.name}`;
    fileInput.value = "";
    await loadDocs();
    await selectDoc(doc.id);
  } catch (err) {
    $("uploadStatus").textContent = `Error: ${err.message}`;
  } finally {
    uploadBtn.disabled = !fileInput.files.length;
  }
});

// ---------- Chat ----------

function chatFor(id) {
  if (!state.chats.has(id)) state.chats.set(id, []);
  return state.chats.get(id);
}

function renderChat() {
  const box = $("messages");
  box.innerHTML = "";
  for (const entry of chatFor(state.currentId)) box.appendChild(buildMessage(entry));
  box.scrollTop = box.scrollHeight;
}

function buildMessage(entry) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${entry.role}`;

  const text = document.createElement("div");
  text.className = "answer";
  text.textContent = entry.content;
  wrap.appendChild(text);

  if (entry.role === "assistant" && entry.sources) {
    const searches = entry.sources.searches;
    if (searches.length) {
      const meta = document.createElement("div");
      meta.className = "meta";
      meta.textContent = `🔎 Web searches: ${searches.join(", ")}`;
      wrap.appendChild(meta);
    }

    if (entry.sources.web_sources.length) {
      const d = document.createElement("details");
      d.open = true;
      d.innerHTML = "<summary>Web sources</summary>";
      const list = document.createElement("div");
      list.className = "web-list";
      for (const w of entry.sources.web_sources) {
        const row = document.createElement("div");
        row.style.margin = "6px 0";
        const a = document.createElement("a");
        a.href = /^https?:\/\//.test(w.url) ? w.url : "#";
        a.target = "_blank";
        a.rel = "noopener noreferrer";
        a.textContent = w.title || w.url;
        const snip = document.createElement("div");
        snip.textContent = w.snippet || "";
        snip.style.color = "#6b7280";
        row.append(a, snip);
        list.appendChild(row);
      }
      d.appendChild(list);
      wrap.appendChild(d);
    }

    if (entry.sources.doc_sources.length) {
      const d = document.createElement("details");
      d.innerHTML = "<summary>Document excerpts used</summary>";
      for (const s of entry.sources.doc_sources) {
        const pre = document.createElement("pre");
        pre.textContent = `[Excerpt ${s.excerpt}] (similarity ${s.score})\n\n${s.text}`;
        d.appendChild(pre);
      }
      wrap.appendChild(d);
    }
  }
  return wrap;
}

$("askForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = $("question").value.trim();
  if (!question || !state.currentId) return;

  const docId = state.currentId;
  const history = chatFor(docId);
  history.push({ role: "user", content: question });
  $("question").value = "";
  $("askBtn").disabled = true;

  const pending = { role: "assistant", content: "Thinking…" };
  history.push(pending);
  renderChat();

  try {
    const res = await api("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        doc_id: docId,
        question,
        use_web: $("useWeb").checked,
        top_k: 4,
      }),
    });
    pending.content = res.answer;
    pending.sources = res;
  } catch (err) {
    pending.content = `Error: ${err.message}`;
  } finally {
    $("askBtn").disabled = false;
    if (state.currentId === docId) renderChat();
  }
});

// ---------- Init ----------

loadDocs();