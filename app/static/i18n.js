/* Interface copy, in one place, switchable at runtime.
 *
 * English is the source of truth: a key missing from another language falls
 * back to it rather than showing the key. The choice is remembered per browser;
 * on the first visit it follows the browser's own language, so a deployment
 * does not have to be configured for its readers.
 *
 * Strings carrying values write them as {name} and pass an object to t().
 * Functional strings stay as they are — question templates, tool output and
 * document text are not interface copy and are never translated here.
 */

const DICT = {
  en: {
    "nav.chat": "Chat",
    "nav.upload": "Upload",
    "nav.api": "API console",
    "nav.signout": "Sign out",
    "nav.corpus": "Corpus",
    "nav.theme": "Switch theme",
    "nav.lang": "切换中文",

    "gate.title": "Sign in to ChillRAG",
    "gate.sub": "Ask your documents. Answers cite the files and pages they come from.",
    "gate.username": "Username",
    "gate.password": "Password",
    "gate.signin": "Sign in",
    "gate.failed": "Sign-in failed ({status})",

    "chat.new": "New chat",
    "chat.empty.title": "Ask your documents.",
    "chat.empty.sub": "Retrieval, reasoning and citation, grounded in your corpus.",
    "chat.empty.none": "No corpus yet — upload documents first.",
    "chat.composer.placeholder": "Ask anything about the selected corpus…",
    "chat.composer.hint": "answers cite files and pages · Enter to send · Shift+Enter for a newline",
    "chat.send": "Send",
    "chat.upload": "Upload documents",
    "chat.examples": [
      "What was the operating revenue?",
      "Summarize the main risk factors.",
      "List the largest shareholders and their stakes.",
    ],

    "sessions.none": "No conversations yet",
    "sessions.noMatch": "No conversations match",
    "sessions.foot": "{n} saved in this browser",
    "sessions.new": "New chat",
    "sessions.today": "Today",
    "sessions.yesterday": "Yesterday",
    "sessions.earlier": "Earlier",
    "sessions.delete": "Delete",
    "sessions.confirm": "Delete again to confirm",
    "sessions.rename": "Rename",
    "sessions.untitled": "Untitled",
    "sessions.filter": "Filter conversations…",

    "time.now": "just now",
    "time.minutes": "{n} min ago",
    "time.hours": "{n} h ago",
    "time.days": "{n} d ago",
    "time.elapsed": "{m}m {s}s",

    "msg.copy": "Copy",
    "msg.copied": "Copied ✓",
    "msg.noAnswer": "(no answer)",

    "status.thinking": "Thinking…",
    "status.answering": "Answering…",
    "status.working": "Working…",
    "status.elapsed": "{n}s",

    "tool.get_context": "Reading nearby context",
    "tool.get_toc": "Reading the table of contents",
    "tool.get_materials": "Listing documents",
    "tool.calculate": "Calculating",
    "tool.faiss_search": "Semantic search",
    "tool.bm25_search": "Keyword search",
    "tool.grep_search": "Exact-text search",
    "tool.args": "Arguments",
    "tool.other": "Tool",

    "upload.title": "Add documents",
    "upload.sub": "Drop files here and they become a searchable corpus: parsed, split into passages, and indexed.",
    "upload.corpus.label": "Corpus",
    "upload.corpus.new": "New corpus…",
    "upload.corpus.name": "Corpus name",
    "upload.corpus.placeholder": "e.g. 天津银行 2024 年报",
    "upload.corpus.hint": "Letters, digits, spaces and dashes. It names both the source folder and its cache.",
    "upload.corpus.id": "source id {id}",
    "upload.drop": "Drop files here, or click to choose",
    "upload.drop.sub": "PDF, Word, Excel, PowerPoint, images, Markdown",
    "upload.pick": "Choose files",
    "upload.list": "{n} file(s) ready",
    "upload.remove": "Remove",
    "upload.go": "Upload and ingest",
    "upload.uploading": "Uploading… {pct}%",
    "upload.parsing": "Parsing and indexing… {elapsed}",
    "upload.parsing.hint": "Parsing runs on the server: a large annual report can take minutes. This page can stay open, or you can come back later.",
    "upload.file.pending": "waiting",
    "upload.file.parsing": "parsing…",
    "upload.file.done": "{n} passages",
    "upload.file.none": "no text found",
    "upload.file.failed": "failed",
    "upload.done.title": "Ready to search",
    "upload.done.body": "{documents} document(s) · {chunks} passages",
    "upload.done.chat": "Ask about it",
    "upload.done.more": "Add more files",
    "upload.reject.suffix": "{name}: this file type is not supported",
    "upload.reject.read": "{name}: could not be read",
    "upload.needFiles": "Choose at least one file.",
    "upload.needName": "Name the corpus before uploading.",
    "upload.errFailed": "Ingest failed on the server. Check the service log for the reason.",
  },

  zh: {
    "nav.chat": "问答",
    "nav.upload": "上传",
    "nav.api": "接口调试",
    "nav.signout": "退出",
    "nav.corpus": "语料",
    "nav.theme": "切换主题",
    "nav.lang": "Switch to English",

    "gate.title": "登录 ChillRAG",
    "gate.sub": "向你的文档提问，答案会标注来自哪个文件、哪一页。",
    "gate.username": "用户名",
    "gate.password": "密码",
    "gate.signin": "登录",
    "gate.failed": "登录失败（{status}）",

    "chat.new": "新建对话",
    "chat.empty.title": "向你的文档提问。",
    "chat.empty.sub": "检索、推理、引用，全部基于你自己的语料。",
    "chat.empty.none": "还没有语料，请先上传文档。",
    "chat.composer.placeholder": "就选中的语料提问…",
    "chat.composer.hint": "答案标注文件与页码 · Enter 发送 · Shift+Enter 换行",
    "chat.send": "发送",
    "chat.upload": "上传文档",
    "chat.examples": [
      "营业收入是多少？",
      "总结主要风险因素。",
      "列出持股比例最高的股东。",
    ],

    "sessions.none": "还没有对话",
    "sessions.noMatch": "没有匹配的对话",
    "sessions.foot": "本浏览器保存 {n} 条",
    "sessions.new": "新建对话",
    "sessions.today": "今天",
    "sessions.yesterday": "昨天",
    "sessions.earlier": "更早",
    "sessions.delete": "删除",
    "sessions.confirm": "再点一次确认删除",
    "sessions.rename": "重命名",
    "sessions.untitled": "未命名",
    "sessions.filter": "筛选对话…",

    "time.now": "刚刚",
    "time.minutes": "{n} 分钟前",
    "time.hours": "{n} 小时前",
    "time.days": "{n} 天前",
    "time.elapsed": "{m}分{s}秒",

    "msg.copy": "复制",
    "msg.copied": "已复制 ✓",
    "msg.noAnswer": "（没有回答）",

    "status.thinking": "思考中…",
    "status.answering": "作答中…",
    "status.working": "处理中…",
    "status.elapsed": "{n} 秒",

    "tool.get_context": "读取上下文",
    "tool.get_toc": "读取目录",
    "tool.get_materials": "列出文档",
    "tool.calculate": "计算",
    "tool.faiss_search": "语义检索",
    "tool.bm25_search": "关键词检索",
    "tool.grep_search": "原文检索",
    "tool.args": "参数",
    "tool.other": "工具",

    "upload.title": "添加文档",
    "upload.sub": "把文件放进来，它就变成可检索的语料：解析、切分、建索引。",
    "upload.corpus.label": "语料",
    "upload.corpus.new": "新建语料…",
    "upload.corpus.name": "语料名称",
    "upload.corpus.placeholder": "例如：天津银行 2024 年报",
    "upload.corpus.hint": "可用文字、数字、空格和短横线。它同时是源文件夹名和缓存目录名。",
    "upload.corpus.id": "source id {id}",
    "upload.drop": "把文件拖到这里，或点击选择",
    "upload.drop.sub": "PDF、Word、Excel、PowerPoint、图片、Markdown",
    "upload.pick": "选择文件",
    "upload.list": "待上传 {n} 个文件",
    "upload.remove": "移除",
    "upload.go": "上传并解析",
    "upload.uploading": "上传中… {pct}%",
    "upload.parsing": "解析与建索引… {elapsed}",
    "upload.parsing.hint": "解析在服务端进行：一份大的年报可能要几分钟。可以留在本页等待，也可以稍后再回来。",
    "upload.file.pending": "排队中",
    "upload.file.parsing": "解析中…",
    "upload.file.done": "{n} 个片段",
    "upload.file.none": "未提取到文字",
    "upload.file.failed": "失败",
    "upload.done.title": "可以检索了",
    "upload.done.body": "{documents} 篇文档 · {chunks} 个片段",
    "upload.done.chat": "去提问",
    "upload.done.more": "继续上传",
    "upload.reject.suffix": "{name}：不支持这种文件类型",
    "upload.reject.read": "{name}：读取失败",
    "upload.needFiles": "请至少选择一个文件。",
    "upload.needName": "请先给语料起个名字。",
    "upload.errFailed": "服务端解析失败，具体原因见服务日志。",
  },
};

const LANG_KEY = "chillrag.lang";
let lang = "en";

function detectLang() {
  try {
    const saved = localStorage.getItem(LANG_KEY);
    if (saved && DICT[saved]) return saved;
  } catch { /* private mode: fall through to the browser's own language */ }
  return (navigator.language || "en").toLowerCase().startsWith("zh") ? "zh" : "en";
}

function t(key, vars) {
  const table = DICT[lang] || DICT.en;
  const text = table[key] !== undefined ? table[key] : (DICT.en[key] !== undefined ? DICT.en[key] : key);
  if (!vars || typeof text !== "string") return text;
  return text.replace(/\{(\w+)\}/g, (whole, name) => (vars[name] !== undefined ? vars[name] : whole));
}

/* Fill every element that declares a key. Dynamic regions re-render themselves
   on the "chillrag:lang" event instead. */
function applyI18n(root) {
  const scope = root || document;
  for (const el of scope.querySelectorAll("[data-i18n]")) el.textContent = t(el.dataset.i18n);
  for (const el of scope.querySelectorAll("[data-i18n-placeholder]")) el.placeholder = t(el.dataset.i18nPlaceholder);
  for (const el of scope.querySelectorAll("[data-i18n-title]")) el.title = t(el.dataset.i18nTitle);
  document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
}

function getLang() { return lang; }

function setLang(next) {
  if (!DICT[next] || next === lang) return;
  lang = next;
  try { localStorage.setItem(LANG_KEY, next); } catch { /* not fatal */ }
  applyI18n();
  window.dispatchEvent(new CustomEvent("chillrag:lang", { detail: next }));
}

lang = detectLang();
