/* Shared page chrome: identity, theme, language, nav — and the small helpers
 * every page would otherwise redefine.
 *
 * Loads after i18n.js (it calls t()) and before the page's own script. A page
 * declares itself with <body data-page="chat">, gives the nav an empty
 * #nav element, and calls bootPage() to hide the gate and start.
 */

"use strict";

const $ = (id) => document.getElementById(id);
const $$ = (selector, root) => Array.from((root || document).querySelectorAll(selector));

const PAGE_URL = { chat: "/", upload: "/upload", api: "/api-test" };
const THEME_KEY = "chillrag.theme";

/* ============================== formatting ============================== */

function escapeHtml(value) {
  return String(value).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function humanSize(bytes) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) { value /= 1024; unit += 1; }
  return (value < 10 && unit > 0 ? value.toFixed(1) : Math.round(value)) + " " + units[unit];
}

function relativeTime(stamp) {
  const seconds = Math.max(0, (Date.now() - stamp) / 1000);
  if (seconds < 60) return t("time.now");
  if (seconds < 3600) return t("time.minutes", { n: Math.floor(seconds / 60) });
  if (seconds < 86400) return t("time.hours", { n: Math.floor(seconds / 3600) });
  return t("time.days", { n: Math.floor(seconds / 86400) });
}

/* ================================= theme ================================ */

function effectiveTheme() {
  const chosen = document.documentElement.dataset.theme;
  if (chosen) return chosen;
  return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem(THEME_KEY); } catch { /* private mode */ }
  if (saved === "dark" || saved === "light") document.documentElement.dataset.theme = saved;
}

function toggleTheme() {
  document.documentElement.dataset.theme = effectiveTheme() === "dark" ? "light" : "dark";
  try { localStorage.setItem(THEME_KEY, document.documentElement.dataset.theme); } catch { /* not fatal */ }
}

/* ================================ request =============================== */

/* The service answers with one envelope everywhere: {code, msg, data}. Returns
   `data` and throws on anything else, so callers only handle success. */
async function api(path, options) {
  const response = await fetch(path, Object.assign({ credentials: "same-origin" }, options || {}));
  const body = await response.json().catch(() => null);
  if (!response.ok || (body && body.code !== 0)) {
    const error = new Error((body && body.msg) || ("HTTP " + response.status));
    error.status = response.status;
    throw error;
  }
  return body ? body.data : null;
}

function postJSON(path, payload) {
  return api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

/* ============================== identity =============================== */

function setIdentity(me) {
  const slot = $("me");
  if (slot) slot.textContent = me ? me.username + " · " + me.role : "";
  return me || null;
}

/* The signed-in user, or null — a 401 here is the normal signed-out answer,
   not an error to report. */
async function currentUser() {
  try {
    return await api("/api/auth/me");
  } catch (error) {
    if (error.status === 401) return null;
    throw error;
  }
}

async function signOut() {
  try { await fetch("/api/auth/logout", { method: "POST" }); } catch { /* reload anyway */ }
  location.reload();
}

/* ================================= nav ================================= */

function refreshLangButton() {
  const button = $("lang-toggle");
  if (button) button.textContent = getLang() === "zh" ? "EN" : "中";
}

/* Prepends the brand and the page links and appends the controls, leaving
   whatever the page wrote in the middle (its own selectors, mostly). */
function renderNav() {
  const nav = $("nav");
  if (!nav) return;
  const page = document.body.dataset.page || "";

  const brand = document.createElement("a");
  brand.className = "nav-brand";
  brand.href = PAGE_URL.chat;
  brand.innerHTML = '<span class="mark">❄</span><span class="name">ChillRAG</span>';

  const links = document.createElement("div");
  links.className = "nav-links";
  for (const key of Object.keys(PAGE_URL)) {
    const link = document.createElement("a");
    link.className = "nav-link";
    link.href = PAGE_URL[key];
    link.dataset.i18n = "nav." + key;
    if (key === page) link.setAttribute("aria-current", "page");
    links.appendChild(link);
  }

  const right = document.createElement("div");
  right.className = "nav-right";
  right.innerHTML =
    '<button class="nav-icon" id="lang-toggle" data-i18n-title="nav.lang"></button>' +
    '<button class="nav-icon" id="theme-toggle" data-i18n-title="nav.theme">◐</button>' +
    '<span class="who" id="me"></span>' +
    '<button class="nav-link" id="signout" data-i18n="nav.signout"></button>';

  nav.prepend(links);
  nav.prepend(brand);
  nav.appendChild(right);

  $("lang-toggle").addEventListener("click", () => setLang(getLang() === "zh" ? "en" : "zh"));
  $("theme-toggle").addEventListener("click", toggleTheme);
  $("signout").addEventListener("click", signOut);
  refreshLangButton();
}

/* ================================= gate ================================ */

function renderGate(gate) {
  gate.innerHTML =
    '<form class="gate-card" id="gate-form">' +
      '<div class="mark">❄</div>' +
      '<h1 data-i18n="gate.title"></h1>' +
      '<p class="sub" data-i18n="gate.sub"></p>' +
      '<div class="field"><label for="gate-user" data-i18n="gate.username"></label>' +
        '<input class="input" id="gate-user" autocomplete="username" autocapitalize="off"></div>' +
      '<div class="field"><label for="gate-pass" data-i18n="gate.password"></label>' +
        '<input class="input" id="gate-pass" type="password" autocomplete="current-password"></div>' +
      '<button class="btn btn-primary go" type="submit" data-i18n="gate.signin"></button>' +
      '<div class="notice" id="gate-msg"></div>' +
    "</form>";

  $("gate-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const message = $("gate-msg");
    message.textContent = "";
    try {
      await postJSON("/space/api/passport/login", {
        username: $("gate-user").value,
        passwd: $("gate-pass").value,
      });
    } catch (error) {
      // A rejected password is expected input, so it reads in the page's own
      // language; anything else is the server's to explain.
      message.textContent = error.status === 401 ? t("gate.failed", { status: 401 }) : error.message;
      return;
    }
    // The cookie is set; a reload is the shortest path to a page that knows it.
    location.reload();
  });

  applyI18n(gate);
  $("gate-user").focus();
}

/* ================================= boot =============================== */

/* Chrome first, then the gate: a visitor without a cookie sees the login card
   and nothing of the page behind it. `onReady` runs only when signed in. */
async function bootPage(onReady) {
  initTheme();
  renderNav();
  applyI18n();
  window.addEventListener("chillrag:lang", () => { applyI18n(); refreshLangButton(); });

  const me = setIdentity(await currentUser());
  const gate = $("gate");
  if (me) {
    if (gate) gate.hidden = true;
  } else if (gate) {
    gate.hidden = false;
    renderGate(gate);
    return;
  }
  if (onReady) await onReady(me);
}
