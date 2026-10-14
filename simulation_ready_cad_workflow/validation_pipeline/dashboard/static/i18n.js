/*
 * Dashboard language switch (Chinese / English).
 *
 * The UI is written in Chinese. In English mode this script translates what the
 * browser renders: text nodes and the title, aria-label, placeholder and alt
 * attributes (and the value of button-type inputs), using the catalog in
 * i18n/en.js. A MutationObserver keeps translating whatever app.js renders
 * later. Agent output and user content are marked translate="no" (or
 * data-no-i18n) and are never rewritten; pre, code, textarea, script and style
 * contents are skipped too. Chinese mode loads no catalog and observes nothing.
 *
 * Choice: ?lang=zh|en (persisted) > localStorage > "lang" cookie > browser
 * language (zh* -> zh, otherwise en). window.i18n.setLang() stores it and
 * reloads; any element with data-lang-toggle switches language on click.
 *
 * Lookup (canonical text = whitespace collapsed, separator punctuation at both
 * ends kept aside; the same rule as i18n_catalog.canonical in Python):
 * exact key -> template key with {0}, {1}... (captured values are translated
 * in turn; a template with under three fixed Chinese characters only applies
 * when they translate fully) -> the text split at the coarsest separators it
 * contains (" · " "；" "。", then "，" "：", then brackets, then a space before
 * a number), each part looked up again. Anything not found stays unchanged.
 */
(function (root) {
  "use strict";

  var STORAGE_KEY = "quest-dashboard-lang";
  var COOKIE_NAME = "lang";
  var LANGS = { zh: true, en: true };
  var CJK = /[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]/;
  var CJK_ALL = /[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]/g;
  // Must match EDGE_CHARS in dashboard/i18n_catalog.py.
  var EDGE = " \t\r\n\u3000\u00b7\u2022|\uff5c\uff0c,\uff1b;\uff1a:\u3001\u3002.\uff01!\uff1f?\u2026\u2192\u2190\u2197\u2194\u25b6\u275a\u00d7\u2713\u2717\u2304\u25cf\u25cb\u25c6\u25c7\u25a0\u25a1\u2605\u2606\u26a0";
  // Fallback split points, coarse to fine: phrases, clauses, brackets/arrows,
  // then a space before a number ("全部 3").
  var SEPARATORS = [
    /(\s*[\u00b7\u2022|\uff5c\uff1b\u3002\uff01\uff1f]\s*|\s*;\s+)/,
    /(\s*[\uff0c\u3001\uff1a]\s*|\s*[:,]\s+)/,
    /(\s*[\uff08\uff09]\s*|\s+[\/\u2192\u2190]\s+)/,
    /(\s+)(?=\d)/,
  ];
  var PUNCTUATION = {
    "\uff0c": ", ", "\u3002": ". ", "\uff1b": "; ", "\uff1a": ": ", "\u3001": ", ",
    "\uff01": "! ", "\uff1f": "? ", "\uff08": " (", "\uff09": ") ", "\uff5c": " | ", "\u3000": " ",
  };
  var ATTRIBUTES = ["title", "aria-label", "placeholder", "alt", "value"];
  var BUTTON_INPUTS = { button: true, submit: true, reset: true };
  var SKIP_CONTENT = { PRE: true, CODE: true, TEXTAREA: true, SCRIPT: true, STYLE: true };
  var MEMO_LIMIT = 20000;
  // Templates with adjacent placeholders backtrack heavily on long text; long
  // text is only split, and its parts are short enough to try templates on.
  var TEMPLATE_TEXT_LIMIT = 400;

  function hasCjk(text) {
    return CJK.test(text);
  }

  function englishPunctuation(text) {
    var out = "";
    for (var i = 0; i < text.length; i += 1) out += PUNCTUATION[text[i]] || text[i];
    return out;
  }

  function tidy(text) {
    return text.replace(/ {2,}/g, " ").replace(/ +\)/g, ")").replace(/\( +/g, "(").trim();
  }

  // [leading punctuation, core, trailing punctuation]; brackets around the
  // whole core (as in "（实验性）") count as punctuation. Mirrors
  // canonical() in dashboard/i18n_catalog.py.
  function splitEdges(text) {
    var start = 0;
    var end = text.length;
    while (start < end && EDGE.indexOf(text[start]) >= 0) start += 1;
    while (end > start && EDGE.indexOf(text[end - 1]) >= 0) end -= 1;
    var pair = text[start] + text[end - 1];
    if (end - start > 2 && (pair === "\uff08\uff09" || pair === "()") && !/[()\uff08\uff09]/.test(text.slice(start + 1, end - 1))) {
      start += 1;
      end -= 1;
      while (start < end && EDGE.indexOf(text[start]) >= 0) start += 1;
      while (end > start && EDGE.indexOf(text[end - 1]) >= 0) end -= 1;
    }
    return [text.slice(0, start), text.slice(start, end), text.slice(end)];
  }

  function escapeRegExp(text) {
    return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  function compile(catalog) {
    var exact = new Map();
    var templates = [];
    Object.keys(catalog || {}).forEach(function (key) {
      var value = catalog[key];
      if (typeof value !== "string" || !value) return;
      exact.set(key, value);
      if (!/\{\d+\}/.test(key)) return;
      var parts = key.split(/\{(\d+)\}/);
      var pattern = "";
      var order = [];
      var literal = "";
      var literalLength = 0;
      for (var i = 0; i < parts.length; i += 1) {
        if (i % 2 === 0) {
          pattern += escapeRegExp(parts[i]);
          literalLength += parts[i].length;
          if (parts[i].length > literal.length) literal = parts[i];
        } else {
          pattern += "([\\s\\S]*?)";
          order.push(parts[i]);
        }
      }
      // With little fixed Chinese text ("{0}分{1}") a template matches too
      // much, so it only applies when every captured value translates fully.
      var strict = (parts.filter(function (_part, index) { return index % 2 === 0; }).join("").match(CJK_ALL) || []).length < 3;
      templates.push({ re: new RegExp("^" + pattern + "$"), order: order, value: value, literal: literal, weight: literalLength, strict: strict });
    });
    templates.sort(function (a, b) { return b.weight - a.weight; });
    return { exact: exact, templates: templates };
  }

  /** Build a translator for one catalog; pure apart from its memo. */
  function createTranslator(catalog) {
    var compiled = compile(catalog);
    var memo = new Map();

    function remember(key, value) {
      if (memo.size >= MEMO_LIMIT) memo.clear();
      memo.set(key, value);
      return value;
    }

    // Exact entry or template for a core string (no edge punctuation); null
    // when neither applies.
    function direct(core) {
      var key = "\u0002" + core;
      if (memo.has(key)) return memo.get(key);
      var found = compiled.exact.get(core);
      if (found != null) return remember(key, found);
      var templates = core.length <= TEMPLATE_TEXT_LIMIT ? compiled.templates : [];
      for (var i = 0; i < templates.length; i += 1) {
        var template = templates[i];
        if (core.indexOf(template.literal) < 0) continue;
        var match = template.re.exec(core);
        if (!match) continue;
        var values = {};
        var complete = true;
        for (var j = 0; j < template.order.length && complete; j += 1) {
          var capture = match[j + 1];
          var value = hasCjk(capture) ? translate(capture) : englishPunctuation(capture);
          complete = !(template.strict && hasCjk(value));
          values[template.order[j]] = value;
        }
        if (!complete) continue;
        var filled = template.value.replace(/\{(\d+)\}/g, function (all, index) {
          return Object.prototype.hasOwnProperty.call(values, index) ? values[index] : all;
        });
        return remember(key, tidy(filled));
      }
      return remember(key, null);
    }

    // Translate a core string; null when unknown. ``level`` is the first
    // SEPARATORS level the fallback may split at.
    function lookup(core, level) {
      var key = level + "\u0001" + core;
      if (memo.has(key)) return memo.get(key);
      var found = direct(core);
      return remember(key, found != null ? found : segments(core, level));
    }

    // Split at the coarsest separators present. From each part, the longest
    // run of parts that is itself an entry wins ("3 min ago" + "运行命令：{0}"
    // in "3 分钟前：运行命令：x"); otherwise the part is looked up alone and may
    // split further. Partial results are kept.
    function segments(core, level) {
      for (; level < SEPARATORS.length; level += 1) {
        var parts = core.split(SEPARATORS[level]);
        if (parts.length < 2) continue;
        var out = "";
        var translated = false;
        for (var i = 0; i < parts.length; ) {
          var end = i;
          var result = null;
          for (var j = parts.length - 1; j > i && result == null; j -= 2) {
            result = tryBody(parts.slice(i, j + 1).join(""), level, true);
            if (result != null) end = j;
          }
          if (result == null) result = tryBody(parts[i], level + 1, false);
          if (result != null) translated = true;
          out += result != null ? result : parts[i];
          if (end + 1 < parts.length) out += englishPunctuation(parts[end + 1]);
          i = end + 2;
        }
        return translated ? tidy(out) : null;
      }
      return null;
    }

    // Translate trimmed text, keeping edge punctuation; null when unknown.
    // ``directOnly`` skips the separator fallback.
    function tryBody(body, level, directOnly) {
      if (!body || !hasCjk(body)) return null;
      var edges = splitEdges(body.replace(/\s+/g, " "));
      if (!hasCjk(edges[1])) return null;
      var result = directOnly ? direct(edges[1]) : lookup(edges[1], level || 0);
      if (result == null) return null;
      var lead = englishPunctuation(edges[0]).replace(/^\s+/, "");
      // Keep the space after a trailing comma or colon (an element may follow).
      var tail = englishPunctuation(edges[2]).replace(/\s{2,}/g, " ").replace(/([.!?)])\s+$/, "$1");
      return lead + result + tail;
    }

    /** Translate a string, keeping its leading and trailing whitespace. */
    function translate(text) {
      if (text == null) return text;
      text = String(text);
      if (!hasCjk(text)) return text;
      var match = /^(\s*)([\s\S]*?)(\s*)$/.exec(text);
      var result = tryBody(match[2]);
      return result == null ? text : match[1] + result + match[3];
    }

    function optedOut(element) {
      return element.hasAttribute("data-no-i18n") || element.getAttribute("translate") === "no";
    }

    function skipsContent(element) {
      return SKIP_CONTENT[String(element.nodeName).toUpperCase()] === true || optedOut(element);
    }

    // True when an ancestor (not the node itself) hides the node's content.
    function insideSkipped(node) {
      for (var parent = node.parentNode; parent && parent.nodeType === 1; parent = parent.parentNode) {
        if (skipsContent(parent)) return true;
      }
      return false;
    }

    function translateAttribute(element, name) {
      if (name === "value") {
        var tag = String(element.nodeName).toUpperCase();
        var type = String(element.getAttribute("type") || "").toLowerCase();
        if (tag !== "BUTTON" && !(tag === "INPUT" && BUTTON_INPUTS[type])) return;
      }
      var current = element.getAttribute(name);
      if (!current || !hasCjk(current)) return;
      var next = translate(current);
      if (next !== current) element.setAttribute(name, next);
    }

    function translateText(node) {
      var current = node.nodeValue;
      if (!current || !hasCjk(current)) return;
      var next = translate(current);
      if (next !== current) node.nodeValue = next;
    }

    /** Translate a node and its subtree (the caller checks ancestors). */
    function translateTree(start) {
      var stack = [start];
      while (stack.length) {
        var node = stack.pop();
        if (node.nodeType === 3) { translateText(node); continue; }
        if (node.nodeType !== 1 && node.nodeType !== 9 && node.nodeType !== 11) continue;
        if (node.nodeType === 1) {
          if (optedOut(node)) continue;
          for (var i = 0; i < ATTRIBUTES.length; i += 1) translateAttribute(node, ATTRIBUTES[i]);
          if (skipsContent(node)) continue;
        }
        var children = node.childNodes || [];
        for (var j = children.length - 1; j >= 0; j -= 1) stack.push(children[j]);
      }
    }

    /** Apply MutationObserver records; returns nothing. */
    function applyMutations(records) {
      for (var i = 0; i < records.length; i += 1) {
        var record = records[i];
        if (record.type === "childList") {
          var added = record.addedNodes || [];
          for (var j = 0; j < added.length; j += 1) {
            var node = added[j];
            if (node.isConnected === false || insideSkipped(node)) continue;
            translateTree(node);
          }
        } else if (record.type === "characterData") {
          if (!insideSkipped(record.target)) translateText(record.target);
        } else if (record.type === "attributes") {
          var target = record.target;
          if (!optedOut(target) && !insideSkipped(target)) translateAttribute(target, record.attributeName);
        }
      }
    }

    return {
      translate: translate,
      lookup: function (core) { return lookup(core, 0); },
      translateTree: translateTree,
      applyMutations: applyMutations,
      insideSkipped: insideSkipped,
    };
  }

  // ------------------------------------------------------------ language choice

  function readStorage() {
    try { return root.localStorage ? root.localStorage.getItem(STORAGE_KEY) : null; } catch (_error) { return null; }
  }

  function readCookie() {
    try {
      var match = /(?:^|;\s*)lang=([^;]*)/.exec(root.document ? root.document.cookie || "" : "");
      return match ? decodeURIComponent(match[1]) : null;
    } catch (_error) { return null; }
  }

  function persist(lang) {
    try { if (root.localStorage) root.localStorage.setItem(STORAGE_KEY, lang); } catch (_error) { /* private mode */ }
    try {
      if (root.document) root.document.cookie = COOKIE_NAME + "=" + lang + "; path=/; max-age=31536000; SameSite=Lax";
    } catch (_error) { /* file:// snapshots may refuse cookies */ }
  }

  function detect() {
    var query = null;
    try { query = new URLSearchParams(root.location ? root.location.search : "").get("lang"); } catch (_error) { query = null; }
    if (query && LANGS[query]) { persist(query); return query; }
    var stored = readStorage();
    var cookie = readCookie();
    if (stored && LANGS[stored]) {
      if (cookie !== stored) persist(stored);
      return stored;
    }
    if (cookie && LANGS[cookie]) return cookie;
    var browser = (root.navigator && (root.navigator.language || (root.navigator.languages || [])[0])) || "";
    return /^zh/i.test(browser) ? "zh" : "en";
  }

  var lang = detect();
  var active = null;
  var catalogs = root.QUEST_I18N_CATALOGS = root.QUEST_I18N_CATALOGS || {};

  function setLang(next) {
    if (!LANGS[next]) return;
    persist(next);
    try {
      var url = new URL(root.location.href);
      if (url.searchParams.has("lang")) {
        url.searchParams.delete("lang");
        root.location.replace(url.href);
        return;
      }
    } catch (_error) { /* fall through to a plain reload */ }
    root.location.reload();
  }

  /** Translate a string that never reaches the DOM (alert, confirm, Notification…). */
  function t(text) {
    return lang === "en" && active ? active.translate(text) : text;
  }

  function toggleMarkup(className) {
    var label = lang === "en" ? "切换到中文" : "Switch to English";
    return '<button type="button" class="lang-toggle' + (className ? " " + className : "") +
      '" data-lang-toggle translate="no" title="' + label + '" aria-label="' + label + '">' +
      '<span lang="zh"' + (lang === "zh" ? ' class="is-active"' : "") + ">中文</span>" +
      '<span aria-hidden="true">/</span>' +
      '<span lang="en"' + (lang === "en" ? ' class="is-active"' : "") + ">English</span></button>";
  }

  function activate() {
    if (active || !catalogs.en || !root.document) return;
    active = createTranslator(catalogs.en);
    var document = root.document;
    active.translateTree(document.documentElement);
    if (typeof root.MutationObserver !== "function") return;
    var observer = new root.MutationObserver(function (records) {
      active.applyMutations(records);
      // Drop the records our own writes queued; nothing else runs meanwhile.
      observer.takeRecords();
    });
    observer.observe(document, {
      childList: true,
      subtree: true,
      characterData: true,
      attributes: true,
      attributeFilter: ATTRIBUTES,
    });
  }

  root.i18n = {
    get lang() { return lang; },
    setLang: setLang,
    t: t,
    toggleMarkup: toggleMarkup,
    createTranslator: createTranslator,
  };
  root.t = t;

  if (root.document && root.document.addEventListener) {
    root.document.addEventListener("click", function (event) {
      var target = event.target && event.target.closest ? event.target.closest("[data-lang-toggle]") : null;
      if (!target) return;
      event.preventDefault();
      setLang(target.getAttribute("data-lang-toggle") || (lang === "en" ? "zh" : "en"));
    });
  }

  if (lang === "en" && root.document) {
    if (root.document.documentElement) root.document.documentElement.lang = "en";
    if (catalogs.en) {
      activate();
    } else {
      catalogs.onload = function (name) { if (name === "en") activate(); };
      var current = root.document.currentScript;
      var source = current && current.src ? current.src.replace(/i18n\.js(?:[?#].*)?$/, "i18n/en.js") : "";
      if (source && source !== current.src) {
        var script = root.document.createElement("script");
        script.src = source;
        (root.document.head || root.document.documentElement).appendChild(script);
      }
    }
  }
})(typeof window !== "undefined" ? window : globalThis);
