const queryInput = document.querySelector("#query");
const searchButton = document.querySelector("#search-button");
const resetButton = document.querySelector("#reset-button");
const resultArea = document.querySelector("#result-area");
const diagnosticsToggle = document.querySelector("#diagnostics-toggle");
const diagnosticsPanel = document.querySelector("#diagnostics-panel");
const diagnosticsValues = document.querySelector("#diagnostics-values");
const indexFileInput = document.querySelector("#index-file");
const indexMessage = document.querySelector("#index-message");

function escapeHtml(value) {
  const element = document.createElement("span");
  element.textContent = value;
  return element.innerHTML;
}

function emptyState(title, message) {
  resultArea.innerHTML = `
    <div class="empty-state">
      <div class="empty-mark">⌕</div>
      <h2>${escapeHtml(title)}</h2>
      <p>${escapeHtml(message)}</p>
    </div>`;
}

function renderResults(data) {
  if (!data.suggestions.length) {
    emptyState("No matches found", "Try another phrase or use fewer characters.");
    return;
  }

  const cards = data.suggestions.map((item) => `
    <article class="result-card">
      <div class="result-top">
        <p class="sentence">${escapeHtml(item.completed_sentence)}</p>
        <span class="score">score ${item.score}</span>
      </div>
      <div class="source"><span>File name:</span> ${escapeHtml(item.source_text)} <b>·</b> <span>Line:</span> ${item.offset}</div>
    </article>`).join("");

  resultArea.innerHTML = `
    <div class="results-header">
      <h2>${data.suggestions.length} matching suggestions</h2>
      <span class="timing">${data.elapsed_ms} ms</span>
    </div>
    <div class="result-list">${cards}</div>`;
}

function renderDiagnostics(data) {
  if (!data) {
    diagnosticsPanel.hidden = true;
    return;
  }

  const searchPaths = {
    "direct-only": "Direct lookup only",
    "short-query-variants": "One-character variant lookup",
    "trigram-anchors": "FTS trigram anchor lookup",
    "full-corpus-fallback": "Full-corpus fallback",
    empty: "No searchable input",
  };
  const rows = [
    ["Input after cleanup", JSON.stringify(data.normalized_query)],
    ["Direct-match rows returned (max 5)", `${data.direct_match_count} in ${data.direct_lookup_ms} ms`],
    ["Search path", searchPaths[data.search_path] || data.search_path],
  ];
  if (data.generated_variant_count) {
    rows.push(["Legal typo variants generated", data.generated_variant_count]);
  }
  rows.push(["Engine total", `${data.total_ms} ms`]);
  diagnosticsValues.innerHTML = rows.map(([label, value]) => `
    <div><dt>${escapeHtml(String(label))}</dt><dd>${escapeHtml(String(value))}</dd></div>`).join("");
  diagnosticsPanel.hidden = false;
}

async function search() {
  const query = queryInput.value;
  if (!query.trim()) {
    emptyState("Ready to search", "Type text and press Enter.");
    return;
  }
  if (query === "#") {
    queryInput.value = "";
    emptyState("Search reset", "You can start a new search.");
    queryInput.focus();
    return;
  }

  searchButton.disabled = true;
  searchButton.textContent = "Searching...";
  try {
    const response = await fetch("/api/suggestions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, diagnostics: diagnosticsToggle.checked }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Search failed.");
    renderResults(data);
    renderDiagnostics(data.diagnostics);
  } catch (error) {
    resultArea.innerHTML = `<div class="error">${escapeHtml(error.message)}</div>`;
  } finally {
    searchButton.disabled = false;
    searchButton.textContent = "Search";
  }
}

queryInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") search();
});
searchButton.addEventListener("click", search);
resetButton.addEventListener("click", () => {
  queryInput.value = "";
  emptyState("Search reset", "You can start a new search.");
  queryInput.focus();
});

diagnosticsToggle.addEventListener("change", () => {
  if (!diagnosticsToggle.checked) diagnosticsPanel.hidden = true;
});

indexFileInput.addEventListener("change", async () => {
  const [file] = indexFileInput.files;
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".sqlite3")) {
    indexMessage.textContent = "Please select an index.sqlite3 file.";
    indexFileInput.value = "";
    return;
  }

  indexMessage.textContent = `Uploading ${file.name} and validating it against the active archive...`;
  try {
    const response = await fetch("/api/index", {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream", "X-Index-Filename": file.name },
      body: file,
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Index upload failed.");
    indexMessage.textContent = `Uploaded index is active: ${Number(data.indexed_sentence_count).toLocaleString()} lines ready.`;
  } catch (error) {
    indexMessage.textContent = error.message;
  } finally {
    indexFileInput.value = "";
  }
});
