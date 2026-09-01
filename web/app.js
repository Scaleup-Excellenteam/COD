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

  const directEnough = data.search_path === "direct-only";
  const pathMessages = {
    "short-query-variants": "Because this is a short query, the engine created legal one-character alternatives and checked them in the index.",
    "trigram-anchors": "The trigram index first narrowed the corpus to likely sentences, then the engine checked one-character corrections.",
    "full-corpus-fallback": "For a one-character query, the engine must check the full corpus to preserve the one-typo rule.",
  };
  const correctionMessages = (data.correction_details || []).slice(0, 3).map((detail) => {
    if (detail.operation === "replace") {
      return `Found “${detail.matched_text}” by changing “${detail.from_character}” to “${detail.to_character}” at character ${detail.position} (score ${detail.score}).`;
    }
    if (detail.operation === "remove-extra") {
      return `Found “${detail.matched_text}” by removing the extra “${detail.from_character}” at character ${detail.position} (score ${detail.score}).`;
    }
    return `Found “${detail.matched_text}” by adding the missing “${detail.to_character}” at character ${detail.position} (score ${detail.score}).`;
  });
  const correctionText = directEnough
    ? "Skipped. Five direct matches were already available, and they always rank above a one-character correction."
    : [
        pathMessages[data.search_path] || "The engine checked the legal one-character correction paths.",
        `It considered replacing one character, removing one extra character, or adding one missing character.${data.generated_variant_count ? ` For this short query it generated ${data.generated_variant_count} unique legal alternatives.` : ""}`,
        ...correctionMessages,
        !correctionMessages.length ? "No corrected result entered the final suggestions." : "",
      ].filter(Boolean).join(" ");
  const steps = [
    ["1", "Prepare the input", `The engine searched for “${data.normalized_query}” after applying the required cleanup rules.`],
    ["2", "Look for direct matches", `It returned ${data.direct_match_count} direct match${data.direct_match_count === 1 ? "" : "es"} (up to 5) in ${data.direct_lookup_ms} ms.`],
    ["3", "Check one typing error", correctionText],
    ["4", "Rank and return", `The visible suggestions are the best legal matches after scoring. Engine time: ${data.total_ms} ms.`],
  ];
  diagnosticsValues.innerHTML = steps.map(([number, title, text]) => `
    <article class="story-step">
      <span class="story-number">${number}</span>
      <div><h3>${escapeHtml(title)}</h3><p>${escapeHtml(text)}</p></div>
    </article>`).join("");
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
