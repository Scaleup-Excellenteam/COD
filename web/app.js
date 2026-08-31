const queryInput = document.querySelector("#query");
const searchButton = document.querySelector("#search-button");
const resetButton = document.querySelector("#reset-button");
const resultArea = document.querySelector("#result-area");

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
    emptyState("לא נמצאו השלמות", "נסה להקליד ביטוי אחר או פחות תווים.");
    return;
  }

  const cards = data.suggestions.map((item) => `
    <article class="result-card">
      <div class="result-top">
        <p class="sentence">${escapeHtml(item.completed_sentence)}</p>
        <span class="score">score ${item.score}</span>
      </div>
      <div class="source">${escapeHtml(item.source_text)}:${item.offset}</div>
    </article>`).join("");

  resultArea.innerHTML = `
    <div class="results-header">
      <h2>${data.suggestions.length} הצעות נמצאו</h2>
      <span class="timing">${data.elapsed_ms} ms</span>
    </div>
    <div class="result-list">${cards}</div>`;
}

async function search() {
  const query = queryInput.value;
  if (!query.trim()) {
    emptyState("מוכן לחיפוש", "הקלד טקסט ולחץ Enter.");
    return;
  }
  if (query === "#") {
    queryInput.value = "";
    emptyState("החיפוש אופס", "אפשר להתחיל חיפוש חדש.");
    queryInput.focus();
    return;
  }

  searchButton.disabled = true;
  searchButton.textContent = "מחפש...";
  try {
    const response = await fetch("/api/suggestions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "החיפוש נכשל.");
    renderResults(data);
  } catch (error) {
    resultArea.innerHTML = `<div class="error">${escapeHtml(error.message)}</div>`;
  } finally {
    searchButton.disabled = false;
    searchButton.textContent = "חיפוש";
  }
}

queryInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") search();
});
searchButton.addEventListener("click", search);
resetButton.addEventListener("click", () => {
  queryInput.value = "";
  emptyState("החיפוש אופס", "אפשר להתחיל חיפוש חדש.");
  queryInput.focus();
});
