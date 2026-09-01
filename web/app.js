const queryInput = document.querySelector("#query");
const searchButton = document.querySelector("#search-button");
const voiceButton = document.querySelector("#voice-button");
const voiceStatus = document.querySelector("#voice-status");
const resetButton = document.querySelector("#reset-button");
const resultArea = document.querySelector("#result-area");
const diagnosticsPanel = document.querySelector("#diagnostics-panel");
const diagnosticsValues = document.querySelector("#diagnostics-values");
const indexFileInput = document.querySelector("#index-file");
const indexMessage = document.querySelector("#index-message");

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;
let isListening = false;
let queryBeforeVoiceInput = "";
let finalTranscript = "";

function setVoiceStatus(message) {
  voiceStatus.textContent = message;
  voiceStatus.hidden = !message;
}

function setListening(listening) {
  isListening = listening;
  voiceButton.setAttribute("aria-pressed", String(listening));
  voiceButton.setAttribute("aria-label", listening ? "Stop voice typing" : "Start voice typing");
  voiceButton.title = listening ? "Stop voice typing" : "Start voice typing";
  voiceButton.classList.toggle("is-listening", listening);
  voiceButton.textContent = listening ? "■" : "🎙";
}

function voiceErrorMessage(error) {
  const messages = {
    "not-allowed": "Microphone permission was denied. Allow it in your browser and try again.",
    "service-not-allowed": "Speech recognition is unavailable in this browser.",
    "no-speech": "No speech was detected. Try again.",
    "audio-capture": "No microphone was found.",
    network: "The browser's speech service is unavailable. Check your connection and try again.",
    "language-not-supported": "Your browser does not support voice typing in the selected language.",
  };
  return messages[error] || "Voice typing stopped unexpectedly. Try again.";
}

function selectedVoiceLanguage() {
  const preferredLanguages = navigator.languages || [navigator.language || "en-US"];
  return preferredLanguages.some((language) => language.toLowerCase().startsWith("he"))
    ? "he-IL"
    : "en-US";
}

function containsHebrew(text) {
  return /[\u0590-\u05ff]/u.test(text);
}

async function translateHebrewToEnglish(text) {
  const response = await fetch("/api/translate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Hebrew translation failed.");
  return data.translated_text;
}

function finishVoiceInput(transcript) {
  queryInput.value = `${queryBeforeVoiceInput}${transcript}`.trimStart();
  setVoiceStatus("");
  search();
}

function setupVoiceTyping() {
  if (!SpeechRecognition) {
    voiceButton.hidden = true;
    return;
  }

  recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = true;
  recognition.maxAlternatives = 1;

  recognition.onstart = () => {
    setListening(true);
    setVoiceStatus("Listening… speak your search phrase.");
  };

  recognition.onresult = (event) => {
    let interimTranscript = "";
    for (let index = event.resultIndex; index < event.results.length; index += 1) {
      const transcript = event.results[index][0].transcript;
      if (event.results[index].isFinal) {
        finalTranscript += transcript;
      } else {
        interimTranscript += transcript;
      }
    }
    queryInput.value = `${queryBeforeVoiceInput}${finalTranscript}${interimTranscript}`.trimStart();
    if (finalTranscript) setVoiceStatus("Voice input received. Searching…");
  };

  recognition.onerror = (event) => {
    setVoiceStatus(voiceErrorMessage(event.error));
  };

  recognition.onend = () => {
    const transcript = finalTranscript.trim();
    setListening(false);
    if (transcript) {
      finishVoiceInput(transcript);
    } else if (!voiceStatus.textContent) {
      setVoiceStatus("Voice typing stopped.");
    }
    finalTranscript = "";
  };

  voiceButton.addEventListener("click", () => {
    if (isListening) {
      recognition.stop();
      return;
    }
    queryBeforeVoiceInput = queryInput.value.trim() ? `${queryInput.value.trim()} ` : "";
    finalTranscript = "";
    recognition.lang = selectedVoiceLanguage();
    try {
      recognition.start();
    } catch (_error) {
      setVoiceStatus("Voice typing is already starting. Please wait.");
    }
  });
}

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

  const correctedTextBySuggestion = new Map(
    (data.diagnostics.selected_corrections || []).map((detail) => [
      detail.suggestion_number,
      detail.matched_text,
    ])
  );
  const cards = data.suggestions.map((item, index) => `
    <article class="result-card">
      <div class="result-top">
        <p class="sentence">${highlightMatch(
          item.completed_sentence,
          correctedTextBySuggestion.get(index + 1) || data.diagnostics.normalized_query
        )}</p>
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

function highlightMatch(text, normalizedQuery) {
  const characters = Array.from(text);
  const normalized = [];
  const sourcePositions = [];
  let previousWasSpace = false;
  characters.forEach((character, index) => {
    if (/^[\p{L}\p{N}]$/u.test(character)) {
      normalized.push(character.toLowerCase());
      sourcePositions.push(index);
      previousWasSpace = false;
    } else if (/^\s$/u.test(character) && normalized.length && !previousWasSpace) {
      normalized.push(" ");
      sourcePositions.push(index);
      previousWasSpace = true;
    }
  });
  while (normalized.at(-1) === " ") {
    normalized.pop();
    sourcePositions.pop();
  }
  const start = normalized.join("").indexOf(normalizedQuery);
  if (start < 0) return escapeHtml(text);

  const firstCharacter = sourcePositions[start];
  const lastCharacter = sourcePositions[start + normalizedQuery.length - 1];
  return `${escapeHtml(characters.slice(0, firstCharacter).join(""))}<strong class="match-highlight">${escapeHtml(characters.slice(firstCharacter, lastCharacter + 1).join(""))}</strong>${escapeHtml(characters.slice(lastCharacter + 1).join(""))}`;
}

function renderDiagnostics(data) {
  if (!data) {
    diagnosticsPanel.hidden = true;
    return;
  }

  diagnosticsValues.innerHTML = (data.log_story || []).map((message, index) => {
    if (index === 3 && Object.keys(data.correction_trace || {}).length) {
      return `<li class="log-step-with-details">
        <span>${escapeHtml(message)}</span>
        <button class="expand-button" type="button" aria-expanded="false">Expand</button>
        ${renderCorrectionTrace(data.correction_trace)}
      </li>`;
    }
    return `<li>${escapeHtml(message)}</li>`;
  }).join("");
  diagnosticsPanel.hidden = false;
}

function renderCorrectionTrace(trace) {
  const removeExtra = trace.remove_extra || [];
  const replace = trace.replace || [];
  const addMissing = trace.add_missing || [];
  const list = (items, sentence) => `<ul>${items.map((item) =>
    `<li>${escapeHtml(sentence(item))}</li>`
  ).join("")}</ul>`;
  return `<div class="correction-trace" hidden>
    <p><code>?</code> means one character from the candidate sentence.</p>
    <h4>Remove an extra typed character</h4>
    ${list(removeExtra, (item) => `Remove “${item.character}” at character ${item.position}: “${item.pattern}”`)}
    <h4>Replace one typed character</h4>
    ${list(replace, (item) => `Character ${item.position}: “${item.pattern}”`)}
    <h4>Add one missing character</h4>
    ${list(addMissing, (item) => `Character ${item.position}: “${item.pattern}”`)}
  </div>`;
}

async function search() {
  let query = queryInput.value;
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
    if (containsHebrew(query)) {
      setVoiceStatus("Translating Hebrew to English…");
      query = await translateHebrewToEnglish(query);
      queryInput.value = query;
      setVoiceStatus("");
    }
    const response = await fetch("/api/suggestions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
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

diagnosticsValues.addEventListener("click", (event) => {
  const button = event.target.closest(".expand-button");
  if (!button) return;
  const details = button.parentElement.querySelector(".correction-trace");
  const isExpanded = button.getAttribute("aria-expanded") === "true";
  button.setAttribute("aria-expanded", String(!isExpanded));
  button.textContent = isExpanded ? "Expand" : "Collapse";
  details.hidden = isExpanded;
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
    renderActiveIndex(data);
  } catch (error) {
    indexMessage.textContent = error.message;
  } finally {
    indexFileInput.value = "";
  }
});

function renderActiveIndex(data) {
  indexMessage.textContent = `Active index: ${data.index_name} · ${Number(data.indexed_sentence_count).toLocaleString()} lines`;
}

async function loadActiveIndex() {
  try {
    const response = await fetch("/api/status");
    if (!response.ok) throw new Error("Index status is unavailable.");
    renderActiveIndex(await response.json());
  } catch (error) {
    indexMessage.textContent = "Active index status is unavailable. Run the site with web_app.py.";
  }
}

loadActiveIndex();
setupVoiceTyping();
