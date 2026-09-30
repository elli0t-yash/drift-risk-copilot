# Drift Risk Copilot — Frontend Integration Guide
# This file is instructions for Claude Code integrating the 
# frontend with the Drift Risk Copilot backend API.

## Base URL
LIVE: https://drift-risk-copilot-99506253437.asia-south1.run.app
LOCAL: http://localhost:8080 (for local backend dev)

Set as an environment variable: NEXT_PUBLIC_API_BASE_URL
Fall back to the live URL if not set.

## CORS
The backend allows all origins. No proxy needed.

---

## Endpoints

### GET /health
Returns: { "status": "ok", "version": "0.1.0" }
Use on page load to show the "● API Live" / "● API Offline" 
indicator in the navbar.

---

### GET /scenarios
Returns the 3 historical scenario presets.
Response shape:
[
  {
    "id": "covid_crash",
    "name": "COVID Crash (Mar 2020)",
    "date_range": "Feb 19 – Mar 23, 2020",
    "description": "Nifty fell 38% in 33 days...",
    "shocks_pct": { "MARKET": -38.0, "BRENT": -55.0, ... },
    "propagate": false
  },
  { "id": "ilfs_contagion", ... },
  { "id": "taper_tantrum_2013", ... }
]
Use to populate the three scenario shortcut buttons.
When a scenario button is clicked, pre-fill the chat input 
with: "What would the [scenario.name] have done to my portfolio?"
and submit immediately.

---

### POST /portfolio/upload
Upload a CSV or XLSX file to auto-populate the portfolio builder.
Request: multipart/form-data, field name "file"

CSV format accepted (two layouts):
  Layout A — weight-based:
    ticker,weight
    RELIANCE.NS,0.10
    HDFCBANK.NS,0.10

  Layout B — value-based:
    ticker,shares,avg_price_inr
    RELIANCE.NS,10,2850.00

Response (200):
{
  "portfolio": {
    "holdings": [
      { "ticker": "RELIANCE.NS", "weight": 0.10 },
      ...
    ],
    "total_value_inr": 10000000.0
  },
  "layout_detected": "weight",
  "tickers_normalised": [],
  "row_count": 10
}
On error: 400 with { "error": "...", "code": "..." }

After a successful upload, populate the portfolio builder 
with the returned holdings and total_value_inr.

---

### GET /auth/upstox/status
Returns: { "configured": true }
Call on page load. If configured is false, hide the 
"Connect with Upstox" button entirely.

### GET /auth/upstox/login
Redirect the user's browser to this URL directly:
  window.location.href = `${API_BASE_URL}/auth/upstox/login`
The backend handles the OAuth redirect to Upstox.
Do NOT fetch this as an API call — it must be a full 
browser navigation.

### GET /auth/upstox/callback
The backend handles this — the frontend never calls it 
directly. After Upstox redirects back, the backend 
returns JSON directly to the browser:
{
  "portfolio": { "holdings": [...], "total_value_inr": ... },
  "holdings_count": 10,
  "data_as_of": "2026-09-28T00:00:00Z"
}
To handle this: on page load, check if the current URL 
path is /auth/upstox/callback. If it is, read the 
response body (the backend already rendered it as JSON), 
parse it, populate the portfolio builder, then redirect 
to / using window.history.replaceState.

---

### POST /ask  ← MOST IMPORTANT ENDPOINT
This is the core endpoint. Call it on every user message.

Request body:
{
  "portfolio": {
    "holdings": [
      { "ticker": "RELIANCE.NS", "weight": 0.10 },
      { "ticker": "HDFCBANK.NS", "weight": 0.10 }
      // ... at least 2 holdings
    ],
    "total_value_inr": 10000000.0
  },
  "message": "What would a Nifty crash do to my portfolio?",
  "conversation_history": [
    // All prior turns in this session
    // Start with [] on first message
    // Append each turn after a successful response
    { "role": "user", "content": "Prior user message" },
    { "role": "assistant", "content": "Prior AI response" }
  ]
}

Response (200):
{
  "experiment": { "type": "FactorShock", ... },
  "trace": { ... },  // full EvidenceTrace — use for PDF
  "narration": "Your portfolio would lose approximately ₹20.6L...",
  "grounding_warnings": [],  // empty = all good
  "result_id": "uuid-for-pdf-download",
  "assistant_turn": {
    "role": "assistant",
    "content": "Your portfolio would lose approximately ₹20.6L..."
  },
  "suggestion": "Run a stress test for the IL&FS scenario",
  "agent_execution_trace": {
    "id": "uuid",
    "tool_plans": [
      { "tool": "factor_shock", "reason": "..." }
    ],
    "gemini_calls": 3,
    "total_latency_ms": 7200
  }
}

After a successful response:
1. Append to conversation_history:
   { "role": "user", "content": <the message you sent> }
   then response.assistant_turn
2. Render response.narration as the AI message in chat
3. Show response.suggestion as a tappable chip below 
   the AI message — clicking it submits that text as 
   the next message
4. If response.grounding_warnings.length > 0, show a 
   yellow banner: "⚠ Some numbers could not be verified"
5. Store response.result_id for the PDF download button
6. Show regime badge from:
   response.trace.model_params.regime_state.current_label
   "Bull" → green, "Bear" → amber, "Crisis" → red

On error 422: show in chat as a soft grey message:
  "I can only help with portfolio risk questions."
On error 5xx: show red banner "Something went wrong. Try again."
Do NOT append failed turns to conversation_history.

---

### GET /report/{result_id}
Download a PDF report for a completed /ask result.
result_id comes from the /ask response.

Call as:
  const response = await fetch(`${API_BASE_URL}/report/${resultId}`)
  const blob = await response.blob()
  const url = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = url
  a.download = `drift-report-${experimentType}-${Date.now()}.pdf`
  a.click()

Show a "↓ PDF" icon button on each AI message that has 
a result_id. Only show it after the response is received, 
not while loading.

---

### GET /execution-trace/{id}
Returns the full agent execution trace for an /ask result.
id = response.agent_execution_trace.id from /ask response.

Optional — show a "View Trace" link in a developer/debug 
panel. Not needed for the main demo flow.

---

### POST /experiment  ← Direct experiment, no AI narration
Use only for the "Direct" mode if you implement it.
Request:
{
  "portfolio": { ... },
  "experiment": {
    "type": "RiskDecomposition"
    // or FactorShock, CvarRebalance, 
    // ReverseStress, PolicyCheck, 
    // RiskDrift, PortfolioPerformance
  }
}
Returns the raw EvidenceTrace. No narration, no suggestion.
Most users should use /ask instead.

---

## Portfolio validation rules
Enforce these client-side before any API call:
- At least 2 holdings
- All weights > 0
- Weights sum to 1.0 ± 0.01 (show live indicator)
- total_value_inr > 0

Show inline errors — do not submit if validation fails.

---

## Default portfolio (pre-load on first visit)
{
  "holdings": [
    { "ticker": "RELIANCE.NS", "weight": 0.10 },
    { "ticker": "HDFCBANK.NS", "weight": 0.10 },
    { "ticker": "ICICIBANK.NS", "weight": 0.10 },
    { "ticker": "INFY.NS",     "weight": 0.10 },
    { "ticker": "TCS.NS",      "weight": 0.10 },
    { "ticker": "LT.NS",       "weight": 0.10 },
    { "ticker": "ITC.NS",      "weight": 0.10 },
    { "ticker": "KOTAKBANK.NS","weight": 0.10 },
    { "ticker": "BHARTIARTL.NS","weight": 0.10 },
    { "ticker": "TMPV.NS",     "weight": 0.10 }
  ],
  "total_value_inr": 10000000.0
}

---

## Conversation history management
// Initialise
let conversationHistory = []

// On each successful /ask response:
conversationHistory.push(
  { role: "user", content: userMessage },
  response.assistant_turn  // { role: "assistant", content: "..." }
)

// Pass on every /ask call:
body.conversation_history = conversationHistory

// On "Clear conversation" button:
conversationHistory = []

// Keep the last 20 turns max to avoid token bloat:
if (conversationHistory.length > 20) {
  conversationHistory = conversationHistory.slice(-20)
}

---

## Regime badge
Source: response.trace.model_params.regime_state.current_label
Values: "Bull" | "Bear" | "Crisis"
Show as a small pill in the chat header, updated after 
each /ask response.
Colours: Bull #10B981, Bear #F59E0B, Crisis #EF4444

---

## Loading states
- Disable send button and scenario buttons during a request
- Show a pulsing indicator in the chat area
- Expected response time: 5–10 seconds (Gemini calls)
- Do not show a timeout error until 60 seconds

---

## Key things that will break if done wrong
1. Not sending conversation_history — the AI loses context 
   and "what should I do about it?" won't work
2. Not appending assistant_turn after each response — 
   same problem
3. Fetching /auth/upstox/login instead of redirecting — 
   OAuth will fail silently
4. Calling /report/{id} before result_id exists — 
   will 404
5. Submitting with weights not summing to 1.0 — 
   will 400
6. Sending fewer than 2 holdings — will 400

---

## Testing the integration
Before building any UI, verify the API is live:
  curl https://drift-risk-copilot-99506253437.asia-south1.run.app/health
Expected: {"status":"ok","version":"0.1.0"}

Then verify /ask works:
  curl -X POST \
    https://drift-risk-copilot-99506253437.asia-south1.run.app/ask \
    -H "Content-Type: application/json" \
    -d '{
      "portfolio": {
        "holdings": [
          {"ticker":"RELIANCE.NS","weight":0.5},
          {"ticker":"HDFCBANK.NS","weight":0.5}
        ],
        "total_value_inr": 1000000
      },
      "message": "What is my risk?",
      "conversation_history": []
    }'
Expected: 200 with narration field non-empty.
