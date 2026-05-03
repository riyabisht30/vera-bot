# Vera Bot — magicpin AI Challenge Submission

## Approach

Single-prompt composer using Llama 3.3 70B (Groq) at temperature=0 with trigger-kind dispatch. Each trigger kind (`research_digest`, `perf_dip`, `recall_due`, `festival_upcoming`, `dormant_with_vera`, etc.) gets a specialised `PRIORITY` instruction prepended to a shared base system prompt that encodes all 5 scoring dimensions. All four context layers (category, merchant, trigger, customer) are injected into every call.

## Key design decisions

- **Dispatch layer** routes by `trigger.kind` before calling LLM, keeping prompts focused and trigger-relevant rather than generic
- **Specificity enforcement** — system prompt requires a real number from context in the first 8 words; scored examples (10/10 vs 0/10) are shown inline so the model calibrates correctly
- **Language detection** — checks `merchant.identity.languages`; injects Hinglish instruction for Hindi-speaking merchants to avoid the 2-point merchant-fit penalty for pure English
- **Post-LLM validation** rejects URLs, multiple CTAs, and messages >400 chars with automatic re-prompt before returning the result
- **Anti-repetition** — last message from `conversation_history` is surfaced in the prompt with an explicit "do NOT repeat this angle" instruction
- **Auto-reply detection** uses phrase matching (turn 1: nudge owner, turn 2: wait 4h, turn 3: end)
- **Intent transition detection** catches "yes/confirm/haan/kar do" at turn ≥ 2 to switch from qualify→execute mode, injecting `INTENT DETECTED` instruction to prevent further qualifying questions
- **Out-of-scope deflection** catches unrelated asks (GST, legal, medical advice) and redirects back to the conversation topic
- **Suppression keys** follow format `<trigger_kind>::<merchant_id>::<week>` for weekly dedup; tracked in-memory per session

## Tradeoffs

- In-memory state means a server restart loses context; acceptable for the test window but would need a persistent store (Redis/Postgres) in production
- Single LLM call per `compose()` without retrieval; adding digest-item embedding retrieval would improve specificity on `research_digest` triggers by surfacing the most relevant clinical finding rather than the whole digest
- `llama-3.1-8b-instant` used as runtime model for latency (sub-3s per call, fits within judge's 15s tick timeout); `llama-3.3-70b-versatile` produces higher-quality messages but exceeds the free-tier daily token budget at scale

## What would help most with more context

- **Real merchant conversation history at scale** — to calibrate when NOT to send (merchants who always ignore Vera need a different strategy, not more messages)
- **Peer benchmark data by city + locality** — CTR peer medians vary significantly by micro-market, not just category; locality-level comparison ("3 dentists in Lajpat Nagar average 28 calls/week") would sharpen social-proof messages considerably
- **Actual slot availability per merchant** — real booking calendar data would unlock precise recall/appointment messages instead of placeholder slot times
- **Merchant's Google post history** — knowing what was posted (not just when) would let the bot suggest genuinely new content angles

## Running the bot

```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here
uvicorn bot:app --host 0.0.0.0 --port 8080
```

## Generating submission.jsonl

```bash
# Generate expanded dataset (run from challenge/ folder)
python dataset/generate_dataset.py --seed-dir dataset --out expanded

# Compose all 30 test pairs
cd ../vera-bot && python generate_submission.py
```

## Live deployment

Bot is deployed at: `https://web-production-7c1f0.up.railway.app`

Healthcheck: `GET /v1/healthz` → `{"status": "ok", ...}`
