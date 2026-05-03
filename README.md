# Vera Bot — magicpin AI Challenge Submission

## Approach

**Single LLM composer (Llama 3.3 70B via Groq) with trigger-kind dispatch and 4-context injection.**

Every outbound message is produced by feeding all four context layers (category, merchant, trigger, customer) into a single Gemini call with a carefully tuned system prompt. The prompt enforces all scoring dimensions explicitly:

- **Specificity** — system prompt requires citing the exact number/date/source from context; fabrication is a hard-banned instruction.
- **Category fit** — category voice rules (`vocab_allowed`, `vocab_taboo`, tone) are injected verbatim; each trigger kind gets a framing hint (e.g., `research_digest` → "clinical peer tone, source citation at end").
- **Merchant fit** — merchant's `owner_first_name`, performance numbers, active offers, signals, and `customer_aggregate` fields are all passed; the prompt instructs the model to anchor on the merchant's specific state.
- **Trigger relevance** — the trigger object (including payload, urgency, kind) is passed in full; the framing hint for each `kind` instructs *why now*.
- **Engagement compulsion** — 8 compulsion levers are listed in the system prompt; the model is instructed to use 1-2 per message with a single, end-of-message CTA.

## Multi-turn conversation handling

| Signal | Bot action |
|---|---|
| Auto-reply detected (keyword match) | Send one "looks like auto-reply, reply YES" prompt |
| Same auto-reply twice | Wait 24 hours |
| Auto-reply 3× in a row | End conversation |
| Merchant says YES / "let's do it" | Immediately switch to action mode (no more qualifying) |
| Merchant says "not interested" / "stop" | End + suppress |
| Off-topic question (e.g., GST filing) | Politely decline, redirect to original topic |

## Hard constraints enforced

- URLs stripped from all outbound bodies (Meta policy, -3 penalty)
- Suppression keys tracked in-memory — no duplicate sends for the same event
- Anti-repetition guard — same body never sent twice in one conversation
- Temperature = 0 for deterministic output

## What additional context would have helped most

1. **Real slot availability per merchant** — the dataset has `available_slots` in some triggers but not all; real booking calendar data would unlock much stronger recall/appointment messages.
2. **Merchant's actual Google post history** — knowing *what* was posted (not just *when*) would let the bot avoid repetition and suggest genuinely new content.
3. **Per-merchant language preference from conversation history** — the seed data shows languages `["en", "hi"]` for most merchants but production Vera would know which language the merchant *actually* replies in.
4. **Category peer benchmarks at locality level** — the current `peer_stats` is city-scoped; locality-level comparison ("3 dentists in Lajpat Nagar") would dramatically sharpen social-proof messages.

## Running the bot

```bash
pip install -r requirements.txt
uvicorn bot:app --host 0.0.0.0 --port 8080
```

Set `GROQ_API_KEY` environment variable (or edit the default in `bot.py`).

## Generating submission.jsonl

```bash
# First generate expanded dataset (from challenge folder)
cd ../challenge && python generate_dataset.py --out ./expanded

# Then compose the 30 test pairs
cd ../vera-bot && python generate_submission.py
```
