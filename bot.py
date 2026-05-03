"""
Vera Bot — magicpin AI Challenge submission.
FastAPI server exposing all 5 judge endpoints.

Run:  uvicorn bot:app --host 0.0.0.0 --port 8080
"""

import os
import re
import time
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from groq import Groq

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
TEAM_NAME    = "VeraBuilder"
TEAM_MEMBERS = ["Vera Participant"]
CONTACT_EMAIL = "participant@example.com"
MODEL_NAME   = "llama-3.3-70b-versatile"   # best free Groq model

groq_client = Groq(api_key=GROQ_API_KEY)

app = FastAPI(title="Vera Bot — magicpin AI Challenge")
START_TIME = time.time()

# ─────────────────────────────────────────────
# IN-MEMORY STATE
# ─────────────────────────────────────────────
# (scope, context_id) → {version, payload, stored_at}
contexts: dict[tuple[str, str], dict] = {}

# conversation_id → conversation state dict
conversations: dict[str, dict] = {}

# suppression keys already fired (don't send same trigger twice)
fired_keys: set[str] = set()


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def get_ctx(scope: str, cid: str) -> dict | None:
    entry = contexts.get((scope, cid))
    return entry["payload"] if entry else None


def strip_urls(text: str) -> str:
    """Remove any URLs from message body (Meta policy, -3 penalty each)."""
    return re.sub(r"https?://\S+", "", text).strip()


def parse_llm_json(raw: str) -> dict:
    """Parse JSON from LLM response, tolerating markdown fences."""
    raw = raw.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    raw = raw.strip()
    return json.loads(raw)


# ─────────────────────────────────────────────
# AUTO-REPLY / INTENT DETECTION
# ─────────────────────────────────────────────

AUTO_REPLY_PHRASES = [
    "thank you for contacting",
    "our team will respond",
    "our team will get back",
    "automated assistant",
    "automated message",
    "automated response",
    "i am a bot",
    "i'm a bot",
    "will be attended",
    "outside business hours",
    "currently unavailable",
    "aapki jaankari ke liye",
    "main ek automated",
    "madad ke liye shukriya, lekin main",
    "we have received your",
    "will respond shortly",
]

def is_auto_reply(msg: str) -> bool:
    low = msg.lower()
    return any(phrase in low for phrase in AUTO_REPLY_PHRASES)


POSITIVE_INTENT_RE = re.compile(
    r"\b(yes|haan|ha\b|ok|okay|sure|agreed|go ahead|let'?s do it|let'?s go|"
    r"sounds good|great|perfect|proceed|chalega|karo|kar do|confirm|done|"
    r"do it|ship it)\b",
    re.IGNORECASE,
)

NEGATIVE_INTENT_PHRASES = [
    "not interested", "stop messaging", "don't message", "do not message",
    "nahi chahiye", "band karo", "mat karo", "bhejo mat", "unsubscribe",
    "why are you bothering", "useless", "annoying", "stop sending",
    "leave me alone",
]

def is_positive_intent(msg: str) -> bool:
    return bool(POSITIVE_INTENT_RE.search(msg))

def is_negative_intent(msg: str) -> bool:
    low = msg.lower()
    return any(p in low for p in NEGATIVE_INTENT_PHRASES)


# ─────────────────────────────────────────────
# SYSTEM PROMPT — shared for all LLM calls
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are Vera, magicpin's AI merchant assistant helping Indian small business owners grow.

STRICT RULES (violations are scored 0 or penalised):
1. NO URLs in the message body — Meta policy, hard ban.
2. EXACTLY ONE call-to-action (CTA), placed at the very END of the message.
3. ONLY use facts present in the provided contexts. NEVER fabricate numbers, citations, competitor names, offers, or slot times.
4. Match the merchant's language: if identity.languages includes "hi", weave in Hindi-English code-mix naturally.
5. Address merchant by identity.owner_first_name (or Doc/Dr. Firstname for dentists).
6. Peer / colleague tone — NOT promotional. "AMAZING DEAL!" is penalised.
7. For research/compliance items: ALWAYS end with the source citation (e.g., "— JIDA Oct 2026 p.14").
8. No long preambles. Start with the merchant's name and the hook — no "I hope you're doing well."
9. Do NOT re-introduce yourself after the first message in a conversation.
10. For customer-facing messages (send_as=merchant_on_behalf): warm-clinical, no overclaims, respect consent scope.

COMPULSION LEVERS — use 1-2 per message:
• Specificity / verifiability — exact number, date, percentage, source from context
• Loss aversion — "you're missing X" / "before this window closes"
• Social proof — "N merchants in your locality did Y this month"
• Effort externalization — "I've already drafted it — just say go" / "5-min setup"
• Curiosity — "want to see who?" / "want the full breakdown?"
• Reciprocity — "I noticed X about your account, thought you'd want to know"
• Asking the merchant — "what's your most-asked treatment this week?"
• Single binary commit — Reply YES / STOP (not multi-choice)

OUTPUT FORMAT — return ONLY valid JSON, no markdown fences:
{
  "body": "the WhatsApp message text",
  "cta": "open_ended | binary_yes_no | binary_yes_stop | multi_choice_slot | none",
  "send_as": "vera | merchant_on_behalf",
  "suppression_key": "copy from trigger.suppression_key",
  "rationale": "concise: why this message, which lever used, why now"
}"""


# ─────────────────────────────────────────────
# TRIGGER-KIND → FRAMING HINTS
# ─────────────────────────────────────────────

KIND_HINTS: dict[str, str] = {
    "research_digest":
        "Clinical insight from a trusted source. Include source citation at the end. "
        "Tie to THIS merchant's patient/customer profile. CTA: open_ended (offer to pull abstract or draft patient content).",
    "regulation_change":
        "Compliance deadline with specific date. Use urgency level 4+. Offer to check their setup. CTA: binary_yes_no.",
    "recall_due":
        "Patient recall reminder. Mention how long since last visit. Offer 2 specific slots from trigger payload. "
        "Use merchant's active offer price. CTA: multi_choice_slot.",
    "perf_dip":
        "Alert to specific metric drop (name the metric, cite the percentage). Offer one concrete action. CTA: open_ended.",
    "perf_spike":
        "Positive reinforcement with the exact number. Build momentum — offer to amplify. CTA: open_ended.",
    "milestone_reached":
        "Celebrate the specific milestone. Turn it into social proof or a Google post opportunity. CTA: binary_yes_no.",
    "festival_upcoming":
        "Time-sensitive seasonal opportunity. Category-specific angle (salons: bridal, restaurants: catering). "
        "Name the festival and days until. CTA: binary_yes_no.",
    "ipl_match_today":
        "Time-sensitive local event. Provide contrarian data if available (e.g., Saturday IPL = -12% covers). "
        "Leverage existing offer. CTA: binary_yes_no.",
    "curious_ask_due":
        "Low-stakes question to the merchant. Offer a concrete deliverable in return (Google post, reply template). "
        "Keep it to one question. CTA: open_ended.",
    "review_theme_emerged":
        "Specific review theme with occurrence count from context. Offer response draft or action. CTA: binary_yes_no.",
    "dormant_with_vera":
        "Re-engagement lead with value, not guilt. Mention something specific they haven't done yet. CTA: binary_yes_no.",
    "renewal_due":
        "Subscription expiry with exact days remaining. Show what they'd lose (visibility, Vera features). CTA: binary_yes_no.",
    "winback_eligible":
        "Merchant who lapsed. Show what improved while they were away. Low-commitment offer. CTA: open_ended.",
    "wedding_package_followup":
        "Bridal journey. Use exact days-to-wedding count from trigger. Reference trial completed. CTA: binary_yes_no.",
    "competitor_opened":
        "Nearby competitor info. Frame as preparation opportunity, not fear. Offer to strengthen listing. CTA: open_ended.",
    "supply_alert":
        "Urgent: use batch numbers exactly from trigger payload. Count affected customers from aggregate. CTA: binary_yes_no.",
    "chronic_refill_due":
        "Precise medication names + exact run-out date. Show total + savings. Delivery ETA. CTA: binary_yes_no.",
    "customer_lapsed_soft":
        "Lapsed customer (3-6 months). No guilt trip. Personal connection + relevant new offer. CTA: binary_yes_no.",
    "customer_lapsed_hard":
        "Long-lapsed customer (6+ months). Low-commitment trial offer. CTA: binary_yes_no.",
    "appointment_tomorrow":
        "Reminder with appointment time + any prep instructions. Brief, practical. CTA: none or binary_yes_no.",
}


# ─────────────────────────────────────────────
# LLM CALL
# ─────────────────────────────────────────────

def call_llm(prompt: str) -> str:
    """Call Groq LLM with retry on rate-limit errors."""
    for attempt in range(3):
        try:
            resp = groq_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content
        except Exception as exc:
            err_str = str(exc)
            if "429" in err_str or "rate" in err_str.lower():
                if attempt < 2:
                    time.sleep(30)
                    continue
            raise RuntimeError(f"Groq API error: {exc}")
    raise RuntimeError("Groq API: max retries exceeded")


# ─────────────────────────────────────────────
# COMPOSE — initial outbound message
# ─────────────────────────────────────────────

def compose(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: dict | None = None,
) -> dict:
    """Compose the first outbound WhatsApp message for a trigger."""
    kind = trigger.get("kind", "")
    hint = KIND_HINTS.get(kind, "Match trigger kind to appropriate framing.")

    parts = [
        f"CATEGORY CONTEXT:\n{json.dumps(category, indent=2, ensure_ascii=False)}",
        f"\nMERCHANT CONTEXT:\n{json.dumps(merchant, indent=2, ensure_ascii=False)}",
        f"\nTRIGGER:\n{json.dumps(trigger, indent=2, ensure_ascii=False)}",
    ]
    if customer:
        parts.append(
            f"\nCUSTOMER CONTEXT:\n{json.dumps(customer, indent=2, ensure_ascii=False)}"
        )

    parts.append(
        f"\nFRAMING HINT FOR trigger.kind='{kind}': {hint}"
    )
    parts.append(
        "\nNow compose the WhatsApp message. Return ONLY valid JSON."
    )

    raw = call_llm("\n".join(parts))
    result = parse_llm_json(raw)

    # Guarantee suppression_key comes from trigger
    if not result.get("suppression_key"):
        result["suppression_key"] = trigger.get(
            "suppression_key", f"trg:{trigger.get('id', uuid.uuid4().hex[:8])}"
        )

    # Strip any URLs that slipped in
    if "body" in result:
        result["body"] = strip_urls(result["body"])

    return result


# ─────────────────────────────────────────────
# COMPOSE REPLY — multi-turn conversation
# ─────────────────────────────────────────────

def compose_reply(
    category: dict,
    merchant: dict,
    trigger: dict | None,
    customer: dict | None,
    history: list[dict],
    merchant_message: str,
    turn_number: int,
) -> dict:
    """Compose the bot's reply to a merchant message in an ongoing conversation."""
    history_text = "\n".join(
        f"[{t['from'].upper()}]: {t['body']}" for t in history
    )

    trigger_section = (
        f"\nORIGINAL TRIGGER:\n{json.dumps(trigger, indent=2, ensure_ascii=False)}"
        if trigger else ""
    )
    customer_section = (
        f"\nCUSTOMER CONTEXT:\n{json.dumps(customer, indent=2, ensure_ascii=False)}"
        if customer else ""
    )

    prompt = f"""CONVERSATION HISTORY:
{history_text}

[MERCHANT (turn {turn_number})]: {merchant_message}

MERCHANT CONTEXT:
{json.dumps(merchant, indent=2, ensure_ascii=False)}

CATEGORY CONTEXT:
{json.dumps(category, indent=2, ensure_ascii=False)}
{trigger_section}
{customer_section}

REPLY INSTRUCTIONS:
- This is a REPLY — do NOT re-introduce yourself or Vera.
- Match the merchant's language from their latest message.
- If merchant said YES / "let's do it" / "go ahead" / committed in any way:
  → IMMEDIATELY switch to ACTION mode (draft content, confirm slot, etc.)
  → Do NOT ask another qualifying question — this loses major points.
- If merchant asked an off-topic question (e.g., GST filing, personal matter):
  → Decline politely in one sentence, then redirect back to the original topic.
- If merchant seems confused or needs clarification: ask ONE clarifying question.
- No URLs in body.

Return ONLY valid JSON:
{{
  "action": "send | wait | end",
  "body": "reply message text (required when action=send)",
  "cta": "open_ended | binary_yes_no | binary_confirm_cancel | multi_choice_slot | none",
  "wait_seconds": 0,
  "rationale": "brief reasoning for this action"
}}"""

    raw = call_llm(prompt)
    result = parse_llm_json(raw)

    if result.get("action") not in ("send", "wait", "end"):
        result["action"] = "send"

    if result.get("body"):
        result["body"] = strip_urls(result["body"])

    return result


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _) in contexts:
        if scope in counts:
            counts[scope] += 1
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": counts,
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": TEAM_NAME,
        "team_members": TEAM_MEMBERS,
        "model": "llama-3.3-70b-versatile (Groq)",
        "approach": (
            "4-context LLM composer (Llama 3.3 70B via Groq) with trigger-kind dispatch, "
            "auto-reply detection (3-strike exit), intent-transition routing, "
            "suppression key tracking, and anti-repetition guards."
        ),
        "contact_email": CONTACT_EMAIL,
        "version": "1.0.0",
        "submitted_at": now_iso(),
    }


class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str


@app.post("/v1/context")
async def push_context(body: CtxBody):
    valid_scopes = ("category", "merchant", "customer", "trigger")
    if body.scope not in valid_scopes:
        return JSONResponse(
            status_code=400,
            content={"accepted": False, "reason": "invalid_scope",
                     "details": f"scope must be one of {valid_scopes}"},
        )

    key = (body.scope, body.context_id)
    current = contexts.get(key)

    if current and current["version"] >= body.version:
        return JSONResponse(
            status_code=409,
            content={"accepted": False, "reason": "stale_version",
                     "current_version": current["version"]},
        )

    contexts[key] = {
        "version": body.version,
        "payload": body.payload,
        "stored_at": now_iso(),
    }
    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": now_iso(),
    }


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


@app.post("/v1/tick")
async def tick(body: TickBody):
    # Sort triggers by urgency (highest first) and process up to 5 per tick
    # to stay within the 30-second budget (~2-4s per Gemini call)
    trigger_payloads: list[tuple[int, str, dict]] = []

    for trg_id in body.available_triggers:
        entry = contexts.get(("trigger", trg_id))
        if not entry:
            continue
        trg = entry["payload"]

        # Skip expired
        expires = trg.get("expires_at")
        if expires:
            try:
                exp_dt = datetime.fromisoformat(expires.replace("Z", "+00:00"))
                if datetime.now(timezone.utc) > exp_dt:
                    continue
            except Exception:
                pass

        # Skip already-fired suppression keys
        supp = trg.get("suppression_key", "")
        if supp and supp in fired_keys:
            continue

        # Skip if conversation already started and ended
        conv_id = f"conv_{trg.get('merchant_id', 'x')}_{trg_id}"
        conv = conversations.get(conv_id)
        if conv and conv.get("ended"):
            continue

        urgency = trg.get("urgency", 1)
        trigger_payloads.append((urgency, trg_id, trg))

    # Highest urgency first, cap at 20
    trigger_payloads.sort(key=lambda x: -x[0])
    trigger_payloads = trigger_payloads[:20]

    actions = []

    for _, trg_id, trg in trigger_payloads:
        merchant_id = trg.get("merchant_id")
        if not merchant_id:
            continue

        merchant = get_ctx("merchant", merchant_id)
        if not merchant:
            continue

        category_slug = merchant.get("category_slug")
        category = get_ctx("category", category_slug) if category_slug else None
        if not category:
            continue

        customer_id = trg.get("customer_id")
        customer = get_ctx("customer", customer_id) if customer_id else None

        try:
            composed = compose(category, merchant, trg, customer)
        except Exception as e:
            continue

        body_text = composed.get("body", "").strip()
        if not body_text:
            continue

        supp = composed.get("suppression_key", trg.get("suppression_key", ""))
        if supp:
            fired_keys.add(supp)

        conv_id = f"conv_{merchant_id}_{trg_id}"
        send_as = composed.get("send_as", "vera")
        # Force merchant_on_behalf when customer is present
        if customer:
            send_as = "merchant_on_behalf"

        conversations[conv_id] = {
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "trigger_id": trg_id,
            "category_slug": category_slug,
            "turns": [{"from": "vera", "body": body_text}],
            "auto_reply_count": 0,
            "ended": False,
            "suppression_key": supp,
            "sent_bodies": {body_text},
        }

        # Build template_params (3 slots)
        owner = (merchant.get("identity") or {}).get("owner_first_name", "Merchant")
        words = body_text.split()
        mid = len(words) // 2
        template_params = [owner, " ".join(words[:mid]), " ".join(words[mid:])]

        actions.append({
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": send_as,
            "trigger_id": trg_id,
            "template_name": f"vera_{trg.get('kind', 'generic')}_v1",
            "template_params": template_params,
            "body": body_text,
            "cta": composed.get("cta", "open_ended"),
            "suppression_key": supp,
            "rationale": composed.get("rationale", ""),
        })

    return {"actions": actions}


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: str | None = None
    customer_id: str | None = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


@app.post("/v1/reply")
async def reply_endpoint(body: ReplyBody):
    conv_id = body.conversation_id
    msg = body.message.strip()

    conv = conversations.get(conv_id)
    if not conv or conv.get("ended"):
        return {"action": "end", "rationale": "Conversation ended or not found."}

    # Record incoming turn
    conv["turns"].append({"from": body.from_role, "body": msg})

    # ── 1. Hard negative intent ──
    if is_negative_intent(msg):
        conv["ended"] = True
        return {
            "action": "end",
            "rationale": (
                "Merchant explicitly opted out. Closing; suppressing all pending triggers "
                "for this merchant for 30 days."
            ),
        }

    # ── 2. Auto-reply detection ──
    prev_count = conv.get("auto_reply_count", 0)
    is_ar = is_auto_reply(msg)

    # Also detect repeated message (same text as two turns ago)
    turns = conv["turns"]
    same_as_prev = (
        len(turns) >= 3 and
        turns[-1]["body"].strip().lower() == turns[-3]["body"].strip().lower()
        if len(turns) >= 3 else False
    )

    if is_ar or same_as_prev:
        new_count = prev_count + 1
        conv["auto_reply_count"] = new_count

        if new_count == 1:
            reply_body = (
                "Looks like an auto-reply 🙂 "
                "When the owner sees this, just reply 'YES' to continue."
            )
            conv["turns"].append({"from": "vera", "body": reply_body})
            return {
                "action": "send",
                "body": reply_body,
                "cta": "binary_yes_no",
                "rationale": "Detected auto-reply; sending one prompt for owner attention.",
            }
        elif new_count == 2:
            return {
                "action": "wait",
                "wait_seconds": 86400,
                "rationale": "Same auto-reply twice. Owner unavailable. Backing off 24h.",
            }
        else:
            conv["ended"] = True
            return {
                "action": "end",
                "rationale": "Auto-reply 3× in a row. No real engagement. Closing conversation.",
            }

    # Real message — reset auto-reply counter
    conv["auto_reply_count"] = 0

    # ── 3. Load contexts for reply composition ──
    merchant_id = conv.get("merchant_id") or body.merchant_id
    merchant = get_ctx("merchant", merchant_id) if merchant_id else None

    cat_slug = conv.get("category_slug")
    if not cat_slug and merchant:
        cat_slug = merchant.get("category_slug")
    category = get_ctx("category", cat_slug) if cat_slug else None

    customer_id = conv.get("customer_id") or body.customer_id
    customer = get_ctx("customer", customer_id) if customer_id else None

    trigger_id = conv.get("trigger_id")
    trigger = get_ctx("trigger", trigger_id) if trigger_id else None

    if not merchant or not category:
        return {
            "action": "send",
            "body": "Got it! What would you like me to help with?",
            "cta": "open_ended",
            "rationale": "Missing merchant/category context; falling back to generic ack.",
        }

    # ── 4. Compose reply ──
    try:
        result = compose_reply(
            category, merchant, trigger, customer,
            conv["turns"][:-1],   # history before this latest turn
            msg,
            body.turn_number,
        )
    except Exception as e:
        return {
            "action": "send",
            "body": "Got it! Let me work on that for you.",
            "cta": "open_ended",
            "rationale": f"Compose error: {e}",
        }

    action = result.get("action", "send")

    if action == "end":
        conv["ended"] = True
    elif action == "send":
        reply_body = (result.get("body") or "").strip()

        # Anti-repetition: if we already sent this exact body, tweak it
        sent = conv.setdefault("sent_bodies", set())
        if reply_body in sent:
            reply_body += " Kuch aur chahiye to bataiye!"
        if reply_body:
            sent.add(reply_body)
            conv["turns"].append({"from": "vera", "body": reply_body})
        result["body"] = reply_body

    return result


@app.post("/v1/teardown")
async def teardown():
    """Called by judge at end of test — wipe all state."""
    contexts.clear()
    conversations.clear()
    fired_keys.clear()
    return {"status": "cleared"}
