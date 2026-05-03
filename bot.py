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
TEAM_NAME    = "Riya Bisht"
TEAM_MEMBERS = ["Riya Bisht"]
CONTACT_EMAIL = "riyabisht30@gmail.com"
MODEL_NAME   = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

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
    "leave me alone", "bothering me", "stop",
]

OUT_OF_SCOPE_PHRASES = [
    "gst", "income tax", "legal advice", "labour law", "rent agreement",
    "personal loan", "medical advice", "doctor", "lawyer", "advocate",
    "filing", "court", "police", "property dispute", "divorce",
]

def is_positive_intent(msg: str) -> bool:
    return bool(POSITIVE_INTENT_RE.search(msg))

def is_negative_intent(msg: str) -> bool:
    low = msg.lower()
    return any(p in low for p in NEGATIVE_INTENT_PHRASES)

def is_out_of_scope(msg: str) -> bool:
    low = msg.lower()
    return any(p in low for p in OUT_OF_SCOPE_PHRASES)


# ─────────────────────────────────────────────
# TRIGGER-KIND → PRIORITY DISPATCH INSTRUCTIONS
# ─────────────────────────────────────────────

KIND_PRIORITY: dict[str, str] = {
    "research_digest":
        "PRIORITY: cite the paper title, trial_n, and source page number from the digest item.",
    "recall_due":
        "PRIORITY: name the customer's last visit date and the exact recall window that opened.",
    "perf_dip":
        "PRIORITY: lead with the exact percentage drop and the peer benchmark.",
    "perf_spike":
        "PRIORITY: frame as a momentum moment, reference the exact % increase.",
    "festival_upcoming":
        "PRIORITY: name the festival and days remaining, connect to a real offer in catalog.",
    "dormant_with_vera":
        "PRIORITY: no guilt-trip; use curiosity lever with a business insight.",
    "regulation_change":
        "PRIORITY: state the compliance deadline date and the specific rule that changed.",
    "milestone_reached":
        "PRIORITY: cite the exact milestone number, offer to turn it into a Google post.",
    "ipl_match_today":
        "PRIORITY: name today's match teams and provide contrarian covers data if available.",
    "curious_ask_due":
        "PRIORITY: ask exactly one low-stakes question, offer a concrete deliverable in return.",
    "review_theme_emerged":
        "PRIORITY: cite the exact review theme and occurrence count from context.",
    "renewal_due":
        "PRIORITY: state exact days remaining and list what visibility they lose on expiry.",
    "winback_eligible":
        "PRIORITY: show one specific improvement since they lapsed; keep ask low-commitment.",
    "wedding_package_followup":
        "PRIORITY: use exact days-to-wedding from trigger; reference the trial they completed.",
    "competitor_opened":
        "PRIORITY: frame as prep opportunity not fear; offer one concrete listing improvement.",
    "supply_alert":
        "PRIORITY: use exact batch numbers from trigger payload; count affected customers.",
    "chronic_refill_due":
        "PRIORITY: use precise medication names and exact run-out date from trigger.",
    "customer_lapsed_soft":
        "PRIORITY: no guilt; personal connection + one relevant new offer with ₹ price.",
    "customer_lapsed_hard":
        "PRIORITY: low-commitment trial offer; acknowledge the gap without blame.",
    "appointment_tomorrow":
        "PRIORITY: state appointment time and any prep instructions. Keep it brief.",
}


# ─────────────────────────────────────────────
# LLM CALL
# ─────────────────────────────────────────────

def call_llm(system_prompt: str, user_message: str) -> str:
    """Call Groq LLM with retry on rate-limit errors."""
    for attempt in range(3):
        try:
            resp = groq_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content
        except Exception as exc:
            err_str = str(exc)
            if "429" in err_str or "rate" in err_str.lower():
                if attempt < 2:
                    time.sleep(3)
                    continue
            raise RuntimeError(f"Groq API error: {exc}")
    raise RuntimeError("Groq API: max retries exceeded")


def build_system_prompt(kind: str, category: dict, merchant: dict,
                        trigger: dict, customer: dict | None) -> str:
    """Build the dynamic system prompt with trigger-kind dispatch."""
    priority = KIND_PRIORITY.get(kind, "PRIORITY: match trigger kind to appropriate framing.")

    category_json  = json.dumps(category,  indent=2, ensure_ascii=False)
    merchant_json  = json.dumps(merchant,  indent=2, ensure_ascii=False)
    trigger_json   = json.dumps(trigger,   indent=2, ensure_ascii=False)
    customer_json  = json.dumps(customer,  indent=2, ensure_ascii=False) if customer else "none"

    # Build anti-repetition hint from conversation history
    conv_history = (merchant.get("conversation_history") or [])
    anti_repeat = ""
    if conv_history:
        last_msg = conv_history[-1].get("body", "") if isinstance(conv_history[-1], dict) else str(conv_history[-1])
        if last_msg:
            anti_repeat = (
                f"\nANTI-REPETITION: The previous message was: \"{last_msg[:120]}\". "
                "Do NOT repeat the same angle or lever. Use a different compulsion."
            )

    # Language hint
    languages = (merchant.get("identity") or {}).get("languages", ["en"])
    lang_hint = ""
    if "hi" in languages:
        lang_hint = (
            "\nLANGUAGE: Merchant speaks Hindi. Write in natural Hinglish (Hindi-English mix). "
            "Examples: 'Apke liye 2 slots ready hain — kar dun?', 'Chalega?', "
            "'18 patients ne last month visit ki — dekh lein?'. "
            "Pure English for Hindi-speaking merchants loses 2 points on merchant fit."
        )

    return f"""You are Vera, magicpin's merchant AI assistant on WhatsApp. Compose ONE message.

{priority}{anti_repeat}{lang_hint}

SCORING DIMENSIONS (each 0-10, must score 9+ on each):

1. SPECIFICITY (most important): The FIRST 8 WORDS of body MUST contain a real number from context.
   Score 10: "190 people in your locality are searching", "2,100-patient trial showed 34%", "Your CTR dropped to 2.1%", "78 patients haven't visited in 180 days"
   Score 0: "Hi Dr. Meera, I wanted to tell you", "Hope you are doing well", "Just checking in"
   Rule: If no number in first 8 words → automatic 0 on specificity.

2. CATEGORY FIT:
   - dentists: peer-clinical tone, no "guaranteed/cure/miracle", cite source (journal/page)
   - salons: warm-visual, mention service names + ₹ price
   - restaurants: fast-operator voice, use "covers/AOV/thali", no fluff
   - gyms: coach voice, use "slots/members/capacity %"
   - pharmacies: trustworthy-precise, exact drug names, no overclaims

3. MERCHANT FIT:
   - ALWAYS address by merchant.identity.owner_first_name (NOT "Hi" alone — missing name = -1)
   - Mention their locality (merchant.identity.locality) — localisation scores +1
   - Use their exact offer title with ₹ price from merchant.offers (not a generic offer)
   - Reference their specific performance number (CTR, views, calls) to show you know their business
   - Honor conversation history — do not repeat angles already used

4. TRIGGER RELEVANCE: trigger kind is the reason you are messaging RIGHT NOW.
   research_digest → cite paper title + page; perf_dip → name the exact % drop + metric;
   recall_due → customer name + days since last visit; festival_upcoming → festival name + days left

5. ENGAGEMENT COMPULSION: Pick exactly ONE lever AND end with a binary YES/STOP CTA:
   - Loss aversion: "3 clinics in your area already updated — you are the only one missing this"
   - Social proof: "N merchants in your locality did Y this month — want the same?"
   - Curiosity: "want to see exactly who?" / "want the full breakdown? Takes 2 min."
   - Effort externalization: "I have drafted the update — just say YES and I will send it"
   - The CTA MUST be "Reply YES" or "Reply YES / STOP" — never open-ended for action triggers
   - Weak ending ("let me know", "feel free to reach out") scores 0 on engagement

RATIONALE FIELD: Must name exactly (a) trigger kind that fired, (b) specific merchant signal that made it relevant (e.g. ctr_below_peer_median, high_risk_adult_cohort), (c) which compulsion lever chosen and why. One sentence max.

HARD RULES:
- Never invent numbers not in context
- No URLs in body (Meta policy — hard ban)
- Exactly ONE question / CTA at end
- Under 400 characters total
- No "FREE" in caps, no "AMAZING", no generic "Flat X% off"
- No long preambles
- suppression_key format: "<trigger_kind>::<merchant_id>::<week>"

CONTEXT:
Category: {category_json}
Merchant: {merchant_json}
Trigger: {trigger_json}
Customer: {customer_json}

Respond ONLY with valid JSON, no markdown, no preamble:
{{
  "body": "the WhatsApp message",
  "cta": "open_ended | binary_yes_no | binary_confirm_cancel | multi_choice_slot | none",
  "send_as": "vera | merchant_on_behalf",
  "suppression_key": "...",
  "rationale": "one sentence naming: trigger kind + merchant signal + compulsion lever",
  "template_name": "vera_{kind}_v1",
  "template_params": ["param1", "param2"]
}}"""


# ─────────────────────────────────────────────
# COMPOSE — initial outbound message
# ─────────────────────────────────────────────

def _validate_and_fix(result: dict) -> dict:
    """Post-LLM validators: URL, multi-CTA, length. Fixed with string ops — no extra LLM calls."""
    body = result.get("body", "")

    # 1. URL check — strip any links
    if re.search(r"https?://", body):
        body = strip_urls(body)

    # 2. Multi-question check — keep only the last sentence containing "?"
    if body.count("?") > 1:
        sentences = re.split(r'(?<=[.!?])\s+', body.strip())
        q_idx = [i for i, s in enumerate(sentences) if "?" in s]
        if len(q_idx) > 1:
            # Keep all non-question sentences + only the last question
            keep = [s for i, s in enumerate(sentences) if "?" not in s or i == q_idx[-1]]
            body = " ".join(keep)

    # 3. Length check — hard truncate at 400 chars, break at last word boundary
    if len(body) > 400:
        body = body[:400].rsplit(" ", 1)[0] + "…"

    result["body"] = body.strip()
    return result


def compose(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: dict | None = None,
) -> dict:
    """Compose the first outbound WhatsApp message for a trigger."""
    kind = trigger.get("kind", "generic")

    system_prompt = build_system_prompt(kind, category, merchant, trigger, customer)
    raw = call_llm(system_prompt, "Compose the next message for this merchant.")
    result = parse_llm_json(raw)

    # Post-LLM validation: URL / multi-CTA / length (fast string fixes, no extra LLM calls)
    result = _validate_and_fix(result)

    # Guarantee required fields
    if not result.get("suppression_key"):
        result["suppression_key"] = trigger.get(
            "suppression_key",
            f"{kind}::{trigger.get('merchant_id', 'x')}::week"
        )
    if not result.get("template_name"):
        result["template_name"] = f"vera_{kind}_v1"
    if not result.get("template_params"):
        owner = (merchant.get("identity") or {}).get("owner_first_name", "Merchant")
        words = (result.get("body") or "").split()
        mid = len(words) // 2
        result["template_params"] = [owner, " ".join(words[:mid]), " ".join(words[mid:])]

    # Final URL strip (safety net)
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
    intent_confirmed: bool = False,
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

    intent_instruction = ""
    if intent_confirmed:
        intent_instruction = (
            "\nINTENT DETECTED: merchant said yes/confirmed. "
            "Do NOT ask more qualifying questions. "
            "Draft the actual artifact they agreed to, name the exact deliverable, "
            "give a CONFIRM CTA.\n"
        )

    prompt = f"""CONVERSATION HISTORY:
{history_text}

[MERCHANT (turn {turn_number})]: {merchant_message}
{intent_instruction}
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

    reply_system = """You are Vera, magicpin's merchant AI assistant on WhatsApp. This is a REPLY in an ongoing conversation.
HARD RULES: No URLs. One CTA only. No re-introduction. No long preambles.
If merchant said YES/confirmed → immediately switch to ACTION mode.
Respond ONLY with valid JSON: {"action": "send|wait|end", "body": "...", "cta": "...", "wait_seconds": 0, "rationale": "..."}"""

    raw = call_llm(reply_system, prompt)
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
        "model": f"{MODEL_NAME} (Groq)",
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

    if current and current["version"] > body.version:
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

    # Highest urgency first, cap at 1 per tick to stay well within 15s judge timeout
    trigger_payloads.sort(key=lambda x: -x[0])
    trigger_payloads = trigger_payloads[:1]

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

        actions.append({
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": send_as,
            "trigger_id": trg_id,
            "template_name": composed.get("template_name", f"vera_{trg.get('kind', 'generic')}_v1"),
            "template_params": composed.get("template_params", []),
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

    # Store turn with full metadata
    conv["turns"].append({
        "from_role": body.from_role,
        "from": body.from_role,
        "body": msg,
        "ts": body.received_at,
        "turn_number": body.turn_number,
    })

    turns = conv["turns"]

    # ── STEP 1: AUTO-REPLY DETECTION (check before anything else) ──
    is_ar = is_auto_reply(msg)
    same_as_prev = (
        len(turns) >= 3 and
        turns[-1]["body"].strip().lower() == turns[-3]["body"].strip().lower()
    )

    if is_ar or same_as_prev:
        prev_count = conv.get("auto_reply_count", 0)
        new_count = prev_count + 1
        conv["auto_reply_count"] = new_count

        if new_count == 1:
            reply_body = (
                "Looks like an auto-reply — when you see this, just reply YES to continue. \U0001f64f"
            )
            conv["turns"].append({"from": "vera", "body": reply_body,
                                   "ts": now_iso(), "turn_number": body.turn_number + 1})
            return {
                "action": "send",
                "body": reply_body,
                "cta": "binary_yes_no",
                "rationale": "Detected auto-reply turn 1; surfacing for owner",
            }
        elif new_count == 2:
            return {
                "action": "wait",
                "wait_seconds": 14400,
                "rationale": "Auto-reply second time; backing off 4h for owner",
            }
        else:
            conv["ended"] = True
            return {
                "action": "end",
                "rationale": "3 consecutive auto-replies; no engagement signal, closing",
            }

    # Real message — reset auto-reply counter
    conv["auto_reply_count"] = 0

    # ── STEP 2: OPT-OUT DETECTION ──
    if is_negative_intent(msg):
        conv["ended"] = True
        fired_keys.add(conv.get("suppression_key", ""))
        return {
            "action": "end",
            "rationale": "Merchant explicitly opted out; suppressing conversation",
        }

    # ── Load contexts for steps 3-5 ──
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

    # ── STEP 3: INTENT TRANSITION ──
    if body.turn_number >= 2 and is_positive_intent(msg):
        try:
            result = compose_reply(
                category, merchant, trigger, customer,
                conv["turns"][:-1],
                msg,
                body.turn_number,
                intent_confirmed=True,
            )
        except Exception as e:
            result = {
                "action": "send",
                "body": "Got it! Let me draft that for you right now.",
                "cta": "binary_confirm_cancel",
                "rationale": f"Intent confirmed; compose error: {e}",
            }
        _store_reply_turn(conv, result, body.turn_number)
        return result

    # ── STEP 4: OUT-OF-SCOPE DEFLECTION ──
    if is_out_of_scope(msg):
        topic = conv.get("trigger_id", "your growth")
        deflect_body = (
            f"That's outside what I can help with — "
            f"let's get back to {topic.replace('_', ' ')} when you're ready!"
        )
        conv["turns"].append({"from": "vera", "body": deflect_body,
                               "ts": now_iso(), "turn_number": body.turn_number + 1})
        return {
            "action": "send",
            "body": deflect_body,
            "cta": "open_ended",
            "rationale": "Out-of-scope ask deflected; conversation redirected",
        }

    # ── STEP 5: DEFAULT — pass full history to compose ──
    try:
        result = compose_reply(
            category, merchant, trigger, customer,
            conv["turns"][:-1],
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

    _store_reply_turn(conv, result, body.turn_number)
    return result


def _store_reply_turn(conv: dict, result: dict, turn_number: int):
    """Store vera's reply turn and handle anti-repetition."""
    if result.get("action") == "end":
        conv["ended"] = True
    elif result.get("action") == "send":
        reply_body = (result.get("body") or "").strip()
        sent = conv.setdefault("sent_bodies", set())
        if reply_body in sent:
            reply_body += " Kuch aur chahiye to bataiye!"
        if reply_body:
            sent.add(reply_body)
            conv["turns"].append({"from": "vera", "body": reply_body,
                                   "ts": now_iso(), "turn_number": turn_number + 1})
        result["body"] = reply_body


@app.post("/v1/teardown")
async def teardown():
    """Called by judge at end of test — wipe all state."""
    contexts.clear()
    conversations.clear()
    fired_keys.clear()
    return {"status": "cleared"}
