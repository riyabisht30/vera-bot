"""
Generate submission.jsonl — 30 canonical test pairs.

Usage:
    python generate_submission.py

Reads from: ../challenge/expanded/
Writes to:  ./submission.jsonl
"""

import json
import os
import sys
import time
from pathlib import Path

# Fix Windows console encoding for Unicode characters (₹, Hindi, etc.)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add bot module to path for shared helpers
sys.path.insert(0, str(Path(__file__).parent))
from bot import compose, get_ctx, contexts

EXPANDED_DIR = Path(__file__).parent.parent / "challenge" / "dataset" / "expanded"
OUT_FILE = Path(__file__).parent / "submission.jsonl"

TEST_IDS = [f"T{i:02d}" for i in range(1, 31)]


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_dataset(expanded_dir: Path):
    """Load all dataset files into the global contexts store."""
    print("Loading dataset...")

    # Categories
    for cat_file in (expanded_dir / "categories").glob("*.json"):
        data = load_json(cat_file)
        slug = data.get("slug", cat_file.stem)
        contexts[("category", slug)] = {"version": 1, "payload": data}
        print(f"  category: {slug}")

    # Merchants
    merchant_dir = expanded_dir / "merchants"
    if merchant_dir.exists():
        for mf in sorted(merchant_dir.glob("*.json")):
            data = load_json(mf)
            mid = data.get("merchant_id", mf.stem)
            contexts[("merchant", mid)] = {"version": 1, "payload": data}
    else:
        # Fall back to seed file
        seed_file = expanded_dir.parent.parent / "dataset" / "merchants_seed.json"
        if seed_file.exists():
            seed = load_json(seed_file)
            for m in seed.get("merchants", []):
                mid = m.get("merchant_id", "")
                contexts[("merchant", mid)] = {"version": 1, "payload": m}
    print(f"  merchants loaded: {sum(1 for k in contexts if k[0]=='merchant')}")

    # Customers
    customer_dir = expanded_dir / "customers"
    if customer_dir.exists():
        for cf in sorted(customer_dir.glob("*.json")):
            data = load_json(cf)
            cid = data.get("customer_id", cf.stem)
            contexts[("customer", cid)] = {"version": 1, "payload": data}
    else:
        seed_file = expanded_dir.parent.parent / "dataset" / "customers_seed.json"
        if seed_file.exists():
            seed = load_json(seed_file)
            for c in seed.get("customers", []):
                cid = c.get("customer_id", "")
                contexts[("customer", cid)] = {"version": 1, "payload": c}
    print(f"  customers loaded: {sum(1 for k in contexts if k[0]=='customer')}")

    # Triggers
    trigger_dir = expanded_dir / "triggers"
    if trigger_dir.exists():
        for tf in sorted(trigger_dir.glob("*.json")):
            data = load_json(tf)
            tid = data.get("id", tf.stem)
            contexts[("trigger", tid)] = {"version": 1, "payload": data}
    else:
        seed_file = expanded_dir.parent.parent / "dataset" / "triggers_seed.json"
        if seed_file.exists():
            seed = load_json(seed_file)
            for t in seed.get("triggers", []):
                tid = t.get("id", "")
                contexts[("trigger", tid)] = {"version": 1, "payload": t}
    print(f"  triggers loaded: {sum(1 for k in contexts if k[0]=='trigger')}")


def get_test_pairs(expanded_dir: Path) -> list[dict]:
    """Load the 30 canonical test pairs."""
    tp_file = expanded_dir / "test_pairs.json"
    if tp_file.exists():
        data = load_json(tp_file)
        pairs = data if isinstance(data, list) else data.get("pairs", [])
        print(f"Loaded {len(pairs)} test pairs from test_pairs.json")
        return pairs

    # Fallback: build 30 pairs from available triggers (sorted by urgency desc)
    print("test_pairs.json not found — building pairs from triggers seed...")
    all_triggers = [
        v["payload"] for (scope, _), v in contexts.items() if scope == "trigger"
    ]
    all_triggers.sort(key=lambda t: -t.get("urgency", 1))

    pairs = []
    seen_merchants: set[str] = set()
    for trg in all_triggers:
        mid = trg.get("merchant_id", "")
        if not mid:
            continue
        pairs.append({
            "merchant_id": mid,
            "trigger_id": trg.get("id", ""),
            "customer_id": trg.get("customer_id"),
        })
        if len(pairs) >= 30:
            break

    # Pad to 30 if needed (re-use triggers with different merchants)
    if len(pairs) < 30:
        for trg in all_triggers:
            if len(pairs) >= 30:
                break
            mid = trg.get("merchant_id", "")
            if not mid:
                continue
            pairs.append({
                "merchant_id": mid,
                "trigger_id": trg.get("id", ""),
                "customer_id": trg.get("customer_id"),
            })

    print(f"Built {len(pairs)} test pairs")
    return pairs[:30]


def process_pair(test_id: str, pair: dict) -> dict | None:
    merchant_id  = pair.get("merchant_id", "")
    trigger_id   = pair.get("trigger_id", "")
    customer_id  = pair.get("customer_id")

    merchant = get_ctx("merchant", merchant_id)
    trigger  = get_ctx("trigger", trigger_id)

    if not merchant:
        print(f"  [{test_id}] SKIP — merchant not found: {merchant_id}")
        return None
    if not trigger:
        print(f"  [{test_id}] SKIP — trigger not found: {trigger_id}")
        return None

    cat_slug = merchant.get("category_slug")
    category = get_ctx("category", cat_slug) if cat_slug else None
    if not category:
        print(f"  [{test_id}] SKIP — category not found: {cat_slug}")
        return None

    customer = get_ctx("customer", customer_id) if customer_id else None

    t0 = time.time()
    try:
        result = compose(category, merchant, trigger, customer)
    except Exception as e:
        print(f"  [{test_id}] ERROR composing: {e}")
        return {"test_id": test_id, "error": str(e)}

    elapsed = time.time() - t0
    body_text = (result.get("body") or "").strip()
    if not body_text:
        print(f"  [{test_id}] WARN — empty body")
        return {"test_id": test_id, "error": "empty body"}

    name_safe = merchant.get('identity', {}).get('name', merchant_id)[:30].encode('ascii', 'replace').decode()
    body_safe = body_text[:60].encode('ascii', 'replace').decode()
    print(f"  {test_id}: {name_safe} — {trigger.get('kind', '?')} ... OK ({elapsed:.1f}s)")

    return {
        "test_id": test_id,
        "merchant_id": merchant_id,
        "trigger_id": trigger_id,
        "customer_id": customer_id,
        "body": body_text,
        "cta": result.get("cta", "open_ended"),
        "send_as": result.get("send_as", "vera"),
        "suppression_key": result.get("suppression_key", ""),
        "rationale": result.get("rationale", ""),
    }


def main():
    expanded_dir = EXPANDED_DIR

    # If expanded dir doesn't exist, fall back to seed files
    if not expanded_dir.exists():
        print(f"Expanded dataset not found at {expanded_dir}")
        print("Falling back to seed files in challenge/dataset/...")
        expanded_dir = expanded_dir.parent.parent / "dataset"

    load_dataset(expanded_dir)

    pairs = get_test_pairs(EXPANDED_DIR if EXPANDED_DIR.exists() else expanded_dir)

    # Load already-generated pairs so we can resume if interrupted (skip errors)
    existing: dict[str, dict] = {}
    if OUT_FILE.exists():
        with open(OUT_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "error" not in row:   # only keep successful entries
                    existing[row["test_id"]] = row
        print(f"Resuming: {len(existing)} already done (errors will be retried).\n")

    print(f"\nComposing {len(pairs)} messages...\n")
    results = list(existing.values())
    for i, pair in enumerate(pairs[:30]):
        test_id = f"T{i+1:02d}"
        if test_id in existing:
            print(f"  [{test_id}] SKIP - already done")
            continue
        row = process_pair(test_id, pair)
        if row:
            results.append(row)
            # Write incrementally so we don't lose progress on crash
            with open(OUT_FILE, "w", encoding="utf-8") as f:
                for r in sorted(results, key=lambda x: x["test_id"]):
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        time.sleep(1)   # Groq free tier: ~30 req/min, 1s gap is enough

    # Final sorted write
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for row in sorted(results, key=lambda x: x["test_id"]):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    succeeded = sum(1 for r in results if "error" not in r)
    print(f"\nDone. {succeeded}/30 succeeded.")
    if succeeded < 30:
        print(f"  WARNING: {30 - succeeded} pairs failed — check errors above.")


if __name__ == "__main__":
    main()
