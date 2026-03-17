"""
fetch_precons.py — Download all Commander precon decklists from Scryfall

Run this ONCE on your machine to build the local card database.
After this runs, the game engine never needs to call Scryfall again.

Usage:
    cd engine
    pip install -r requirements.txt
    python fetch_precons.py

What it does:
    1. Downloads the Scryfall bulk card data (~100MB, cached locally)
    2. Finds every card printed in a Commander precon set
    3. Reconstructs each precon decklist using Scryfall's deck data
    4. Saves everything to card_cache/ as JSON
    5. Reports what was found and any gaps

Output files:
    card_cache/cards.json       — All card definitions (name → data)
    card_cache/precons.json     — All precon decklists (set → decks → cards)
    card_cache/fetch_report.txt — Summary of what was downloaded
"""

import json
import time
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Optional

CACHE_DIR = Path(__file__).parent / "card_cache"
CACHE_DIR.mkdir(exist_ok=True)

BULK_DATA_PATH  = CACHE_DIR / "scryfall_bulk.json"
CARDS_PATH      = CACHE_DIR / "cards.json"
PRECONS_PATH    = CACHE_DIR / "precons.json"
REPORT_PATH     = CACHE_DIR / "fetch_report.txt"

SCRYFALL_HEADERS = {
    "User-Agent": "CommanderArena/1.0 (contact: your@email.com)",
    "Accept": "application/json",
}

# Scryfall rate limit: 50-100ms between requests
RATE_LIMIT_DELAY = 0.1


def get(url: str) -> dict:
    """Make a GET request to Scryfall with rate limiting."""
    time.sleep(RATE_LIMIT_DELAY)
    req = urllib.request.Request(url, headers=SCRYFALL_HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def download_bulk_cards():
    """
    Download Scryfall's bulk card data.
    This is a single ~100MB file with every card ever printed.
    Much more efficient than fetching cards one by one.
    We cache it locally — only re-download if it's more than 7 days old.
    """
    import os

    # Check if cache is fresh (less than 7 days old)
    if BULK_DATA_PATH.exists():
        age_days = (time.time() - BULK_DATA_PATH.stat().st_mtime) / 86400
        if age_days < 7:
            print(f"[Bulk] Using cached bulk data ({age_days:.1f} days old)")
            with open(BULK_DATA_PATH) as f:
                return json.load(f)
        else:
            print(f"[Bulk] Cache is {age_days:.1f} days old — refreshing")

    # Get the bulk data download URL from Scryfall
    print("[Bulk] Fetching bulk data manifest...")
    manifest = get("https://api.scryfall.com/bulk-data")

    # Find the "Oracle Cards" bulk file — one entry per unique card name
    oracle_entry = next(
        (e for e in manifest["data"] if e["type"] == "oracle_cards"),
        None
    )
    if not oracle_entry:
        raise RuntimeError("Could not find oracle_cards bulk data")

    download_url = oracle_entry["download_uri"]
    size_mb = oracle_entry.get("size", 0) / 1_000_000

    print(f"[Bulk] Downloading {size_mb:.0f}MB card database...")
    print(f"       This only happens once — subsequent runs use the cache")

    req = urllib.request.Request(download_url, headers=SCRYFALL_HEADERS)
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.load(resp)

    with open(BULK_DATA_PATH, "w") as f:
        json.dump(data, f)

    print(f"[Bulk] Downloaded {len(data):,} cards")
    return data


def get_all_commander_sets() -> list[dict]:
    """Fetch all sets with type 'commander' from Scryfall."""
    print("[Sets] Fetching commander set list...")
    data = get("https://api.scryfall.com/sets")
    cmd_sets = [s for s in data["data"] if s["set_type"] == "commander"]
    print(f"[Sets] Found {len(cmd_sets)} commander sets")
    return cmd_sets


def get_deck_list(deck_id: str) -> Optional[dict]:
    """Fetch a specific decklist by its Scryfall ID."""
    try:
        return get(f"https://api.scryfall.com/decks/{deck_id}/export/json")
    except Exception as e:
        print(f"  Warning: could not fetch deck {deck_id}: {e}")
        return None


def get_precon_decks_for_set(set_code: str, set_name: str) -> list[dict]:
    """
    Get all precon decks for a given set.
    Scryfall doesn't have a direct 'list all decks for set' endpoint,
    so we search for cards with deck= parameter and reconstruct from there.
    
    Alternative approach: use the /cards/search endpoint with set filter,
    then group by collector number ranges (each precon has a distinct range).
    """
    decks = []

    # Search for all cards in this set
    url = (f"https://api.scryfall.com/cards/search"
           f"?q=set:{set_code}+is:commander&unique=cards")

    try:
        commanders = []
        while url:
            page = get(url)
            commanders.extend(page.get("data", []))
            url = page.get("next_page")

        print(f"  Found {len(commanders)} commanders in {set_code}")

        # Each unique commander = one precon deck
        for cmd_card in commanders:
            deck = reconstruct_deck_from_set(set_code, cmd_card)
            if deck:
                decks.append(deck)

    except Exception as e:
        print(f"  Error fetching decks for {set_code}: {e}")

    return decks


def reconstruct_deck_from_set(set_code: str, commander_card: dict) -> Optional[dict]:
    """
    Reconstruct a precon deck by finding all cards with the same
    collector number prefix in the set (each precon occupies a range).
    """
    cmd_name = commander_card["name"]

    # Get all cards in this set to find the full decklist
    url = f"https://api.scryfall.com/cards/search?q=set:{set_code}&order=set&unique=prints"

    all_set_cards = []
    try:
        while url:
            page = get(url)
            all_set_cards.extend(page.get("data", []))
            url = page.get("next_page")
    except:
        return None

    # Scryfall precon sets organize cards by deck — cards 1-100 = deck 1, etc.
    # We can use the commander's collector number to identify the deck range
    try:
        cmd_num = int(commander_card.get("collector_number", "0").rstrip("★").rstrip("s"))
    except:
        cmd_num = 0

    # Each precon is 100 cards. Commander sets typically have 4-5 decks.
    # Find the deck this commander belongs to by its collector number range
    deck_size = 100
    deck_index = (cmd_num - 1) // deck_size
    start = deck_index * deck_size + 1
    end   = start + deck_size - 1

    deck_cards = []
    for card in all_set_cards:
        try:
            num = int(card.get("collector_number", "0").rstrip("★").rstrip("s"))
            if start <= num <= end:
                deck_cards.append(card["name"])
        except:
            continue

    if len(deck_cards) < 90:  # Sanity check — a deck should have close to 100 cards
        return None

    # Deduplicate while preserving order, handle basic lands
    seen = {}
    final_list = []
    for name in deck_cards:
        if name in seen:
            seen[name] += 1
        else:
            seen[name] = 1
        final_list.append(name)

    colors = commander_card.get("color_identity", [])

    return {
        "name":      f"{cmd_name} deck",
        "commander": cmd_name,
        "colors":    colors,
        "set_code":  set_code,
        "cards":     final_list,
    }


def build_card_definitions(all_cards_raw: list[dict]) -> dict:
    """
    Build a name → card data dictionary from the bulk download.
    Strips down to only the fields we need to keep file size manageable.
    """
    cards = {}
    for card in all_cards_raw:
        name = card.get("name", "")
        if not name:
            continue

        # Handle double-faced cards
        if "card_faces" in card and card.get("layout") in ("transform", "modal_dfc"):
            face = card["card_faces"][0]
            oracle_text = face.get("oracle_text", "")
            mana_cost   = face.get("mana_cost", card.get("mana_cost", ""))
            power       = face.get("power")
            toughness   = face.get("toughness")
        else:
            oracle_text = card.get("oracle_text", "")
            mana_cost   = card.get("mana_cost", "")
            power       = card.get("power")
            toughness   = card.get("toughness")

        cards[name] = {
            "scryfall_id":  card.get("id", ""),
            "name":         name,
            "mana_cost":    mana_cost,
            "cmc":          card.get("cmc", 0),
            "colors":       card.get("colors", []),
            "color_identity": card.get("color_identity", []),
            "type_line":    card.get("type_line", ""),
            "oracle_text":  oracle_text,
            "power":        power,
            "toughness":    toughness,
            "loyalty":      card.get("loyalty"),
            "keywords":     card.get("keywords", []),
            "legalities":   {"commander": card.get("legalities", {}).get("commander", "not_legal")},
            "is_legendary": "Legendary" in card.get("type_line", ""),
        }

    return cards


def main():
    print("=" * 60)
    print("Commander Arena — Precon Database Builder")
    print("=" * 60)
    print()

    # ── Step 1: Download bulk card data ───────────────────────────
    all_cards_raw = download_bulk_cards()
    print(f"\n[Cards] Building card definition database...")
    card_defs = build_card_definitions(all_cards_raw)
    print(f"[Cards] {len(card_defs):,} unique cards indexed")

    with open(CARDS_PATH, "w") as f:
        json.dump(card_defs, f)
    print(f"[Cards] Saved to {CARDS_PATH}")

    # ── Step 2: Skip deck reconstruction ─────────────────────────
    # precons.json was built manually from verified decklists.
    # fetch_precons.py only updates cards.json (the card data).
    # Do NOT overwrite precons.json here.
    total_decks = 0
    if PRECONS_PATH.exists():
        with open(PRECONS_PATH) as f:
            existing = json.load(f)
        total_decks = existing.get("meta", {}).get("total_decks", 0)
        print(f"\n[Decks] Keeping existing precons.json ({total_decks} decks)")
        print(f"        (precons.json was built from verified decklists — not overwriting)")

    # ── Step 5: Write report ──────────────────────────────────────
    report = [
        "Commander Arena — Precon Fetch Report",
        "=" * 50,
        f"Total sets processed:  (skipped — using existing precons.json)",
        f"Total precon decks:    {total_decks} (from existing precons.json)",
        f"Total unique cards:    {len(card_defs):,}",
        "",
        "Sets and deck counts:",
    ]
    # Deck report skipped — precons.json managed separately

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(report))

    print()
    print("=" * 60)
    print(f"✅ Done!")
    print(f"   {len(cmd_sets)} sets processed")
    print(f"   {total_decks} precon decks saved")
    print(f"   {len(card_defs):,} cards in database")
    print()
    print(f"   Files written to: {CACHE_DIR}")
    print(f"   Run the engine now — no more Scryfall calls needed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
