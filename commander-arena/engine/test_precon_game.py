"""
test_precon_game.py — End-to-end test with real precon decklists

Loads two precon decks from precons.json, builds CardDefinition objects
using card_parser's effect tagger (no Scryfall needed — just the names
and whatever data we have), then runs a full simulator game.

Usage:
    cd engine
    python test_precon_game.py

    # Run a specific matchup:
    python test_precon_game.py --p1 "Heavenly Inferno" --p2 "Devour for Power"

    # Run multiple games:
    python test_precon_game.py --games 5 --verbose
"""

import sys
import json
import argparse
import uuid
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'ml'))

from game_state import (
    GameState, Player, CardDefinition, CardInstance,
    Color, CardType, Zone
)
from card_parser import tag_effects, parse_colors, parse_card_types, parse_subtypes
from simulator import GameSimulator, summarize_results


# ─────────────────────────────────────────────────────────────────────────────
# Load precon data
# ─────────────────────────────────────────────────────────────────────────────

PRECONS_PATH = Path(__file__).parent / "card_cache" / "precons.json"

def load_precons() -> dict:
    if not PRECONS_PATH.exists():
        # Try one level up
        alt = Path(__file__).parent / "precons.json"
        if alt.exists():
            with open(alt) as f:
                return json.load(f)
        raise FileNotFoundError(f"precons.json not found at {PRECONS_PATH}")
    with open(PRECONS_PATH) as f:
        return json.load(f)


def list_all_decks(precons: dict) -> list[tuple[str, str]]:
    """Return list of (set_name, deck_name) for all decks."""
    result = []
    for set_name, decks in precons.get("sets", {}).items():
        for deck_name in decks:
            result.append((set_name, deck_name))
    return result


def find_deck(precons: dict, deck_name: str) -> dict | None:
    """Find a deck by name (partial match, case-insensitive)."""
    needle = deck_name.lower()
    for set_name, decks in precons.get("sets", {}).items():
        for name, deck in decks.items():
            if needle in name.lower() or needle in set_name.lower():
                return deck
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Build CardDefinition from a card name
# ─────────────────────────────────────────────────────────────────────────────

# Minimal hardcoded data for common card types so we don't need Scryfall
# This lets us run games without the full card database downloaded
# Format: name → (mana_cost, cmc, type_line, power, toughness, oracle_text, keywords)
KNOWN_CARDS: dict[str, tuple] = {
    # Basic lands
    "Plains":    ("{T}: Add {W}.", 0, "Basic Land — Plains",    None, None, "{T}: Add {W}.",    []),
    "Island":    ("{T}: Add {U}.", 0, "Basic Land — Island",    None, None, "{T}: Add {U}.",    []),
    "Swamp":     ("{T}: Add {B}.", 0, "Basic Land — Swamp",     None, None, "{T}: Add {B}.",    []),
    "Mountain":  ("{T}: Add {R}.", 0, "Basic Land — Mountain",  None, None, "{T}: Add {R}.",    []),
    "Forest":    ("{T}: Add {G}.", 0, "Basic Land — Forest",    None, None, "{T}: Add {G}.",    []),
    # Common utility lands
    "Command Tower":       ("", 0, "Land", None, None, "{T}: Add one mana of any color in your commander's color identity.", []),
    "Sol Ring":            ("{1}", 1, "Artifact", None, None, "{T}: Add {C}{C}.", []),
    "Arcane Signet":       ("{2}", 2, "Artifact", None, None, "{T}: Add one mana of any color in your commander's color identity.", []),
    "Lightning Greaves":   ("{2}", 2, "Artifact — Equipment", None, None, "Equipped creature has haste and shroud. Equip {0}.", ["Haste", "Shroud"]),
    "Swiftfoot Boots":     ("{2}", 2, "Artifact — Equipment", None, None, "Equipped creature has haste and hexproof. Equip {1}.", ["Haste", "Hexproof"]),
    "Swords to Plowshares":("{W}", 1, "Instant", None, None, "Exile target creature.", []),
    "Path to Exile":       ("{W}", 1, "Instant", None, None, "Exile target creature.", []),
    "Cultivate":           ("{2}{G}", 3, "Sorcery", None, None, "Search your library for up to two basic land cards, reveal those cards, and put one onto the battlefield tapped and the other into your hand. Then shuffle.", []),
    "Kodama's Reach":      ("{2}{G}", 3, "Sorcery", None, None, "Search your library for up to two basic land cards, reveal those cards, and put one onto the battlefield tapped and the other into your hand. Then shuffle.", []),
    "Rampant Growth":      ("{1}{G}", 2, "Sorcery", None, None, "Search your library for a basic land card and put that card onto the battlefield tapped. Then shuffle.", []),
    "Wrath of God":        ("{2}{W}{W}", 4, "Sorcery", None, None, "Destroy all creatures. They can't be regenerated.", []),
    "Damnation":           ("{2}{B}{B}", 4, "Sorcery", None, None, "Destroy all creatures. They can't be regenerated.", []),
    "Blasphemous Act":     ("{8}{R}", 9, "Sorcery", None, None, "Blasphemous Act costs {1} less to cast for each creature on the battlefield. Blasphemous Act deals 13 damage to each creature.", []),
    "Counterspell":        ("{U}{U}", 2, "Instant", None, None, "Counter target spell.", []),
    "Cyclonic Rift":       ("{1}{U}", 2, "Instant", None, None, "Return target nonland permanent you don't control to its owner's hand. Overload {6}{U}.", []),
    "Rhystic Study":       ("{2}{U}", 3, "Enchantment", None, None, "Whenever an opponent casts a spell, you may draw a card unless that player pays {1}.", []),
    "Skullclamp":          ("{1}", 1, "Artifact — Equipment", None, None, "Equipped creature gets +1/-1. Whenever equipped creature dies, draw two cards. Equip {1}.", []),
    "Darksteel Ingot":     ("{3}", 3, "Artifact", None, None, "Indestructible. {T}: Add one mana of any color.", ["Indestructible"]),
    "Boros Signet":        ("{2}", 2, "Artifact", None, None, "{1}, {T}: Add {R}{W}.", []),
    "Orzhov Signet":       ("{2}", 2, "Artifact", None, None, "{1}, {T}: Add {W}{B}.", []),
    "Rakdos Signet":       ("{2}", 2, "Artifact", None, None, "{1}, {T}: Add {B}{R}.", []),
    "Gruul Signet":        ("{2}", 2, "Artifact", None, None, "{1}, {T}: Add {R}{G}.", []),
    "Dimir Signet":        ("{2}", 2, "Artifact", None, None, "{1}, {T}: Add {U}{B}.", []),
    "Diabolic Tutor":      ("{2}{B}{B}", 4, "Sorcery", None, None, "Search your library for a card and put that card into your hand. Then shuffle.", []),
    "Krenko, Mob Boss":    ("{2}{R}{R}", 4, "Legendary Creature — Goblin Warrior", 3, 3, "{T}: Create X 1/1 red Goblin creature tokens, where X is the number of Goblins you control.", []),
    "Serra Angel":         ("{3}{W}{W}", 5, "Creature — Angel", 4, 4, "Flying, vigilance", ["Flying", "Vigilance"]),
    "Sol Ring":            ("{1}", 1, "Artifact", None, None, "{T}: Add {C}{C}.", []),
}

# Basic land type → color mapping
LAND_COLORS = {
    "plains": Color.WHITE, "island": Color.BLUE, "swamp": Color.BLACK,
    "mountain": Color.RED, "forest": Color.GREEN,
}

def _parse_int(val) -> int | None:
    try:    return int(val)
    except: return None

def build_card_definition(name: str) -> CardDefinition:
    """
    Build a CardDefinition from a card name.
    Uses KNOWN_CARDS for common cards, otherwise builds a generic definition.
    """
    # Check known cards first
    if name in KNOWN_CARDS:
        mana, cmc, type_line, power, toughness, oracle, keywords = KNOWN_CARDS[name]
        card_types = parse_card_types(type_line)
        colors     = parse_colors([])  # Will infer below
        # Infer colors from mana cost
        for sym, color in [("{W}",Color.WHITE),("{U}",Color.BLUE),("{B}",Color.BLACK),
                           ("{R}",Color.RED),  ("{G}",Color.GREEN)]:
            if sym in mana and color not in colors:
                colors.append(color)
        # Basic lands
        for land_name, color in LAND_COLORS.items():
            if land_name in type_line.lower():
                colors = [color]
                break

        tags = tag_effects(oracle, card_types, keywords)
        return CardDefinition(
            scryfall_id  = f"known-{name.lower().replace(' ','-')}",
            name         = name,
            mana_cost    = mana,
            cmc          = float(cmc),
            colors       = colors,
            card_types   = card_types,
            subtypes     = parse_subtypes(type_line),
            keywords     = keywords,
            oracle_text  = oracle,
            power        = _parse_int(power),
            toughness    = _parse_int(toughness),
            loyalty      = None,
            is_legendary = "legendary" in type_line.lower(),
            effect_tags  = tags,
        )

    # Unknown card — build a generic definition based on name heuristics
    # This allows games to run even without the full Scryfall database
    name_lower = name.lower()

    # Guess type from name patterns
    if any(x in name_lower for x in ["plains","island","swamp","mountain","forest",
                                      "refuge","garrison","basilica","carnarium",
                                      "tower","bog","cave","grove","vale","barren",
                                      "cavern","moor","heath","tundra","taiga",
                                      "savannah","scrubland","badlands","plateau",
                                      "bayou","tropical","underground","volcanic"]):
        type_line = "Land"
        mana, cmc, power, toughness, oracle = "", 0, None, None, "{T}: Add {C}."
        keywords = []
    elif any(x in name_lower for x in ["angel","demon","dragon","elf","goblin","zombie",
                                        "knight","soldier","wizard","rogue","shaman"]):
        type_line = "Creature"
        mana, cmc, power, toughness = "{3}{G}", 3, 2, 2
        oracle, keywords = "", []
    elif any(x in name_lower for x in ["signet","ring","staff","sword","axe","shield",
                                        "sphere","ingot","chalice","medallion"]):
        type_line = "Artifact"
        mana, cmc, power, toughness = "{2}", 2, None, None
        oracle, keywords = "{T}: Add {C}.", []
    elif any(x in name_lower for x in ["wrath","damnation","judgment","devastation",
                                        "obliterate","armageddon","plague"]):
        type_line = "Sorcery"
        mana, cmc, power, toughness = "{3}{W}{W}", 5, None, None
        oracle, keywords = "Destroy all creatures.", []
    elif any(x in name_lower for x in ["growth","reach","cultivate","harvest",
                                        "rampant","farseek","expedition"]):
        type_line = "Sorcery"
        mana, cmc, power, toughness = "{1}{G}", 2, None, None
        oracle, keywords = ("Search your library for a basic land card and put it "
                           "onto the battlefield tapped.", [])
    else:
        # Generic creature
        type_line = "Creature"
        mana, cmc, power, toughness = "{2}{G}", 3, 2, 2
        oracle, keywords = "", []

    card_types = parse_card_types(type_line)
    tags = tag_effects(oracle, card_types, keywords)

    return CardDefinition(
        scryfall_id  = f"generic-{name.lower().replace(' ','-')[:30]}",
        name         = name,
        mana_cost    = mana,
        cmc          = float(cmc),
        colors       = [],
        card_types   = card_types,
        subtypes     = parse_subtypes(type_line),
        keywords     = keywords,
        oracle_text  = oracle,
        power        = _parse_int(power),
        toughness    = _parse_int(toughness),
        loyalty      = None,
        is_legendary = "legendary" in type_line.lower(),
        effect_tags  = tags,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Load the cards.json downloaded by fetch_precons.py
# ─────────────────────────────────────────────────────────────────────────────

CARDS_JSON_PATH = Path(__file__).parent / "card_cache" / "cards.json"
_cards_db: dict | None = None

def _get_cards_db() -> dict:
    """Load cards.json once and cache it."""
    global _cards_db
    if _cards_db is not None:
        return _cards_db
    if CARDS_JSON_PATH.exists():
        with open(CARDS_JSON_PATH) as f:
            _cards_db = json.load(f)
        print(f"  [CardDB] Loaded {len(_cards_db):,} cards from cards.json")
    else:
        _cards_db = {}
        print(f"  [CardDB] cards.json not found — using generic definitions only")
        print(f"           Run fetch_precons.py to download full card data")
    return _cards_db


def build_card_definition_from_db(name: str, db: dict):
    """
    Build a CardDefinition from the cards.json downloaded by fetch_precons.py.
    Returns None if card not in database.
    """
    data = db.get(name)
    if not data:
        return None

    type_line  = data.get("type_line", "")
    oracle     = data.get("oracle_text", "")
    keywords   = data.get("keywords", [])
    card_types = parse_card_types(type_line)
    colors     = parse_colors(data.get("colors", []))
    tags       = tag_effects(oracle, card_types, keywords)

    def _parse_pt(val):
        try:    return int(val)
        except: return None

    return CardDefinition(
        scryfall_id  = data.get("scryfall_id", f"scryfall-{name}"),
        name         = name,
        mana_cost    = data.get("mana_cost", ""),
        cmc          = float(data.get("cmc", 0)),
        colors       = colors,
        card_types   = card_types,
        subtypes     = parse_subtypes(type_line),
        keywords     = keywords,
        oracle_text  = oracle,
        power        = _parse_pt(data.get("power")),
        toughness    = _parse_pt(data.get("toughness")),
        loyalty      = _parse_pt(data.get("loyalty")),
        is_legendary = data.get("is_legendary", False),
        is_commander_legal = (
            data.get("legalities", {}).get("commander", "legal") == "legal"
        ),
        effect_tags  = tags,
    )


def build_deck_from_precon(deck_data: dict) -> tuple[list[CardDefinition], list[str]]:
    """
    Build a list of CardDefinitions from a precon deck entry.
    Uses cards.json from fetch_precons.py if available, falls back to generics.
    Returns (card_definitions, commander_names).
    """
    cards      = []
    commanders = deck_data.get("commanders", [])
    db         = _get_cards_db()

    for card_name in deck_data.get("cards", []):
        # Try full Scryfall data first
        defn = build_card_definition_from_db(card_name, db)
        # Fall back to generic if not found
        if defn is None:
            defn = build_card_definition(card_name)
        cards.append(defn)

    return cards, commanders


# ─────────────────────────────────────────────────────────────────────────────
# Run a game
# ─────────────────────────────────────────────────────────────────────────────

def run_precon_game(
    deck1_name: str,
    deck2_name: str,
    precons:    dict,
    turn_limit: int  = 50,
    verbose:    bool = False,
    seed:       int | None = None,
):
    deck1_data = find_deck(precons, deck1_name)
    deck2_data = find_deck(precons, deck2_name)

    if not deck1_data:
        print(f"❌ Deck not found: '{deck1_name}'")
        print("Available decks:")
        for sname, dname in list_all_decks(precons)[:20]:
            print(f"  {dname} ({sname})")
        return None

    if not deck2_data:
        print(f"❌ Deck not found: '{deck2_name}'")
        return None

    print(f"\n{'='*60}")
    print(f"  {deck1_data['name']}  vs  {deck2_data['name']}")
    print(f"  Turn limit: {turn_limit}  |  Seed: {seed}")
    print('='*60)

    cards1, cmds1 = build_deck_from_precon(deck1_data)
    cards2, cmds2 = build_deck_from_precon(deck2_data)

    print(f"  Deck 1: {deck1_data['name']} — {len(cards1)} cards, "
          f"commanders: {cmds1}")
    print(f"  Deck 2: {deck2_data['name']} — {len(cards2)} cards, "
          f"commanders: {cmds2}")

    db     = _get_cards_db()
    known1 = sum(1 for c in deck1_data.get("cards",[]) if c in db)
    known2 = sum(1 for c in deck2_data.get("cards",[]) if c in db)
    print(f"  Known card data: {known1}/{len(cards1)} | {known2}/{len(cards2)}")

    # Tag distribution
    all_tags1 = {}
    for c in cards1:
        for t in c.effect_tags:
            all_tags1[t] = all_tags1.get(t, 0) + 1
    tag_summary = ", ".join(f"{t}:{n}" for t,n in
                            sorted(all_tags1.items(), key=lambda x:-x[1])[:8])
    print(f"  Deck 1 effect tags: {tag_summary}")

    p1 = Player(player_id=0, name=deck1_data['name'][:20])
    p2 = Player(player_id=1, name=deck2_data['name'][:20])

    sim = GameSimulator(
        player_decks=[
            (p1, cards1, cmds1),
            (p2, cards2, cmds2),
        ],
        turn_limit = turn_limit,
        verbose    = verbose,
        seed       = seed,
    )

    result = sim.run()

    print(f"\n  Winner:  {result.winner_name}")
    print(f"  Turns:   {result.turns}")
    print(f"  Timeout: {result.timeout}")
    print(f"  Records: {len(result.records)}")

    # Show final life totals
    for p in sim.state.players:
        print(f"  {p.name}: {p.life} life")

    # Show any errors from the log
    errors = [l for l in result.log if "[ERROR]" in l]
    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for e in errors[:5]:
            print(f"    {e}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run Commander precon games")
    parser.add_argument("--p1",     default="Heavenly Inferno",
                        help="Deck 1 name (partial match)")
    parser.add_argument("--p2",     default="Devour for Power",
                        help="Deck 2 name (partial match)")
    parser.add_argument("--games",  type=int, default=1,
                        help="Number of games to run")
    parser.add_argument("--turns",  type=int, default=50,
                        help="Turn limit per game")
    parser.add_argument("--verbose",action="store_true",
                        help="Print full game log")
    parser.add_argument("--list",   action="store_true",
                        help="List all available decks and exit")
    parser.add_argument("--seed",   type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    precons = load_precons()

    if args.list:
        print(f"\nAll precon decks ({precons['meta']['total_decks']} total):\n")
        for set_name, deck_name in list_all_decks(precons):
            print(f"  {deck_name:40}  ({set_name})")
        return

    results = []
    for i in range(args.games):
        seed = args.seed if args.seed is not None else random.randint(0, 99999)
        result = run_precon_game(
            deck1_name = args.p1,
            deck2_name = args.p2,
            precons    = precons,
            turn_limit = args.turns,
            verbose    = args.verbose,
            seed       = seed,
        )
        if result:
            results.append(result)

    if len(results) > 1:
        summarize_results(results)


if __name__ == "__main__":
    main()
