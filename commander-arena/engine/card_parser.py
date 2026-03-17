"""
card_parser.py — Loads card data from Scryfall and tags effects

This replaces ALL of the fetchCard / Claude-based card parsing in the JS app.
We download card data once, store it locally as JSON, and never call Scryfall
at game time. Zero API cost for card data.

The critical function here is tag_effects() — it reads oracle text and assigns
effect_tags like ["ramp", "removal", "draw"] to each card. These tags are what
the heuristic AI and ML encoder use to understand what a card does.
"""

from __future__ import annotations
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Optional

from game_state import CardDefinition, Color, CardType


# ─────────────────────────────────────────────────────────────────────────────
# Local cache — we store downloaded card data so we never hit Scryfall twice
# ─────────────────────────────────────────────────────────────────────────────

CACHE_DIR = Path(__file__).parent / "card_cache"
CACHE_DIR.mkdir(exist_ok=True)

CARD_DB_PATH    = CACHE_DIR / "cards.json"       # All cards we've loaded
PRECON_DB_PATH  = CACHE_DIR / "precons.json"     # Precon deck lists


# ─────────────────────────────────────────────────────────────────────────────
# Effect tagging — the AI's understanding of what a card DOES
# ─────────────────────────────────────────────────────────────────────────────

# Pattern → tag mappings
# Ordered from most specific to least specific
# Each pattern is checked against the card's oracle text (lowercased)

EFFECT_PATTERNS: list[tuple[str, str]] = [
    # Ramp (mana acceleration)
    (r"add \{[wubrgc]\}",              "mana"),    # Basic mana ability
    (r"search your library for a.* land", "ramp"),
    (r"put.* land.* onto the battlefield", "ramp"),
    (r"add.* mana",                    "ramp"),

    # Card draw
    (r"draw (\w+) cards?",             "draw"),
    (r"draw cards? equal to",          "draw"),
    (r"look at the top",               "scry"),

    # Removal (targeted)
    (r"destroy target",                "removal"),
    (r"exile target",                  "removal"),
    (r"return target.*to.*hand",       "bounce"),
    (r"deal \d+ damage to",            "damage"),
    (r"deal .* damage to any target",  "removal"),

    # Board wipes (mass removal)
    (r"destroy all",                   "board_wipe"),
    (r"exile all",                     "board_wipe"),
    (r"each (creature|player|opponent) (gets?|takes?|loses?)", "board_wipe"),

    # Counters
    (r"counter target spell",          "counter"),
    (r"counter target (creature|instant|sorcery|artifact|enchantment)", "counter"),

    # Tutors (search effects)
    (r"search your library for a? ?card", "tutor"),

    # Token creation
    (r"create (a|an|\d+|x) .*token",   "token"),

    # Life gain
    (r"you gain \d+ life",             "lifegain"),
    (r"you gain life equal to",        "lifegain"),
    (r"lifelink",                      "lifegain"),

    # Graveyard recursion
    (r"return.* from.* graveyard.* to", "recursion"),
    (r"return target.* card from",     "recursion"),

    # Protection / hexproof
    (r"hexproof",                      "protection"),
    (r"indestructible",                "protection"),
    (r"protection from",               "protection"),

    # Pump (stat boosts)
    (r"gets? \+\d+/\+\d+",            "pump"),
    (r"put \d+ \+1/\+1 counters?",    "counters"),

    # Discard / hand hate
    (r"discard",                       "discard"),

    # Extra turns / extra combat
    (r"take an extra turn",            "extra_turn"),
    (r"additional combat phase",       "extra_combat"),

    # Copy effects
    (r"copy target",                   "copy"),
    (r"create a (token that|copy)",    "copy"),

    # Sacrifice outlets
    (r"sacrifice (a|another) creature", "sacrifice"),

    # Landfall / triggers
    (r"whenever.* land enters",        "landfall"),
    (r"whenever you cast",             "spellslinger"),
    (r"whenever a creature (enters|dies)", "etb_trigger"),

    # Commander-specific
    (r"commander tax",                 "commander_matters"),
    (r"command zone",                  "commander_matters"),
    (r"partner",                       "partner"),
]


def tag_effects(oracle_text: str, card_types: list[CardType], keywords: list[str]) -> list[str]:
    """
    Given a card's oracle text, return a list of effect tags.
    These tags are what the AI uses to understand what the card does
    without needing to parse the full rules text at decision time.
    """
    tags: set[str] = set()
    text = oracle_text.lower()

    # Check oracle text patterns
    for pattern, tag in EFFECT_PATTERNS:
        if re.search(pattern, text):
            tags.add(tag)

    # Check keywords (from Scryfall's parsed keyword list)
    kw_lower = [k.lower() for k in keywords]
    if "flying"      in kw_lower: tags.add("evasion")
    if "trample"     in kw_lower: tags.add("evasion")
    if "deathtouch"  in kw_lower: tags.add("deathtouch")
    if "lifelink"    in kw_lower: tags.add("lifegain")
    if "haste"       in kw_lower: tags.add("haste")
    if "vigilance"   in kw_lower: tags.add("vigilance")
    if "hexproof"    in kw_lower: tags.add("protection")
    if "indestructible" in kw_lower: tags.add("protection")

    # Type-based inference
    if CardType.LAND in card_types:
        tags.add("mana")

    return sorted(tags)


# ─────────────────────────────────────────────────────────────────────────────
# Scryfall data loading
# ─────────────────────────────────────────────────────────────────────────────

def parse_colors(scryfall_colors: list[str]) -> list[Color]:
    mapping = {"W": Color.WHITE, "U": Color.BLUE, "B": Color.BLACK,
               "R": Color.RED,   "G": Color.GREEN}
    return [mapping[c] for c in scryfall_colors if c in mapping]


def parse_card_types(type_line: str) -> list[CardType]:
    types = []
    line = type_line.lower()
    if "creature"     in line: types.append(CardType.CREATURE)
    if "instant"      in line: types.append(CardType.INSTANT)
    if "sorcery"      in line: types.append(CardType.SORCERY)
    if "enchantment"  in line: types.append(CardType.ENCHANTMENT)
    if "artifact"     in line: types.append(CardType.ARTIFACT)
    if "land"         in line: types.append(CardType.LAND)
    if "planeswalker" in line: types.append(CardType.PLANESWALKER)
    if "battle"       in line: types.append(CardType.BATTLE)
    return types


def parse_subtypes(type_line: str) -> list[str]:
    """Extract subtypes from "Legendary Creature — Elf Druid" → ["Elf", "Druid"]"""
    if "—" in type_line:
        return type_line.split("—")[1].strip().split()
    if "-" in type_line and type_line.index("-") > 10:
        return type_line.split("-")[1].strip().split()
    return []


def scryfall_card_to_definition(data: dict) -> CardDefinition:
    """
    Convert a raw Scryfall API response to a CardDefinition.
    Handles double-faced cards by using face 0.
    """
    # Double-faced cards — use front face for most data
    if "card_faces" in data and data.get("layout") in ("transform", "modal_dfc", "flip"):
        face = data["card_faces"][0]
        oracle_text = face.get("oracle_text", "")
        mana_cost   = face.get("mana_cost", data.get("mana_cost", ""))
        colors      = parse_colors(face.get("colors", data.get("colors", [])))
        power       = _parse_pt(face.get("power"))
        toughness   = _parse_pt(face.get("toughness"))
        type_line   = face.get("type_line", data.get("type_line", ""))
    else:
        oracle_text = data.get("oracle_text", "")
        mana_cost   = data.get("mana_cost", "")
        colors      = parse_colors(data.get("colors", []))
        power       = _parse_pt(data.get("power"))
        toughness   = _parse_pt(data.get("toughness"))
        type_line   = data.get("type_line", "")

    card_types = parse_card_types(type_line)
    keywords   = data.get("keywords", [])
    effect_tags = tag_effects(oracle_text, card_types, keywords)

    return CardDefinition(
        scryfall_id  = data["id"],
        name         = data["name"],
        mana_cost    = mana_cost,
        cmc          = float(data.get("cmc", 0)),
        colors       = colors,
        card_types   = card_types,
        subtypes     = parse_subtypes(type_line),
        keywords     = keywords,
        oracle_text  = oracle_text,
        power        = power,
        toughness    = toughness,
        loyalty      = _parse_pt(data.get("loyalty")),
        is_legendary = "Legendary" in type_line,
        is_commander_legal = _check_commander_legal(data),
        effect_tags  = effect_tags,
    )


def _parse_pt(val) -> Optional[int]:
    """Parse power/toughness — can be *, X, or a number."""
    if val is None: return None
    try:    return int(val)
    except: return None   # *, X, etc.


def _check_commander_legal(data: dict) -> bool:
    """Check if card is legal in Commander format."""
    legalities = data.get("legalities", {})
    return legalities.get("commander", "not_legal") == "legal"


# ─────────────────────────────────────────────────────────────────────────────
# Card database — persistent local storage
# ─────────────────────────────────────────────────────────────────────────────

class CardDatabase:
    """
    Local card database. Loads from disk, downloads from Scryfall if missing.
    At game time: zero network calls, zero API cost.
    """

    def __init__(self):
        self._cards: dict[str, CardDefinition] = {}   # name → CardDefinition
        self._load_from_disk()

    def _load_from_disk(self):
        """Load any previously downloaded cards."""
        if not CARD_DB_PATH.exists():
            return
        with open(CARD_DB_PATH) as f:
            raw = json.load(f)
        # Detect format: fetch_precons saves "type_line" (Scryfall raw).
        # card_parser saves "card_types" (internal enum format).
        # If Scryfall raw format, skip — test_precon_game reads it directly.
        sample = next(iter(raw.values()), {}) if raw else {}
        if "type_line" in sample and "card_types" not in sample:
            return
        loaded = 0
        for name, data in raw.items():
            try:
                self._cards[name] = self._dict_to_definition(data)
                loaded += 1
            except Exception:
                pass
        if loaded:
            print(f"[CardDB] Loaded {loaded} cards from cache")

    def _save_to_disk(self):
        """Persist current cards to disk."""
        raw = {name: self._definition_to_dict(defn)
               for name, defn in self._cards.items()}
        with open(CARD_DB_PATH, "w") as f:
            json.dump(raw, f, indent=2)

    def get(self, name: str) -> Optional[CardDefinition]:
        """Get a card by name. Returns None if not in database."""
        return self._cards.get(name)

    def fetch_and_store(self, name: str) -> Optional[CardDefinition]:
        """
        Download a card from Scryfall and store it locally.
        Only called during deck loading — never during a live game.
        Respects Scryfall's rate limit (50-100ms between requests).
        """
        if name in self._cards:
            return self._cards[name]

        url = f"https://api.scryfall.com/cards/named?exact={urllib.parse.quote(name)}"
        try:
            time.sleep(0.1)  # Scryfall rate limit — be a good citizen
            req = urllib.request.Request(url, headers={"User-Agent": "CommanderArena/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.load(resp)

            defn = scryfall_card_to_definition(data)
            self._cards[name] = defn
            self._save_to_disk()
            return defn

        except Exception as e:
            print(f"[CardDB] Failed to fetch {name}: {e}")
            return None

    def load_precon_deck(self, deck_name: str, card_names: list[str]) -> list[CardDefinition]:
        """
        Load all cards for a precon deck.
        Downloads any missing cards from Scryfall (once only).
        Returns list of CardDefinitions in deck order.
        """
        result = []
        missing = [n for n in card_names if n not in self._cards]

        if missing:
            print(f"[CardDB] Downloading {len(missing)} cards for {deck_name}...")
            for i, name in enumerate(missing):
                defn = self.fetch_and_store(name)
                if defn:
                    result.append(defn)
                if i % 10 == 0:
                    print(f"  {i}/{len(missing)}...")

        for name in card_names:
            defn = self._cards.get(name)
            if defn:
                result.append(defn)

        return result

    # ── Serialization helpers ──────────────────────────────────────────────

    def _definition_to_dict(self, defn: CardDefinition) -> dict:
        return {
            "scryfall_id":  defn.scryfall_id,
            "name":         defn.name,
            "mana_cost":    defn.mana_cost,
            "cmc":          defn.cmc,
            "colors":       [c.value for c in defn.colors],
            "card_types":   [t.name for t in defn.card_types],
            "subtypes":     defn.subtypes,
            "keywords":     defn.keywords,
            "oracle_text":  defn.oracle_text,
            "power":        defn.power,
            "toughness":    defn.toughness,
            "loyalty":      defn.loyalty,
            "is_legendary": defn.is_legendary,
            "is_commander_legal": defn.is_commander_legal,
            "effect_tags":  defn.effect_tags,
        }

    def _dict_to_definition(self, d: dict) -> CardDefinition:
        color_map = {"W": Color.WHITE, "U": Color.BLUE, "B": Color.BLACK,
                     "R": Color.RED,   "G": Color.GREEN, "C": Color.COLORLESS}
        return CardDefinition(
            scryfall_id  = d["scryfall_id"],
            name         = d["name"],
            mana_cost    = d["mana_cost"],
            cmc          = d["cmc"],
            colors       = [color_map[c] for c in d["colors"] if c in color_map],
            card_types   = [CardType[t] for t in d["card_types"]],
            subtypes     = d["subtypes"],
            keywords     = d["keywords"],
            oracle_text  = d["oracle_text"],
            power        = d["power"],
            toughness    = d["toughness"],
            loyalty      = d.get("loyalty"),
            is_legendary = d["is_legendary"],
            is_commander_legal = d.get("is_commander_legal", True),
            effect_tags  = d.get("effect_tags", []),
        )


# Singleton — import this everywhere
import urllib.parse
card_db = CardDatabase()
