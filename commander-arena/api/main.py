"""
api/main.py — FastAPI Bridge

Connects the Python game engine to the JS frontend.
Runs alongside the existing Express server (different port).

Endpoints:
  POST /api/engine/decide   — AI makes a turn decision, returns same JSON format
                              as the Claude API response so frontend needs minimal changes
  POST /api/engine/game     — Start a new headless game (for training/testing)
  GET  /api/engine/health   — Health check
  GET  /api/engine/decks    — List all available precon decks

The /api/engine/decide endpoint accepts:
  {
    "player_name": "Kaalia",
    "hand": [{"name": "Sol Ring", "mana_cost": "{1}", ...}],
    "battlefield": [...],
    "opponents": [...],
    "life": 40,
    "turn": 5,
    "phase": "MAIN1",
    "mana_available": 4,
    "commander": "Kaalia of the Vast"
  }

And returns the same format as Claude's API so the frontend works unchanged:
  {
    "content": [{
      "type": "text",
      "text": '{"plays":["Sol Ring"],"attackers":["Dragon"],"attack_target":"Josh","reasoning":"..."}'
    }]
  }

Usage:
  cd api
  pip install fastapi uvicorn
  uvicorn main:app --port 3002 --reload

Then in server.js, add:
  app.post('/api/engine/decide', proxy_to_python)
"""

import sys
import json
import random
import uuid
from pathlib import Path
from typing import Optional

# Add engine to path
ENGINE_DIR = Path(__file__).parent.parent / "engine"
sys.path.insert(0, str(ENGINE_DIR))
sys.path.insert(0, str(Path(__file__).parent.parent / "ml"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import engine components
from game_state import (
    GameState, Player, CardDefinition, CardInstance,
    Color, CardType, Zone, Phase
)
from heuristic_ai import HeuristicAI
from actions import ActionGenerator, ActionType
from rules import RulesEngine

# Load precon data
PRECONS_PATH = ENGINE_DIR / "card_cache" / "precons.json"
CARDS_PATH   = ENGINE_DIR / "card_cache" / "cards.json"

_precons: dict = {}
_cards_db: dict = {}

def _load_data():
    global _precons, _cards_db
    if PRECONS_PATH.exists():
        with open(PRECONS_PATH) as f:
            _precons = json.load(f)
        print(f"[API] Loaded {_precons.get('meta',{}).get('total_decks',0)} precon decks")
    if CARDS_PATH.exists():
        with open(CARDS_PATH) as f:
            _cards_db = json.load(f)
        print(f"[API] Loaded {len(_cards_db):,} cards")

_load_data()

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Commander Arena Engine",
    description="Python game engine API for Commander Arena",
    version="1.0.0",
)

# Allow requests from the Express server and browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", "http://localhost:3000",
                   "http://localhost:5173", "*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────────────────────────

class CardInfo(BaseModel):
    name:       str
    mana_cost:  str  = ""
    type_line:  str  = ""
    power:      Optional[str] = None
    toughness:  Optional[str] = None
    oracle_text:str  = ""
    keywords:   list[str] = []
    tapped:     bool = False

class OpponentInfo(BaseModel):
    name:        str
    life:        int
    battlefield: list[CardInfo] = []
    hand_size:   int = 7

class DecideRequest(BaseModel):
    """
    Game state snapshot sent from the JS frontend.
    Matches what aiDecideTurn() in index.html already knows about.
    """
    player_name:    str
    commander:      str  = ""
    hand:           list[CardInfo] = []
    battlefield:    list[CardInfo] = []
    opponents:      list[OpponentInfo] = []
    life:           int  = 40
    turn:           int  = 1
    phase:          str  = "MAIN1"
    mana_available: int  = 0
    lands_played:   bool = False

class DecideResponse(BaseModel):
    """
    Response formatted to match Claude API response structure.
    Frontend code that reads data.content[0].text will work unchanged.
    """
    content: list[dict]


# ─────────────────────────────────────────────────────────────────────────────
# Card building helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_card_types(type_line: str) -> list:
    types = []
    line  = type_line.lower()
    if "creature"    in line: types.append(CardType.CREATURE)
    if "instant"     in line: types.append(CardType.INSTANT)
    if "sorcery"     in line: types.append(CardType.SORCERY)
    if "enchantment" in line: types.append(CardType.ENCHANTMENT)
    if "artifact"    in line: types.append(CardType.ARTIFACT)
    if "land"        in line: types.append(CardType.LAND)
    if "planeswalker"in line: types.append(CardType.PLANESWALKER)
    return types or [CardType.CREATURE]

def _parse_colors(mana_cost: str) -> list:
    colors = []
    if "{W}" in mana_cost: colors.append(Color.WHITE)
    if "{U}" in mana_cost: colors.append(Color.BLUE)
    if "{B}" in mana_cost: colors.append(Color.BLACK)
    if "{R}" in mana_cost: colors.append(Color.RED)
    if "{G}" in mana_cost: colors.append(Color.GREEN)
    return colors

def _parse_cmc(mana_cost: str) -> float:
    import re
    total = 0
    for m in re.findall(r'\{([^}]+)\}', mana_cost):
        if m.isdigit(): total += int(m)
        elif m.upper() not in ('X','Y','Z'): total += 1
    return float(total)

def _parse_pt(val) -> Optional[int]:
    try:    return int(val)
    except: return None

def card_info_to_definition(card: CardInfo) -> CardDefinition:
    """Convert a CardInfo from the frontend into a CardDefinition."""
    # Try cards.json first for accurate data
    db_data = _cards_db.get(card.name)
    if db_data:
        type_line = db_data.get("type_line", card.type_line)
        oracle    = db_data.get("oracle_text", card.oracle_text)
        keywords  = db_data.get("keywords", card.keywords)
        mana_cost = db_data.get("mana_cost", card.mana_cost)
    else:
        type_line = card.type_line
        oracle    = card.oracle_text
        keywords  = card.keywords
        mana_cost = card.mana_cost

    from card_parser import tag_effects, parse_card_types, parse_subtypes
    card_types = parse_card_types(type_line) or _parse_card_types(type_line)
    subtypes   = parse_subtypes(type_line)
    tags = tag_effects(oracle, card_types, keywords)

    return CardDefinition(
        scryfall_id  = f"api-{card.name.lower().replace(' ','-')[:30]}",
        name         = card.name,
        mana_cost    = mana_cost,
        cmc          = _parse_cmc(mana_cost),
        colors       = _parse_colors(mana_cost),
        card_types   = card_types,
        subtypes     = [],
        keywords     = keywords,
        oracle_text  = oracle,
        power        = _parse_pt(card.power),
        toughness    = _parse_pt(card.toughness),
        loyalty      = None,
        is_legendary = "legendary" in type_line.lower(),
        effect_tags  = tags,
    )

def build_instance(defn: CardDefinition, player_id: int,
                   zone: Zone, tapped: bool = False) -> CardInstance:
    inst = CardInstance(
        instance_id   = str(uuid.uuid4()),
        definition    = defn,
        owner_id      = player_id,
        controller_id = player_id,
        zone          = zone,
        tapped        = tapped,
        summoning_sick= False,
    )
    return inst


# ─────────────────────────────────────────────────────────────────────────────
# Core decision logic
# ─────────────────────────────────────────────────────────────────────────────

def make_ai_decision(req: DecideRequest) -> dict:
    """
    Build a minimal GameState from the frontend snapshot,
    run the heuristic AI, and return a decision plan.

    Returns dict: {plays, attackers, attack_target, reasoning}
    """
    # ── Build state ────────────────────────────────────────────────
    state = GameState()

    # AI player (id=1, same as frontend convention where 0=human)
    ai_player = Player(player_id=1, name=req.player_name, life=req.life)
    # Human player (placeholder — we need them in state for targeting)
    opp_life  = req.opponents[0].life if req.opponents else 40
    opp_name  = req.opponents[0].name if req.opponents else "Human"
    human     = Player(player_id=0, name=opp_name, life=opp_life)

    state.players       = [human, ai_player]
    state.active_player = 1
    state.priority_player = 1

    # Map phase string to Phase enum
    phase_map = {
        "MAIN1": Phase.MAIN1, "MAIN2": Phase.MAIN2,
        "DECLARE_ATTACKERS": Phase.DECLARE_ATTACKERS,
        "BEGIN_COMBAT": Phase.BEGIN_COMBAT,
        "UPKEEP": Phase.UPKEEP, "DRAW": Phase.DRAW,
        "END": Phase.END,
    }
    state.phase      = phase_map.get(req.phase, Phase.MAIN1)
    state.turn_number = req.turn

    # ── Populate AI hand ───────────────────────────────────────────
    for card_info in req.hand:
        defn = card_info_to_definition(card_info)
        inst = build_instance(defn, 1, Zone.HAND)
        state.all_instances[inst.instance_id] = inst
        ai_player.hand.append(inst.instance_id)

    # ── Populate AI battlefield ────────────────────────────────────
    for card_info in req.battlefield:
        defn = card_info_to_definition(card_info)
        inst = build_instance(defn, 1, Zone.BATTLEFIELD, tapped=card_info.tapped)
        state.all_instances[inst.instance_id] = inst

    # ── Populate opponent battlefields ────────────────────────────
    for i, opp_info in enumerate(req.opponents):
        opp_pid = i  # 0, 1, 2... (human is 0)
        for card_info in opp_info.battlefield:
            defn = card_info_to_definition(card_info)
            inst = build_instance(defn, opp_pid, Zone.BATTLEFIELD,
                                  tapped=card_info.tapped)
            state.all_instances[inst.instance_id] = inst

    # ── Set land count ─────────────────────────────────────────────
    ai_player.lands_played_this_turn = 1 if req.lands_played else 0

    # ── Run heuristic AI ───────────────────────────────────────────
    ai = HeuristicAI(player_id=1, randomness=0.05)

    # Get legal actions
    gen     = ActionGenerator(state)
    actions = gen.get_legal_actions()

    # Score and rank all actions
    scored = [(ai._score_action(state, a), a) for a in actions]
    scored.sort(key=lambda x: x[0], reverse=True)

    # ── Build decision plan ────────────────────────────────────────
    plays        = []
    attackers    = []
    attack_target = req.opponents[0].name if req.opponents else "none"
    reasoning_parts = []

    # Cards to play this turn (sorcery speed, positive score)
    for score, action in scored:
        if score <= 0:
            break
        if action.action_type in (ActionType.CAST_SPELL, ActionType.CAST_COMMANDER,
                                   ActionType.PLAY_LAND):
            inst = state.get_instance(action.card_iid)
            if inst and inst.definition.name not in plays:
                plays.append(inst.definition.name)
                reasoning_parts.append(
                    f"Cast {inst.definition.name} (score: {score:.1f})"
                )
        elif action.action_type == ActionType.DECLARE_ATTACKER:
            inst = state.get_instance(action.card_iid)
            if inst and inst.definition.name not in attackers:
                attackers.append(inst.definition.name)
                reasoning_parts.append(
                    f"Attack with {inst.definition.name} (score: {score:.1f})"
                )
                if action.target_pids:
                    tgt = state.get_player(action.target_pids[0])
                    if tgt:
                        attack_target = tgt.name

    reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Pass — no good actions available"

    return {
        "plays":         plays,
        "attackers":     attackers,
        "attack_target": attack_target,
        "spell_targets": {},
        "reasoning":     reasoning,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/engine/health")
def health():
    return {
        "status":  "ok",
        "decks":   _precons.get("meta", {}).get("total_decks", 0),
        "cards":   len(_cards_db),
        "engine":  "heuristic_v1",
    }


@app.get("/api/engine/decks")
def list_decks():
    """List all available precon decks."""
    decks = []
    for set_name, set_decks in _precons.get("sets", {}).items():
        for deck_name, deck_data in set_decks.items():
            decks.append({
                "name":      deck_name,
                "set":       set_name,
                "commanders": deck_data.get("commanders", []),
                "colors":    deck_data.get("colors", []),
            })
    return {"decks": decks, "total": len(decks)}


@app.post("/api/engine/decide", response_model=DecideResponse)
def decide(req: DecideRequest):
    """
    Make an AI turn decision based on the current game state.

    Returns response in Claude API format so the frontend works unchanged:
    { "content": [{ "type": "text", "text": "{...json plan...}" }] }
    """
    try:
        plan = make_ai_decision(req)
        plan_json = json.dumps(plan)

        # Wrap in Claude API response format
        return DecideResponse(content=[{
            "type": "text",
            "text": plan_json,
        }])

    except Exception as e:
        import traceback
        print(f"[/api/engine/decide] Error: {e}")
        traceback.print_exc()
        # Return a safe fallback — pass turn
        fallback = json.dumps({
            "plays": [], "attackers": [], "attack_target": "none",
            "spell_targets": {}, "reasoning": f"Engine error: {str(e)}"
        })
        return DecideResponse(content=[{"type": "text", "text": fallback}])


@app.post("/api/engine/game")
def run_headless_game(body: dict):
    """
    Run a complete headless game between two precon decks.
    Used for training data generation and testing.

    Body: { "deck1": "Heavenly Inferno", "deck2": "Devour for Power",
            "turns": 50, "seed": 42 }
    """
    try:
        # Import here to avoid circular imports at module level
        sys.path.insert(0, str(ENGINE_DIR))
        from simulator import GameSimulator
        from test_precon_game import (
            find_deck, build_deck_from_precon, _get_cards_db
        )

        deck1_name = body.get("deck1", "Heavenly Inferno")
        deck2_name = body.get("deck2", "Devour for Power")
        turn_limit = body.get("turns", 50)
        seed       = body.get("seed", random.randint(0, 99999))

        deck1_data = find_deck(_precons, deck1_name)
        deck2_data = find_deck(_precons, deck2_name)

        if not deck1_data:
            raise HTTPException(400, f"Deck not found: {deck1_name}")
        if not deck2_data:
            raise HTTPException(400, f"Deck not found: {deck2_name}")

        cards1, cmds1 = build_deck_from_precon(deck1_data)
        cards2, cmds2 = build_deck_from_precon(deck2_data)

        p1 = Player(player_id=0, name=deck1_data["name"][:20])
        p2 = Player(player_id=1, name=deck2_data["name"][:20])

        sim = GameSimulator(
            player_decks=[(p1, cards1, cmds1), (p2, cards2, cmds2)],
            turn_limit=turn_limit,
            seed=seed,
        )
        result = sim.run()

        return {
            "winner":   result.winner_name,
            "turns":    result.turns,
            "timeout":  result.timeout,
            "records":  len(result.records),
            "life": {
                p.name: p.life
                for p in sim.state.players
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))



@app.get("/api/engine/deck/{deck_name}")
def get_deck(deck_name: str):
    """Return full card list for a named precon deck."""
    from urllib.parse import unquote
    name = unquote(deck_name)
    for set_name, set_decks in _precons.get("sets", {}).items():
        for dname, deck_data in set_decks.items():
            if dname.lower() == name.lower():
                return {
                    "name":       dname,
                    "set":        set_name,
                    "commanders": deck_data.get("commanders", []),
                    "colors":     deck_data.get("colors", []),
                    "cards":      deck_data.get("cards", []),
                    "card_count": deck_data.get("card_count", 0),
                }
    raise HTTPException(404, f"Deck not found: {name}")

# ─────────────────────────────────────────────────────────────────────────────
# Run directly
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=3002, reload=True)
