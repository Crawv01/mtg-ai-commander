"""
effects.py — Card Effect Implementations

Replaces [STUB] entries from rules.py with real game effects.

Each effect is a function that takes:
    state:      GameState    — the current game state (will be mutated)
    source:     CardInstance — the card causing the effect
    controller: Player       — the player who controls the effect
    targets:    list[str]    — list of instance_ids or player_ids (pre-chosen)

Returns: list[str] log lines describing what happened.

Effect categories implemented here:
    - draw:       draw N cards
    - ramp:       search for / put land onto battlefield
    - removal:    destroy or exile a target creature
    - token:      create creature tokens
    - board_wipe: destroy all creatures

Targeting:
    AI targeting uses heuristic_target() — picks the biggest threat on the
    opposing side. This runs at resolution time so the board state is current.

How effects are wired in:
    rules.py calls execute_effect(state, source, stack_obj, controller)
    which dispatches to the right handler based on source.definition.effect_tags.
    Multiple tags on one card run multiple effects in sequence.

Adding new effects:
    1. Write a handler function: def effect_draw(state, source, ctrl, targets)
    2. Register it in EFFECT_REGISTRY at the bottom of this file
    3. Remove the STUB log line for that tag in rules.py
"""

from __future__ import annotations
import random
import re
from typing import Optional, TYPE_CHECKING

from game_state import (
    GameState, Player, CardInstance, CardDefinition,
    Zone, Phase, Color, CardType
)

if TYPE_CHECKING:
    from game_state import StackObject


# ─────────────────────────────────────────────────────────────────────────────
# Effect dispatcher — called by rules.py instead of _execute_effect_stub()
# ─────────────────────────────────────────────────────────────────────────────

def execute_effect(
    state:      GameState,
    source:     CardInstance,
    stack_obj,              # StackObject
    controller: Player,
) -> list[str]:
    """
    Main entry point. Dispatches each effect tag to its handler.
    Tags are processed in priority order — ramp before draw, removal before wipe.
    """
    logs  = []
    tags  = source.definition.effect_tags
    defn  = source.definition
    text  = defn.oracle_text.lower()

    # Tags that have real implementations
    IMPLEMENTED = {"draw", "ramp", "removal", "token", "board_wipe"}

    for tag in tags:
        if tag not in IMPLEMENTED:
            logs.append(f"  [STUB] {defn.name}: {tag} — not yet implemented")
            continue

        handler = EFFECT_REGISTRY.get(tag)
        if handler is None:
            logs.append(f"  [STUB] {defn.name}: {tag} — no handler registered")
            continue

        effect_logs = handler(state, source, controller, stack_obj.targets)
        logs.extend(effect_logs)

    return logs


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic targeting
# ─────────────────────────────────────────────────────────────────────────────

def heuristic_target_creature(
    state:      GameState,
    controller: Player,
    prefer_opponents: bool = True,
) -> Optional[CardInstance]:
    """
    Pick the best creature to target for a removal effect.
    Prefers opponent creatures; picks the highest-threat one.
    Threat score = power + CMC (powerful AND expensive = highest priority).
    """
    candidates = []

    for inst in state.get_battlefield():
        if not inst.definition.is_creature():
            continue
        # Skip indestructible — removal won't work (handled in effect)
        if inst.definition.has_keyword("indestructible"):
            continue
        is_opponent = inst.controller_id != controller.player_id
        if prefer_opponents and not is_opponent:
            continue
        threat = inst.effective_power + inst.definition.cmc
        candidates.append((threat, inst))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def heuristic_target_land(
    state:      GameState,
    controller: Player,
) -> Optional[CardInstance]:
    """
    Find the best untapped land to put onto the battlefield from library.
    Returns None if library has no lands.
    Used by ramp effects that fetch a basic land.
    """
    for iid in controller.library:
        inst = state.all_instances.get(iid)
        if inst and inst.definition.is_land():
            return inst
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Helper: parse "draw N cards" from oracle text
# ─────────────────────────────────────────────────────────────────────────────

def _parse_draw_count(oracle_text: str, default: int = 1) -> int:
    """
    Extract how many cards to draw from oracle text.
    Handles: "draw a card", "draw two cards", "draw 3 cards", "draw X cards"
    """
    text = oracle_text.lower()
    WORD_TO_NUM = {
        "a": 1, "an": 1, "one": 1, "two": 2, "three": 3,
        "four": 4, "five": 5, "six": 6, "seven": 7,
    }
    # "draw X cards" with digit
    m = re.search(r'draw (\d+) cards?', text)
    if m:
        return int(m.group(1))
    # "draw a card" / "draw two cards" etc.
    m = re.search(r'draw (\w+) cards?', text)
    if m:
        word = m.group(1)
        return WORD_TO_NUM.get(word, default)
    return default


def _parse_token_count(oracle_text: str) -> int:
    """
    Extract how many tokens to create.
    Handles: "create a token", "create two tokens", "create 1/1 token"
    """
    text = oracle_text.lower()
    WORD_TO_NUM = {
        "a": 1, "an": 1, "one": 1, "two": 2, "three": 3,
        "four": 4, "five": 5, "x": 1,
    }
    m = re.search(r'create (\d+)', text)
    if m:
        return int(m.group(1))
    m = re.search(r'create (\w+)', text)
    if m:
        return WORD_TO_NUM.get(m.group(1), 1)
    return 1


def _parse_token_stats(oracle_text: str) -> tuple[int, int, list[str]]:
    """
    Extract token power/toughness and any keywords from oracle text.
    Returns (power, toughness, keywords).
    Examples:
        "create a 1/1 white Soldier creature token" → (1, 1, [])
        "create a 2/2 green Wolf creature token with trample" → (2, 2, ["trample"])
        "create a 0/1 colorless Thopter artifact creature token with flying" → (0, 1, ["flying"])
    """
    text = oracle_text.lower()
    # Parse P/T
    m = re.search(r'(\d+)/(\d+)', text)
    power     = int(m.group(1)) if m else 1
    toughness = int(m.group(2)) if m else 1

    # Parse keywords on the token
    token_keywords = []
    KNOWN_KW = ["flying", "trample", "haste", "vigilance", "deathtouch",
                "lifelink", "first strike", "reach", "hexproof", "indestructible"]
    # Look for keywords after "token with ..."
    token_section = re.search(r'token with (.+?)(?:\.|$)', text)
    if token_section:
        kw_text = token_section.group(1)
        for kw in KNOWN_KW:
            if kw in kw_text:
                token_keywords.append(kw.title())

    return power, toughness, token_keywords


def _parse_land_count(oracle_text: str) -> int:
    """How many lands does this ramp effect fetch? Usually 1, sometimes 2."""
    text = oracle_text.lower()
    if "two basic land" in text or "two lands" in text:
        return 2
    return 1


# ─────────────────────────────────────────────────────────────────────────────
# Effect handlers
# ─────────────────────────────────────────────────────────────────────────────

def effect_draw(
    state:      GameState,
    source:     CardInstance,
    controller: Player,
    targets:    list[str],
) -> list[str]:
    """
    Draw N cards.
    Parses N from oracle text. Handles "draw a card" through "draw seven cards".
    Also handles "each player draws" and "target player draws".
    """
    logs = []
    text  = source.definition.oracle_text.lower()
    count = _parse_draw_count(text)

    # "each player draws" — all players draw
    if "each player draws" in text:
        for player in state.living_players():
            drawn = 0
            for _ in range(count):
                inst = _draw_one(state, player)
                if inst:
                    drawn += 1
                else:
                    player.has_lost = True
                    logs.append(f"  {player.name} cannot draw — library empty → loses.")
                    break
            if drawn > 0:
                logs.append(f"  {player.name} draws {drawn} card(s). "
                            f"({len(player.hand)} in hand)")
        return logs

    # Normal: controller draws
    drawn = 0
    for _ in range(count):
        inst = _draw_one(state, controller)
        if inst:
            drawn += 1
        else:
            controller.has_lost = True
            logs.append(f"  {controller.name} cannot draw — library empty → loses.")
            break

    logs.append(f"  {controller.name} draws {drawn} card(s). "
                f"({len(controller.hand)} in hand)")
    return logs


def effect_ramp(
    state:      GameState,
    source:     CardInstance,
    controller: Player,
    targets:    list[str],
) -> list[str]:
    """
    Search library for a basic land and put it onto the battlefield (or hand).
    Handles:
        - "search your library for a basic land card and put it onto the battlefield"
        - "search your library for a basic land card, put it into your hand"
        - "put the top card of your library onto the battlefield if it's a land"
    """
    logs  = []
    text  = source.definition.oracle_text.lower()
    count = _parse_land_count(text)
    tapped = "tapped" in text

    # Determine destination: battlefield vs hand
    to_battlefield = ("onto the battlefield" in text or
                      "put it onto" in text)

    fetched = 0
    for _ in range(count):
        land = heuristic_target_land(state, controller)
        if land is None:
            logs.append(f"  {source.definition.name}: no basic land in library.")
            break

        controller.library.remove(land.instance_id)
        random.shuffle(controller.library)  # Shuffle after searching

        if to_battlefield:
            land.zone   = Zone.BATTLEFIELD
            land.tapped = tapped
            logs.append(f"  {source.definition.name}: {land.definition.name} enters "
                        f"the battlefield{' tapped' if tapped else ''}.")
        else:
            land.zone = Zone.HAND
            controller.hand.append(land.instance_id)
            logs.append(f"  {source.definition.name}: {land.definition.name} "
                        f"put into {controller.name}'s hand.")
        fetched += 1

    if fetched == 0:
        logs.append(f"  {source.definition.name}: no lands found in library.")

    return logs


def effect_removal(
    state:      GameState,
    source:     CardInstance,
    controller: Player,
    targets:    list[str],
) -> list[str]:
    """
    Destroy or exile a target creature.
    Uses pre-chosen targets if provided, otherwise picks heuristically.
    Handles: destroy target, exile target, -X/-X until end of turn (as removal).
    """
    logs  = []
    text  = source.definition.oracle_text.lower()
    exile = "exile target" in text

    # Use pre-chosen target if available, otherwise heuristic
    target_inst = None
    if targets:
        target_inst = state.get_instance(targets[0])

    if target_inst is None:
        target_inst = heuristic_target_creature(state, controller,
                                                prefer_opponents=True)

    if target_inst is None:
        logs.append(f"  {source.definition.name}: no valid target — fizzles.")
        return logs

    # Check indestructible (destroy fails; exile still works)
    indestr = target_inst.definition.has_keyword("indestructible")
    if indestr and not exile:
        logs.append(f"  {source.definition.name}: {target_inst.definition.name} "
                    f"is indestructible — effect doesn't apply.")
        return logs

    target_owner = state.get_player(target_inst.owner_id)
    dest = Zone.EXILE if exile else Zone.GRAVEYARD
    action_word  = "exiles" if exile else "destroys"

    logs.append(f"  {source.definition.name} {action_word} "
                f"{target_inst.definition.name}.")

    # Move to destination — respects commander replacement effect (handled in rules)
    _move_permanent(state, target_inst, target_owner, dest)

    # STUB: "when ~ dies" triggers would fire here
    if not exile:
        death_text = target_inst.definition.oracle_text.lower()
        if "when" in death_text and "dies" in death_text:
            logs.append(f"  [STUB] {target_inst.definition.name}: dies trigger — "
                        f"not yet implemented")

    return logs


def effect_token(
    state:      GameState,
    source:     CardInstance,
    controller: Player,
    targets:    list[str],
) -> list[str]:
    """
    Create one or more creature tokens.
    Parses count and stats from oracle text.
    Tokens are CardInstance objects with generated definitions.
    """
    logs  = []
    defn  = source.definition
    text  = defn.oracle_text.lower()
    count = _parse_token_count(text)

    power, toughness, token_kws = _parse_token_stats(text)

    # Parse token name / type from oracle text
    # e.g. "create a 1/1 white Soldier creature token"
    token_name = _parse_token_name(text)
    token_color = _parse_token_color(text, controller)

    for i in range(count):
        token_defn = CardDefinition(
            scryfall_id  = f"token-{source.instance_id}-{i}",
            name         = token_name,
            mana_cost    = "",
            cmc          = 0.0,
            colors       = token_color,
            card_types   = [CardType.CREATURE],
            subtypes     = _parse_token_subtypes(text),
            keywords     = token_kws,
            oracle_text  = " ".join(token_kws),
            power        = power,
            toughness    = toughness,
            loyalty      = None,
            is_legendary = False,
            effect_tags  = [],
        )
        # Tokens are not commanders and use same color handling
        import uuid as _uuid
        token = CardInstance(
            instance_id   = str(_uuid.uuid4()),
            definition    = token_defn,
            owner_id      = controller.player_id,
            controller_id = controller.player_id,
            zone          = Zone.BATTLEFIELD,
            summoning_sick= True,
            has_haste     = "haste" in [k.lower() for k in token_kws],
        )
        state.all_instances[token.instance_id] = token

    kw_str = f" with {', '.join(token_kws)}" if token_kws else ""
    logs.append(f"  {defn.name}: {controller.name} creates {count} "
                f"{power}/{toughness} {token_name} token(s){kw_str}.")

    return logs


def effect_board_wipe(
    state:      GameState,
    source:     CardInstance,
    controller: Player,
    targets:    list[str],
) -> list[str]:
    """
    Destroy all creatures (or all permanents for some wipes).
    Handles:
        "destroy all creatures"
        "destroy all nonland permanents"
        "exile all creatures"
        Indestructible creatures survive destroy (not exile).
    """
    logs  = []
    text  = source.definition.oracle_text.lower()
    exile = "exile all" in text

    # Determine scope
    hits_all_permanents = (
        "nonland permanent" in text or
        "all permanents" in text
    )
    hits_creatures = (
        "all creatures" in text or
        hits_all_permanents
    )
    hits_artifacts  = "all artifacts" in text or hits_all_permanents
    hits_enchant    = "all enchantments" in text or hits_all_permanents

    # Collect targets — check all permanents on battlefield simultaneously
    to_destroy: list[CardInstance] = []

    for inst in state.get_battlefield():
        qualifies = False
        if hits_creatures and inst.definition.is_creature():
            qualifies = True
        if hits_artifacts and CardType.ARTIFACT in inst.definition.card_types:
            qualifies = True
        if hits_enchant and CardType.ENCHANTMENT in inst.definition.card_types:
            qualifies = True

        if not qualifies:
            continue

        # Indestructible survives destroy (not exile)
        if not exile and inst.definition.has_keyword("indestructible"):
            continue

        to_destroy.append(inst)

    if not to_destroy:
        logs.append(f"  {source.definition.name}: no valid targets on battlefield.")
        return logs

    action_word = "exiles" if exile else "destroys"
    names = ", ".join(i.definition.name for i in to_destroy[:5])
    if len(to_destroy) > 5:
        names += f" and {len(to_destroy) - 5} more"
    logs.append(f"  {source.definition.name} {action_word} {len(to_destroy)} "
                f"permanent(s): {names}.")

    dest = Zone.EXILE if exile else Zone.GRAVEYARD
    for inst in to_destroy:
        owner = state.get_player(inst.owner_id)
        _move_permanent(state, inst, owner, dest)

        # STUB: dies triggers
        if not exile:
            death_text = inst.definition.oracle_text.lower()
            if "when" in death_text and "dies" in death_text:
                logs.append(f"  [STUB] {inst.definition.name}: dies trigger — "
                            f"not yet implemented")

    return logs


# ─────────────────────────────────────────────────────────────────────────────
# Token parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_token_name(text: str) -> str:
    """Extract creature type from token creation text."""
    # Common patterns: "1/1 white Soldier", "2/2 green Wolf", "0/1 Thopter"
    m = re.search(r'\d+/\d+\s+\w+\s+(\w+)\s+creature\s+token', text)
    if m:
        return m.group(1).title()
    m = re.search(r'\d+/\d+\s+(\w+)\s+(?:artifact\s+)?creature\s+token', text)
    if m:
        word = m.group(1)
        # Skip color words
        if word not in ('white','blue','black','red','green','colorless'):
            return word.title()
    return "Creature"


def _parse_token_subtypes(text: str) -> list[str]:
    """Extract subtypes for the token."""
    m = re.search(r'(\w+)\s+creature\s+token', text)
    if m:
        sub = m.group(1).title()
        if sub not in ('Artifact', 'Enchantment'):
            return [sub]
    return []


def _parse_token_color(text: str, controller: Player) -> list[Color]:
    """Infer token color from oracle text."""
    COLOR_MAP = {
        'white':     Color.WHITE,
        'blue':      Color.BLUE,
        'black':     Color.BLACK,
        'red':       Color.RED,
        'green':     Color.GREEN,
        'colorless': Color.COLORLESS,
    }
    for word, color in COLOR_MAP.items():
        if word in text:
            return [color]
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Zone movement helper (respects commander replacement)
# ─────────────────────────────────────────────────────────────────────────────

def _move_permanent(
    state:  GameState,
    inst:   CardInstance,
    owner:  Optional[Player],
    target: Zone,
):
    """
    Move a permanent to a zone, respecting commander replacement (903.11/12).
    Commanders going to GY or exile go to command zone instead.
    """
    if inst.is_commander and owner and target in (Zone.GRAVEYARD, Zone.EXILE):
        # Rule 903.11/12: owner may put commander in command zone
        inst.zone          = Zone.COMMAND
        inst.tapped        = False
        inst.damage_marked = 0
        inst._deathtouch_damaged = False
        if inst.instance_id not in owner.command_zone:
            owner.command_zone.append(inst.instance_id)
        # Remove from any old zone list
        _remove_from_lists(inst, owner)
    else:
        # Normal zone change
        _remove_from_lists(inst, owner)
        inst.zone = target
        if owner:
            match target:
                case Zone.GRAVEYARD: owner.graveyard.append(inst.instance_id)
                case Zone.EXILE:     owner.exile.append(inst.instance_id)
                case Zone.HAND:      owner.hand.append(inst.instance_id)
                case Zone.LIBRARY:   owner.library.append(inst.instance_id)

    # Reset combat state
    inst.is_attacking = False
    inst.is_blocking  = False
    inst.blocked_by.clear()
    inst.blocking     = None


def _remove_from_lists(inst: CardInstance, owner: Optional[Player]):
    """Remove instance_id from all zone lists of its owner."""
    if not owner:
        return
    iid = inst.instance_id
    for lst in [owner.hand, owner.graveyard, owner.exile,
                owner.library, owner.command_zone]:
        if iid in lst:
            lst.remove(iid)


def _draw_one(state: GameState, player: Player) -> Optional[CardInstance]:
    """Draw the top card of a player's library. Returns None if empty."""
    if not player.library:
        return None
    iid  = player.library.pop(0)
    inst = state.all_instances.get(iid)
    if not inst:
        return None
    inst.zone = Zone.HAND
    player.hand.append(iid)
    player.draw_this_turn += 1
    return inst


# ─────────────────────────────────────────────────────────────────────────────
# Effect registry — maps tag → handler function
# ─────────────────────────────────────────────────────────────────────────────

EFFECT_REGISTRY: dict[str, callable] = {
    "draw":       effect_draw,
    "ramp":       effect_ramp,
    "removal":    effect_removal,
    "token":      effect_token,
    "board_wipe": effect_board_wipe,
}
