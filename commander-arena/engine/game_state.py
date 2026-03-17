"""
game_state.py — Complete game state representation

This is the foundation everything else builds on.
The ML model will receive a *encoded* version of this state as input.
The heuristic AI will read this directly.
The rules engine will mutate this.

Design principles:
  - Pure data, no game logic here
  - Every field must be serializable (for saving/loading/training data)
  - Immutable snapshots via copy() for MCTS lookahead
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import copy
import uuid


# ─────────────────────────────────────────────────────────────────────────────
# Enums — named constants so we never use magic strings
# ─────────────────────────────────────────────────────────────────────────────

class Zone(Enum):
    """Where a card can physically be."""
    LIBRARY    = auto()
    HAND       = auto()
    BATTLEFIELD= auto()
    GRAVEYARD  = auto()
    EXILE      = auto()
    COMMAND    = auto()   # Commander zone
    STACK      = auto()   # Currently being cast


class Phase(Enum):
    """Turn phases in order."""
    UNTAP       = auto()
    UPKEEP      = auto()
    DRAW        = auto()
    MAIN1       = auto()
    BEGIN_COMBAT= auto()
    DECLARE_ATTACKERS = auto()
    DECLARE_BLOCKERS  = auto()
    COMBAT_DAMAGE     = auto()
    END_COMBAT  = auto()
    MAIN2       = auto()
    END         = auto()
    CLEANUP     = auto()


class Color(Enum):
    WHITE = "W"
    BLUE  = "U"
    BLACK = "B"
    RED   = "R"
    GREEN = "G"
    COLORLESS = "C"


class CardType(Enum):
    CREATURE    = auto()
    INSTANT     = auto()
    SORCERY     = auto()
    ENCHANTMENT = auto()
    ARTIFACT    = auto()
    LAND        = auto()
    PLANESWALKER= auto()
    BATTLE      = auto()


# ─────────────────────────────────────────────────────────────────────────────
# Card — static definition (what the card IS, not where it is)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CardDefinition:
    """
    The static definition of a card — its rules text, stats, types.
    Loaded once from Scryfall data and never mutated.
    Think of this as the card's entry in the rulebook.
    """
    scryfall_id:  str
    name:         str
    mana_cost:    str            # "{2}{G}{G}" format
    cmc:          float          # Converted mana cost
    colors:       list[Color]
    card_types:   list[CardType]
    subtypes:     list[str]      # ["Elf", "Druid"] etc.
    keywords:     list[str]      # ["Flying", "Trample"] etc.
    oracle_text:  str
    power:        Optional[int]  # None for non-creatures
    toughness:    Optional[int]
    loyalty:      Optional[int]  # Planeswalkers only
    is_legendary: bool = False
    is_commander_legal: bool = True

    # Effect tags — what categories of effects this card has
    # Populated by card_parser.py when we load from Scryfall
    # Used by the heuristic AI and ML encoder
    effect_tags: list[str] = field(default_factory=list)
    # Examples: ["draw", "removal", "ramp", "counter", "token", "pump",
    #            "board_wipe", "tutor", "recursion", "protection"]

    def is_creature(self)  -> bool: return CardType.CREATURE     in self.card_types
    def is_instant(self)   -> bool: return CardType.INSTANT      in self.card_types
    def is_sorcery(self)   -> bool: return CardType.SORCERY      in self.card_types
    def is_land(self)      -> bool: return CardType.LAND         in self.card_types
    def is_permanent(self) -> bool:
        return any(t in self.card_types for t in [
            CardType.CREATURE, CardType.ENCHANTMENT,
            CardType.ARTIFACT, CardType.PLANESWALKER,
            CardType.LAND, CardType.BATTLE
        ])

    def has_keyword(self, kw: str) -> bool:
        return kw.lower() in [k.lower() for k in self.keywords]


# ─────────────────────────────────────────────────────────────────────────────
# CardInstance — a specific copy of a card in play
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CardInstance:
    """
    A specific physical copy of a card.
    Multiple instances can exist of the same CardDefinition (e.g., two Mountains).
    This tracks runtime state: tapped, counters, damage, attachments.
    """
    instance_id:  str              # Unique ID for this physical copy
    definition:   CardDefinition   # What card this is
    owner_id:     int              # Which player owns this card
    controller_id: int             # Who currently controls it (can differ from owner)
    zone:         Zone

    # Battlefield state (only relevant when zone == BATTLEFIELD)
    tapped:           bool = False
    damage_marked:    int  = 0     # Damage this turn (resets at cleanup)
    counters:         dict = field(default_factory=dict)
    # Examples: {"p1p1": 3, "m1m1": 0, "loyalty": 4}

    # Combat state (reset each combat)
    is_attacking:     bool = False
    is_blocking:      bool = False
    blocked_by:       list[str] = field(default_factory=list)   # instance_ids
    blocking:         Optional[str] = None                       # instance_id

    # Status effects
    summoning_sick:   bool = True   # New creatures can't attack/tap until next turn
    has_haste:        bool = False  # Overrides summoning sickness
    is_phased_out:    bool = False

    # Auras/equipment attached to this
    attachments:      list[str] = field(default_factory=list)   # instance_ids
    attached_to:      Optional[str] = None                      # instance_id

    # Commander tracking
    is_commander:     bool = False
    times_cast:       int  = 0     # For commander tax calculation

    @property
    def effective_power(self) -> int:
        base = self.definition.power or 0
        p1p1 = self.counters.get("p1p1", 0)
        m1m1 = self.counters.get("m1m1", 0)
        return base + p1p1 - m1m1

    @property
    def effective_toughness(self) -> int:
        base = self.definition.toughness or 0
        p1p1 = self.counters.get("p1p1", 0)
        m1m1 = self.counters.get("m1m1", 0)
        return base + p1p1 - m1m1

    @property
    def commander_tax(self) -> int:
        return self.times_cast * 2

    def can_attack(self) -> bool:
        if self.tapped or self.is_phased_out: return False
        if self.summoning_sick and not self.has_haste: return False
        if not self.definition.is_creature(): return False
        return True

    def can_block(self) -> bool:
        if self.tapped or self.is_phased_out: return False
        if not self.definition.is_creature(): return False
        return True

    def would_die(self) -> bool:
        """Returns True if this creature has lethal damage marked."""
        if not self.definition.is_creature(): return False
        if self.definition.has_keyword("indestructible"): return False
        return self.damage_marked >= self.effective_toughness

    def copy(self) -> CardInstance:
        return copy.deepcopy(self)


# ─────────────────────────────────────────────────────────────────────────────
# ManaPool — tracks available mana for the current phase
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ManaPool:
    """
    Available mana right now.
    Empties at phase transitions (mana burn rule was removed but we still track it).
    """
    white:     int = 0
    blue:      int = 0
    black:     int = 0
    red:       int = 0
    green:     int = 0
    colorless: int = 0

    @property
    def total(self) -> int:
        return self.white + self.blue + self.black + self.red + self.green + self.colorless

    def add(self, color: Color, amount: int = 1):
        match color:
            case Color.WHITE:     self.white     += amount
            case Color.BLUE:      self.blue      += amount
            case Color.BLACK:     self.black     += amount
            case Color.RED:       self.red       += amount
            case Color.GREEN:     self.green     += amount
            case Color.COLORLESS: self.colorless += amount

    def can_pay(self, cost: dict[Color, int]) -> bool:
        """Check if this pool can pay a given mana cost."""
        temp = copy.copy(self)
        for color, amount in cost.items():
            if color == Color.COLORLESS:
                # Generic mana — can be paid with anything
                if temp.total < amount: return False
                # Spend the most abundant color first (simplification)
                remaining = amount
                for attr in ['white','blue','black','red','green','colorless']:
                    take = min(getattr(temp, attr), remaining)
                    setattr(temp, attr, getattr(temp, attr) - take)
                    remaining -= take
                    if remaining == 0: break
            else:
                attr = color.name.lower()
                if getattr(temp, attr) < amount: return False
                setattr(temp, attr, getattr(temp, attr) - amount)
        return True

    def empty(self):
        self.white = self.blue = self.black = self.red = self.green = self.colorless = 0

    def __repr__(self):
        parts = []
        if self.white:     parts.append(f"{self.white}W")
        if self.blue:      parts.append(f"{self.blue}U")
        if self.black:     parts.append(f"{self.black}B")
        if self.red:       parts.append(f"{self.red}R")
        if self.green:     parts.append(f"{self.green}G")
        if self.colorless: parts.append(f"{self.colorless}C")
        return "{" + ", ".join(parts) + "}" if parts else "{empty}"


# ─────────────────────────────────────────────────────────────────────────────
# Player — one player's complete state
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Player:
    """
    Everything about one player.
    The ML model will encode each player's state separately.
    """
    player_id:   int
    name:        str
    is_human:    bool = False

    # Life and loss conditions
    life:        int  = 40    # Commander starts at 40
    poison:      int  = 0     # 10 = lose
    has_lost:    bool = False

    # Commander damage received (tracked per source commander)
    # key = player_id of the commander's controller
    commander_damage: dict[int, int] = field(default_factory=dict)

    # Zones — lists of CardInstance IDs
    # Actual CardInstance objects live in GameState.all_instances
    library:     list[str] = field(default_factory=list)   # instance_ids, ordered
    hand:        list[str] = field(default_factory=list)
    graveyard:   list[str] = field(default_factory=list)
    exile:       list[str] = field(default_factory=list)
    command_zone:list[str] = field(default_factory=list)

    # Mana
    mana_pool:   ManaPool = field(default_factory=ManaPool)
    lands_played_this_turn: int = 0

    # Turn tracking
    draw_this_turn:   int = 0

    def check_loss_conditions(self) -> Optional[str]:
        """Returns reason string if player has lost, None otherwise."""
        if self.life <= 0:    return "life total reached 0"
        if self.poison >= 10: return "10 poison counters"
        if any(dmg >= 21 for dmg in self.commander_damage.values()):
            return "21 commander damage from a single commander"
        return None


# ─────────────────────────────────────────────────────────────────────────────
# StackObject — something on the stack waiting to resolve
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StackObject:
    """Represents a spell or ability on the stack."""
    stack_id:       str
    source_id:      str           # instance_id of the card/permanent generating this
    controller_id:  int
    effect_key:     str           # Matches a key in effects/registry.py
    targets:        list[str] = field(default_factory=list)  # instance_ids or player_ids
    x_value:        int = 0       # For X spells


# ─────────────────────────────────────────────────────────────────────────────
# GameState — the complete, authoritative game state
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GameState:
    """
    The entire state of a game at a single moment in time.

    This is what gets:
    - Passed to the heuristic AI to make decisions
    - Encoded into a tensor for the ML model
    - Saved to disk for training data
    - Sent over the API to the JS frontend

    Critical design rule: ALL game objects live in all_instances.
    Players only store lists of instance_ids.
    This prevents duplication and makes deep copies clean.
    """
    game_id:     str = field(default_factory=lambda: str(uuid.uuid4()))

    # Players (index 0 = human or first player)
    players:     list[Player] = field(default_factory=list)

    # ALL card instances in the game, keyed by instance_id
    # This is the single source of truth for every card
    all_instances: dict[str, CardInstance] = field(default_factory=dict)

    # The stack (index 0 = bottom, index -1 = top)
    stack:       list[StackObject] = field(default_factory=list)

    # Turn structure
    turn_number:    int   = 1
    active_player:  int   = 0    # player_id of whose turn it is
    priority_player: int  = 0   # player_id who currently has priority
    phase:          Phase = Phase.UNTAP
    combat_damage_assigned: bool = False

    # Attackers/blockers for current combat
    # key = attacker instance_id, value = list of blocker instance_ids
    declared_attackers: dict[str, list[str]] = field(default_factory=dict)
    # key = blocker instance_id, value = attacker instance_id
    declared_blockers:  dict[str, str]       = field(default_factory=dict)

    # Game log (for training data / display)
    log: list[str] = field(default_factory=list)

    # Is the game over?
    winner_id: Optional[int] = None

    # ── Convenience accessors ──────────────────────────────────────

    def get_instance(self, iid: str) -> Optional[CardInstance]:
        return self.all_instances.get(iid)

    def get_player(self, pid: int) -> Optional[Player]:
        for p in self.players:
            if p.player_id == pid: return p
        return None

    def get_active_player(self) -> Player:
        return self.get_player(self.active_player)

    def get_battlefield(self, controller_id: Optional[int] = None) -> list[CardInstance]:
        """All permanents on battlefield, optionally filtered by controller."""
        result = [
            inst for inst in self.all_instances.values()
            if inst.zone == Zone.BATTLEFIELD
        ]
        if controller_id is not None:
            result = [i for i in result if i.controller_id == controller_id]
        return result

    def get_hand(self, player_id: int) -> list[CardInstance]:
        p = self.get_player(player_id)
        if not p: return []
        return [self.all_instances[iid] for iid in p.hand if iid in self.all_instances]

    def get_opponents(self, player_id: int) -> list[Player]:
        return [p for p in self.players if p.player_id != player_id and not p.has_lost]

    def living_players(self) -> list[Player]:
        return [p for p in self.players if not p.has_lost]

    def add_to_log(self, msg: str):
        self.log.append(msg)

    # ── Deep copy for MCTS lookahead ──────────────────────────────

    def snapshot(self) -> GameState:
        """
        Returns a deep copy of the game state.
        MCTS uses this to simulate future turns without touching the real state.
        """
        return copy.deepcopy(self)

    # ── Serialization ─────────────────────────────────────────────

    def to_dict(self) -> dict:
        """
        Convert to a plain dict for JSON serialization.
        Used by the API to send state to the JS frontend.
        """
        # Full serialization will be implemented in serializer.py
        # This stub exists so the API can call it now
        return {
            "game_id":      self.game_id,
            "turn":         self.turn_number,
            "phase":        self.phase.name,
            "active":       self.active_player,
            "winner":       self.winner_id,
            "player_count": len(self.players),
        }

    def __repr__(self):
        players_str = ", ".join(f"{p.name}({p.life})" for p in self.players)
        return f"<GameState turn={self.turn_number} phase={self.phase.name} [{players_str}]>"
