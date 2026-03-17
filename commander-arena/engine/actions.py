"""
actions.py — Legal action generation

Every decision an AI makes is an Action.
The ML model outputs a probability distribution over all possible actions.
The heuristic AI scores each action and picks the best.

Design principle: actions are data, not functions.
The rules engine in rules.py knows how to *execute* each action type.
This file only defines what actions ARE and which ones are LEGAL.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from game_state import GameState, CardInstance, Player


# ─────────────────────────────────────────────────────────────────────────────
# Action types
# ─────────────────────────────────────────────────────────────────────────────

class ActionType(Enum):
    # Priority actions (can be taken when you have priority)
    PASS_PRIORITY    = auto()   # Do nothing, pass to next player
    CAST_SPELL       = auto()   # Cast a card from hand
    PLAY_LAND        = auto()   # Play a land (once per turn)
    ACTIVATE_ABILITY = auto()   # Use a tap/cost ability
    CAST_COMMANDER   = auto()   # Cast commander from command zone

    # Combat actions
    DECLARE_ATTACKER = auto()   # Declare a creature as attacker
    DECLARE_BLOCKER  = auto()   # Declare a creature as blocker
    ASSIGN_DAMAGE    = auto()   # Order blockers for damage assignment

    # Special choices (responding to prompts)
    CHOOSE_TARGET    = auto()   # Choose a target for a spell/ability
    CHOOSE_MODE      = auto()   # Choose a mode (modal spells)
    CHOOSE_X         = auto()   # Choose X value for X spells
    DISCARD_CARD     = auto()   # Choose what to discard
    SACRIFICE        = auto()   # Choose what to sacrifice

    # Commander specific
    MOVE_TO_COMMAND  = auto()   # Move commander from GY/exile to command zone

    # Turn structure
    ADVANCE_PHASE    = auto()   # Move to next phase (when no actions remain)


# ─────────────────────────────────────────────────────────────────────────────
# Action dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Action:
    """
    A single thing a player can do at this moment.

    frozen=True means it's immutable — actions are descriptions of intent,
    not mutable objects. This makes them safe to use as dict keys and in sets.

    The ML model will learn to assign a score to each Action given a GameState.
    """
    action_type:  ActionType
    actor_id:     int           # player_id making this action

    # For card-based actions
    card_iid:     Optional[str] = None   # instance_id of the card being acted on

    # For targeted actions
    target_iids:  tuple[str, ...] = field(default_factory=tuple)  # target instance_ids
    target_pids:  tuple[int, ...] = field(default_factory=tuple)  # target player_ids

    # For special choices
    choice_value: Optional[int] = None   # X value, mode index, etc.

    # Human-readable description (for logging and UI)
    description:  str = ""

    def __repr__(self):
        return f"<Action {self.action_type.name}: {self.description or self.card_iid or ''}>"


# ─────────────────────────────────────────────────────────────────────────────
# Legal action generator
# ─────────────────────────────────────────────────────────────────────────────

class ActionGenerator:
    """
    Given a GameState, produces all legal actions for the current player.

    This is called:
    1. By the heuristic AI to get the list of actions to score
    2. By the ML training loop to verify the model doesn't pick illegal actions
    3. By the rules engine to validate a proposed action before executing it

    The completeness of this generator directly determines how correctly
    the AI plays. If a legal action isn't generated, the AI can never take it.
    """

    def __init__(self, state: GameState):
        self.state = state

    def get_legal_actions(self) -> list[Action]:
        """
        Returns all legal actions for the priority player right now.
        Always includes PASS_PRIORITY (you can always do nothing).
        """
        from game_state import Phase, Zone

        state = self.state
        pid   = state.priority_player
        player = state.get_player(pid)

        if player is None or player.has_lost:
            return [self._pass(pid)]

        actions = [self._pass(pid)]

        # ── Phase-specific actions ─────────────────────────────────

        if state.phase == Phase.DECLARE_ATTACKERS and pid == state.active_player:
            actions.extend(self._attacker_actions(player))

        elif state.phase == Phase.DECLARE_BLOCKERS and pid != state.active_player:
            actions.extend(self._blocker_actions(player))

        else:
            # Main phases and instants at any time
            if state.phase in (Phase.MAIN1, Phase.MAIN2):
                # Rule 505.5a/b: lands and sorcery-speed spells require empty stack
                if not state.stack:
                    actions.extend(self._land_actions(player))
                    actions.extend(self._cast_actions(player, sorcery_speed=True))
                    actions.extend(self._commander_cast_actions(player))
                else:
                    # Stack not empty — only instants and flash
                    actions.extend(self._cast_actions(player, sorcery_speed=False))
            else:
                # Instant speed only (upkeep, end step, in response to something)
                actions.extend(self._cast_actions(player, sorcery_speed=False))

            actions.extend(self._activated_ability_actions(player))

        return actions

    # ── Private generators ─────────────────────────────────────────

    def _pass(self, pid: int) -> Action:
        return Action(ActionType.PASS_PRIORITY, actor_id=pid,
                      description="Pass priority")

    def _land_actions(self, player: Player) -> list[Action]:
        """Playing a land — once per turn, sorcery speed."""
        from game_state import Zone

        if player.lands_played_this_turn > 0: return []
        if self.state.active_player != player.player_id: return []

        actions = []
        for iid in player.hand:
            inst = self.state.get_instance(iid)
            if inst and inst.definition.is_land():
                actions.append(Action(
                    ActionType.PLAY_LAND,
                    actor_id=player.player_id,
                    card_iid=iid,
                    description=f"Play {inst.definition.name}"
                ))
        return actions

    def _cast_actions(self, player: Player, sorcery_speed: bool) -> list[Action]:
        """Casting spells from hand."""
        actions = []
        bf = self.state.get_battlefield(player.player_id)

        # Calculate total mana available (lands + mana rocks + dorks)
        available_mana = self._calculate_available_mana(player, bf)

        for iid in player.hand:
            inst = self.state.get_instance(iid)
            if not inst: continue
            defn = inst.definition
            if defn.is_land(): continue

            # Sorcery-speed restriction
            if not defn.has_keyword("flash") and not sorcery_speed:
                if not (defn.is_instant()): continue

            # Can we afford it?
            cost = self._parse_mana_cost(defn.mana_cost, inst)
            if available_mana >= cost:
                actions.append(Action(
                    ActionType.CAST_SPELL,
                    actor_id=player.player_id,
                    card_iid=iid,
                    description=f"Cast {defn.name} ({defn.mana_cost})"
                ))

        return actions

    def _commander_cast_actions(self, player: Player) -> list[Action]:
        """Cast commander from command zone."""
        actions = []
        bf = self.state.get_battlefield(player.player_id)
        available_mana = self._calculate_available_mana(player, bf)

        for iid in player.command_zone:
            inst = self.state.get_instance(iid)
            if not inst: continue
            defn = inst.definition

            cost = self._parse_mana_cost(defn.mana_cost, inst) + inst.commander_tax
            if available_mana >= cost:
                actions.append(Action(
                    ActionType.CAST_COMMANDER,
                    actor_id=player.player_id,
                    card_iid=iid,
                    description=f"Cast commander {defn.name} (tax: +{inst.commander_tax})"
                ))
        return actions

    def _attacker_actions(self, player: Player) -> list[Action]:
        """Declare attackers in combat."""
        actions = []
        opponents = self.state.get_opponents(player.player_id)

        for inst in self.state.get_battlefield(player.player_id):
            if not inst.can_attack(): continue

            for opp in opponents:
                actions.append(Action(
                    ActionType.DECLARE_ATTACKER,
                    actor_id=player.player_id,
                    card_iid=inst.instance_id,
                    target_pids=(opp.player_id,),
                    description=f"Attack {opp.name} with {inst.definition.name}"
                ))

        return actions

    def _blocker_actions(self, player: Player) -> list[Action]:
        """Declare blockers."""
        actions = []
        attackers = [
            self.state.get_instance(iid)
            for iid in self.state.declared_attackers
            if self.state.get_instance(iid)
        ]
        # Filter to attackers targeting this player
        attackers = [
            a for a in attackers
            if player.player_id in self.state.declared_attackers.get(a.instance_id, [])
        ]

        for blocker in self.state.get_battlefield(player.player_id):
            if not blocker.can_block(): continue
            for attacker in attackers:
                if self._can_block(blocker, attacker):
                    actions.append(Action(
                        ActionType.DECLARE_BLOCKER,
                        actor_id=player.player_id,
                        card_iid=blocker.instance_id,
                        target_iids=(attacker.instance_id,),
                        description=f"Block {attacker.definition.name} with {blocker.definition.name}"
                    ))
        return actions

    def _activated_ability_actions(self, player: Player) -> list[Action]:
        """Activated abilities on permanents (e.g., {T}: draw a card)."""
        # Will be expanded as we add more cards
        # For now, handle basic mana abilities
        actions = []
        for inst in self.state.get_battlefield(player.player_id):
            if inst.tapped: continue
            if "mana" in inst.definition.effect_tags:
                actions.append(Action(
                    ActionType.ACTIVATE_ABILITY,
                    actor_id=player.player_id,
                    card_iid=inst.instance_id,
                    description=f"Tap {inst.definition.name} for mana"
                ))
        return actions

    # ── Helpers ────────────────────────────────────────────────────

    def _calculate_available_mana(self, player: Player, battlefield: list) -> int:
        """
        Rough estimate of mana available this turn.
        Full mana system will be in mana.py — this is sufficient for action generation.
        """
        # Count untapped lands and mana producers
        total = player.mana_pool.total
        for inst in battlefield:
            if inst.tapped: continue
            if inst.definition.is_land(): total += 1
            elif "mana" in inst.definition.effect_tags: total += 1
        return total

    def _parse_mana_cost(self, mana_cost: str, inst: CardInstance) -> int:
        """
        Convert "{2}{G}{G}" to a generic mana count.
        Full pip-by-pip cost checking happens in mana.py.
        This simplified version is enough for action generation filtering.
        """
        if not mana_cost: return 0
        import re
        total = 0
        for match in re.findall(r'\{([^}]+)\}', mana_cost):
            if match.isdigit():
                total += int(match)
            elif match == 'X':
                pass  # X costs handled separately
            else:
                total += 1  # Each colored pip = 1 mana
        return total

    def _can_block(self, blocker: CardInstance, attacker: CardInstance) -> bool:
        """Check if blocker can legally block attacker."""
        a = attacker.definition
        b = blocker.definition

        # Flying — can only be blocked by flying or reach
        if a.has_keyword("flying"):
            if not (b.has_keyword("flying") or b.has_keyword("reach")):
                return False

        # Shadow — can only block shadow
        if a.has_keyword("shadow") and not b.has_keyword("shadow"):
            return False

        # Fear — only artifact or black
        if a.has_keyword("fear"):
            if CardType.ARTIFACT not in b.definition.card_types:
                from game_state import Color
                if Color.BLACK not in b.definition.colors:
                    return False

        # Intimidate — only artifact or shares color
        if a.has_keyword("intimidate"):
            from game_state import CardType
            if CardType.ARTIFACT not in b.definition.card_types:
                if not any(c in b.definition.colors for c in a.colors):
                    return False

        return True
