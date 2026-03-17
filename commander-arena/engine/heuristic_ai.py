"""
heuristic_ai.py — Rule-based AI decision engine

This is the AI that plays the game before any ML.
It works by:
  1. Getting all legal actions from ActionGenerator
  2. Scoring each action using hand-coded game knowledge
  3. Picking the highest-scoring action

Two reasons this exists:
  A. It's the minimum shippable AI — free, fast, no API costs
  B. It's the training opponent for ML self-play — the ML learns by
     beating this AI, then beating better versions of itself

The heuristic AI will never be as good as a trained ML model,
but a well-tuned heuristic AI beats a mediocre ML model every time.
Build this well.

Design: each score_*() method handles one action type and returns a float.
Higher = better. The AI always picks the max. Ties broken randomly.
"""

from __future__ import annotations
import random
from typing import TYPE_CHECKING

from actions import Action, ActionType, ActionGenerator
from game_state import GameState, CardInstance, CardType, Zone

if TYPE_CHECKING:
    from game_state import Player


class HeuristicAI:
    """
    Rule-based AI that makes reasonable MTG decisions without any training.

    Tuning guide:
    - Weights are in WEIGHTS dict below — adjust these to change play style
    - Each score method should return a value between -100 and +100
    - PASS_PRIORITY always scores 0 (the baseline "do nothing" option)
    - An action scoring > 0 means "better than doing nothing"
    - An action scoring < 0 means "actively harmful, avoid unless forced"
    """

    # ── Tunable weights ────────────────────────────────────────────────────
    # Adjust these values to change AI aggression, priorities, etc.
    WEIGHTS = {
        # Board development
        "ramp_land":         8.0,   # Playing a land
        "ramp_rock":         7.0,   # Casting a mana rock
        "creature_per_power":1.5,   # Creature value = power * this
        "creature_vigilance":2.0,   # Extra value for vigilance (can attack + block)
        "commander_bonus":   5.0,   # Extra motivation to cast commander

        # Removal / interaction
        "removal_per_threat":4.0,   # Value of removing a high-threat creature
        "board_wipe":        6.0,   # Mass removal value (context-dependent)
        "counter_spell":     5.0,   # Counterspell value

        # Card advantage
        "draw_per_card":     3.5,   # Each extra card drawn
        "tutor":             8.0,   # Find exactly the card you need

        # Combat
        "attack_damage":     2.0,   # Damage per attack
        "lethal_kill":      20.0,   # Killing a player
        "block_prevent":     2.5,   # Damage prevented by blocking

        # Risk weights (subtracted)
        "lose_creature":    -3.0,   # Cost of losing a creature in combat
        "tap_out_risk":     -2.0,   # Risk of having no mana to respond
    }

    def __init__(self, player_id: int, randomness: float = 0.1):
        """
        player_id:   Which player this AI controls
        randomness:  0.0 = always picks best action, 1.0 = completely random
                     A small amount of randomness prevents deterministic loops
                     and makes self-play training data more diverse
        """
        self.player_id  = player_id
        self.randomness = randomness

    def choose_action(self, state: GameState) -> Action:
        """
        Main entry point. Given a game state, return the best action.
        Called by the rules engine whenever this AI has priority.
        """
        generator = ActionGenerator(state)
        legal_actions = generator.get_legal_actions()

        if not legal_actions:
            # Should never happen (PASS_PRIORITY is always legal) but safety first
            from actions import ActionType
            return Action(ActionType.PASS_PRIORITY, actor_id=self.player_id,
                         description="Pass (no actions)")

        # Score every legal action
        scored = [(self._score_action(state, a), a) for a in legal_actions]

        # Add noise proportional to randomness setting
        if self.randomness > 0:
            scored = [(s + random.gauss(0, self.randomness * 10), a) for s, a in scored]

        # Pick the highest scoring action
        scored.sort(key=lambda x: x[0], reverse=True)

        best_score, best_action = scored[0]
        return best_action

    # ── Main scoring dispatcher ────────────────────────────────────────────

    def _score_action(self, state: GameState, action: Action) -> float:
        """Route each action type to its scoring method."""
        player = state.get_player(self.player_id)
        if not player: return 0.0

        match action.action_type:
            case ActionType.PASS_PRIORITY:
                return self._score_pass(state, player)
            case ActionType.PLAY_LAND:
                return self._score_play_land(state, player, action)
            case ActionType.CAST_SPELL:
                return self._score_cast_spell(state, player, action)
            case ActionType.CAST_COMMANDER:
                return self._score_cast_commander(state, player, action)
            case ActionType.DECLARE_ATTACKER:
                return self._score_attack(state, player, action)
            case ActionType.DECLARE_BLOCKER:
                return self._score_block(state, player, action)
            case ActionType.ACTIVATE_ABILITY:
                return self._score_activate(state, player, action)
            case _:
                return 0.0

    # ── Scoring methods ────────────────────────────────────────────────────

    def _score_pass(self, state: GameState, player: Player) -> float:
        """
        Passing is usually 0, but sometimes good:
        - End step: save mana for responses
        - When significantly ahead: don't overextend into board wipes
        """
        if state.phase.name in ("END", "UPKEEP"):
            return 2.0  # Slightly prefer passing at non-action points

        # If way ahead on board, passing is safer (don't overextend)
        my_power  = sum(i.effective_power for i in state.get_battlefield(self.player_id)
                        if i.definition.is_creature())
        opp_power = sum(
            sum(i.effective_power for i in state.get_battlefield(p.player_id)
                if i.definition.is_creature())
            for p in state.get_opponents(self.player_id)
        )
        if my_power > opp_power * 2:
            return 1.5  # Slightly prefer passing when very far ahead

        return 0.0

    def _score_play_land(self, state: GameState, player: Player, action: Action) -> float:
        """
        Playing a land is almost always correct.
        Value decreases in late game (diminishing returns past 7 lands).
        """
        w = self.WEIGHTS
        lands_on_bf = sum(
            1 for i in state.get_battlefield(self.player_id)
            if i.definition.is_land()
        )

        # Early game: land every turn is critical
        if lands_on_bf < 4:  return w["ramp_land"] + 3.0
        if lands_on_bf < 7:  return w["ramp_land"]
        return w["ramp_land"] * 0.5  # Late game: lands are less impactful

    def _score_cast_spell(self, state: GameState, player: Player, action: Action) -> float:
        """Score casting a spell based on its effect tags and the board state."""
        inst = state.get_instance(action.card_iid)
        if not inst: return 0.0

        defn = inst.definition
        w    = self.WEIGHTS
        score = 0.0

        # ── Effect-based scoring ──────────────────────────────────

        if "ramp" in defn.effect_tags or "mana" in defn.effect_tags:
            score += w["ramp_rock"]

        if "draw" in defn.effect_tags:
            # Estimate cards drawn (crude — effect_tags will store count later)
            score += w["draw_per_card"] * 2

        if "tutor" in defn.effect_tags:
            score += w["tutor"]

        if "removal" in defn.effect_tags:
            # Value removal more when opponents have big threats
            biggest_threat = self._biggest_opponent_threat(state)
            if biggest_threat:
                score += w["removal_per_threat"] * (biggest_threat.effective_power / 3)

        if "board_wipe" in defn.effect_tags:
            # Board wipes are better when opponents have more creatures
            opp_creatures = sum(
                len([i for i in state.get_battlefield(p.player_id)
                     if i.definition.is_creature()])
                for p in state.get_opponents(self.player_id)
            )
            my_creatures = len([i for i in state.get_battlefield(self.player_id)
                                if i.definition.is_creature()])
            # Good wipe = opponents have more than us
            ratio = opp_creatures / max(my_creatures + 1, 1)
            score += w["board_wipe"] * ratio

        if "counter" in defn.effect_tags:
            score += w["counter_spell"]

        # ── Creature-specific scoring ─────────────────────────────

        if defn.is_creature():
            score += defn.power * w["creature_per_power"] if defn.power else 1.0
            if defn.has_keyword("vigilance"): score += w["creature_vigilance"]
            if defn.has_keyword("flying"):    score += 2.0
            if defn.has_keyword("deathtouch"):score += 2.5
            if defn.has_keyword("lifelink"):  score += 1.5
            if defn.has_keyword("haste"):     score += 2.0  # Can attack immediately

        # ── Mana efficiency bonus ─────────────────────────────────
        # Cards that cost less than their "fair" value score higher
        # CMC vs impact (crude approximation)
        lands = len([i for i in state.get_battlefield(self.player_id)
                     if i.definition.is_land()])
        if defn.cmc <= 2 and lands >= 4:
            score += 1.5  # Cheap cards are efficient to cast

        # ── Timing considerations ─────────────────────────────────
        # Don't tap out completely if opponents can still do things
        total_mana = sum(
            1 for i in state.get_battlefield(self.player_id)
            if (i.definition.is_land() or "mana" in i.definition.effect_tags)
            and not i.tapped
        )
        if total_mana - inst.definition.cmc <= 0:
            score += w["tap_out_risk"]  # Negative — risks being unable to respond

        return score

    def _score_cast_commander(self, state: GameState, player: Player, action: Action) -> float:
        """Casting your commander is generally high priority."""
        inst = state.get_instance(action.card_iid)
        if not inst: return 0.0

        base = self._score_cast_spell(state, player, action)
        return base + self.WEIGHTS["commander_bonus"]

    def _score_attack(self, state: GameState, player: Player, action: Action) -> float:
        """Score a potential attack."""
        w       = self.WEIGHTS
        attacker= state.get_instance(action.card_iid)
        if not attacker: return 0.0

        target_pid = action.target_pids[0] if action.target_pids else None
        if target_pid is None: return 0.0

        target_player = state.get_player(target_pid)
        if not target_player: return 0.0

        # Would this attack be lethal?
        if attacker.effective_power >= target_player.life:
            return self.WEIGHTS["lethal_kill"]

        # Base attack value = damage dealt
        score = attacker.effective_power * w["attack_damage"]

        # Prefer attacking the player closest to death
        life_ratio = 1 - (target_player.life / 40)
        score *= (1 + life_ratio)

        # Prefer attacking players with no blockers
        opp_bf = state.get_battlefield(target_pid)
        opp_creatures = [i for i in opp_bf if i.definition.is_creature() and not i.tapped]

        if not opp_creatures:
            score += 3.0  # Undefended!
        else:
            # Estimate if we'd trade or die
            biggest_blocker = max(opp_creatures, key=lambda i: i.effective_power,
                                  default=None)
            if biggest_blocker:
                if biggest_blocker.effective_power >= attacker.effective_toughness:
                    score += w["lose_creature"]  # We'd die

        # Bonus for trample (excess damage goes through)
        if attacker.definition.has_keyword("trample"):
            score += 1.5

        # Bonus for flying (harder to block)
        if attacker.definition.has_keyword("flying"):
            score += 1.0

        return score

    def _score_block(self, state: GameState, player: Player, action: Action) -> float:
        """Score a potential block."""
        w        = self.WEIGHTS
        blocker  = state.get_instance(action.card_iid)
        attacker_iid = action.target_iids[0] if action.target_iids else None
        if not blocker or not attacker_iid: return 0.0

        attacker = state.get_instance(attacker_iid)
        if not attacker: return 0.0

        score = 0.0

        # Value of preventing the attack damage
        score += attacker.effective_power * w["block_prevent"]

        # Would we kill the attacker?
        if blocker.effective_power >= attacker.effective_toughness:
            score += attacker.definition.cmc * 1.5  # Destroying expensive attacker = good

        # Would we die?
        if attacker.effective_power >= blocker.effective_toughness:
            score += w["lose_creature"]  # Losing our blocker = bad

        # Deathtouch: even a 0/1 can kill anything
        if blocker.definition.has_keyword("deathtouch"):
            score += 3.0  # Efficient trade

        # Never block with our commander unless absolutely necessary
        if blocker.is_commander:
            score -= 5.0

        return score

    def _score_activate(self, state: GameState, player: Player, action: Action) -> float:
        """Score an activated ability."""
        inst = state.get_instance(action.card_iid)
        if not inst: return 0.0

        # Mana abilities: generally good to activate
        if "mana" in inst.definition.effect_tags:
            # Good to activate if we have spells to cast this turn
            hand = state.get_hand(self.player_id)
            if any(c.definition.cmc > 0 for c in hand):
                return 5.0
            return 1.0  # Still fine to float mana

        return 2.0  # Generic activated ability

    # ── Utility methods ────────────────────────────────────────────────────

    def _biggest_opponent_threat(self, state: GameState) -> CardInstance | None:
        """Find the most dangerous creature an opponent controls."""
        all_threats = []
        for opp in state.get_opponents(self.player_id):
            creatures = [
                i for i in state.get_battlefield(opp.player_id)
                if i.definition.is_creature()
            ]
            all_threats.extend(creatures)

        if not all_threats: return None
        return max(all_threats, key=lambda i: i.effective_power + i.definition.cmc)

    def _my_board_value(self, state: GameState) -> float:
        """Rough estimate of our board strength."""
        total = 0.0
        for inst in state.get_battlefield(self.player_id):
            if inst.definition.is_creature():
                total += inst.effective_power + inst.effective_toughness
            elif "mana" in inst.definition.effect_tags:
                total += 2.0
        return total
