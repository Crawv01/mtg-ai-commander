"""
rules.py — The Rules Engine

Implements Magic: The Gathering turn structure per the Comprehensive Rules
(MagicCompRules_21031101). All rule references are cited inline.

Three layers, all connected:
  1. Turn structure  — untap, upkeep, draw, main, combat, end, cleanup
  2. Mana + casting  — paying costs, stack, resolving
  3. Combat          — attackers, blockers, damage assignment, death check

Rules corrections vs. first draft (verified against comp rules):
  - 103.7c: In multiplayer, NO player skips draw on turn 1 (only 2-player skips)
  - 502.3:  No priority during untap step (confirmed correct)
  - 503.2:  Active player gets priority during upkeep after triggers
  - 504.3:  Active player gets priority after drawing (before main phase)
  - 507.3:  Active player gets priority during beginning of combat step
  - 508.1:  Attackers declared all-at-once then priority given (not one-by-one)
  - 509.1:  Only attacked players may declare blockers
  - 510.1c: Blocked creature assigns damage in declared order (lethal first)
  - 514.3:  No priority during cleanup unless SBAs/triggers fire
  - 903.6:  Commanders separated BEFORE library shuffle, not after
  - 903.11/12: Commander going to GY/exile is OPTIONAL (player chooses)
  - 903.14a: Commander damage is a state-based action (704)
  - 704.5h: Deathtouch damage kills via SBA check, not during damage step

Stub policy:
  Effects not yet implemented log "[STUB] CardName: effect — not yet implemented"
  The game runs to completion; stubs just don't fire their effects.
"""

from __future__ import annotations
import random
import uuid
from typing import Optional

from game_state import (
    GameState, Player, CardInstance, CardDefinition,
    Zone, Phase, Color, CardType, ManaPool, StackObject
)
from actions import Action, ActionType


PHASE_ORDER = [
    Phase.UNTAP,
    Phase.UPKEEP,
    Phase.DRAW,
    Phase.MAIN1,
    Phase.BEGIN_COMBAT,
    Phase.DECLARE_ATTACKERS,
    Phase.DECLARE_BLOCKERS,
    Phase.COMBAT_DAMAGE,
    Phase.END_COMBAT,
    Phase.MAIN2,
    Phase.END,
    Phase.CLEANUP,
]


class RulesEngine:
    """
    Executes actions and advances game state per the Comprehensive Rules.
    All state lives in GameState. RulesEngine is stateless.
    """

    def __init__(self, state: GameState):
        self.state = state

    # =========================================================================
    # PUBLIC ENTRY POINT
    # =========================================================================

    def execute_action(self, action: Action) -> list[str]:
        """
        Execute a player action. Mutates self.state. Returns log lines.
        This is the only method the AI / simulator needs to call directly.
        """
        logs = []

        match action.action_type:
            case ActionType.PASS_PRIORITY:
                logs += self._handle_pass_priority(action)
            case ActionType.PLAY_LAND:
                logs += self._handle_play_land(action)
            case ActionType.CAST_SPELL | ActionType.CAST_COMMANDER:
                logs += self._handle_cast_spell(action)
            case ActionType.ACTIVATE_ABILITY:
                logs += self._handle_activate_ability(action)
            case ActionType.DECLARE_ATTACKER:
                logs += self._handle_declare_attacker(action)
            case ActionType.DECLARE_BLOCKER:
                logs += self._handle_declare_blocker(action)
            case ActionType.MOVE_TO_COMMAND:
                logs += self._handle_move_to_command(action)
            case _:
                logs.append(f"[WARN] Unhandled action: {action.action_type}")

        # Rule 116.5, 704.3: SBAs checked before any player gets priority
        logs += self._check_state_based_actions()
        logs += self._check_game_over()

        for line in logs:
            self.state.add_to_log(line)

        return logs

    # =========================================================================
    # GAME SETUP
    # =========================================================================

    def setup_game(
        self,
        player_decks: list[tuple[Player, list[CardDefinition], list[str]]]
    ) -> list[str]:
        """
        Initialize a game from (Player, decklist, commander_names) tuples.

        Rule 903.6: Commanders go to command zone FIRST, then shuffle remaining 99.
        Rule 903.7: Life set to 40, each player draws 7.
        Rule 103.7c: In multiplayer (Commander), NO player skips draw on turn 1.
        """
        logs = []
        self.state.players = []

        for player, decklist, commander_names in player_decks:
            self.state.players.append(player)
            player.library.clear()
            player.hand.clear()
            player.command_zone.clear()
            player.life = 40

            commanders_placed = []
            library_iids = []

            for defn in decklist:
                inst = self._create_instance(defn, player.player_id)
                if defn.name in commander_names:
                    # Rule 903.6: commander goes to command zone before shuffle
                    inst.is_commander = True
                    inst.zone = Zone.COMMAND
                    player.command_zone.append(inst.instance_id)
                    commanders_placed.append(defn.name)
                else:
                    inst.zone = Zone.LIBRARY
                    library_iids.append(inst.instance_id)

            # Rule 903.6: shuffle only the non-commander 99 cards
            random.shuffle(library_iids)
            player.library = library_iids

            cmd_str = ", ".join(commanders_placed) if commanders_placed else "none"
            logs.append(f"{player.name}: commanders in command zone: [{cmd_str}]. "
                        f"Library: {len(player.library)} cards.")

        # Determine turn order (rule 103.1: random)
        random.shuffle(self.state.players)

        # Rule 903.7: each player draws opening hand of 7
        for player in self.state.players:
            for _ in range(7):
                self._draw_card(player)

        self.state.active_player   = self.state.players[0].player_id
        self.state.priority_player = self.state.active_player
        self.state.phase           = Phase.UNTAP
        self.state.turn_number     = 1

        order = " → ".join(p.name for p in self.state.players)
        logs.append(f"Turn order: {order}")
        logs.append(f"{self.state.players[0].name} goes first.")
        logs.append("(Rule 103.7c: no player skips draw in multiplayer.)")

        return logs

    # =========================================================================
    # TURN STRUCTURE
    # =========================================================================

    def begin_turn(self) -> list[str]:
        """
        Start a new turn: run untap (no priority), then open upkeep priority window.
        The simulator calls this, then loops execute_action() until upkeep passes.
        """
        logs = []
        player = self.state.get_active_player()
        logs.append(f"\n── Turn {self.state.turn_number}: {player.name} ──")

        # UNTAP (502): automatic, no priority (rule 502.3)
        self.state.phase = Phase.UNTAP
        logs += self._run_untap_step(player)

        # UPKEEP (503): triggers go on stack, then active player gets priority
        self.state.phase = Phase.UPKEEP
        logs += self._run_upkeep_triggers(player)
        self.state.priority_player = player.player_id
        logs.append(f"Upkeep. {player.name} has priority.")

        return logs

    def _run_untap_step(self, player: Player) -> list[str]:
        """
        Rule 502.1: Phase (stub).
        Rule 502.2: Untap all permanents active player controls simultaneously.
        Rule 502.3: No priority given.
        """
        logs = []

        # Rule 502.1: STUB — phasing not yet implemented

        # Rule 502.2: untap all controlled permanents simultaneously
        untapped = 0
        for inst in self.state.get_battlefield(player.player_id):
            if inst.tapped:
                inst.tapped = False
                untapped += 1
            # Summoning sickness removed — creature controlled since turn began
            if inst.definition.is_creature():
                inst.summoning_sick = False

        player.lands_played_this_turn = 0
        player.draw_this_turn = 0

        if untapped:
            logs.append(f"{player.name} untaps {untapped} permanent(s).")
        return logs

    def _run_upkeep_triggers(self, player: Player) -> list[str]:
        """Rule 503.1: Beginning-of-upkeep triggers go on stack (stubbed)."""
        logs = []
        for inst in self.state.get_battlefield():
            text = inst.definition.oracle_text.lower()
            if "beginning of your upkeep" in text or "beginning of each upkeep" in text:
                ctrl = self.state.get_player(inst.controller_id)
                is_relevant = (ctrl and ctrl.player_id == player.player_id) \
                              or "each upkeep" in text
                if is_relevant:
                    logs.append(f"[STUB] {inst.definition.name}: upkeep trigger — "
                                f"not yet implemented")
        return logs

    def _advance_to_draw(self) -> list[str]:
        """
        Called when all players pass in upkeep with empty stack.
        Rule 504.1: Active player draws (turn-based action, no stack).
        Rule 504.3: Active player gets priority after drawing.
        """
        logs = []
        player = self.state.get_active_player()
        self.state.phase = Phase.DRAW

        drawn = self._draw_card(player)
        if drawn:
            logs.append(f"{player.name} draws. ({len(player.hand)} in hand)")
        else:
            # Rule 704.5b: attempted draw from empty library → loses
            player.has_lost = True
            logs.append(f"{player.name} tries to draw from empty library → loses.")

        # Rule 504.2: any draw-step triggers go on stack (stub)
        # Rule 504.3: active player gets priority
        self.state.priority_player = player.player_id
        logs.append(f"Draw step. {player.name} has priority.")
        return logs

    def _advance_to_main1(self) -> list[str]:
        """Called when all players pass in draw step with empty stack."""
        logs = []
        player = self.state.get_active_player()
        self.state.phase = Phase.MAIN1
        self.state.priority_player = player.player_id
        logs.append(f"{player.name} enters Main Phase 1.")
        return logs

    def end_turn(self) -> list[str]:
        """
        Run end step (triggers), cleanup, then pass turn.
        Called when active player passes in MAIN2 with empty stack.
        """
        logs = []
        player = self.state.get_active_player()

        # END STEP (513)
        self.state.phase = Phase.END
        logs += self._run_end_triggers(player)
        # Rule 513.2: active player gets priority (simulator handles this
        # by looping through any actions before cleanup)

        # CLEANUP (514)
        self.state.phase = Phase.CLEANUP
        logs += self._run_cleanup(player)

        logs += self._advance_turn()
        return logs

    def _run_end_triggers(self, player: Player) -> list[str]:
        """Rule 513.1: 'at the beginning of the end step' triggers."""
        logs = []
        for inst in self.state.get_battlefield():
            text = inst.definition.oracle_text.lower()
            if "beginning of" in text and "end step" in text:
                logs.append(f"[STUB] {inst.definition.name}: end step trigger — "
                            f"not yet implemented")
        return logs

    def _run_cleanup(self, player: Player) -> list[str]:
        """
        Rule 514.1: Discard to max hand size (7). Turn-based action, no stack.
        Rule 514.2: Remove all damage; end 'until EOT' effects. Simultaneous.
        Rule 514.3: No priority unless SBAs or triggers fire.
        """
        logs = []

        # Rule 514.1: discard to hand size
        while len(player.hand) > 7:
            iid  = player.hand[-1]
            inst = self.state.get_instance(iid)
            name = inst.definition.name if inst else "unknown"
            self._move_to_zone(inst, Zone.GRAVEYARD, player)
            logs.append(f"{player.name} discards {name} (hand size).")

        # Rule 514.2: remove all damage from ALL permanents simultaneously
        for inst in self.state.get_battlefield():
            inst.damage_marked = 0
            inst._deathtouch_damaged = False
            # STUB: "until end of turn" effects end here

        # Reset combat tracking
        self.state.declared_attackers.clear()
        self.state.declared_blockers.clear()
        for inst in self.state.get_battlefield():
            inst.is_attacking = False
            inst.is_blocking  = False
            inst.blocked_by.clear()
            inst.blocking     = None

        # Empty mana pools (rule 500.4)
        for p in self.state.players:
            p.mana_pool.empty()

        return logs

    def _advance_turn(self) -> list[str]:
        """Move to next living player's turn."""
        logs = []
        ids = [p.player_id for p in self.state.players]
        cur = ids.index(self.state.active_player)

        for offset in range(1, len(ids) + 1):
            next_pid = ids[(cur + offset) % len(ids)]
            nxt      = self.state.get_player(next_pid)
            if nxt and not nxt.has_lost:
                self.state.active_player   = next_pid
                self.state.priority_player = next_pid
                if next_pid == self.state.players[0].player_id:
                    self.state.turn_number += 1
                logs.append(f"Turn passes to {nxt.name}.")
                break

        return logs

    # =========================================================================
    # PRIORITY
    # =========================================================================

    def _handle_pass_priority(self, action: Action) -> list[str]:
        """
        Rule 116.3d: Player passes → next player in turn order gets priority.
        Rule 116.4:  All players pass in succession →
                       stack non-empty: resolve top
                       stack empty:     end current phase/step
        """
        logs = []
        state  = self.state
        living = state.living_players()
        if not living:
            return logs

        ids      = [p.player_id for p in living]
        cur_idx  = ids.index(state.priority_player) if state.priority_player in ids else 0
        next_pid = ids[(cur_idx + 1) % len(ids)]

        if next_pid == state.active_player:
            # All players have passed in succession
            if state.stack:
                logs += self._resolve_top_of_stack()
                # Rule 116.3b: active player gets priority after resolution
                state.priority_player = state.active_player
            else:
                logs += self._advance_phase()
        else:
            state.priority_player = next_pid
            nxt = state.get_player(next_pid)
            if nxt:
                logs.append(f"Priority → {nxt.name}.")

        return logs

    def _advance_phase(self) -> list[str]:
        """Drive the turn forward when all players pass with empty stack."""
        logs = []
        state = self.state
        phase = state.phase

        if phase == Phase.UPKEEP:
            logs += self._advance_to_draw()

        elif phase == Phase.DRAW:
            logs += self._advance_to_main1()

        elif phase == Phase.MAIN1:
            # Rule 507: Beginning of combat step
            # Rule 507.1: In Commander (attack multiple players), no single
            #             defending player chosen — all opponents can be attacked.
            # Rule 507.2: Triggers at beginning of combat (stub)
            # Rule 507.3: Active player gets priority
            state.phase = Phase.BEGIN_COMBAT
            state.priority_player = state.active_player
            logs.append(f"Beginning of combat. {state.get_active_player().name} has priority.")

        elif phase == Phase.BEGIN_COMBAT:
            # Rule 508: Declare attackers step
            state.phase = Phase.DECLARE_ATTACKERS
            state.priority_player = state.active_player
            logs.append("Declare attackers step.")

        elif phase == Phase.DECLARE_ATTACKERS:
            if not state.declared_attackers:
                # Rule 506.1: skip blockers and damage if no attackers
                state.phase = Phase.END_COMBAT
                state.priority_player = state.active_player
                logs.append("No attackers. End of combat.")
            else:
                # Rule 509: Declare blockers step
                # Priority goes to first attacked player in turn order
                state.phase = Phase.DECLARE_BLOCKERS
                attacked = self._get_attacked_player_ids()
                ids2 = [p.player_id for p in state.players if not p.has_lost]
                cur2 = ids2.index(state.active_player)
                for off in range(1, len(ids2) + 1):
                    pid = ids2[(cur2 + off) % len(ids2)]
                    if pid in attacked:
                        state.priority_player = pid
                        pname = state.get_player(pid).name
                        logs.append(f"Declare blockers. {pname} to declare blockers.")
                        break

        elif phase == Phase.DECLARE_BLOCKERS:
            # All defending players have passed → combat damage
            state.phase = Phase.COMBAT_DAMAGE
            logs += self._run_combat_damage()
            # Rule 511: End of combat
            state.phase = Phase.END_COMBAT
            logs += self._run_end_of_combat_triggers()
            state.priority_player = state.active_player
            logs.append("End of combat.")

        elif phase == Phase.END_COMBAT:
            state.phase = Phase.MAIN2
            state.priority_player = state.active_player
            logs.append(f"{state.get_active_player().name} enters Main Phase 2.")

        elif phase == Phase.MAIN2:
            logs += self.end_turn()

        elif phase == Phase.END:
            state.phase = Phase.CLEANUP
            logs += self._run_cleanup(state.get_active_player())
            logs += self._advance_turn()

        return logs

    def _run_end_of_combat_triggers(self) -> list[str]:
        """Rule 511.1: 'at end of combat' triggers."""
        logs = []
        for inst in self.state.get_battlefield():
            if "end of combat" in inst.definition.oracle_text.lower():
                logs.append(f"[STUB] {inst.definition.name}: end of combat trigger — "
                            f"not yet implemented")
        # Rule 511.3: all creatures/planeswalkers removed from combat
        for inst in self.state.get_battlefield():
            inst.is_attacking = False
            inst.is_blocking  = False
        return logs

    # =========================================================================
    # MANA & CASTING
    # =========================================================================

    def _handle_play_land(self, action: Action) -> list[str]:
        """
        Rule 305: Playing a land.
        Rule 505.5b: Main phase only, stack empty, once per turn, active player only.
        Not a spell — no stack, no responses.
        """
        logs = []
        player = self.state.get_player(action.actor_id)
        inst   = self.state.get_instance(action.card_iid)

        if not player or not inst:
            return ["[ERROR] play_land: invalid player or card"]
        if player.player_id != self.state.active_player:
            return ["[ERROR] play_land: not active player"]
        if self.state.phase not in (Phase.MAIN1, Phase.MAIN2):
            return ["[ERROR] play_land: not a main phase"]
        if self.state.stack:
            return ["[ERROR] play_land: stack not empty"]
        if player.lands_played_this_turn >= 1:
            return [f"[ERROR] {player.name} already played a land this turn"]

        self._move_to_zone(inst, Zone.BATTLEFIELD, player)
        inst.tapped         = False
        inst.summoning_sick = False
        player.lands_played_this_turn += 1

        logs.append(f"{player.name} plays {inst.definition.name}.")

        # STUB: ETB effects on lands
        action_tags = [t for t in inst.definition.effect_tags if t != "mana"]
        if action_tags:
            logs.append(f"[STUB] {inst.definition.name}: ETB {action_tags[0]} — "
                        f"not yet implemented")

        return logs

    def _handle_cast_spell(self, action: Action) -> list[str]:
        """
        Rule 601.2: Cast a spell — move to stack, announce choices, pay costs.
        """
        logs = []
        player = self.state.get_player(action.actor_id)
        inst   = self.state.get_instance(action.card_iid)

        if not player or not inst:
            return ["[ERROR] cast_spell: invalid player or card"]

        defn = inst.definition

        # Rule 505.5a: sorcery-speed restriction
        if not defn.is_instant() and not defn.has_keyword("flash"):
            if self.state.active_player != player.player_id:
                return [f"[ERROR] {defn.name} is sorcery speed — not active player"]
            if self.state.phase not in (Phase.MAIN1, Phase.MAIN2):
                return [f"[ERROR] {defn.name} is sorcery speed — wrong phase "
                        f"({self.state.phase.name})"]
            if self.state.stack:
                return [f"[ERROR] {defn.name} is sorcery speed — stack not empty"]

        cost = self._calculate_cost(inst)
        paid = self._pay_mana_cost(player, cost)
        if paid is None:
            return [f"[ERROR] {player.name} cannot afford {defn.name} "
                    f"(needs {cost}, available {self._available_mana(player)})"]
        logs += paid

        # Remove from source zone
        if action.card_iid in player.hand:
            player.hand.remove(action.card_iid)
        elif action.card_iid in player.command_zone:
            player.command_zone.remove(action.card_iid)
            inst.times_cast += 1  # Tax applies on next cast

        inst.zone = Zone.STACK

        stack_obj = StackObject(
            stack_id      = str(uuid.uuid4()),
            source_id     = inst.instance_id,
            controller_id = player.player_id,
            effect_key    = defn.name,
            targets       = list(action.target_iids),
        )
        self.state.stack.append(stack_obj)

        tax_note = ""
        if action.action_type == ActionType.CAST_COMMANDER and inst.times_cast > 1:
            tax_note = f" (tax +{(inst.times_cast - 1) * 2})"
        logs.append(f"{player.name} casts {defn.name}{tax_note}. "
                    f"[stack: {len(self.state.stack)}]")

        # Rule 116.3c: caster gets priority after casting
        self.state.priority_player = player.player_id
        return logs

    def _calculate_cost(self, inst: CardInstance) -> int:
        """
        Total generic mana cost including commander tax.
        Rule 903.10: +{2} per previous cast from command zone.
        """
        import re
        cost = 0
        for m in re.findall(r'\{([^}]+)\}', inst.definition.mana_cost):
            if m.isdigit():
                cost += int(m)
            elif m.upper() not in ('X', 'Y', 'Z'):
                cost += 1
        if inst.is_commander:
            cost += inst.commander_tax
        return cost

    def _available_mana(self, player: Player) -> int:
        total = player.mana_pool.total
        for inst in self.state.get_battlefield(player.player_id):
            if not inst.tapped:
                if inst.definition.is_land() or "mana" in inst.definition.effect_tags:
                    total += 1
        return total

    def _pay_mana_cost(self, player: Player, cost: int) -> Optional[list[str]]:
        """
        Pay cost by using mana pool then tapping producers.
        Returns log lines or None if can't afford.
        """
        logs = []
        remaining = cost

        # Drain mana pool first
        for attr in ['colorless','white','blue','black','red','green']:
            if remaining <= 0:
                break
            take = min(getattr(player.mana_pool, attr), remaining)
            setattr(player.mana_pool, attr, getattr(player.mana_pool, attr) - take)
            remaining -= take

        if remaining == 0:
            return logs

        # Tap producers (lands first, then mana rocks)
        producers = sorted(
            [i for i in self.state.get_battlefield(player.player_id)
             if not i.tapped and
             (i.definition.is_land() or "mana" in i.definition.effect_tags)],
            key=lambda i: 0 if i.definition.is_land() else 1
        )

        tapped = []
        for prod in producers:
            if remaining <= 0:
                break
            prod.tapped = True
            tapped.append(prod.definition.name)
            remaining -= 1

        if remaining > 0:
            for inst in self.state.get_battlefield(player.player_id):
                if inst.definition.name in tapped:
                    inst.tapped = False
            return None

        if tapped:
            logs.append(f"  {player.name} taps: {', '.join(tapped)}.")
        return logs

    def _handle_activate_ability(self, action: Action) -> list[str]:
        """
        Rule 605: Mana abilities resolve immediately without the stack.
        """
        logs = []
        player = self.state.get_player(action.actor_id)
        inst   = self.state.get_instance(action.card_iid)

        if not player or not inst:
            return ["[ERROR] activate_ability: invalid"]
        if inst.tapped:
            return [f"[ERROR] {inst.definition.name} already tapped"]

        inst.tapped = True
        player.mana_pool.add(Color.COLORLESS, 1)
        logs.append(f"[STUB] {player.name} taps {inst.definition.name} for {{C}} "
                    f"(actual color not yet implemented — rule 605).")
        return logs

    # =========================================================================
    # STACK RESOLUTION
    # =========================================================================

    def _resolve_top_of_stack(self) -> list[str]:
        """
        Rule 608: Resolve topmost stack object.
        Permanents enter the battlefield.
        Non-permanents execute effect then go to graveyard.
        """
        logs = []
        if not self.state.stack:
            return logs

        stack_obj  = self.state.stack.pop()
        inst       = self.state.get_instance(stack_obj.source_id)

        if not inst:
            logs.append("[WARN] Resolving stack object with missing card — skipped")
            return logs

        defn       = inst.definition
        controller = self.state.get_player(stack_obj.controller_id)
        logs.append(f"{defn.name} resolves.")

        if defn.is_permanent():
            # Rule 608.3: permanent spell → enter battlefield
            self._move_to_zone(inst, Zone.BATTLEFIELD, controller)
            inst.summoning_sick = True
            inst.has_haste      = defn.has_keyword("haste")
            logs.append(f"  {defn.name} enters the battlefield"
                        f"{' (haste)' if inst.has_haste else ''}.")

            # ETB effects — run if card has effect tags
            etb_tags = [t for t in defn.effect_tags
                        if t not in ("mana","evasion","protection",
                                     "deathtouch","lifegain","haste","vigilance")]
            if etb_tags:
                try:
                    from effects import execute_effect
                    logs += execute_effect(self.state, inst, stack_obj, controller)
                except ImportError:
                    logs.append(f"  [STUB] {defn.name}: ETB {etb_tags[0]} — "
                                f"not yet implemented")
        else:
            # Try real effects first; fall back to stub for unimplemented tags
            try:
                from effects import execute_effect
                logs += execute_effect(self.state, inst, stack_obj, controller)
            except ImportError:
                logs += self._execute_effect_stub(inst, stack_obj, controller)
            self._move_to_zone(inst, Zone.GRAVEYARD, controller)

        # Rule 116.3b: active player gets priority after resolution
        self.state.priority_player = self.state.active_player
        return logs

    def _execute_effect_stub(self, inst, stack_obj, controller) -> list[str]:
        """Log what each effect tag would do — replace in effects.py."""
        DESCS = {
            "draw":        "draw cards",
            "removal":     "destroy/exile target",
            "board_wipe":  "destroy all creatures",
            "counter":     "counter target spell",
            "ramp":        "search for land",
            "token":       "create token(s)",
            "bounce":      "return target to hand",
            "damage":      "deal damage",
            "tutor":       "search library",
            "recursion":   "return from graveyard",
            "pump":        "give +X/+X",
            "discard":     "opponent discards",
            "lifegain":    "gain life",
            "scry":        "scry N",
            "copy":        "copy spell/permanent",
            "extra_turn":  "extra turn",
            "extra_combat":"extra combat",
            "sacrifice":   "sacrifice permanent",
            "counters":    "+1/+1 counters",
        }
        tags = inst.definition.effect_tags
        if not tags:
            return [f"  [STUB] {inst.definition.name}: no tags — nothing happens"]
        return [f"  [STUB] {inst.definition.name}: {DESCS.get(t, t)} — "
                f"not yet implemented"
                for t in tags]

    # =========================================================================
    # COMBAT
    # =========================================================================

    def _handle_declare_attacker(self, action: Action) -> list[str]:
        """
        Rule 508.1: Declare attacker.
        Attackers are declared as a batch (simulator collects them all),
        but each individual declaration is registered here.
        Priority is given only AFTER the full batch is declared.
        """
        logs = []
        player   = self.state.get_player(action.actor_id)
        attacker = self.state.get_instance(action.card_iid)

        if not player or not attacker:
            return ["[ERROR] declare_attacker: invalid"]
        if not attacker.can_attack():
            return [f"[ERROR] {attacker.definition.name} cannot attack"]

        target_pid = action.target_pids[0] if action.target_pids else None
        if target_pid is None:
            return ["[ERROR] declare_attacker: no target"]

        target = self.state.get_player(target_pid)
        if not target:
            return ["[ERROR] declare_attacker: target not found"]

        attacker.is_attacking = True
        if attacker.instance_id not in self.state.declared_attackers:
            self.state.declared_attackers[attacker.instance_id] = []
        if target_pid not in self.state.declared_attackers[attacker.instance_id]:
            self.state.declared_attackers[attacker.instance_id].append(target_pid)

        # Rule 508.1f: tap unless vigilance
        if not attacker.definition.has_keyword("vigilance"):
            attacker.tapped = True

        logs.append(f"{player.name} attacks {target.name} with {attacker.definition.name}.")

        # STUB: "whenever ~ attacks" triggers
        if "attacks" in attacker.definition.oracle_text.lower():
            logs.append(f"  [STUB] {attacker.definition.name}: attack trigger — not yet implemented")

        return logs

    def _handle_declare_blocker(self, action: Action) -> list[str]:
        """
        Rule 509.1: Declare blocker.
        Only players being attacked may block.
        """
        logs = []
        player  = self.state.get_player(action.actor_id)
        blocker = self.state.get_instance(action.card_iid)

        if not player or not blocker:
            return ["[ERROR] declare_blocker: invalid"]

        # Rule 509.1: only defending (attacked) players block
        if player.player_id not in self._get_attacked_player_ids():
            return [f"[ERROR] {player.name} is not being attacked"]

        attacker_iid = action.target_iids[0] if action.target_iids else None
        if not attacker_iid:
            return ["[ERROR] declare_blocker: no attacker specified"]

        attacker = self.state.get_instance(attacker_iid)
        if not attacker or not attacker.is_attacking:
            return ["[ERROR] declare_blocker: invalid attacker"]

        if blocker.tapped:
            return [f"[ERROR] {blocker.definition.name} is tapped"]

        blocker.is_blocking = True
        blocker.blocking    = attacker_iid
        if blocker.instance_id not in attacker.blocked_by:
            attacker.blocked_by.append(blocker.instance_id)
        self.state.declared_blockers[blocker.instance_id] = attacker_iid

        logs.append(f"{player.name} blocks {attacker.definition.name} "
                    f"with {blocker.definition.name}.")
        return logs

    def _get_attacked_player_ids(self) -> set[int]:
        attacked = set()
        for pids in self.state.declared_attackers.values():
            attacked.update(pids)
        return attacked

    def _run_combat_damage(self) -> list[str]:
        """
        Rule 510: Combat damage step.
        510.1: Active player announces damage order, then defending player.
        510.2: All damage dealt simultaneously.
        510.1c: Must assign lethal to each earlier blocker before moving to next.

        STUB: First strike / double strike creates additional pre-damage step (506.1).
        """
        logs = []
        logs.append("Combat damage.")

        # STUB: first strike pre-damage step

        for attacker_iid, target_pids in self.state.declared_attackers.items():
            attacker = self.state.get_instance(attacker_iid)
            if not attacker or attacker.effective_power <= 0:
                continue

            blockers = [
                self.state.get_instance(biid)
                for biid in attacker.blocked_by
                if self.state.get_instance(biid)
            ]

            if blockers:
                logs += self._assign_blocked_damage(attacker, blockers)
            else:
                # Rule 510.1b: unblocked → damage to attacked player
                for pid in target_pids:
                    target = self.state.get_player(pid)
                    if target:
                        logs += self._deal_combat_damage_to_player(
                            attacker, target, attacker.effective_power
                        )

        return logs

    def _assign_blocked_damage(
        self,
        attacker: CardInstance,
        blockers: list[CardInstance],
    ) -> list[str]:
        """
        Rule 510.1c: Assign damage to blockers in order.
        Must assign lethal to each blocker before assigning to the next.
        Deathtouch: 1 point of damage is lethal (via 704.5h SBA).
        Trample: excess past lethal goes to attacked player.
        Lifelink: gain life equal to damage dealt.
        """
        logs = []
        has_trample  = attacker.definition.has_keyword("trample")
        has_death    = attacker.definition.has_keyword("deathtouch")
        has_lifelink = attacker.definition.has_keyword("lifelink")
        atk_ctrl     = self.state.get_player(attacker.controller_id)

        remaining   = attacker.effective_power
        total_dealt = 0

        for blocker in blockers:
            if remaining <= 0:
                break

            # Lethal damage needed for this blocker
            # Rule 704.5h: any damage from deathtouch source kills
            if has_death:
                lethal_needed = 1
            else:
                lethal_needed = max(0,
                    blocker.effective_toughness - blocker.damage_marked)

            # Must assign at least lethal before moving on (510.1c)
            assign = min(remaining, lethal_needed) if lethal_needed > 0 else 0

            # Can assign more (but AI assigns exactly lethal for simplicity)
            if assign > 0:
                blocker.damage_marked += assign
                if has_death:
                    blocker._deathtouch_damaged = True
                remaining   -= assign
                total_dealt += assign
                logs.append(f"  {attacker.definition.name} → "
                            f"{blocker.definition.name}: {assign} damage.")

        # Trample: remaining damage hits the attacked player
        if has_trample and remaining > 0:
            for pid in self.state.declared_attackers.get(attacker.instance_id, []):
                tgt = self.state.get_player(pid)
                if tgt:
                    logs += self._deal_combat_damage_to_player(attacker, tgt, remaining)
                    total_dealt += remaining

        # Lifelink on attacker
        if has_lifelink and atk_ctrl and total_dealt > 0:
            atk_ctrl.life += total_dealt
            logs.append(f"  Lifelink: {atk_ctrl.name} gains {total_dealt} life "
                        f"({atk_ctrl.life} total).")

        # Each blocker deals its power to the attacker
        for blocker in blockers:
            if blocker.effective_power > 0:
                attacker.damage_marked += blocker.effective_power
                if blocker.definition.has_keyword("deathtouch"):
                    attacker._deathtouch_damaged = True
                logs.append(f"  {blocker.definition.name} → "
                            f"{attacker.definition.name}: {blocker.effective_power} damage.")

                # Blocker lifelink
                if blocker.definition.has_keyword("lifelink"):
                    bctrl = self.state.get_player(blocker.controller_id)
                    if bctrl:
                        bctrl.life += blocker.effective_power
                        logs.append(f"  Lifelink: {bctrl.name} gains "
                                    f"{blocker.effective_power} life.")

        return logs

    def _deal_combat_damage_to_player(
        self,
        source: CardInstance,
        target: Player,
        amount: int,
    ) -> list[str]:
        """
        Deal combat damage to a player.
        Rule 903.14a: Track commander damage.
        """
        logs = []
        if amount <= 0:
            return logs

        target.life -= amount
        logs.append(f"  {source.definition.name} → {target.name}: {amount} damage. "
                    f"({target.name}: {target.life} life)")

        # Rule 903.14a: commander damage tracking (combat damage only)
        if source.is_commander:
            ctrl = source.controller_id
            prev = target.commander_damage.get(ctrl, 0)
            target.commander_damage[ctrl] = prev + amount
            total = target.commander_damage[ctrl]
            logs.append(f"  Commander damage to {target.name}: {total} total "
                        f"(from {source.definition.name}).")

        # Lifelink on unblocked attacker
        if source.definition.has_keyword("lifelink"):
            atk_ctrl = self.state.get_player(source.controller_id)
            if atk_ctrl:
                atk_ctrl.life += amount
                logs.append(f"  Lifelink: {atk_ctrl.name} gains {amount} life.")

        return logs

    # =========================================================================
    # COMMANDER ZONE
    # =========================================================================

    def _handle_move_to_command(self, action: Action) -> list[str]:
        """
        Rule 903.11/12: Owner MAY move commander to command zone instead of GY/exile.
        This action represents the player making that choice.
        """
        logs = []
        player = self.state.get_player(action.actor_id)
        inst   = self.state.get_instance(action.card_iid)

        if not player or not inst or not inst.is_commander:
            return ["[ERROR] move_to_command: not a commander"]
        if inst.zone not in (Zone.GRAVEYARD, Zone.EXILE):
            return [f"[ERROR] commander is in {inst.zone.name}, not GY/exile"]

        self._move_to_zone(inst, Zone.COMMAND, player)
        inst.tapped              = False
        inst.damage_marked       = 0
        inst._deathtouch_damaged = False
        logs.append(f"{inst.definition.name} moves to the command zone "
                    f"(rule 903.{'11' if inst.zone == Zone.GRAVEYARD else '12'}).")
        return logs

    # =========================================================================
    # STATE-BASED ACTIONS (Rule 704)
    # =========================================================================

    def _check_state_based_actions(self) -> list[str]:
        """
        Rule 704.3: Check SBAs before each player gets priority.
        Repeat until no SBAs fire.

        Implemented SBAs:
          704.5a: 0 or less life → loses
          704.5b: drew from empty library → loses (set by _draw_card)
          704.5c: 10+ poison → loses
          704.5f: 0 or less toughness → graveyard
          704.5g: lethal damage → destroyed
          704.5h: deathtouch damage → destroyed
          903.14a: 21+ commander damage from one source → loses
        """
        logs = []
        changed = True

        while changed:
            changed = False

            # Check creatures on battlefield
            for inst in list(self.state.get_battlefield()):
                if not inst.definition.is_creature():
                    continue
                owner = self.state.get_player(inst.owner_id)

                # 704.5f: toughness 0 or less
                if inst.effective_toughness <= 0:
                    logs.append(f"SBA 704.5f: {inst.definition.name} "
                                f"has {inst.effective_toughness} toughness.")
                    logs += self._permanent_to_zone(inst, owner, Zone.GRAVEYARD)
                    changed = True
                    continue

                # Skip indestructible for damage-based SBAs
                indestr = inst.definition.has_keyword("indestructible")

                # 704.5h: any damage from deathtouch source
                if not indestr and getattr(inst, '_deathtouch_damaged', False):
                    logs.append(f"SBA 704.5h: {inst.definition.name} "
                                f"damaged by deathtouch → destroyed.")
                    logs += self._permanent_to_zone(inst, owner, Zone.GRAVEYARD)
                    changed = True
                    continue

                # 704.5g: lethal damage
                if (not indestr and
                        inst.effective_toughness > 0 and
                        inst.damage_marked >= inst.effective_toughness):
                    logs.append(f"SBA 704.5g: {inst.definition.name} "
                                f"has lethal damage ({inst.damage_marked}/"
                                f"{inst.effective_toughness}).")
                    logs += self._permanent_to_zone(inst, owner, Zone.GRAVEYARD)
                    changed = True
                    continue

            # Check players
            for player in self.state.players:
                if player.has_lost:
                    continue

                # 704.5a
                if player.life <= 0:
                    player.has_lost = True
                    logs.append(f"SBA 704.5a: {player.name} at {player.life} life → loses.")
                    logs += self._remove_player_permanents(player)
                    changed = True

                # 704.5c
                elif player.poison >= 10:
                    player.has_lost = True
                    logs.append(f"SBA 704.5c: {player.name} has {player.poison} poison → loses.")
                    logs += self._remove_player_permanents(player)
                    changed = True

                # 903.14a
                elif any(dmg >= 21 for dmg in player.commander_damage.values()):
                    player.has_lost = True
                    worst_src = max(player.commander_damage,
                                   key=player.commander_damage.get)
                    total = player.commander_damage[worst_src]
                    logs.append(f"SBA 903.14a: {player.name} took {total} "
                                f"commander damage → loses.")
                    logs += self._remove_player_permanents(player)
                    changed = True

        return logs

    def _permanent_to_zone(
        self,
        inst:   CardInstance,
        owner:  Optional[Player],
        target: Zone,
    ) -> list[str]:
        """
        Move a permanent to a zone.
        Rule 903.11/12: Commander MAY go to command zone instead of GY/exile.
        AI always chooses command zone (optimal play).
        """
        logs = []

        if inst.is_commander and owner and target in (Zone.GRAVEYARD, Zone.EXILE):
            # Rule 903.11/12: replacement effect — go to command zone instead
            self._move_to_zone(inst, Zone.COMMAND, owner)
            inst.tapped              = False
            inst.damage_marked       = 0
            inst._deathtouch_damaged = False
            logs.append(f"  {inst.definition.name} → command zone (rule 903.11/12).")
        else:
            self._move_to_zone(inst, target, owner)
            inst._deathtouch_damaged = False

            if target == Zone.GRAVEYARD:
                text = inst.definition.oracle_text.lower()
                if "when" in text and ("dies" in text or "graveyard" in text):
                    logs.append(f"  [STUB] {inst.definition.name}: dies trigger — "
                                f"not yet implemented")

        # Reset combat state
        inst.is_attacking = False
        inst.is_blocking  = False
        inst.blocked_by.clear()
        inst.blocking     = None

        return logs

    def _remove_player_permanents(self, player: Player) -> list[str]:
        """Remove all permanents a losing player controls."""
        logs = []
        for inst in self.state.get_battlefield(player.player_id):
            inst.zone = Zone.EXILE
            logs.append(f"  {inst.definition.name} exiled (controller lost).")
        return logs

    # =========================================================================
    # GAME OVER
    # =========================================================================

    def _check_game_over(self) -> list[str]:
        """Rule 104.2a: player wins when all opponents have left."""
        logs = []
        living = self.state.living_players()

        if len(living) <= 1 and self.state.winner_id is None:
            if len(living) == 1:
                self.state.winner_id = living[0].player_id
                logs.append(f"\n🏆 {living[0].name} wins!")
            else:
                self.state.winner_id = -1
                logs.append("\nDraw — all players lost simultaneously.")

        return logs

    # =========================================================================
    # ZONE TRANSITIONS
    # =========================================================================

    def _move_to_zone(self, inst: CardInstance, target: Zone,
                      player: Optional[Player] = None):
        """Single correct path for all zone changes."""
        owner = player or self.state.get_player(inst.owner_id)
        if not owner:
            return
        self._remove_from_zone_list(inst, owner, inst.zone)
        inst.zone = target
        match target:
            case Zone.HAND:       owner.hand.append(inst.instance_id)
            case Zone.BATTLEFIELD: pass
            case Zone.GRAVEYARD:  owner.graveyard.append(inst.instance_id)
            case Zone.EXILE:      owner.exile.append(inst.instance_id)
            case Zone.COMMAND:
                if inst.instance_id not in owner.command_zone:
                    owner.command_zone.append(inst.instance_id)
            case Zone.LIBRARY:    owner.library.append(inst.instance_id)

    def _remove_from_zone_list(self, inst: CardInstance, owner: Player, zone: Zone):
        iid = inst.instance_id
        match zone:
            case Zone.HAND:       owner.hand.remove(iid) if iid in owner.hand else None
            case Zone.GRAVEYARD:  owner.graveyard.remove(iid) if iid in owner.graveyard else None
            case Zone.EXILE:      owner.exile.remove(iid) if iid in owner.exile else None
            case Zone.COMMAND:    owner.command_zone.remove(iid) if iid in owner.command_zone else None
            case Zone.LIBRARY:    owner.library.remove(iid) if iid in owner.library else None

    # =========================================================================
    # CARD DRAW
    # =========================================================================

    def _draw_card(self, player: Player) -> Optional[CardInstance]:
        """
        Rule 504.1: Draw top card of library.
        Returns None if library empty (caller handles 704.5b loss).
        """
        if not player.library:
            return None
        iid  = player.library.pop(0)
        inst = self.state.all_instances.get(iid)
        if not inst:
            return None
        inst.zone = Zone.HAND
        player.hand.append(iid)
        player.draw_this_turn += 1
        return inst

    def _create_instance(self, defn: CardDefinition, owner_id: int) -> CardInstance:
        inst = CardInstance(
            instance_id   = str(uuid.uuid4()),
            definition    = defn,
            owner_id      = owner_id,
            controller_id = owner_id,
            zone          = Zone.LIBRARY,
        )
        self.state.all_instances[inst.instance_id] = inst
        return inst
