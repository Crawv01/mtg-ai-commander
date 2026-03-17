"""
simulator.py — Headless Game Simulator

Runs complete Commander games between AI players to generate training data.

Key design decisions:
  - Attackers declared as a batch (rules-accurate per 508.1)
  - Turn limit fallback: highest life total wins (better training signal than draw)
  - Hard phase-advance safety: if a phase sees too many passes, force advance
    (prevents infinite loops when effects are stubbed and no damage is dealt)
  - MTGEncoder import is optional — works without torch installed
"""

from __future__ import annotations
import random
import uuid
import time
from dataclasses import dataclass, field
from typing import Optional

from game_state import GameState, Player, CardDefinition, Phase, Zone
from actions import Action, ActionType, ActionGenerator
from rules import RulesEngine
from heuristic_ai import HeuristicAI

# Optional ML encoder — works without torch
try:
    from ml.model import MTGEncoder
    _HAS_ML = True
except ImportError:
    _HAS_ML = False
    class MTGEncoder:                           # type: ignore
        STATE_SIZE  = 512
        ACTION_SIZE = 256
        @staticmethod
        def encode_state(state):      return [0.0] * 512
        @staticmethod
        def encode_action_mask(acts): return [0.0] * 256
        @staticmethod
        def action_to_index(action):  return action.action_type.value % 256


# ─────────────────────────────────────────────────────────────────────────────
# Training record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingRecord:
    game_id:    str
    turn:       int
    phase:      str
    player_id:  int
    state_vec:  list[float]
    action_idx: int
    action_mask:list[float]
    winner_id:  int = -1

    @property
    def outcome(self) -> float:
        if self.winner_id == -1:                  return 0.5
        if self.winner_id == self.player_id:      return 1.0
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Game result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GameResult:
    game_id:      str
    winner_id:    int
    winner_name:  str
    turns:        int
    duration_sec: float
    records:      list[TrainingRecord]
    log:          list[str]
    timeout:      bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Simulator
# ─────────────────────────────────────────────────────────────────────────────

class GameSimulator:
    """
    Runs a complete Commander game between AI players.

    Usage:
        sim = GameSimulator(
            player_decks=[
                (player1, decklist1, ["CommanderName1"]),
                (player2, decklist2, ["CommanderName2"]),
            ]
        )
        result = sim.run()
    """

    DEFAULT_TURN_LIMIT   = 50
    # Max times priority can pass in ONE phase before we force-advance.
    # This prevents infinite loops when effects are stubbed and nothing happens.
    MAX_PASSES_PER_PHASE = 12   # 4 players × 3 passes each is plenty

    def __init__(
        self,
        player_decks: list[tuple[Player, list[CardDefinition], list[str]]],
        ai_players:   Optional[dict[int, HeuristicAI]] = None,
        ml_model      = None,
        turn_limit:   int  = DEFAULT_TURN_LIMIT,
        seed:         Optional[int] = None,
        verbose:      bool = False,
    ):
        if seed is not None:
            random.seed(seed)

        self.player_decks = player_decks
        self.turn_limit   = turn_limit
        self.verbose      = verbose
        self.ml_model     = ml_model
        self.game_id      = str(uuid.uuid4())[:8]

        self.state  = GameState(game_id=self.game_id)
        self.engine = RulesEngine(self.state)

        # Build AI for each player
        self.ai: dict[int, HeuristicAI] = {}
        if ai_players:
            self.ai = ai_players
        else:
            for player, _, _ in player_decks:
                self.ai[player.player_id] = HeuristicAI(
                    player_id  = player.player_id,
                    randomness = 0.1,
                )

        self._records: list[TrainingRecord] = []

    # =========================================================================
    # MAIN RUN
    # =========================================================================

    def run(self) -> GameResult:
        start = time.time()
        logs  = []

        setup_logs = self.engine.setup_game(self.player_decks)
        logs.extend(setup_logs)
        self._log(setup_logs)

        timeout = False

        while self.state.winner_id is None:

            if self.state.turn_number > self.turn_limit:
                timeout   = True
                wid, wname = self._resolve_timeout()
                self.state.winner_id = wid
                msg = f"[TIMEOUT] Turn {self.turn_limit} limit. Winner by life: {wname}"
                logs.append(msg)
                self._log([msg])
                break

            turn_logs = self.engine.begin_turn()
            logs.extend(turn_logs)
            self._log(turn_logs)

            if self.state.winner_id is not None:
                break

            loop_logs = self._run_priority_loop()
            logs.extend(loop_logs)

        # Fill winner into all records
        winner_id = self.state.winner_id if self.state.winner_id is not None else -1
        for rec in self._records:
            rec.winner_id = winner_id

        winner      = self.state.get_player(winner_id)
        winner_name = winner.name if winner else "Draw"
        duration    = time.time() - start

        summary = (f"Game {self.game_id}: {winner_name} wins. "
                   f"Turns={self.state.turn_number} "
                   f"Time={duration:.2f}s Records={len(self._records)}")
        logs.append(summary)
        self._log([summary])

        return GameResult(
            game_id      = self.game_id,
            winner_id    = winner_id,
            winner_name  = winner_name,
            turns        = self.state.turn_number,
            duration_sec = duration,
            records      = self._records,
            log          = logs,
            timeout      = timeout,
        )

    # =========================================================================
    # PRIORITY LOOP
    # =========================================================================

    def _run_priority_loop(self) -> list[str]:
        """
        Core game loop. Runs from upkeep through end of turn.

        Safety design:
          - phase_pass_count tracks consecutive passes in the current phase
          - If it exceeds MAX_PASSES_PER_PHASE, we force a PASS_PRIORITY from
            the active player to advance the phase automatically
          - This prevents infinite loops when effects are stubbed
        """
        logs = []
        current_active   = self.state.active_player
        phase_pass_count = 0
        last_phase       = self.state.phase

        while (self.state.winner_id is None and
               self.state.active_player == current_active):

            # Reset pass counter when phase changes
            if self.state.phase != last_phase:
                phase_pass_count = 0
                last_phase = self.state.phase

            pid    = self.state.priority_player
            player = self.state.get_player(pid)

            if not player or player.has_lost:
                action = Action(ActionType.PASS_PRIORITY, actor_id=pid,
                               description="auto-pass (lost)")
                logs.extend(self.engine.execute_action(action))
                phase_pass_count += 1
                continue

            # ── Special combat phases ──────────────────────────────────────
            if self.state.phase == Phase.DECLARE_ATTACKERS:
                phase_logs = self._handle_declare_attackers()
                logs.extend(phase_logs)
                phase_pass_count = 0
                continue

            if self.state.phase == Phase.DECLARE_BLOCKERS:
                phase_logs = self._handle_declare_blockers()
                logs.extend(phase_logs)
                phase_pass_count = 0
                continue

            # ── Force advance if stuck ─────────────────────────────────────
            if phase_pass_count >= self.MAX_PASSES_PER_PHASE:
                force_msg = (f"[AUTO] Phase {self.state.phase.name} stuck after "
                             f"{phase_pass_count} passes — forcing advance.")
                logs.append(force_msg)
                self._log([force_msg])
                # Force priority to active player then pass to trigger phase advance
                self.state.priority_player = current_active
                action = Action(ActionType.PASS_PRIORITY,
                               actor_id=current_active,
                               description="force advance")
                action_logs = self.engine.execute_action(action)
                logs.extend(action_logs)
                self._log(action_logs)
                phase_pass_count = 0
                continue

            # ── Normal priority decision ───────────────────────────────────
            action = self._get_action(player)

            if action.action_type == ActionType.PASS_PRIORITY:
                phase_pass_count += 1
            else:
                phase_pass_count = 0

            self._record_decision(player, action)

            action_logs = self.engine.execute_action(action)
            logs.extend(action_logs)
            self._log(action_logs)

        return logs

    # =========================================================================
    # COMBAT BATCHING
    # =========================================================================

    def _handle_declare_attackers(self) -> list[str]:
        """
        Rule 508.1: All attackers declared at once as a single turn-based action.
        AI picks all creatures to attack with, we batch-execute, then pass priority.
        """
        logs = []
        active = self.state.get_active_player()
        ai     = self.ai.get(active.player_id)

        if not ai:
            logs.append(f"{active.name}: no AI — skipping attackers.")
            logs.extend(self.engine.execute_action(
                Action(ActionType.PASS_PRIORITY, actor_id=active.player_id)))
            return logs

        generator        = ActionGenerator(self.state)
        all_legal        = generator.get_legal_actions()
        attacker_options = [a for a in all_legal
                            if a.action_type == ActionType.DECLARE_ATTACKER]

        chosen = self._choose_attackers(ai, attacker_options)

        for atk_action in chosen:
            self._record_decision(active, atk_action)
            action_logs = self.engine.execute_action(atk_action)
            logs.extend(action_logs)
            self._log(action_logs)

        if not chosen:
            logs.append(f"{active.name} declares no attackers.")

        # Done declaring — pass priority (opponents may respond)
        pass_logs = self.engine.execute_action(
            Action(ActionType.PASS_PRIORITY, actor_id=active.player_id,
                   description="done declaring attackers"))
        logs.extend(pass_logs)
        return logs

    def _choose_attackers(
        self,
        ai: HeuristicAI,
        options: list[Action],
    ) -> list[Action]:
        """Select which attacks to declare — score each, take all positives."""
        chosen:      list[Action] = []
        committed:   set[str]     = set()   # Card IIDs already attacking

        for action in options:
            if action.card_iid in committed:
                continue
            score = ai._score_action(self.state, action)
            if score > 0:
                chosen.append(action)
                committed.add(action.card_iid)

        return chosen

    def _handle_declare_blockers(self) -> list[str]:
        """
        Rule 509.1: Each attacked player declares blockers in turn order.
        Batch per player, then pass priority.
        """
        logs = []
        attacked_pids = self.engine._get_attacked_player_ids()

        if not attacked_pids:
            logs.extend(self.engine.execute_action(
                Action(ActionType.PASS_PRIORITY,
                       actor_id=self.state.active_player)))
            return logs

        ids = [p.player_id for p in self.state.players if not p.has_lost]
        cur = ids.index(self.state.active_player)

        for offset in range(1, len(ids) + 1):
            pid = ids[(cur + offset) % len(ids)]
            if pid not in attacked_pids:
                continue

            defender = self.state.get_player(pid)
            ai       = self.ai.get(pid)
            if not ai or not defender:
                continue

            generator   = ActionGenerator(self.state)
            all_legal   = generator.get_legal_actions()
            blk_options = [a for a in all_legal
                           if a.action_type == ActionType.DECLARE_BLOCKER
                           and a.actor_id == pid]

            chosen = self._choose_blockers(ai, blk_options)

            for blk_action in chosen:
                self._record_decision(defender, blk_action)
                action_logs = self.engine.execute_action(blk_action)
                logs.extend(action_logs)
                self._log(action_logs)

            if not chosen:
                logs.append(f"{defender.name} declares no blockers.")

        pass_logs = self.engine.execute_action(
            Action(ActionType.PASS_PRIORITY,
                   actor_id=self.state.priority_player,
                   description="done declaring blockers"))
        logs.extend(pass_logs)
        return logs

    def _choose_blockers(
        self,
        ai: HeuristicAI,
        options: list[Action],
    ) -> list[Action]:
        """Select which blocks to declare — score each, take all positives."""
        chosen:    list[Action] = []
        blocking:  set[str]     = set()

        for action in options:
            if action.card_iid in blocking:
                continue
            score = ai._score_action(self.state, action)
            if score > 0:
                chosen.append(action)
                blocking.add(action.card_iid)

        return chosen

    # =========================================================================
    # AI DECISIONS
    # =========================================================================

    def _get_action(self, player: Player) -> Action:
        if self.ml_model is not None and _HAS_ML:
            return self._ml_action(player)
        ai = self.ai.get(player.player_id)
        if ai:
            return ai.choose_action(self.state)
        return Action(ActionType.PASS_PRIORITY, actor_id=player.player_id)

    def _ml_action(self, player: Player) -> Action:
        import torch
        generator     = ActionGenerator(self.state)
        legal_actions = generator.get_legal_actions()
        state_vec     = MTGEncoder.encode_state(self.state)
        mask          = MTGEncoder.encode_action_mask(legal_actions)
        probs, _      = self.ml_model.predict(
            state_vec.unsqueeze(0), mask.unsqueeze(0))
        probs = probs.squeeze(0)
        best_idx = probs.argmax().item()
        for action in legal_actions:
            if MTGEncoder.action_to_index(action) == best_idx:
                return action
        return legal_actions[0]

    # =========================================================================
    # TRAINING RECORDS
    # =========================================================================

    def _record_decision(self, player: Player, action: Action):
        generator     = ActionGenerator(self.state)
        legal_actions = generator.get_legal_actions()

        raw_state = MTGEncoder.encode_state(self.state)
        raw_mask  = MTGEncoder.encode_action_mask(legal_actions)
        state_vec   = raw_state.tolist()  if hasattr(raw_state,  'tolist') else list(raw_state)
        action_mask = raw_mask.tolist()   if hasattr(raw_mask,   'tolist') else list(raw_mask)

        self._records.append(TrainingRecord(
            game_id    = self.game_id,
            turn       = self.state.turn_number,
            phase      = self.state.phase.name,
            player_id  = player.player_id,
            state_vec  = state_vec,
            action_idx = MTGEncoder.action_to_index(action),
            action_mask= action_mask,
            winner_id  = -1,
        ))

    # =========================================================================
    # TIMEOUT
    # =========================================================================

    def _resolve_timeout(self) -> tuple[int, str]:
        """Highest life total wins. Better training signal than a draw."""
        living = self.state.living_players()
        if not living:
            return -1, "Draw"

        def score(p: Player) -> tuple[int, int]:
            cmd_dealt = sum(
                opp.commander_damage.get(p.player_id, 0)
                for opp in self.state.players
                if opp.player_id != p.player_id
            )
            return (p.life, cmd_dealt)

        winner = max(living, key=score)
        return winner.player_id, winner.name

    # =========================================================================
    # LOGGING
    # =========================================================================

    def _log(self, lines: list[str]):
        if self.verbose:
            for line in lines:
                if line.strip():
                    print(line)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience functions
# ─────────────────────────────────────────────────────────────────────────────

def run_game(
    player_decks: list[tuple[Player, list[CardDefinition], list[str]]],
    verbose: bool = False,
    seed: Optional[int] = None,
) -> GameResult:
    return GameSimulator(player_decks=player_decks, verbose=verbose, seed=seed).run()


def summarize_results(results: list[GameResult]):
    from collections import Counter
    total     = len(results)
    if not total:
        print("No results.")
        return
    draws     = sum(1 for r in results if r.winner_id == -1)
    timeouts  = sum(1 for r in results if r.timeout)
    avg_turns = sum(r.turns for r in results) / total
    total_rec = sum(len(r.records) for r in results)
    wins      = Counter(r.winner_name for r in results if r.winner_id != -1)

    print(f"\n{'='*50}")
    print(f"Results ({total} games)")
    print(f"  Draws/Timeouts: {draws}/{timeouts}")
    print(f"  Avg turns:      {avg_turns:.1f}")
    print(f"  Total records:  {total_rec:,}")
    print(f"  Win distribution:")
    for name, count in wins.most_common():
        print(f"    {name}: {count} ({count/total*100:.0f}%)")
    print('='*50)
