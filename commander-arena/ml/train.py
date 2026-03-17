"""
ml/train.py — Self-play training loop

How ML training works for MTG:

1. GENERATE DATA: Run N games where both sides use the current model
   (or heuristic AI early on). Record every (state, action, outcome).

2. TRAIN: Feed those records into the model. 
   - Policy loss: "Given this state, you chose action X. The game was won.
                   Learn to prefer X in similar states."
   - Value loss:  "Given this state, you predicted 60% win chance.
                   Actual result was a win (100%). Adjust your estimate."

3. EVALUATE: Play the new model vs the old model 100 times.
   If new model wins >55%, it's an improvement. Keep it.

4. REPEAT: Each iteration the model gets a little better.
   After enough iterations, it surpasses the heuristic AI.

This file orchestrates the loop. The actual game simulation is in simulator.py.
"""

from __future__ import annotations
import json
import random
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model import MTGPolicyValueNet, MTGEncoder, get_device


# ─────────────────────────────────────────────────────────────────────────────
# Training record — one decision point from a game
# ─────────────────────────────────────────────────────────────────────────────

class TrainingRecord:
    """
    One data point for training.
    Collected during self-play, stored to disk, loaded during training.
    """
    __slots__ = ["state_vec", "action_idx", "action_mask", "winner_id", "player_id"]

    def __init__(
        self,
        state_vec:   list[float],  # Encoded game state
        action_idx:  int,          # Which action was taken
        action_mask: list[float],  # Which actions were legal
        winner_id:   int,          # Who won the game (-1 if draw)
        player_id:   int,          # Who was making this decision
    ):
        self.state_vec   = state_vec
        self.action_idx  = action_idx
        self.action_mask = action_mask
        self.winner_id   = winner_id
        self.player_id   = player_id

    @property
    def outcome(self) -> float:
        """1.0 if this player won, 0.0 if they lost, 0.5 if draw."""
        if self.winner_id == -1:   return 0.5
        if self.winner_id == self.player_id: return 1.0
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Training data storage
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "training_data"
DATA_DIR.mkdir(exist_ok=True)


def save_game_records(records: list[TrainingRecord], game_id: str):
    """Save all records from one game to disk."""
    path = DATA_DIR / f"game_{game_id}.json"
    data = [{
        "state":   r.state_vec,
        "action":  r.action_idx,
        "mask":    r.action_mask,
        "winner":  r.winner_id,
        "player":  r.player_id,
    } for r in records]
    with open(path, "w") as f:
        json.dump(data, f)


def load_all_records(max_games: int = 10000) -> list[TrainingRecord]:
    """Load training records from disk. Load most recent games first."""
    records = []
    files = sorted(DATA_DIR.glob("game_*.json"), reverse=True)[:max_games]

    for path in files:
        try:
            with open(path) as f:
                data = json.load(f)
            for d in data:
                records.append(TrainingRecord(
                    state_vec   = d["state"],
                    action_idx  = d["action"],
                    action_mask = d["mask"],
                    winner_id   = d["winner"],
                    player_id   = d["player"],
                ))
        except Exception as e:
            print(f"Warning: skipping {path}: {e}")

    print(f"[Train] Loaded {len(records)} records from {len(files)} games")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Core training function
# ─────────────────────────────────────────────────────────────────────────────

def train_on_records(
    model:       MTGPolicyValueNet,
    records:     list[TrainingRecord],
    epochs:      int   = 5,
    batch_size:  int   = 256,
    lr:          float = 1e-3,
    device:      Optional[torch.device] = None,
) -> dict:
    """
    Train the model on collected game records.
    Returns a dict of training metrics.
    """
    if not records:
        print("[Train] No records to train on")
        return {}

    device = device or get_device()
    model = model.to(device)
    model.train()

    # ── Build tensors from records ─────────────────────────────────
    states  = torch.tensor([r.state_vec   for r in records], dtype=torch.float32)
    actions = torch.tensor([r.action_idx  for r in records], dtype=torch.long)
    masks   = torch.tensor([r.action_mask for r in records], dtype=torch.float32)
    outcomes = torch.tensor([r.outcome    for r in records], dtype=torch.float32)
    players = torch.tensor([r.player_id  for r in records], dtype=torch.long)

    dataset = TensorDataset(states, actions, masks, outcomes, players)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    metrics = {"policy_loss": [], "value_loss": [], "total_loss": []}

    for epoch in range(epochs):
        epoch_policy_loss = 0.0
        epoch_value_loss  = 0.0
        batches = 0

        for batch_states, batch_actions, batch_masks, batch_outcomes, batch_players in loader:
            # Move to device
            batch_states   = batch_states.to(device)
            batch_actions  = batch_actions.to(device)
            batch_masks    = batch_masks.to(device)
            batch_outcomes = batch_outcomes.to(device)
            batch_players  = batch_players.to(device)

            # Forward pass
            action_logits, win_probs = model(batch_states, batch_masks)

            # ── Policy loss ──────────────────────────────────────────
            # Cross-entropy: model should prefer the action that led to a win
            # We weight by outcome so wins reinforce more than losses
            policy_loss = F.cross_entropy(action_logits, batch_actions, reduction='none')
            # Weight by outcome — good decisions in winning games matter more
            policy_loss = (policy_loss * batch_outcomes).mean()

            # ── Value loss ───────────────────────────────────────────
            # MSE: model's win probability estimate should match actual outcome
            # Extract this player's predicted win probability
            player_win_probs = win_probs.gather(
                1, batch_players.unsqueeze(1)
            ).squeeze(1)
            value_loss = F.mse_loss(player_win_probs, batch_outcomes)

            # ── Combined loss ────────────────────────────────────────
            # Value loss scaled down — policy learning is more important early
            total_loss = policy_loss + 0.5 * value_loss

            # Backprop
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Prevent exploding gradients
            optimizer.step()

            epoch_policy_loss += policy_loss.item()
            epoch_value_loss  += value_loss.item()
            batches += 1

        scheduler.step()

        avg_policy = epoch_policy_loss / batches
        avg_value  = epoch_value_loss  / batches
        avg_total  = avg_policy + 0.5 * avg_value

        metrics["policy_loss"].append(avg_policy)
        metrics["value_loss"].append(avg_value)
        metrics["total_loss"].append(avg_total)

        print(f"  Epoch {epoch+1}/{epochs} — "
              f"policy: {avg_policy:.4f}, value: {avg_value:.4f}, total: {avg_total:.4f}")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Full training iteration (self-play → train → evaluate)
# ─────────────────────────────────────────────────────────────────────────────

def run_training_iteration(
    iteration:    int,
    model:        MTGPolicyValueNet,
    games_per_iter: int = 50,
    eval_games:   int   = 20,
) -> MTGPolicyValueNet:
    """
    One full training cycle:
    1. Generate games_per_iter games using current model
    2. Train on all accumulated data
    3. Evaluate: does new model beat old model?
    4. Keep whichever is better
    
    Returns the best model after this iteration.
    """
    from simulator import GameSimulator

    print(f"\n{'='*50}")
    print(f"Training Iteration {iteration}")
    print(f"{'='*50}")

    # ── Step 1: Generate self-play data ───────────────────────────
    print(f"\n[Iter {iteration}] Generating {games_per_iter} self-play games...")
    sim = GameSimulator(model=model)
    new_records = 0

    for game_num in range(games_per_iter):
        game_id, records = sim.run_game()
        save_game_records(records, game_id)
        new_records += len(records)

        if (game_num + 1) % 10 == 0:
            print(f"  Generated {game_num + 1}/{games_per_iter} games ({new_records} records)")

    # ── Step 2: Train on all available data ───────────────────────
    print(f"\n[Iter {iteration}] Training on accumulated data...")
    all_records = load_all_records(max_games=5000)  # Last 5000 games

    old_model = MTGPolicyValueNet(
        state_size  = model.state_size,
        action_size = model.action_size,
    )
    old_model.load_state_dict(model.state_dict())  # Copy current model

    device = get_device()
    metrics = train_on_records(model, all_records, epochs=5, device=device)

    # ── Step 3: Evaluate new model vs old model ───────────────────
    print(f"\n[Iter {iteration}] Evaluating new model vs previous...")
    win_rate = evaluate_models(new_model=model, old_model=old_model, games=eval_games)

    print(f"  New model win rate: {win_rate:.1%}")

    if win_rate >= 0.55:
        print(f"  ✅ New model is better — keeping it")
        model.save(f"iter_{iteration:04d}")
        model.save("latest")
        return model
    else:
        print(f"  ❌ New model is not clearly better — reverting")
        old_model.save("latest")
        return old_model


def evaluate_models(
    new_model: MTGPolicyValueNet,
    old_model: MTGPolicyValueNet,
    games:     int = 20,
) -> float:
    """
    Play new_model vs old_model for N games.
    Returns new_model's win rate.
    """
    from simulator import GameSimulator

    new_wins = 0
    for i in range(games):
        # Alternate who plays first to reduce first-mover advantage
        if i % 2 == 0:
            sim = GameSimulator(model_p0=new_model, model_p1=old_model)
            winner_idx = sim.run_game()[0]
            if winner_idx == 0: new_wins += 1
        else:
            sim = GameSimulator(model_p0=old_model, model_p1=new_model)
            winner_idx = sim.run_game()[0]
            if winner_idx == 1: new_wins += 1

    return new_wins / games


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(iterations: int = 100, resume: bool = True):
    """
    Main training loop. Run this to train the AI.
    
    iterations: how many train/eval cycles to run
    resume:     if True, load the latest saved model and continue training
    """
    device = get_device()

    # Load or create model
    checkpoint_path = Path(__file__).parent / "checkpoints" / "latest.pt"
    if resume and checkpoint_path.exists():
        model = MTGPolicyValueNet.load("latest")
        print(f"[Train] Resuming from existing checkpoint")
    else:
        model = MTGPolicyValueNet(
            state_size  = MTGEncoder.STATE_SIZE,
            action_size = MTGEncoder.ACTION_SIZE,
        )
        print(f"[Train] Starting fresh model")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    model = model.to(device)

    # Training loop
    start_iter = 1
    for i in range(start_iter, start_iter + iterations):
        model = run_training_iteration(
            iteration     = i,
            model         = model,
            games_per_iter= 50,
            eval_games    = 20,
        )
        print(f"\n[Train] Iteration {i} complete.")

    print(f"\n✅ Training complete after {iterations} iterations.")
    print(f"   Final model saved to checkpoints/latest.pt")


if __name__ == "__main__":
    main(iterations=100)
