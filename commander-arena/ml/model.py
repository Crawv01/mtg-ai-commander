"""
ml/model.py — Neural network for MTG AI decision making

Architecture: Policy + Value Network (same structure as AlphaZero)

  Input:  Encoded game state (a fixed-size vector describing everything)
  Output: 
    - Policy head: probability distribution over all possible actions
    - Value head:  estimate of how likely each player is to win from here

Why this architecture?
  - The policy head replaces the heuristic AI's scoring function
  - The value head enables MCTS to evaluate positions without playing them out
  - Training signal: did the action we took lead to winning or losing?

The model is intentionally small to start. MTG is complex but our
card set is bounded — we don't need GPT-scale parameters.

Input vector size: ~500 features (calculated in encoder.py)
Hidden layers: 3 × 512 neurons
Output policy: ~200 actions (ActionType × card slots)
Output value: 1 float per player (win probability)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "checkpoints"
MODEL_DIR.mkdir(exist_ok=True)


class MTGPolicyValueNet(nn.Module):
    """
    Combined policy + value network.
    
    Takes a game state vector, outputs:
      - action_logits: unnormalized scores for each possible action
      - value: win probability for the current player (0-1)
    
    We use a shared trunk (the middle layers) so both heads learn
    from the same representation of the game state. This is more
    efficient than two separate networks.
    """

    def __init__(
        self,
        state_size:   int = 512,    # Size of encoded game state vector
        action_size:  int = 256,    # Max number of possible actions
        hidden_size:  int = 512,    # Neurons per hidden layer
        num_layers:   int = 4,      # Depth of the shared trunk
        num_players:  int = 4,      # Commander = 4 players
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.state_size  = state_size
        self.action_size = action_size
        self.num_players = num_players

        # ── Shared trunk ────────────────────────────────────────────
        # Transforms raw game state into a rich internal representation
        # Both policy and value heads read from this

        layers = []
        in_size = state_size
        for i in range(num_layers):
            layers.extend([
                nn.Linear(in_size, hidden_size),
                nn.LayerNorm(hidden_size),   # Stabilizes training
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_size = hidden_size

        self.trunk = nn.Sequential(*layers)

        # ── Policy head ─────────────────────────────────────────────
        # Outputs a score for each possible action
        # Higher score → AI prefers this action
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size),
            # No softmax here — we apply it during inference
            # During training we use cross-entropy loss which expects logits
        )

        # ── Value head ──────────────────────────────────────────────
        # Outputs win probability for each player
        # Used by MCTS to evaluate positions without full rollout
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, num_players),
            nn.Softmax(dim=-1),   # Win probabilities sum to 1
        )

        # Initialize weights — helps training converge faster
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        state_vector: torch.Tensor,          # [batch, state_size]
        action_mask:  torch.Tensor = None,   # [batch, action_size] — 1=legal, 0=illegal
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        state_vector: encoded game state from encoder.py
        action_mask:  which actions are legal right now (illegal ones get -inf)
        
        Returns:
          action_logits: [batch, action_size] — raw scores (apply softmax for probs)
          win_probs:     [batch, num_players] — probability each player wins
        """
        # Shared trunk
        features = self.trunk(state_vector)

        # Policy head
        action_logits = self.policy_head(features)

        # Mask illegal actions — set their logit to -infinity so softmax → 0
        if action_mask is not None:
            action_logits = action_logits.masked_fill(action_mask == 0, float('-inf'))

        # Value head
        win_probs = self.value_head(features)

        return action_logits, win_probs

    def predict(
        self,
        state_vector: torch.Tensor,
        action_mask:  torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inference mode — returns action probabilities (not logits) and win probs.
        Use this during actual gameplay, not during training.
        """
        self.eval()
        with torch.no_grad():
            logits, win_probs = self.forward(state_vector, action_mask)
            action_probs = F.softmax(logits, dim=-1)
        return action_probs, win_probs

    def save(self, name: str = "latest"):
        path = MODEL_DIR / f"{name}.pt"
        torch.save({
            "model_state":  self.state_dict(),
            "state_size":   self.state_size,
            "action_size":  self.action_size,
            "hidden_size":  self.trunk[0].out_features,
            "num_players":  self.num_players,
        }, path)
        print(f"[Model] Saved to {path}")

    @classmethod
    def load(cls, name: str = "latest") -> "MTGPolicyValueNet":
        path = MODEL_DIR / f"{name}.pt"
        checkpoint = torch.load(path, map_location="cpu")
        model = cls(
            state_size  = checkpoint["state_size"],
            action_size = checkpoint["action_size"],
            hidden_size = checkpoint["hidden_size"],
            num_players = checkpoint["num_players"],
        )
        model.load_state_dict(checkpoint["model_state"])
        print(f"[Model] Loaded from {path}")
        return model


class MTGEncoder:
    """
    Converts a GameState into a fixed-size numeric vector the model can process.
    
    This is one of the most important design decisions in the whole project.
    Bad encoding → model can't learn. Good encoding → model learns faster.
    
    Each game state feature becomes one or more numbers:
      - Binary features (tapped/untapped): 1 or 0
      - Count features (cards in hand): raw integer, normalized to 0-1
      - Categorical features (card types): one-hot encoding
    
    The full encoder will live in encoder.py. This stub shows the structure.
    """

    # These must stay constant once we start training
    STATE_SIZE  = 512   # Total size of the state vector
    ACTION_SIZE = 256   # Max actions we ever consider

    @staticmethod
    def encode_state(state) -> torch.Tensor:
        """
        Convert GameState → tensor of shape [STATE_SIZE].
        
        Current stub returns zeros — will be implemented in encoder.py
        once the game state is finalized.
        """
        return torch.zeros(MTGEncoder.STATE_SIZE)

    @staticmethod
    def encode_action_mask(legal_actions: list) -> torch.Tensor:
        """
        Create a mask tensor: 1 for each legal action, 0 for illegal.
        Shape: [ACTION_SIZE]
        """
        mask = torch.zeros(MTGEncoder.ACTION_SIZE)
        for action in legal_actions:
            idx = MTGEncoder.action_to_index(action)
            if idx < MTGEncoder.ACTION_SIZE:
                mask[idx] = 1.0
        return mask

    @staticmethod
    def action_to_index(action) -> int:
        """
        Map an Action to a fixed integer index.
        This mapping must be consistent across all training runs.
        Will be implemented fully in encoder.py.
        """
        from actions import ActionType
        # Stub: just use the action type's value as index
        return action.action_type.value % MTGEncoder.ACTION_SIZE


def get_device() -> torch.device:
    """Pick the best available device: CUDA GPU > Apple Silicon > CPU"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Model] Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Model] Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("[Model] Using CPU (consider a GPU for faster training)")
    return device
