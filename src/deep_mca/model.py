from pathlib import Path

import torch
import torch.nn as nn
from transformers import MambaConfig, MambaModel

from deep_mca.data import PAD_ID, VOCAB_SIZE


class MambaRegressor(nn.Module):
    """Mamba backbone with a linear regression head for throughput prediction."""

    def __init__(
        self,
        hidden_size: int = 256,
        num_layers: int = 4,
        state_size: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        config = MambaConfig(
            vocab_size=VOCAB_SIZE,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            state_size=state_size,
            pad_token_id=PAD_ID,
        )
        self.backbone = MambaModel(config)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1),
        )

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) token IDs
            lengths: (batch,) actual sequence lengths (including BOS/EOS)

        Returns:
            (batch,) predicted throughput values
        """
        outputs = self.backbone(input_ids=input_ids)
        hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)

        # Gather last real token's hidden state (lengths - 1 for 0-indexed)
        last_idx = (lengths - 1).unsqueeze(-1).unsqueeze(-1)  # (batch, 1, 1)
        last_idx = last_idx.expand(-1, -1, hidden.size(-1))  # (batch, 1, hidden)
        pooled = hidden.gather(1, last_idx).squeeze(1)  # (batch, hidden)

        return self.head(pooled).squeeze(-1)  # (batch,)

    @classmethod
    def from_pretrained_backbone(
        cls,
        pretrained_path: str | Path,
        hidden_size: int = 256,
        num_layers: int = 4,
        state_size: int = 16,
        dropout: float = 0.0,
    ) -> "MambaRegressor":
        """Load pretrained backbone weights, initialise regression head fresh."""
        model = cls(
            hidden_size=hidden_size,
            num_layers=num_layers,
            state_size=state_size,
            dropout=dropout,
        )
        state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        model.backbone.load_state_dict(state_dict, strict=False)
        return model
