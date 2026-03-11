# global_attn_pair_query.py
import torch
import torch.nn as nn
from typing import Optional

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.dropout(self.activation(self.linear_1(x))))


class GlobalSketchAttention(nn.Module):
    """
    final_output: (B, P + N*L, C)
    final_mask:   (B, P + N*L)
    """

    def __init__(self,
                 sketch_embed_dim: int,
                 cross_attention_dim: int,
                 num_attention_heads: int = 8,
                 dropout: float = 0.0,
                 prepend_cls: bool = True):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.prepend_cls = prepend_cls

        # global sketch projector
        if sketch_embed_dim != cross_attention_dim:
            self.proj_in = nn.Linear(sketch_embed_dim, cross_attention_dim)
        else:
            self.proj_in = nn.Identity()

        # Cross-Attention: query = pair_embeds
        self.attn_norm = nn.LayerNorm(cross_attention_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cross_attention_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_dropout = nn.Dropout(dropout)

        # FFN
        self.ffn_norm = nn.LayerNorm(cross_attention_dim)
        self.ffn = FeedForward(cross_attention_dim, dropout=dropout)

    def _ensure_flatten_global(self, g: torch.Tensor):
        # allow (B, P, D) or (B, H, W, D)
        if g.dim() == 3:
            return g
        if g.dim() == 4:
            B, H, W, D = g.shape
            return g.view(B, H * W, D)
        raise ValueError(f"Unsupported global_sketch_embeds dim: {g.dim()}")

    def forward(self,
                global_sketch_embeds: torch.Tensor,
                pair_embeds: torch.Tensor,
                pair_mask: Optional[torch.Tensor] = None):
        """
        Args:
            global_sketch_embeds: (B, P, D)
            pair_embeds: (B, N*L, C)
            pair_mask: (B, N*L)  True=keep, False=mask

        Returns:
            final_output: (B, P+N*L, C)
            final_mask: (B, P+N*L)
        """

        # 1. flatten & project sketch → (B, P, C)
        g = self._ensure_flatten_global(global_sketch_embeds)
        key_value = self.proj_in(g)  # (B, P, C)

        # shape checks
        B, P, Cg = key_value.shape
        B2, NL, Cp = pair_embeds.shape
        assert B == B2, f"Batch mismatch"
        assert Cg == Cp == self.cross_attention_dim, "Channel mismatch"

        if pair_mask is not None:
            assert pair_mask.shape == (B, NL)

        # 2. Cross-Attention（pair as query）
        # Query: pair_embeds
        # Key/Value: sketch_embeds
        residual = pair_embeds
        norm_query = self.attn_norm(pair_embeds)

        # Key padding mask for query：mask for pair
        key_padding_mask = ~pair_mask if pair_mask is not None else None
        attn_output, _ = self.cross_attn(
            query=norm_query,
            key=key_value,
            value=key_value,
            key_padding_mask=None 
        )

        refined_pair_embeds = residual + self.attn_dropout(attn_output)

        # 3. FFN
        residual = refined_pair_embeds
        norm_query = self.ffn_norm(refined_pair_embeds)
        refined_pair_embeds = residual + self.ffn(norm_query)

        # 4. concat
        sketch_mask = torch.ones((B, P), dtype=torch.bool, device=pair_embeds.device)

        if self.prepend_cls:
            final_output = torch.cat([key_value, refined_pair_embeds], dim=1)
            final_mask = torch.cat([sketch_mask, pair_mask], dim=1) if pair_mask is not None else None
        else:
            final_output = torch.cat([refined_pair_embeds, key_value], dim=1)
            final_mask = torch.cat([pair_mask, sketch_mask], dim=1) if pair_mask is not None else None

        return final_output, final_mask
