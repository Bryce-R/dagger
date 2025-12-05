import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, d_model=64, n_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d_model * mlp_ratio), d_model),
        )

    def forward(self, x, attn_mask=None):
        # x: (B, T, C)
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = x + attn_out
        x = self.ln1(x)
        mlp_out = self.mlp(x)
        x = x + mlp_out
        x = self.ln2(x)
        return x


class TinyTransformerPolicy(nn.Module):
    def __init__(self, context_len=CONTEXT_LEN, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        self.context_len = context_len
        self.state_emb = nn.Linear(1, d_model)
        self.pos_emb = nn.Embedding(context_len, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model=d_model, n_heads=n_heads) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        x: (B, T, 1) sequence of scalar states
        Returns: (B, 1) predicted action for last time step
        """
        B, T, _ = x.shape
        assert T <= self.context_len

        h = self.state_emb(x)  # (B, T, d_model)

        positions = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        h = h + self.pos_emb(positions)  # broadcast over batch

        # Causal mask: prevent attending to future positions
        # mask shape: (T, T), True means "mask out"
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        for blk in self.blocks:
            h = blk(h, attn_mask=mask)

        h = self.ln_f(h)
        out = self.head(h[:, -1, :])  # (B, 1) -> last token predicts next action
        return out.squeeze(-1)        # (B,)
