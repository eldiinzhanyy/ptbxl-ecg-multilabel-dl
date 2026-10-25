"""Model definitions: 1D CNN, CNN+LSTM, Transformer encoder (see Section III of the paper)."""
import torch
import torch.nn as nn

N_LEADS, N_CLASSES = 12, 5


def conv_block(c_in, c_out, k, p_drop=0.20):
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, k, stride=1, padding=k // 2),
        nn.BatchNorm1d(c_out),
        nn.ReLU(inplace=True),
        nn.Dropout1d(p_drop),
        nn.MaxPool1d(2),
    )


class CNN1D(nn.Module):
    """Four conv blocks (32-64-128-256, k = 7,5,5,3), global average pooling, linear head."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(N_LEADS, 32, 7),
            conv_block(32, 64, 5),
            conv_block(64, 128, 5),
            conv_block(128, 256, 3),
        )
        self.head = nn.Linear(256, N_CLASSES)

    def forward(self, x):                      # x: (B, 12, T)
        return self.head(self.features(x).mean(dim=-1))


class CNNLSTM(nn.Module):
    """First three conv blocks as front-end, 2-layer BiLSTM (128 per direction), pooled + final states."""

    def __init__(self, hidden=128, p_rnn=0.25):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(N_LEADS, 32, 7),
            conv_block(32, 64, 5),
            conv_block(64, 128, 5),
        )
        self.lstm = nn.LSTM(128, hidden, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=p_rnn)
        self.head = nn.Linear(4 * hidden, N_CLASSES)   # mean-pooled (2H) + final states (2H)

    def forward(self, x):
        f = self.features(x).transpose(1, 2)           # (B, T', 128), T' = 125 at 100 Hz
        out, (h, _) = self.lstm(f)                     # out: (B, T', 2H)
        pooled = out.mean(dim=1)
        final = torch.cat([h[-2], h[-1]], dim=1)       # last layer, forward and backward
        return self.head(torch.cat([pooled, final], dim=1))


class TransformerECG(nn.Module):
    """Conv patch embedding (k = s = 5 -> 200 tokens at 100 Hz), 4 pre-LN encoder layers, 8 heads."""

    def __init__(self, d_model=128, n_layers=4, n_heads=8, d_ff=512, p=0.20, n_tokens=200):
        super().__init__()
        self.embed = nn.Conv1d(N_LEADS, d_model, kernel_size=5, stride=5)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout=p, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, norm=nn.LayerNorm(d_model),
                                             enable_nested_tensor=False)
        self.head = nn.Linear(d_model, N_CLASSES)

    def forward(self, x):
        t = self.embed(x).transpose(1, 2) + self.pos   # (B, 200, d_model)
        return self.head(self.encoder(t).mean(dim=1))


def build(name: str) -> nn.Module:
    return {"cnn": CNN1D, "cnn_lstm": CNNLSTM, "transformer": TransformerECG}[name]()


if __name__ == "__main__":
    for n in ("cnn", "cnn_lstm", "transformer"):
        m = build(n)
        y = m(torch.randn(2, N_LEADS, 1000))
        print(n, tuple(y.shape), f"{sum(p.numel() for p in m.parameters()) / 1e6:.3f} M parameters")
