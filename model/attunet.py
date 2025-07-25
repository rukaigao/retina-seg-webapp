import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------ Lightweight Conv Block ------------------
class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout)
        )

    def forward(self, x):
        return self.conv(x)

# ------------------ Attention ------------------
class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        psi = self.relu(self.W_g(g) + self.W_x(x))
        return x * self.psi(psi)

# ------------------ Up Block with Attention ------------------
class Up_Att(nn.Module):
    def __init__(self, ch_in, ch_skip, ch_out, dropout=0.2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.att = Attention_block(ch_in, ch_skip, ch_out // 2)
        self.main_block = ConvBlock(ch_in + ch_skip, ch_out, dropout=dropout)

    def forward(self, x, skip):
        x = self.up(x)
        skip = self.att(x, skip)
        x = torch.cat((skip, x), dim=1)
        return self.main_block(x)

# ------------------ Multi-Head Attention U-Net ------------------
class AttUNetMultiHead(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, dropout=0.2):
        super().__init__()
        ch = base_channels

        # Encoder
        self.enc1 = ConvBlock(in_channels, ch, dropout=dropout)
        self.enc2 = ConvBlock(ch, ch * 2, dropout=dropout)
        self.enc3 = ConvBlock(ch * 2, ch * 4, dropout=dropout)
        self.enc4 = ConvBlock(ch * 4, ch * 8, dropout=dropout)
        self.enc5 = ConvBlock(ch * 8, ch * 16, dropout=dropout)
        self.pool = nn.MaxPool2d(2)

        # Decoders for each head
        self._build_decoders("ava", dropout)
        self._build_decoders("pnv", dropout)
        self._build_decoders("onh", dropout, aux=False)
        self._build_decoders("edges", dropout, aux=False)
        self._build_decoders("vessel", dropout, aux=False)

    def _build_decoders(self, name, dropout, aux=True):
        ch = self.enc1.conv[0].out_channels  # base_channels
        setattr(self, f'up1_{name}', Up_Att(ch*16, ch*8, ch*8, dropout=dropout))
        setattr(self, f'up2_{name}', Up_Att(ch*8, ch*4, ch*4, dropout=dropout))
        setattr(self, f'up3_{name}', Up_Att(ch*4, ch*2, ch*2, dropout=dropout))
        setattr(self, f'up4_{name}', Up_Att(ch*2, ch, ch, dropout=dropout))
        setattr(self, f'out_{name}', nn.Conv2d(ch, 1, kernel_size=1))
        if aux:
            setattr(self, f'aux2_{name}', nn.Conv2d(ch*4, 1, kernel_size=1))
            setattr(self, f'aux3_{name}', nn.Conv2d(ch, 1, kernel_size=1))

    def encode(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        x5 = self.enc5(self.pool(x4))
        return x1, x2, x3, x4, x5

    def forward(self, x, task):
        x1, x2, x3, x4, x5 = self.encode(x)

        def decode(branch):
            u1 = getattr(self, f'up1_{branch}')(x5, x4)
            u2 = getattr(self, f'up2_{branch}')(u1, x3)
            u3 = getattr(self, f'up3_{branch}')(u2, x2)
            u4 = getattr(self, f'up4_{branch}')(u3, x1)
            out = getattr(self, f'out_{branch}')(u4)
            if hasattr(self, f'aux2_{branch}'):
                aux2 = getattr(self, f'aux2_{branch}')(u2)
                aux3 = getattr(self, f'aux3_{branch}')(u4)
                return out, [aux2, aux3]
            else:
                return out

        if task == "A":
            return {
                "ava": decode("ava"),
                "pnv": decode("pnv"),
                "onh": decode("onh"),
                "edges": decode("edges")
            }
        elif task == "B":
            return {
                "vessel": decode("vessel")
            }
        else:
            raise ValueError(f"Unknown task: {task}")