"""LaneResUNet: Custom Residual UNet with Squeeze-and-Excitation Attention
and Multi-Scale Dilated Bottleneck for Single-Class Lane Segmentation.

Designed from scratch for local machine execution (PC/Laptop).
Input:  (B, 3, 48, 48) RGB Image
Output: (B, 1, 48, 48) Binary Lane Mask Logits

Key architectural innovations:
1. Residual Blocks (ResBlock) with identity shortcuts for stable gradient flow.
2. Squeeze-and-Excitation (SEGate) on skip connections to filter background noise (sky/trees/grass).
3. Multi-Scale Dilated Bottleneck (MDFE) to capture wide foreground lanes and distant vanishing points.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEGate(nn.Module):
    """Squeeze-and-Excitation Channel Attention Gate.
    Recalibrates skip-connection features to amplify lane features and suppress background.
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        reduced_ch = max(4, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_ch, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_ch, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        weight = self.fc(x)
        return x * weight


class ResBlock(nn.Module):
    """Residual Convolutional Block: Conv3x3-BN-ReLU -> Conv3x3-BN + Shortcut -> ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        # Shortcut projection if channel dimension changes
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + residual)
        return out


class MultiScaleDilatedBottleneck(nn.Module):
    """Multi-Scale Dilated Context Module at the bottleneck.
    Combines 1x1, standard 3x3, and dilated 3x3 (rate=2) to capture perspective road geometry.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        branch_ch = out_ch // 4
        self.b1 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True)
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True)
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True)
        )
        self.b4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, branch_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(branch_ch * 4, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h, w = x.shape[2:]
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = F.interpolate(self.b4(x), size=(h, w), mode="nearest")
        cat = torch.cat([feat1, feat2, feat3, feat4], dim=1)
        return self.fuse(cat)


class EncoderStage(nn.Module):
    """Downsample (MaxPool) + Residual Block."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.res = ResBlock(in_ch, out_ch)

    def forward(self, x):
        return self.res(self.pool(x))


class DecoderStage(nn.Module):
    """Upsample (ConvTranspose2d) + SE-gated skip concatenation + Residual Block."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.se = SEGate(skip_ch)
        self.res = ResBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        skip = self.se(skip)
        x = torch.cat([skip, x], dim=1)
        return self.res(x)


class LaneResUNet(nn.Module):
    """Complete Custom LaneResUNet for 48x48 Single-Class Lane Segmentation."""
    def __init__(self, in_channels=3, num_classes=1, base_ch=24):
        super().__init__()
        c1 = base_ch          # 24
        c2 = base_ch * 2      # 48
        c3 = base_ch * 4      # 96
        c4 = base_ch * 8      # 192
        c5 = base_ch * 12     # 288 (bottleneck)

        # Encoder Path
        self.enc1 = ResBlock(in_channels, c1)       # 48x48
        self.enc2 = EncoderStage(c1, c2)            # 24x24
        self.enc3 = EncoderStage(c2, c3)            # 12x12
        self.enc4 = EncoderStage(c3, c4)            # 6x6

        # Multi-scale Dilated Bottleneck
        self.pool_bn = nn.MaxPool2d(2)              # 3x3
        self.bottleneck = MultiScaleDilatedBottleneck(c4, c5)

        # Decoder Path with SE-Gated Skip Connections
        self.dec1 = DecoderStage(c5, c4, c4)        # 6x6
        self.dec2 = DecoderStage(c4, c3, c3)        # 12x12
        self.dec3 = DecoderStage(c3, c2, c2)        # 24x24
        self.dec4 = DecoderStage(c2, c1, c1)        # 48x48

        # Output Segmentation Head
        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)        # (B, 24, 48, 48)
        x2 = self.enc2(x1)       # (B, 48, 24, 24)
        x3 = self.enc3(x2)       # (B, 96, 12, 12)
        x4 = self.enc4(x3)       # (B, 192, 6, 6)

        # Bottleneck
        x5 = self.bottleneck(self.pool_bn(x4))  # (B, 288, 3, 3)

        # Decoder
        d1 = self.dec1(x5, x4)   # (B, 192, 6, 6)
        d2 = self.dec2(d1, x3)   # (B, 96, 12, 12)
        d3 = self.dec3(d2, x2)   # (B, 48, 24, 24)
        d4 = self.dec4(d3, x1)   # (B, 24, 48, 48)

        # Head (raw logits)
        return self.head(d4)


# Backward compatibility alias
UNet = LaneResUNet


if __name__ == "__main__":
    model = LaneResUNet()
    dummy = torch.randn(2, 3, 48, 48)
    out = model(dummy)
    n_params = sum(p.numel() for p in model.parameters())
    print("=" * 50)
    print(" MODEL ARCHITECTURE: LaneResUNet")
    print(f" Input Shape  : {tuple(dummy.shape)}")
    print(f" Output Shape : {tuple(out.shape)}")
    print(f" Parameters   : {n_params:,}")
    print("=" * 50)
