import torch.nn as nn

class MMInputProjector(nn.Module):
    def __init__(self, input_dim=1152, output_dim=4096): # SIGLIP so400m dim is 1152
        super().__init__()
        # 使用 MLP 效果通常比单层 Linear 好
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)