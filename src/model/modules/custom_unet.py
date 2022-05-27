import torch
import torch.nn as nn
from typing import List


class Unet(nn.Module):
    def __init__(
        self, channels: List[int] = [1, 64, 128, 256, 512, 1024], num_classes: int = 1, encoder_name: str = "default"
    ):
        super().__init__()

        self.encoder = Encoder(channels)
        self.decoder = Decoder(channels)

        self.head = nn.Conv2d(channels[1], out_channels=num_classes, kernel_size=1)

    def forward(self, x: torch.tensor) -> torch.tensor:
        encoder_outputs = self.encoder(x)

        # encoder_outputs = [i.cuda(1) for i in encoder_outputs]

        x = encoder_outputs[-1]

        x = self.decoder(x, encoder_outputs[::-1][1:])
        x = self.head(x)

        return x


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)

    def forward(self, x):

        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))

        return x2


class Encoder(nn.Module):
    def __init__(self, channels: List[int] = (1, 64, 128, 256, 512, 1024)):
        super().__init__()

        self.blocks = nn.ModuleList(
            [Block(in_channels=channels[i], out_channels=channels[i + 1]) for i in range(len(channels) - 1)]
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.tensor) -> List[torch.tensor]:

        intermediate_outputs = []

        for block in self.blocks:
            x_out = block(x)
            intermediate_outputs.append(x_out)
            x = self.maxpool(x_out)

        return intermediate_outputs


class Decoder(nn.Module):
    def __init__(self, channels: List[int] = (1, 64, 128, 256, 512, 1024)):
        super().__init__()

        self.upconvs = nn.ModuleList(
            [nn.ConvTranspose2d(channels[-i - 1], channels[-i - 2], 2, 2) for i in range(len(channels) - 1)]
        )

        self.blocks = nn.ModuleList([Block(channels[-i - 1], channels[-i - 2]) for i in range(len(channels) - 1)])

    def forward(self, x: torch.tensor, encoder_outpus: List[torch.tensor]):

        for encoder_intermediate_output, upconv, block in zip(encoder_outpus, self.upconvs, self.blocks):

            # print("Input: ", x.size())
            # print("Encoder counterpart: ", encoder_intermediate_output.size())

            x = upconv(x)
            # print("Input upconv: ", x.size())
            x = torch.cat([encoder_intermediate_output, x], dim=1)
            x = block(x)

        return x
