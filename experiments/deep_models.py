import torch
import torch.nn.functional as F
from torch import nn

cfg = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "VGG19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG(nn.Module):
    def __init__(self, vgg_name="VGG19", outputs=10, batch_norm=True, base=0):
        super(VGG, self).__init__()
        self.batch_norm = batch_norm
        base = base if base != 0 else 64
        self.base = base
        self.features = self._make_layers(cfg[vgg_name])
        self.fc = nn.Linear(8 * base, outputs)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                out_channels = x * self.base // 64
                layers += [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
                ]
                if self.batch_norm:
                    layers += [nn.BatchNorm2d(x)]
                layers += [nn.ReLU()]
                in_channels = out_channels
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bn=False):
        super(BasicBlock, self).__init__()
        self.bn = bn
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=not bn
        )
        if bn:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=not bn
        )
        if bn:
            self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if self.bn:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )
            else:
                self.shortcut = nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )

    def forward(self, x):
        if self.bn:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
        else:
            out = F.relu(self.conv1(x))
            out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, bn=False):
        super(Bottleneck, self).__init__()
        self.bn = bn
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        if self.bn:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        if self.bn:
            self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        if self.bn:
            self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
            )
            if self.bn:
                self.shortcut.append(nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        if self.bn:
            out = self.bn1(self.conv1(x))
        else:
            out = self.conv1(x)
        out = F.relu(out)
        if self.bn:
            out = self.bn2(self.conv2(x))
        else:
            out = self.conv2(x)
        out = F.relu(out)
        if self.bn:
            out = self.bn3(self.conv3(x))
        else:
            out = self.conv3(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, bn=False):
        super(ResNet, self).__init__()
        self.bn = bn
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=not bn)
        if bn:
            self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.bn))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.bn:
            out = self.bn1(self.conv1(x))
        else:
            out = self.conv1(x)
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18(outputs=10, batch_norm=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=outputs, bn=batch_norm)


def ResNet34(outputs=10, batch_norm=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=outputs, bn=batch_norm)


def ResNet50(outputs=10, batch_norm=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=outputs, bn=batch_norm)


def ResNet101(outputs=10, batch_norm=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=outputs, bn=batch_norm)


def ResNet152(outputs=10, batch_norm=True):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=outputs, bn=batch_norm)


class VisionTransformerEmbedding(nn.Module):
    def __init__(self, image_size=32, patch_size=4, hidden_size=128):
        super().__init__()
        self.patch_embeddings = nn.Conv2d(3, hidden_size, patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        num_patches = (image_size // patch_size) ** 2
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, num_patches + 1, hidden_size)
        )

    def forward(self, x):
        x = self.patch_embeddings(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embeddings
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        outputs=10,
        image_size=32,
        patch_size=4,
        hidden_size=128,
        num_heads=4,
        num_layers=4,
    ):
        super().__init__()
        self.embedding = VisionTransformerEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                activation="gelu",
                batch_first=True,
                norm_first=True,
                dropout=0,
            ),
            num_layers=num_layers,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, outputs)
        )

    def forward(self, x):
        x = self.encoder(self.embedding(x))
        logits = self.head(x[:, 0])
        return logits


def ViTB16(outputs=10, image_size=224):
    return VisionTransformer(
        outputs=outputs,
        image_size=image_size,
        patch_size=16,
        hidden_size=768,
        num_heads=12,
        num_layers=12,
    )
