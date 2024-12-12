from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
import sys

try:
    sys.path.insert(0, '../')
    import activation
finally:
    pass

class ConvMixerBlock(nn.Module):
    def __init__(self, dim, kernel_size):
        super(ConvMixerBlock, self).__init__()
        self.depthwise_conv = nn.Conv2d(
            dim, dim,
            kernel_size=kernel_size, stride=1,
            padding=kernel_size // 2,
            groups=dim  # Depthwise
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.pointwise_conv = nn.Conv2d(
            dim, dim,
            kernel_size=1, stride=1
        )
        self.activation1 = nn.ReLU()  # 활성화 함수
        self.activation2 = nn.ReLU()

        self.bn2 = nn.BatchNorm2d(16) 
        
    def forward(self, x):
        residual = x.clone()
        x = self.depthwise_conv(x)
        x = self.activation1(x)
        x = self.bn1(x)
        x += residual

        x = self.pointwise_conv(x)
        x = self.activation2(x)
        x = self.bn2(x)

        return x

class ConvMixerPatchEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, depth, kernel_size):
        super(ConvMixerPatchEmbedding, self).__init__()
        # Patch Embedding
        self.patch_embedding = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=patch_size, stride=patch_size, bias=False
        )

        self.bn = nn.BatchNorm2d(out_channels)

        # ConvMixer Blocks
        self.blocks = nn.Sequential(*[
            ConvMixerBlock(out_channels, kernel_size) for _ in range(depth)
        ])

    def forward(self, x):
        x = self.patch_embedding(x)  # 패치 분할
        x = self.bn(x)
        x = self.blocks(x)           # Depthwise & Pointwise Convolution 반복
        return x

class ResNet8WithConvMixerEmbedding(nn.Module):
    def __init__(self, params):
        super().__init__()

        in_channels = params['in_channels']
        out_channels = params['out_channels']
        train_act = params['activation']
        patch_size = params['patch_size']
        depth = params['depth']  # 반복 횟수
        kernel_size = params['kernel_size']  # Depthwise Conv 커널 크기

        # ConvMixer Patch Embedding Layer
        self.patch_embedding = ConvMixerPatchEmbedding(
            in_channels=in_channels,
            out_channels=16,  # Patch Embedding 출력 채널
            patch_size=patch_size,
            depth=depth,
            kernel_size=kernel_size
        )

        # 기존 ResNetCifar 모델
        self.model = ResNetCifar(BasicBlockCifar, [1, 1, 1])

        # Replace conv1 with ConvMixer Patch Embedding
        self.model.conv1 = nn.Identity()  # 기존 conv1 제거
        self.model.fc = nn.Linear(64, out_channels, bias=True)
        
        # Replace activation layers if needed
        replace_layers(train_act, self.patch_embedding, nn.ReLU)
        replace_layers(train_act, self.model, nn.ReLU)

    def forward(self, x):
        x = self.patch_embedding(x)  # ConvMixer Patch Embedding 적용
        x = self.model(x)          # ResNet 나머지 블록 처리

        return x

    def __str__(self):
        model_str = "ResNet8WithConvMixerEmbedding(\n"
        model_str += f"  Patch Embedding: {self.patch_embedding}\n"
        model_str += f"  ResNet Model: {self.model}\n"
        model_str += ")"
        return model_str

def replace_layers(act, model, old):
    if isinstance(act, str):
        if act != 'Default':
            new = getattr(activation, act)
            _replace_layers(model, old, new())
    else:
        train_act_name = act['name']
        train_act_acts = act['activations']
        acts = _get_acts(train_act_acts)

        new = getattr(activation, train_act_name)
        _replace_layers(model, old, new(activations=acts))


def _replace_layers(model, old, new):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            _replace_layers(module, old, new)

        if isinstance(module, old):
            setattr(model, n, new)


def _weights_init_cifar(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


def _get_acts(train_act_acts):
    acts = []

    for act_name in train_act_acts:
        acts.append(getattr(F, act_name))

    return acts

class LambdaLayerCifar(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayerCifar, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlockCifar(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlockCifar, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU()

        if stride != 1 or in_planes != planes:
            if option == 'A':
                tup1 = (0, 0, 0, 0, planes // 4, planes // 4)
                def f(x): return F.pad(x[:, :, ::2, ::2], tup1, 'constant', 0)
                self.shortcut = LambdaLayerCifar(f)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNetCifar(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetCifar, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(
            3,
            16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
    
        self.fc = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.apply(_weights_init_cifar)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class ResNet8(nn.Module):
    def __init__(self, params):
        super().__init__()

        in_channels = params['in_channels']
        out_channels = params['out_channels']

        train_act = params['activation']

        self.model = ResNetCifar(BasicBlockCifar, [1, 1, 1])

        self.model.conv1 = nn.Conv2d(
            in_channels,
            16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.model.fc = nn.Linear(64, out_channels, bias=True)
        replace_layers(train_act, self.model, nn.ReLU)

    def forward(self, x):
        out = self.model(x)
        return out

    def __str__(self):
        model_str = str(self.model)
        return model_str


class ResNet14(nn.Module):
    def __init__(self, params):
        super().__init__()

        in_channels = params['in_channels']
        out_channels = params['out_channels']

        train_act = params['activation']

        self.model = ResNetCifar(BasicBlockCifar, [2, 2, 2])

        self.model.conv1 = nn.Conv2d(
            in_channels,
            16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.model.fc = nn.Linear(64, out_channels, bias=True)
        replace_layers(train_act, self.model, nn.ReLU)

    def forward(self, x):
        out = self.model(x)
        return out

    def __str__(self):
        model_str = str(self.model)
        return model_str