import torch
import torch.nn as nn

from models.networks.block import resnetunet_block as block


class _DummyResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Identity()
        self.layer2 = nn.Identity()
        self.layer3 = nn.Identity()
        self.layer4 = nn.Identity()


def test_resnet_encoder_depth3_input_copies_pretrained_rgb_for_both_triplets(monkeypatch):
    dummy = _DummyResNet()
    torch.manual_seed(0)
    with torch.no_grad():
        dummy.conv1.weight.copy_(torch.randn_like(dummy.conv1.weight))

    monkeypatch.setattr(block, "_load_resnet_pretrained", lambda *args, **kwargs: dummy)
    encoder = block.ResNetEncoder(in_chns=6, load_pretrained=False)

    expected = dummy.conv1.weight.detach()
    actual = encoder.encoder1_conv.weight.detach()

    assert actual.shape == (64, 6, 7, 7)
    assert torch.allclose(actual[:, :3], expected)
    assert torch.allclose(actual[:, 3:6], expected)
