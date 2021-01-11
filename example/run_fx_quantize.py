import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('./example')
from easydict import EasyDict as edict
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from tqdm import tqdm
import pickle

from zfnet import ZFNet, fuse_zfnet
from qqquantize.qconfig import DEFAULT_QAT_MODULE_MAPPING
from qqquantize.quantize import prepare
from qqquantize.observers.fake_quantize import FakeQuantize
from qqquantize.observers.fake_quantize import enable_fake_quant, disable_fake_quant, enable_observer, disable_observer
from qqquantize.savehook import register_intermediate_hooks
from qqquantize.toolbox import fxquantize
import qqquantize.qmodules as qm

CKPT_PATH = 'checkpoint/zfnet_quant.pth'
CIFAR_ROOT = '/home/luojiapeng/root_data_lnk/datasets/cifar'
DEVICE = 'cuda'

def test(net, testloader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(testloader)
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Save checkpoint.
    acc = 100.*correct/total
    print('test_acc: %.3f' % acc)


# Training
def train(net, trainloader, criterion, optimizer):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        _lr = optimizer.param_groups[0]['lr']
    print('training Loss: %.3f | Acc: %.3f%% (%d/%d) | lr: %.6f'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, _lr))


if __name__ == '__main__':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])
    trainset = torchvision.datasets.CIFAR10(
        root=CIFAR_ROOT, train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=4)
    testset = torchvision.datasets.CIFAR10(
        root=CIFAR_ROOT, train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=32, shuffle=False, num_workers=4)
    
    qconfig = edict({
        'activation': FakeQuantize.with_args(bits=8, max_factor=0.8),
        'weight': FakeQuantize.with_args(bits=8, max_factor=0.8),
        'bias': FakeQuantize.with_args(bits=8, max_factor=0.8)
    })

    net = ZFNet(0.5).eval().to(DEVICE)
    fuse_zfnet(net, inplace=True)
    net = prepare(net, qconfig)
    net.load_state_dict(torch.load(CKPT_PATH))
    
    enable_fake_quant(net)
    disable_observer(net)
    print('>>> after quantize test')
    test(net, testloader)

    from qqquantize.qtensor import QTensor
    data_iter = iter(testloader)
    fake_input = next(data_iter)[0]
    
    fake_input = QTensor(fake_input).to(DEVICE)
    def fx_adjust_bits(net, fake_input):
        QMOD_LIST = list(DEFAULT_QAT_MODULE_MAPPING.values())
        for name, mod in net.named_modules():
            if type(mod) in QMOD_LIST:
                for m in mod.modules():
                    if isinstance(m, FakeQuantize):
                        if m.scale > 1:
                            m.scale = torch.tensor([0.0], device=m.scale.device)
                        if m.scale < 2**-7:
                            m.scale = torch.tensor([2**-7], device=m.scale.device)
        def adjust_bits(model, fake_input):
            bit_dict = fxquantize.get_layer_bits(net, fake_input)
            for name, mod in net.named_modules():
                if isinstance(mod, qm.QConv2d):
                    i_bit = bit_dict[name]['inp'][0]
                    changed = fxquantize.conv_bit_adjust(mod, i_bit)
                    if changed:
                        break
            if changed:
                adjust_bits(model, fake_input)
        adjust_bits(net, fake_input)
    bit_dict = fxquantize.get_layer_bits(net, fake_input)
    test(net, testloader)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01,
                          momentum=0.9, weight_decay=5e-4)
    def lr_lambda(epoch):
        if epoch < 50:
            return 1
        elif epoch < 100:
            return 0.2
        else:
            return 0.04
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    for epoch in range(150):
        disable_observer(net)
        train(net, trainloader, criterion, optimizer)
        lr_scheduler.step()
        disable_observer(net)
        test(net, testloader)