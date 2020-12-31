import sys
sys.path.append('./')
sys.path.append('./example')
from easydict import EasyDict as edict
import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm
import pickle

from zfnet import ZFNet
from qqquantize.qconfig import DEFAULT_QAT_MODULE_MAPPING
from qqquantize.quantize import prepare, convert
from qqquantize.observers.fake_quantize import FakeQuantize
from qqquantize.utils import enable_fake_quant, disable_fake_quant, enable_observer, disable_observer
from qqquantize.savehook import register_intermediate_hooks

FLOAT_CKPT = 'example/checkpoint/zfnet_float.pth'
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

if __name__ == '__main__':
    net = ZFNet().eval().to(DEVICE)
    ckpt = torch.load(FLOAT_CKPT)
    net.load_state_dict(ckpt['net'])

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
        testset, batch_size=8, shuffle=False, num_workers=4)
    print('>>> before quantize test')
    test(net, testloader)

    qconfig = edict({
        'activation': FakeQuantize.with_args(bits=8, max_factor=0.75),
        'weight': FakeQuantize.with_args(bits=8, max_factor=0.75)
    })

    # prepare only add observer for activation
    prepare(net, qconfig, inplace=True)
    # convert will convert each layer which contain weight_fake_quant
    convert(net, mapping=DEFAULT_QAT_MODULE_MAPPING, inplace=True)
    disable_fake_quant(net)
    enable_observer(net)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        outputs = net(inputs.to(DEVICE))
    
    enable_fake_quant(net)
    disable_observer(net)
    print('>>> after quantize test')
    test(net, testloader)

    # output quantized intermediate
    hook = register_intermediate_hooks(net)
    loader_iter = iter(testloader)
    inputs, targets = next(loader_iter)
    net(inputs.to(DEVICE))
    inter_data = hook.output_data()
    pickle.dump(inter_data, open('inter_data.pkl', 'wb'))
