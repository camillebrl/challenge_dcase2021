import torch
import torch.nn as nn
import torch.nn.functional as F

# On n'utilise jamais les fonctions forward des blocks, si?

class Block(nn.Module):
    '''Grouped convolution block.'''
    expansion = 2

    def __init__(self, in_planes, cardinality=32, bottleneck_width=4, stride=1):
        super(Block, self).__init__()
        group_width = cardinality * bottleneck_width
        self.conv1 = nn.Conv2d(in_planes, group_width, kernel_size=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3,
                               stride=stride, padding=1, groups=cardinality,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(group_width)
        self.conv3 = nn.Conv2d(group_width, self.expansion * group_width,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * group_width)

        ######################## CBAM ########################
        self.max_pooling_spatial = nn.MaxPool2d(kernel_size=1)
        self.avg_pooling_spatial = nn.MaxPool2d(kernel_size=1)

        self.mlp = nn.Linear(3,64) # voir taille
        ######################################################

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * group_width:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * group_width,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * group_width)
            )

    def cbam(self, out):
        out_channel = out.view((out.size()[0],1)) # out.size()[0] pour garder le shape des batchs
        out_channel_1 = self.max_pooling_channel(out_channel)
        out_channel_2 = self.avg_pooling_channel(out_channel)
        # on applique le même MLP aux 2 sorties:
        out_channel_1 = self.mlp(out_channel_1) 
        out_channel_2 = self.mlp(out_channel_2)
        out_channel = out_channel_1 + out_channel_2
        out_channel = F.relu(out_channel) # relu à la place de softmax pour calquer mcdonnell paper

        out_channel = out_channel.view(out.size())
        out_channel = out_channel * out

        out_spatial = out_channel.view((out_channel.size()[0],2)) # out_channel.size()[0] pour garder le shape des batchs
        out_spatial_max = self.max_pooling_spatial(out_spatial) # changer le pooling
        out_spatial_avg = self.avg_pooling_spatial(out_spatial)
        out_spatial = torch.cat(out_spatial_max,out_spatial_avg)
        out_spatial = self.conv2(out_spatial) 
        out_spatial = F.relu(out_spatial) # relu à la place de softmax pour calquer mcdonnell paper

        out_spatial = out_spatial.view(out.size())
        out_spatial = out_spatial * out

        return out_spatial

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = F.relu(cbam(out))
        out = self.bn2(self.conv2(out))
        out = F.relu(cbam(out))
        out = self.bn3(self.conv3(out))
        out = F.relu(cbam(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        ######################## CBAM ########################
        self.max_pooling_spatial = nn.MaxPool2d(kernel_size=1)
        self.avg_pooling_spatial = nn.MaxPool2d(kernel_size=1)

        self.mlp = nn.Linear(3,64) # voir taille
        ######################################################

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def cbam(self, out):
        out_channel = out.view((out.size()[0],1)) # out.size()[0] pour garder le shape des batchs
        out_channel_1 = self.max_pooling_channel(out_channel)
        out_channel_2 = self.avg_pooling_channel(out_channel)
        # on applique le même MLP aux 2 sorties:
        out_channel_1 = self.mlp(out_channel_1) 
        out_channel_2 = self.mlp(out_channel_2)
        out_channel = out_channel_1 + out_channel_2
        out_channel = F.relu(out_channel) # relu à la place de softmax pour calquer mcdonnell paper

        out_channel = out_channel.view(out.size())
        out_channel = out_channel * out

        out_spatial = out_channel.view((out_channel.size()[0],2)) # out_channel.size()[0] pour garder le shape des batchs
        out_spatial_max = self.max_pooling_spatial(out_spatial) # changer le pooling
        out_spatial_avg = self.avg_pooling_spatial(out_spatial)
        out_spatial = torch.cat(out_spatial_max,out_spatial_avg)
        out_spatial = self.conv2(out_spatial) 
        out_spatial = F.relu(out_spatial) # relu à la place de softmax pour calquer mcdonnell paper

        out_spatial = out_spatial.view(out.size())
        out_spatial = out_spatial * out

        return out_spatial

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = F.relu(cbam(out))
        out = self.bn2(self.conv2(out))
        out = F.relu(cbam(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        ######################## CBAM ########################
        self.conv2 = nn.Conv2d(3,64) # à voir la taille

        self.max_pooling_spatial = nn.MaxPool2d(kernel_size=1)
        self.avg_pooling_spatial = nn.MaxPool2d(kernel_size=1)

        self.mlp = nn.Linear(3,64) # voir taille
        ######################################################

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def cbam(self, out):
        out_channel = out.view((out.size()[0],1)) # out.size()[0] pour garder le shape des batchs
        out_channel_1 = self.max_pooling_channel(out_channel)
        out_channel_2 = self.avg_pooling_channel(out_channel)
        # on applique le même MLP aux 2 sorties:
        out_channel_1 = self.mlp(out_channel_1) 
        out_channel_2 = self.mlp(out_channel_2)
        out_channel = out_channel_1 + out_channel_2
        out_channel = F.softmax(out_channel)

        out_channel = out_channel.view(out.size())
        out_channel = out_channel * out

        out_spatial = out_channel.view((out_channel.size()[0],2)) # out_channel.size()[0] pour garder le shape des batchs
        out_spatial_max = self.max_pooling_spatial(out_spatial) # changer le pooling
        out_spatial_avg = self.avg_pooling_spatial(out_spatial)
        out_spatial = torch.cat(out_spatial_max,out_spatial_avg)
        out_spatial = self.conv2(out_spatial) 
        out_spatial = F.softmax(out_spatial)

        out_spatial = out_spatial.view(out.size())
        out_spatial = out_spatial * out

        out = out_spatial + out # Le shortcut comme dans les blocs ResNet

        return out

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = F.relu(cbam(out))
        out = self.bn2(self.conv2(out))
        out = F.relu(cbam(out))
        out = self.bn3(self.conv3(out))
        out = F.relu(cbam(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_cabm(nn.Module):
    
    def __init__(self, block, num_blocks, div=1, num_classes=10):
        super(ResNet_cabm, self).__init__()
        self.in_planes = 64 // div

        self.conv1 = nn.Conv2d(3, 64 // div, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64 // div)
        self.layer1 = self._make_layer(block, 64 // div, num_blocks[0],
                                       stride=1)
        self.layer2 = self._make_layer(block, 128 // div, num_blocks[1],
                                       stride=2)
        self.layer3 = self._make_layer(block, 256 // div, num_blocks[2],
                                       stride=2)
        self.layer4 = self._make_layer(block, 512 // div, num_blocks[3],
                                       stride=2)
        self.linear = nn.Linear((512 // div) * block.expansion, num_classes)

        ######################## CBAM ########################
        self.conv_spatial = nn.conv_spatial(3,64) # à voir la taille

        self.max_pooling_spatial = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)) # stride de (1,2) pour ne pas diminuer les fréquences mais diminuer le temps (cf McDonnell)
        self.avg_pooling_spatial = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)) # stride de (1,2) pour ne pas diminuer les fréquences mais diminuer le temps (cf McDonnell)

        self.mlp = nn.Linear(3,64) # voir taille
        ######################################################

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            b = block(self.in_planes, planes, stride)
            layers.append(b)
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn1(self.conv1(x))

        ######################## CBAM #########################
        out_channel = out.view((out.size()[0],1)) # out.size()[0] pour garder le shape des batchs
        out_channel_1 = self.max_pooling_channel(out_channel)
        out_channel_2 = self.avg_pooling_channel(out_channel)
        # on applique le même MLP aux 2 sorties:
        out_channel_1 = self.mlp(out_channel_1) 
        out_channel_2 = self.mlp(out_channel_2)
        out_channel = out_channel_1 + out_channel_2
        out_channel = F.softmax(out_channel)

        out_channel = out_channel.view(out.size())
        out_channel = out_channel * out

        out_spatial = out_channel.view((out_channel.size()[0],2)) # out_channel.size()[0] pour garder le shape des batchs
        out_spatial_max = self.max_pooling_spatial(out_spatial) # changer le pooling
        out_spatial_avg = self.avg_pooling_spatial(out_spatial)
        out_spatial = torch.cat(out_spatial_max,out_spatial_avg)
        out_spatial = self.conv_spatial(out_spatial) 
        out_spatial = F.softmax(out_spatial) 

        out_spatial = out_spatial.view(out.size())
        out_spatial = out_spatial * out
        
        out = out_spatial + out # Le shortcut comme dans les blocs ResNet
        #######################################################
        out = F.relu(out)
        
        out = self.layer2(out)
        out = self.layer3(out)
        out_high_freq = self.layer4(out_high_freq)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


nums_blocks = {"ResNet18": [2, 2, 2, 2], "ResNet34": [3, 4, 6, 3],
               "ResNet50": [3, 4, 6, 3], "ResNet101": [3, 4, 23, 3],
               "ResNet152": [3, 8, 36, 3]}

