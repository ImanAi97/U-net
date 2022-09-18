from turtle import forward
from unittest import skip
import torch
import torch.nn as nn
import torchvision.transforms.functional as F

class DoubleConvolution(nn.Module):

    """This Component is used in each layer of U-net as 2 exactly-same Sequential convolutional layer
    with the same output channels,
    padding = 1 (same) added to make the model perform a bit faster 
    """

    def __init__(self, in_channels, out_channels, **kwargs):
        super(DoubleConvolution, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
            ## Bias would be neglected by batchnorm anyway!
            ## So we delete that
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
        )

    def forward(self, x):
        return self.conv(x)


class Unet(nn.Module):
    # Input_channels : 3 (rgb image)
    # output_channels : 1 (binary segmentation)

    def __init__(
        self, in_channels = 3, out_channels = 1,
        features = [64, 128, 256, 512]
    ):
        super(Unet, self).__init__()

        ## We need to use list here, but to have some functionalities we use modulelist:
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # Down Part of Unet:
        for feature in features:
            self.downs.append(DoubleConvolution(in_channels=in_channels, out_channels=feature))
            in_channels = feature

        # Up part of Unet:
        for feature in features[::-1]:
            self.ups.append(
                ## Kernelsize = 2 ----> double the width and the height
                ## out channels  = feature ---> to have space for concatenating skip connections 
                nn.ConvTranspose2d(in_channels = feature * 2, out_channels = feature, kernel_size = 2, stride = 2),
            )
            self.ups.append(DoubleConvolution(in_channels = feature * 2, out_channels = feature))

        self.bottleneck = DoubleConvolution(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size = 1)


    def forward(self, x):
        skip_connections = []
        for down_layer in self.downs:
            conv = down_layer(x)
            skip_connections.append(conv)
            x = self.pool(conv)
        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1] # we use list items in reverse at the up forward of the network

        for idx in range(0, len(self.ups), 2):
            ## go through transpose conv layer 
            x = self.ups[idx](x)
            ## skip_coonection of the layer:
            skip_connection = skip_connections[idx // 2] # 0, 1, 2, 3
            ## concatenating skip_connection

            # In case of shape incompatibility:
            if x.shape != skip_connection.shape:
                x = F.resize(x, size = skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim = 1) # dim 0 : batch, dim 1 : channels
            # go through DoubleConvolution
            x = self.ups[idx + 1](concat_skip)  


        return self.final_conv(x)


if __name__ == "__main__":
    x_rand = torch.randn((3, 3, 161, 161))
    net = Unet(in_channels= 3, out_channels= 1)
    print(net(x_rand).shape)
    print(x_rand.shape)
