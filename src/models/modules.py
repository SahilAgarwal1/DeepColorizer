import torch.nn as nn


class CNN_Halfing_Block(nn.Module):
    def __init__(self, input_planes, output_planes):
        super().__init__()

        # CNN block with parameters to halfen the input image
        self.conv = nn.Sequential(
            nn.Conv2d(input_planes, output_planes, kernel_size= (3,3), stride= (2,2) , \
                      padding= 1),
            nn.BatchNorm2d(output_planes),
            nn.ReLU()

            )

    def forward(self, x):
        return self.conv(x)

class CNN_Upsampling_Block(nn.Module):
    def __init__(self, input_planes, output_planes):
        super().__init__()

        #Upsampling block with scale of 2 followed by Convolution
        self.upAndConv = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2), # upsample by a factor of 2
            nn.Conv2d(input_planes,output_planes,kernel_size=(3,3), stride=(1,1), padding= 1),
            nn.BatchNorm2d(output_planes),
            nn.ReLU()

        )

    def forward(self, x):
        return self.upAndConv(x)


class CNN_Upsampling_Block_Output(nn.Module):
    def __init__(self, input_planes, output_planes):
        super().__init__()

        # Upsampling block with scale of 2 followed by Convolution
        self.upAndConv = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),  # upsample by a factor of 2
            nn.Conv2d(input_planes, output_planes, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(output_planes),
            nn.Sigmoid()

        )

    def forward(self, x):
        return self.upAndConv(x)