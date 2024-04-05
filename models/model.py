import torch.nn as nn
import torch.nn.functional as F


class Convsubblock(nn.Module):
    """
    Conv -> BN -> ReLU or Conv -> ReLU ->BN : (separable/non-separable)
    """

    def __init__(
        self,
        inch,
        outch,
        ks=3,
        stride=1,
        dilation=1,
        dropoutval=0.1,
        preact=True,
        separable=False,
        useBN=True,
    ):
        super().__init__()
        self.conv_sblock = nn.ModuleList()
        if dilation > 1:
            ks_ = dilation * ks - 1
        else:
            ks_ = ks
        padding_ = ks_ // 2

        if separable:
            self.conv_sblock.append(
                nn.Conv2d(
                    inch,
                    inch,
                    ks,
                    stride,
                    padding=padding_,
                    dilation=dilation,
                    bias=False,
                    groups=inch,
                )
            )
            self.conv_sblock.append(
                nn.Conv2d(inch, outch, 1, 1, padding=0, dilation=1, bias=False)
            )
        else:
            self.conv_sblock.append(
                nn.Conv2d(
                    inch,
                    outch,
                    ks,
                    stride,
                    padding=padding_,
                    dilation=dilation,
                    bias=False,
                )
            )

        if preact and useBN:
            self.conv_sblock.append(nn.BatchNorm2d(outch))
            self.conv_sblock.append(nn.ReLU())
            self.conv_sblock.append(nn.Dropout(dropoutval))
        elif not preact and useBN:
            self.conv_sblock.append(nn.ReLU())
            self.conv_sblock.append(nn.BatchNorm2d(outch))
            self.conv_sblock.append(nn.Dropout(dropoutval))
        else:
            self.conv_sblock.append(nn.ReLU())
            self.conv_sblock.append(nn.Dropout(dropoutval))

    def forward(self, x):
        for l in self.conv_sblock:
            x = l(x)
        return x


class ConvBlock(nn.Module):
    """
    pyramid structure
    """

    def __init__(
        self,
        inch,
        outch,
        numlayers=4,
        ks=[3] * 4,
        dropoutval=[0.1] * 4,
        stride=[1] * 4,
        dilation=[1] * 4,
        separable=[False] * 4,
        preact=[True] * 4,
        useBN=[True] * 4,
        lastblock=False,
    ):
        super().__init__()
        # some assert checkings
        assert (
            len(ks) == numlayers
        ), "length of ks list must equal to num of layers in Conv block!"
        assert (
            len(dropoutval) == numlayers
        ), "length of dropoutval list must equal to num of layers in Conv block!"
        assert (
            len(stride) == numlayers
        ), "length of stride list must equal to num of layers in Conv block!"
        assert (
            len(dilation) == numlayers
        ), "length of dilation list must equal to num of layers in Conv block!"
        assert (
            len(separable) == numlayers
        ), "length of separable list must equal to num of layers in Conv block!"
        assert (
            len(preact) == numlayers
        ), "length of preact list must equal to num of layers in Conv block!"
        assert (
            len(useBN) == numlayers
        ), "length of useBN list must equal to num of layers in Conv block!"

        self.convblock = nn.ModuleList()
        for j in range(numlayers):
            if j > 0:
                inch_ = outch
            else:
                inch_ = inch
            if (
                j == numlayers - 1 and not lastblock
            ):  # last layer of every ConvBlock except the final Convblock, set stride=2
                stride_ = 2

            else:  # the last layer in the last conv block
                stride_ = stride[j]  # default stride = 1

            self.convblock.append(
                Convsubblock(
                    inch=inch_,
                    outch=outch,
                    ks=ks[j],
                    stride=stride_,
                    dilation=dilation[j],
                    dropoutval=dropoutval[j],
                    preact=preact[j],
                    separable=separable[j],
                    useBN=useBN[j],
                )
            )

    def forward(self, x):
        for l in self.convblock:
            x = l(x)
        return x


class Net(nn.Module):
    def __init__(
        self,
        inch=[3, 32, 64, 128],
        outch=[32, 64, 128, 256],
        n_classes=10,
        dropoutval=[0.1] * 4,
        ks=[3] * 4,
        numlayers=[4, 3, 3, 2],
        stride=[1] * 4,
        dilation=[1] * 4,
        separable=[False] * 4,
        useBN=[True] * 4,
        preact=[True] * 4,
    ):
        super().__init__()
        self.n_classes = n_classes

        # No Input layer required as input images are 32x32 already

        # conv block1 : 32x32x3 -> 16x16x32, RF: 11
        self.convblk1 = ConvBlock(
            inch=inch[0],
            outch=outch[0],
            numlayers=numlayers[0],
            ks=[ks[0]] * numlayers[0],
            dropoutval=[dropoutval[0]] * numlayers[0],
            stride=[stride[0]] * numlayers[0],
            dilation=[dilation[0]] * numlayers[0],
            separable=[separable[0]] * numlayers[0],
            preact=[preact[0]] * numlayers[0],
            useBN=[useBN[0]] * numlayers[0],
            lastblock=False,
        )
        # conv block2: 16x16x32 -> 8x8x64, RF: 23
        self.convblk2 = ConvBlock(
            inch=inch[1],
            outch=outch[1],
            numlayers=numlayers[1],
            ks=[ks[1]] * numlayers[1],
            dropoutval=[dropoutval[1]] * numlayers[1],
            stride=[stride[1]] * numlayers[1],
            dilation=[dilation[1]] * numlayers[1],
            separable=[separable[1]] * numlayers[1],
            preact=[preact[1]] * numlayers[1],
            useBN=[useBN[1]] * numlayers[1],
            lastblock=False,
        )

        # conv block3: 8x8x64 -> 4x4x128, RF: 39
        self.convblk3 = ConvBlock(
            inch=inch[2],
            outch=outch[2],
            numlayers=numlayers[2],
            ks=[ks[2]] * numlayers[2],
            dropoutval=[dropoutval[2]] * numlayers[2],
            stride=[stride[2]] * numlayers[2],
            dilation=[dilation[2]] * numlayers[2],
            separable=[separable[2]] * numlayers[2],
            preact=[preact[2]] * numlayers[2],
            useBN=[useBN[2]] * numlayers[2],
            lastblock=False,
        )

        # conv block4: 4x4x128 -> 4x4x256, RF: 71
        self.convblk4 = ConvBlock(
            inch=inch[3],
            outch=outch[3],
            numlayers=numlayers[3],
            ks=[ks[3]] * numlayers[3],
            dropoutval=[dropoutval[3]] * numlayers[3],
            stride=[stride[3]] * numlayers[3],
            dilation=[dilation[3]] * numlayers[3],
            separable=[separable[3]] * numlayers[3],
            preact=[preact[3]] * numlayers[3],
            useBN=[useBN[3]] * numlayers[3],
            lastblock=True,
        )

        # GAP layer: 4x4x256 -> 1x1x256, RF: 95
        self.gap = nn.AvgPool2d(kernel_size=4)
        # output layer
        self.out = nn.Conv2d(outch[3], n_classes, 1, padding=0, bias=False)

    def forward(self, x):
        x = self.convblk1(x)
        x = self.convblk2(x)
        x = self.convblk3(x)
        x = self.convblk4(x)
        x = self.gap(x)
        x = self.out(x)
        x = x.view(-1, self.n_classes)
        return F.log_softmax(x, dim=-1)
