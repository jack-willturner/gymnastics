import torch


def test_shift_layer():

    from models.layers import ShiftConv

    # put a little square somewhere in the image
    x = torch.zeros(1, 9, 10, 10)

    h = 4
    w = 4
    x[0, :, h : h + 2, w : w + 2] = 0.1

    layer = ShiftConv(9, 9, kernel_size=3, stride=1, bias=False)

    shifted_x = layer(x)

    """ plotting code 

    import matplotlib.pyplot as plt

    def plot_image(x, ax=None):
        
        if len(x.size()) == 4:
            x = x[0]
        
        if ax is None:
            fig, ax = plt.subplots()
            
            
        if len(x.size()) == 2:
            ax.imshow(x.detach())
        else:
            ax.imshow(x.detach().permute(1,2,0))
    
    fig, ax = plt.subplots(3,3)
    ax = ax.ravel()

    for n, fmap in enumerate(shifted_x[0]):
        plot_image(fmap, ax[n])
    """


def test_shift_when_in_channels_not_divisible_by_groups():

    from models.layers import ShiftConv

    x = torch.zeros(1, 64, 10, 10)

    h = 4
    w = 4
    x[0, :, h : h + 2, w : w + 2] = 0.1

    layer = ShiftConv(64, 90, kernel_size=3, stride=1, bias=False)

    shifted_x = layer(x)


def test_shift_when_out_channels_not_divisible_by_groups():

    from models.layers import ShiftConv

    x = torch.zeros(1, 27, 10, 10)

    h = 4
    w = 4
    x[0, :, h : h + 2, w : w + 2] = 0.1

    layer = ShiftConv(27, 65, kernel_size=3, stride=1, bias=False)

    shifted_x = layer(x)


def test_shift_when_neither_out_nor_in_channels_are_divisible_by_groups():

    from models.layers import ShiftConv

    x = torch.zeros(1, 64, 10, 10)

    h = 4
    w = 4
    x[0, :, h : h + 2, w : w + 2] = 0.1

    layer = ShiftConv(64, 64, kernel_size=3, stride=1, bias=False)

    shifted_x = layer(x)
