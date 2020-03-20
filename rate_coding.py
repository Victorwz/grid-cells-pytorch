import torch


class RateCoder():

    def __init__(self, length, max_val, min_val, pause_length=None, cuda=True, polar=False):
        """
        if polar is True, then min_val will be intepreted as the origin on the unit circle,
        while max_val will be the largest deviation possible
        """
        self.l = length
        self.max_val = max_val
        self.min_val = min_val
        if not pause_length:
            self.pause_length = length
        else:
            self.pause_length = pause_length

        self.step = (max_val - min_val)/length
        self.cuda = cuda
        self.polar = polar

    def rate_code(self, x):
        """
        rate codes along the last dim
        previous dims are assumed batches
        """
        if self.polar:
            x = torch.abs(x)

        if self.min_val > self.max_val:
            # if origin on unit circle is higher than rest of values, flip values
            assert self.polar
            x = -x + self.min_val
            rates = torch.round(((x - self.max_val) / self.min_val) * self.l) + 1.
        else:
            rates = torch.round(((x - self.min_val) / self.max_val) * self.l) + 1.

        rate = torch.ones(x.shape)
        if self.cuda:
            rate = rate.cuda()

        fracs = (rate * rates / self.l).unsqueeze(-1).repeat(1, 1, self.l)
        pause_padding = torch.zeros_like(fracs)
        if self.cuda:
            pause_padding = pause_padding.cuda()

        fracs = torch.cat((fracs, pause_padding), dim=-1)

        mods = torch.cumsum(fracs, dim=-1) % 1. + fracs

        spikes = torch.zeros_like(mods)
        if self.cuda:
            spikes = spikes.cuda()

        spikes[mods > 1.] = 1.
        spikes = spikes.view(spikes.shape[0], -1)

        return spikes


class PadCoder(torch.nn.Module):

    def __init__(self, pad_length):
        super(PadCoder, self).__init__()
        assert pad_length > 0
        self.l = int(pad_length)

    def code(self, x, target=False, value=None):
        """
        expects x with shape [B X T X N]
        with batch size B, T time steps and N inputs
        """
        cuda = x.device != 'cpu'
        x = x.transpose(1, 2)
        zeros = torch.zeros_like(x)
        if cuda:
            zeros = zeros.cuda()

        if target:
            pad = x.unsqueeze(-1).repeat(1, 1, 1, self.l)
        elif target:
            pad = zeros.unsqueeze(-1).repeat(1, 1, 1, self.l)
        else:
            ones = torch.ones_like(x)
            v = value.unsqueeze(-1).repeat(1, ones.shape[-1])
            ones[:] = v
            pad = ones.unsqueeze(-1).repeat(1, 1, 1, self.l)


        x_new = torch.cat((x.unsqueeze(-1), pad), -1)
        x_new = x_new.view(x_new.shape[0], x_new.shape[1], -1)

        assert x_new.shape[-1] == x.shape[-1] * (self.l + 1)
        x_new = x_new.transpose(1, 2)
        return x_new

    def forward(self, x, **kwargs):
        return self.code(x, **kwargs)


