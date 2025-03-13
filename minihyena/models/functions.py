import torch
import torch.nn.functional as F


def fftconv(u, filter, bias, act=True):
    """
    Calculate long convolution using Fast Fourier Transform.
    """
    # fft size must be twice the sequence length to maintain causality
    fft_size = 2 * u.shape[-1] # u : [b, d, l]

    # apply FFT to the inputs
    u_fourier = torch.fft.rfft(u, n=fft_size) / fft_size
    filter_fourier = torch.fft.rfft(filter, n=fft_size) / fft_size

    # apply the convolution by multiplying and IFFT
    # we need to remove padding afterwards
    u_filter = u_fourier * filter_fourier
    y = torch.fft.irfft(u_filter, n=fft_size, norm="forward")

    # apply bias
    out = y + u * bias.unsqueeze(1)

    # apply GeLU activation if needed
    if act:
        out = F.gelu(out)

    return out
