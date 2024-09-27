import torch


def add_training_noise(x,
                       bias_neural_std=None,
                       noise_neural_std=None,
                       noise_neural_walk_std=None,
                       bias_allchans_neural_std=None,
                       device='cpu'):
    """Function to add different types of noise to training input data to make models more robust.
       Identical to the methods in Willet 2021.
    Args:
        x (tensor):                     neural data of shape [batch_size x num_chans x conv_size]
        bias_neural_std (float):        std of bias noise
        noise_neural_std (float):       std of white noise
        noise_neural_walk_std (float):  std of random walk noise
        bias_allchans_neural_std (float): std of bias noise, bias is same across all channels
        device (device):                torch device (cpu or cuda)
    """
    if bias_neural_std:
        # bias is constant across time (i.e. the 3 conv inputs), but different for each channel & batch
        # biases = torch.normal(0, bias_neural_std, x.shape[:2]).unsqueeze(2).repeat(1, 1, x.shape[2])
        biases = torch.normal(torch.zeros(x.shape[:2]), bias_neural_std).unsqueeze(2).repeat(1, 1, x.shape[2])
        x = x + biases.to(device=device)

    if noise_neural_std:
        # adds white noise to each channel and timepoint (independent)
        # noise = torch.normal(0, noise_neural_std, x.shape)
        noise = torch.normal(torch.zeros_like(x), noise_neural_std)
        x = x + noise.to(device=device)

    if noise_neural_walk_std:
        # adds a random walk to each channel (noise is summed across time)
        # noise = torch.normal(0, noise_neural_walk_std, x.shape).cumsum(dim=2)
        noise = torch.normal(torch.zeros_like(x), noise_neural_walk_std).cumsum(dim=2)
        x = x + noise.to(device=device)

    if bias_allchans_neural_std:
        # bias is constant across time (i.e. the 3 conv inputs), and same for each channel
        biases = torch.normal(torch.zeros((x.shape[0], 1, 1)), bias_allchans_neural_std).repeat(1, x.shape[1], x.shape[2])
        x = x + biases.to(device=device)

    return x




