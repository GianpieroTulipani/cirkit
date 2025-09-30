import math
import torch
from torch import nn
from torch.distributions import Normal, Categorical

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class Encoder(nn.Module):
    def __init__(self, channel_input: int, latent_dim: int):
        super(Encoder, self).__init__()

        self.channel_input = channel_input
        self.latent_dim = latent_dim

        self.feature_extractor = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
            )

        self.output_layers = nn.Sequential(nn.Linear(512, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 2*self.latent_dim))

    def forward(self, x, n_samples=None):
        _out = self.feature_extractor(x).view(-1, 4*4*32)
        _out = self.output_layers(_out)

        _mu = _out[:, 0:self.latent_dim]
        _log_var = _out[:, self.latent_dim:]

        dist = Normal(_mu, (0.5 * _log_var).exp())
        if n_samples is None:
            _z = dist.rsample()
        else:
            _z = dist.rsample([n_samples])

        return _mu, _log_var, _z, dist


class CategoricalDecoder(nn.Module):
    def __init__(self, latent_dim: int, output_channel: int, n_bins: int = 256):
        super().__init__()
        self.n_bins = n_bins
        self.output_channel = output_channel
        self.conv_layers = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            View((-1, 32, 4, 4)),
            nn.ConvTranspose2d(32, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 5, stride=2, padding=2, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, output_channel * self.n_bins, 5, stride=2, padding=2, output_padding=1)
        )

    def forward(self, z, n_samples=None):
        if n_samples is None:
            logits = self.conv_layers(z)
            logits = logits.view(z.shape[0], self.output_channel, self.n_bins, 28, 28)
            logits = logits.permute(0, 1, 3, 4, 2).contiguous()
        else:
            K, B = z.shape[0], z.shape[1]
            logits = self.conv_layers(z.view(K*B, -1))
            logits = logits.view(K, B, self.output_channel, self.n_bins, 28, 28)
            logits = logits.permute(0,1,2,4,5,3).contiguous()

        output_dist = Categorical(logits=logits)
        return logits, output_dist


class ConvVAE(nn.Module):
    def __init__(self, input_channel: int, latent_dim: int):
        super().__init__()
        self.encoder = Encoder(input_channel, latent_dim)
        self.decoder = CategoricalDecoder(latent_dim, input_channel, n_bins=256)
        self.prior = Normal(0, 1)

    def forward(self, samples, n_sample=None, mask=None):
        mu, log_var, z, q = self.encoder(samples.float()/255.0, n_sample)

        kl = (q.log_prob(z) - self.prior.log_prob(z)).sum(-1)

        logits, output_dist = self.decoder(z, n_sample)

        if n_sample is None:
            log_pgivenz = output_dist.log_prob(samples)

            if mask is not None:
                log_pgivenz = (log_pgivenz * mask).sum(dim=(1,2,3))
            else:
                log_pgivenz = log_pgivenz.sum(dim=(1,2,3))

            bound = log_pgivenz - kl
            vae_bound = bound.mean()
            iwae_bound = vae_bound
        else:
            K = n_sample
            samples_t = samples.unsqueeze(0).expand(K, -1, -1, -1, -1)
            log_pgivenz = output_dist.log_prob(samples_t)
            if mask is not None:
                mask = mask.unsqueeze(0)
                log_pgivenz = (log_pgivenz * mask).sum(dim=(2,3,4))
            else:
                log_pgivenz = log_pgivenz.sum(dim=(2,3,4))
            _bound = log_pgivenz - kl
            iwae_bound = torch.logsumexp(_bound, dim=0) - math.log(K)
            iwae_bound = iwae_bound.mean()
            vae_bound = _bound.mean()

        output_dict = {
            'q_mean': mu,
            'q_log_var': log_var,
            'latents': z, 'q_dist': q,
            'logits': logits,
            'output_dist': output_dist,
            'kl': kl,
            'likelihood': log_pgivenz,
            'vae_bound': vae_bound,
            'iwae_bound': iwae_bound
        }

        return output_dict