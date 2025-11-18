import torch
import torch.nn as nn
from transformers import RobertaModel, AutoConfig
from transformers.models.bart.modeling_bart import BartDecoder


def sample_norm(mu, log_sigma):
    std_z = torch.randn(mu.size())
    if mu.is_cuda:
        std_z = std_z.cuda()

    return mu + torch.exp(log_sigma) * std_z


class _Inference(nn.Sequential):
    def __init__(self, num_input_channels, latent_dim, disc_variable=True):
        super(_Inference, self).__init__()
        if disc_variable:
            self.add_module('fc', nn.Linear(num_input_channels, num_input_channels // 2))
            self.add_module('relu', nn.ReLU())
            self.add_module('fc2', nn.Linear(num_input_channels // 2, latent_dim))
            self.add_module('log_softmax', nn.LogSoftmax(dim=1))
        else:
            self.add_module('fc', nn.Linear(num_input_channels, latent_dim))


class Sample(nn.Module):
    def __init__(self, temperature):
        super(Sample, self).__init__()
        self._temperature = temperature

    def forward(self, norm_mean, norm_log_sigma, disc_log_alpha, disc_label=None, mixup=False, disc_label_mixup=None,
                mixup_lam=None):
        batch_size = norm_mean.size(0)
        latent_sample = list([])
        latent_sample.append(sample_norm(norm_mean, norm_log_sigma))
        if disc_label is not None:
            disc_label = torch.argmax(disc_label, dim=1)
            if mixup:
                c_a = torch.zeros(disc_log_alpha.size()).cuda()
                c_a = c_a.scatter(1, disc_label.view(-1, 1), 1)
                c_b = torch.zeros(disc_log_alpha.size()).cuda()
                c_b = c_b.scatter(1, disc_label_mixup.view(-1, 1), 1)
                c = mixup_lam * c_a + (1 - mixup_lam) * c_b
            else:
                disc_label = disc_label.long()
                c = torch.zeros(disc_log_alpha.size()).cuda()
                c = c.scatter(1, disc_label, 1)
            latent_sample.append(c)
        else:
            latent_sample.append(self._sample_gumbel_softmax(disc_log_alpha))
        latent_sample = torch.cat(latent_sample, dim=1)
        dim_size = latent_sample.size(1)
        latent_sample = latent_sample.view(batch_size, dim_size, 1, 1)
        return latent_sample

    def _sample_gumbel_softmax(self, log_alpha):
        EPS = 1e-12
        unif = torch.rand(log_alpha.size()).cuda()
        gumbel = -torch.log(-torch.log(unif + EPS) + EPS)
        logit = (log_alpha + gumbel) / self._temperature
        return torch.softmax(logit, dim=1)


class CATCH(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        classifier_dropout = 0.2
        num_targets = 2
        num_class = 2
        self.device = device
        self.temperature = nn.Parameter(torch.tensor(1.0, requires_grad=True, device=device))

        self.encoder = RobertaModel.from_pretrained('FacebookAI/roberta-base')
        for param in self.encoder.base_model.parameters():
            param.requires_grad = False
        self.config = AutoConfig.from_pretrained('FacebookAI/roberta-base')

        self.cmi = torch.nn.Parameter(torch.rand(1, requires_grad=True, device=device))

        hidden_size = self.config.hidden_size

        self.decoder = BartDecoder.from_pretrained('facebook/bart-base')
        self.lm_head = nn.Linear(hidden_size, self.config.vocab_size)
        self.lm_loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.x_sigma = 1
        self.relu = nn.ReLU()

        self.continuous_inference = nn.Sequential()
        self.disc_latent_inference = nn.Sequential()
        conti_mean_inf_module = _Inference(num_input_channels=hidden_size,
                                           latent_dim=hidden_size - num_targets,
                                           disc_variable=False)
        conti_logsigma_inf_module = _Inference(num_input_channels=hidden_size,
                                               latent_dim=hidden_size - num_targets,
                                               disc_variable=False)
        self.continuous_inference.add_module("mean", conti_mean_inf_module)
        self.continuous_inference.add_module("log_sigma", conti_logsigma_inf_module)

        dic_inf = _Inference(num_input_channels=hidden_size, latent_dim=num_targets,
                             disc_variable=True)
        self.disc_latent_inference = dic_inf
        self.sample = Sample(temperature=self.temperature)

        self.kl_beta_c = 0.05

        self.disc_log_prior_param = torch.log(
            torch.tensor([1 / num_targets for _ in range(num_targets)]).view(1, -1).float().cuda())

        self.reconstructor = nn.Linear(hidden_size, hidden_size)
        self.batch_norm = nn.BatchNorm1d(num_features=hidden_size)
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_size - num_targets, hidden_size - num_targets),
            nn.Tanh(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_size - num_targets, num_class),
        )
        self.classify_loss_fn = nn.CrossEntropyLoss()

    def get_lm_loss(self, logits, labels, masks):
        loss = self.lm_loss_fn(logits.view(-1, self.config.vocab_size), labels.view(-1))
        masked_loss = loss * masks.view(-1)
        return torch.mean(masked_loss)

    def forward(self, data):
        input_ids, labels, mask_ids, batch_size = data
        inputs = input_ids['input_ids']
        mask = input_ids['attention_mask']
        decoder_inputs = mask_ids['input_ids']
        decoder_masks = mask_ids['attention_mask']
        decoder_labels = input_ids['input_ids']

        x = self.encoder(inputs, attention_mask=mask)[0]
        x = x[:, 0, :].squeeze(1)
        if batch_size != 1:
            x = self.batch_norm(x)

        norm_mean = self.continuous_inference.mean(x)
        norm_log_sigma = self.continuous_inference.log_sigma(x)
        disc_log_alpha = self.disc_latent_inference(x)
        latent_sample = self.sample(norm_mean, norm_log_sigma, disc_log_alpha)

        latent_sample = latent_sample.squeeze(-1).squeeze(-1)
        decoder_hidden = self.reconstructor(latent_sample)

        decoder_outputs = self.decoder(
            input_ids=decoder_inputs,
            attention_mask=decoder_masks,
            encoder_hidden_states=decoder_hidden.unsqueeze(1))

        lm_logits = self.lm_head(decoder_outputs.last_hidden_state)

        reconstruct_loss = self.get_lm_loss(lm_logits, decoder_labels,
                                            decoder_masks)

        z_mean_sq = norm_mean * norm_mean
        z_log_sigma_sq = 2 * norm_log_sigma
        z_sigma_sq = torch.exp(z_log_sigma_sq)
        continuous_kl_loss = 0.5 * torch.sum(z_mean_sq + z_sigma_sq - z_log_sigma_sq - 1) / batch_size

        prior_kl_loss_l = self.kl_beta_c * torch.abs(continuous_kl_loss - self.cmi)
        elbo_loss_l = reconstruct_loss + prior_kl_loss_l

        classify_sample = sample_norm(norm_mean, norm_log_sigma)

        preds = self.classifier(classify_sample)
        classify_loss = self.classify_loss_fn(preds, labels)

        return preds, elbo_loss_l[0] + classify_loss * 20, labels, batch_size
