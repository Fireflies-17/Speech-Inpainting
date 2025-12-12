import torch
import torch.nn.functional as F
from Inpainting.dataset.km_label import ApplyKmeans


class LossFunction:
    def __init__(self, km_model_path, device='cuda:0'):
        self.device = torch.device(device)
        self.km_model_path = km_model_path
        self.all_embeds = ApplyKmeans(self.km_model_path, device=self.device).C
        self.all_embeds_t = self.all_embeds.T[None, :, :]
        self.center_ = self.all_embeds_t.squeeze(0).mean(dim=0)
        self.all_embeds_t_c = self.all_embeds_t - \
            self.center_.unsqueeze(0).unsqueeze(0)
        self.tau = 0.1
        self.targets = self.compute_targets()

    def compute_targets(self):
        similarity = F.cosine_similarity(self.all_embeds_t_c.unsqueeze(
            2), self.all_embeds_t_c.unsqueeze(0), dim=-1)
        exp_similarity = torch.exp(similarity.squeeze(0) / self.tau)
        result = torch.sum(exp_similarity, dim=-1)
        den = torch.diagonal(exp_similarity)/result
        return den

    def cos_sim(self, output, labels):
        """
        Calculate the cosine similarity-based loss for the given output and labels.

        Args:
            output (torch.Tensor): The output tensor of shape (batch_size, timesteps, features).
            labels (torch.Tensor): The target labels tensor of shape (batch_size, timesteps).

        Returns:
            torch.Tensor: The calculated loss as a scalar tensor.
            torch.Tensor: The predicted labels tensor.
        """
        reshaped_mel_ten = output.view(-1, output.shape[-1])
        target_centroids = self.all_embeds_t_c[0,
                                               labels.view(-1)].view(reshaped_mel_ten.shape)
        cosine_similarity = F.cosine_similarity(
            reshaped_mel_ten, target_centroids)-1
        loss = -cosine_similarity.sum()
        cos_sim_matrix = F.cosine_similarity(
            reshaped_mel_ten.unsqueeze(1), self.all_embeds_t_c, dim=-1)
        pred_labels = torch.argmax(cos_sim_matrix, dim=1)
        return loss, pred_labels.view(labels.shape)

    def cos_sim_target_labels(self, pred_labels, labels):
        """
        Calculate the cosine similarity between the predicted labels and the target labels.

        Args:
            pred_labels (torch.Tensor): The predicted labels tensor of shape (batch_size, timesteps).
            labels (torch.Tensor): The target labels tensor of shape (batch_size, timesteps).

        Returns:
            torch.Tensor: The calculated cosine similarity between the predicted labels and target labels.
        """
        cos_sim_pred_target = F.cosine_similarity(self.all_embeds[:, pred_labels.view(
            -1)]-self.center_.unsqueeze(1), self.all_embeds[:, labels.view(-1)]-self.center_.unsqueeze(1), dim=0)
        return cos_sim_pred_target

    def MSELoss(self, outputs, labels):
        """
        Calculate the Mean Squared Error (MSE) loss and predict the labels.

        Args:
            outputs (torch.Tensor): The output tensor of shape (batch_size, timesteps, features).
            labels (torch.Tensor): The target labels tensor of shape (batch_size, timesteps).

        Returns:
            torch.Tensor: The calculated MSE loss as a scalar tensor.
            torch.Tensor: The predicted labels tensor of shape (batch_size).
        """
        outputs = outputs.view(-1, outputs.shape[-1])
        targets = self.all_embeds[:, labels.view(-1)].T

        pred_labels = torch.argmin(torch.cdist(
            outputs, self.all_embeds.T, p=2), dim=1).view(labels.shape)
        loss = F.mse_loss(outputs, targets, reduction='sum')
        return loss, pred_labels

    def soft_loss(self, outputs, labels):
        """
        Calculate the soft cross-entropy loss and predict the labels.

        Args:
            outputs (torch.Tensor): The output tensor of shape (batch_size, tiemsteps, features).
            labels (torch.Tensor): The target labels tensor of shape (batch_size, timesteps).

        Returns:
            torch.Tensor: The calculated soft cross-entropy loss as a scalar tensor.
            torch.Tensor: The predicted labels tensor of shape (batch_size).

        """
        outputs = outputs.view(-1, outputs.shape[-1])
        loss = F.cross_entropy(
            outputs, labels.view(-1).to(torch.int64), reduction='sum')
        pred_labels = torch.argmax(outputs, dim=1)
        return loss, pred_labels.view(labels.shape)
