import torch
import torch.nn as nn
import torch.nn.functional as F

class DPOLoss(nn.Module):
    def __init__(self, beta=0.1):
        """
        Initializes the DPO Loss module.

        Args:
            beta (float): Scaling factor for the loss. Controls the strength of preference optimization.
        """
        super(DPOLoss, self).__init__()
        self.beta = beta

    def forward(self, 
                pi_logp_chosen: torch.Tensor, 
                pi_logp_rejected: torch.Tensor, 
                ref_logp_chosen: torch.Tensor, 
                ref_logp_rejected: torch.Tensor) -> torch.Tensor:
        """
        Computes the DPO loss.

        Args:
            pi_logp_chosen (torch.Tensor): Log probabilities from the policy model for chosen responses.
            pi_logp_rejected (torch.Tensor): Log probabilities from the policy model for rejected responses.
            ref_logp_chosen (torch.Tensor): Log probabilities from the reference model for chosen responses.
            ref_logp_rejected (torch.Tensor): Log probabilities from the reference model for rejected responses.

        Returns:
            torch.Tensor: Computed DPO loss.
        """
        # Compute log probability differences
        pi_diff = pi_logp_chosen - pi_logp_rejected
        ref_diff = ref_logp_chosen - ref_logp_rejected

        # Compute the scaled difference
        scaled_diff = self.beta * (pi_diff - ref_diff)

        # Apply sigmoid
        sigmoid_output = torch.sigmoid(scaled_diff)

        # Compute negative log likelihood
        loss = -torch.log(sigmoid_output + 1e-8)  # Adding epsilon for numerical stability

        return loss.mean()
