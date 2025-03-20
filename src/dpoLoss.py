import torch
import torch.nn as nn
import torch.nn.functional as F

class DPOLoss(nn.Module):
    def __init__(self, beta, lambda_kl):
        """
        Initializes the DPO Loss module.

        Args:
            beta (float): Scaling factor for the loss. Controls the strength of preference optimization.
            lambda_kl (float): Weight of the KL divergence penalty to prevent model drift in DPO loss.
        """
        super(DPOLoss, self).__init__()
        self.beta = beta
        self.lambda_kl = lambda_kl

    def compute_logprobs(self, logits, labels, selection_mask=None):
        """
        Compute log probabilities.

        Args:
            logits: Tensor of shape (batch_size, num_tokens, vocab_size)
            labels: Tensor of shape (batch_size, num_tokens)
            selection_mask: Tensor for shape (batch_size, num_tokens)

        Returns:
            mean_log_prob: Mean log probability excluding padding tokens.
        """

        if hasattr(logits, "logits"):
            logits = logits.logits

        # Labels are the inputs shifted by one
        labels = labels[:, 1:].clone()

        # Truncate logits to match the labels num_tokens
        logits = logits[:, :-1, :]

        log_probs = F.log_softmax(logits, dim=-1)

        # Gather the log probabilities for the actual labels
        selected_log_probs = torch.gather(
            input=log_probs,
            dim=-1,
            index=labels.unsqueeze(-1)
        ).squeeze(-1)

        if selection_mask is not None:
            mask = selection_mask[:, 1:].clone()

            # Apply the mask to filter out padding tokens
            selected_log_probs = selected_log_probs * mask

            # Calculate the average log probability excluding padding tokens
            # This averages over the tokens, so the shape is (batch_size, num_tokens)
            avg_log_prob = selected_log_probs.sum(-1) / mask.sum(-1)

            return avg_log_prob

        else:
            return selected_log_probs.mean(-1)

    def compute_dpo_loss(
            self, 
            model_chosen_logprobs, # Policy model log P(chosen)
            model_rejected_logprobs, # Policy model log P(rejected)
            reference_chosen_logprobs, # Reference model log P(chosen)
            reference_rejected_logprobs # Reference model log P(rejected)
    ):
        """
        Computes the DPO loss.

        Args:
            model_chosen_logprobs: Log probabilities of the policy model for chosen responses.
            model_rejected_logprobs: Log probabilities of the policy model for rejected responses.
            reference_chosen_logprobs: Log probabilities of the reference model for chosen responses.
            reference_rejected_logprobs: Log probabilities of the reference model for rejected responses.

        Returns:
            (Tensor, Tensor, Tensor):
            - The scalar DPO loss (with KL penalty).
            - The average "chosen" reward (model_chosen_logprobs - reference_chosen_logprobs).
            - The average "rejected" reward (model_rejected_logprobs - reference_rejected_logprobs).
        """
        # Calculate DPO logits: pi_diff - ref_diff
        pi_diff = model_chosen_logprobs - model_rejected_logprobs
        ref_diff = reference_chosen_logprobs - reference_rejected_logprobs
        logits = pi_diff - ref_diff
        print(f"Logits: {logits.mean().item():.4f}")
        # logits = (logits - logits.mean()) / (logits.std() + 1e-8)  # Normalize logits
        
        # DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf), calculate standard DPO loss: -logsigmoid(beta * logits)
        dpo_loss = -F.logsigmoid(self.beta * logits)
        # print(f"Losses: {losses.mean().item():.4f}")

        # Calculate KL penalty: D_KL(reference || model)
        # Convert reference model logprobs to probs since F.kl_div expects probs as target
        reference_chosen_probs = reference_chosen_logprobs.exp()
        reference_rejected_probs = reference_rejected_logprobs.exp()

        # F.kl_div(input=logP, target=Q) computes D_KL(Q || P)
        kl_penalty_chosen = F.kl_div(
            input=model_chosen_logprobs,     # log P(model)
            target=reference_chosen_probs,   # Q(reference)
            reduction="batchmean",
            log_target=False
        )
        
        kl_penalty_rejected = F.kl_div(
            input=model_rejected_logprobs,
            target=reference_rejected_probs,
            reduction="batchmean",
            log_target=False
        )

        # Combine both KL penalties
        kl_penalty = kl_penalty_chosen + kl_penalty_rejected
        
        # 4. Combine into final loss with weighting factor
        losses = dpo_loss + self.lambda_kl * kl_penalty
        
        # Optional values to track progress during training
        chosen_rewards = (model_chosen_logprobs - reference_chosen_logprobs).detach()
        rejected_rewards = (model_rejected_logprobs - reference_rejected_logprobs).detach()

        return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()

    def compute_dpo_loss_batch(self, batch, policy_model, reference_model):
        """
        Compute the DPO loss on an input batch.

        Args:
            batch (dict): Dictionary containing input tensors ("chosen", "rejected", "chosen_mask", "rejected_mask").
            policy_model: Model used for policy logit predictions.
            reference_model: Model used for reference logit predictions.

        Returns:
            torch.Tensor: Computed DPO loss.
        """
        # Compute log probabilities for policy model
        policy_chosen_log_probas = self.compute_logprobs(
            logits=policy_model(batch["chosen"]),
            labels=batch["chosen"],
            selection_mask=batch["chosen_mask"]
        )
        policy_rejected_log_probas = self.compute_logprobs(
            logits=policy_model(batch["rejected"]),
            labels=batch["rejected"],
            selection_mask=batch["rejected_mask"]
        )

        # Compute log probabilities for reference model
        ref_chosen_log_probas = self.compute_logprobs(
            logits=reference_model(batch["chosen"]),
            labels=batch["chosen"],
            selection_mask=batch["chosen_mask"]
        )
        ref_rejected_log_probas = self.compute_logprobs(
            logits=reference_model(batch["rejected"]),
            labels=batch["rejected"],
            selection_mask=batch["rejected_mask"]
        )

        # print(f"Policy chosen log_prob: {policy_chosen_log_probas.mean().item():.4f}")
        # print(f"Ref chosen log_prob: {ref_chosen_log_probas.mean().item():.4f}")
        # print(f"Policy rejected log_prob: {policy_rejected_log_probas.mean().item():.4f}")
        # print(f"Ref rejected log_prob: {ref_rejected_log_probas.mean().item():.4f}")

        # Compute the DPO loss
        loss, chosen_rewards, rejected_rewards = self.compute_dpo_loss(
            model_chosen_logprobs=policy_chosen_log_probas,
            model_rejected_logprobs=policy_rejected_log_probas,
            reference_chosen_logprobs=ref_chosen_log_probas,
            reference_rejected_logprobs=ref_rejected_log_probas
        )
        # print(f"Logits mean: {((policy_chosen_log_probas - policy_rejected_log_probas) - (ref_chosen_log_probas - ref_rejected_log_probas)).mean().item():.4f}")
        return loss, chosen_rewards, rejected_rewards

    def compute_dpo_loss_loader(self, data_loader, policy_model, reference_model, num_batches=None):
        """Apply compute_dpo_loss_batch to a whole data loader"""

        total_loss, total_chosen_rewards, total_rejected_rewards = 0., 0., 0.
        if len(data_loader) == 0:
            return float("nan")

        elif num_batches is None:
            num_batches = len(data_loader)
        else:
            # Reduce the number of batches to match the total number of batches in the data loader
            # if num_batches exceeds the number of batches in the data loader
            num_batches = min(num_batches, len(data_loader))
        for i, batch in enumerate(data_loader):
            if i < num_batches:
                loss, chosen_rewards, rejected_rewards = self.compute_dpo_loss_batch(
                    batch=batch,
                    policy_model=policy_model,
                    reference_model=reference_model,
                )
                total_loss += loss.item()
                total_chosen_rewards += chosen_rewards.item()
                total_rejected_rewards += rejected_rewards.item()

            else:
                break

        # calculate average
        total_loss /= num_batches
        total_chosen_rewards /= num_batches
        total_rejected_rewards /= num_batches
        return total_loss, total_chosen_rewards, total_rejected_rewards
    
    def evaluate_dpo_loss_loader(self, policy_model, reference_model, train_loader=None, val_loader=None, eval_iter=5):
        """Compute the DPO loss for the training and/or validation dataset based on provided loaders"""
        
        policy_model.eval()
        res = {
            "train_loss": float("nan"),
            "train_chosen_reward": float("nan"),
            "train_rejected_reward": float("nan"),
            "val_loss": float("nan"),
            "val_chosen_reward": float("nan"),
            "val_rejected_reward": float("nan")
        }
        
        with torch.no_grad():
            # Calculate the loss and rewards for the training dataset if train_loader is provided
            if train_loader is not None:
                train_loss, train_chosen_rewards, train_rejected_rewards = self.compute_dpo_loss_loader(
                    data_loader=train_loader,
                    policy_model=policy_model,
                    reference_model=reference_model,
                    num_batches=eval_iter
                )
                res["train_loss"] = train_loss
                res["train_chosen_reward"] = train_chosen_rewards
                res["train_rejected_reward"] = train_rejected_rewards

            # Compute the loss and rewards for the validation dataset if val_loader is provided
            if val_loader is not None:
                val_loss, val_chosen_rewards, val_rejected_rewards = self.compute_dpo_loss_loader(
                    data_loader=val_loader,
                    policy_model=policy_model,
                    reference_model=reference_model,
                    num_batches=eval_iter
                )
                res["val_loss"] = val_loss
                res["val_chosen_reward"] = val_chosen_rewards
                res["val_rejected_reward"] = val_rejected_rewards

        policy_model.train()
        return res