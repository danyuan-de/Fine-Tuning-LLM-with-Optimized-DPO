import torch
import torch.nn as nn
import torch.nn.functional as F

class DPOLoss(nn.Module):
    def __init__(self, beta, method, lambda_dpop=0.0, lambda_kl=0.0):
        """
        Initializes the DPO Loss module.

        Args:
            beta (float): Scaling factor for the loss. Controls the strength of preference optimization.
            lambda_dpop (float): Weight for DPOP term to prevent reduction of preferred completion likelihood
        """
        super(DPOLoss, self).__init__()
        self.beta = beta
        self.method = method
        self.lambda_dpop = lambda_dpop
        self.lambda_kl = lambda_kl
        
        # Validate method selection
        valid_methods = ['dpo', 'dpop', 'dpokl', 'dpopkl', 'dpocontrast']
        if self.method not in valid_methods:
            raise ValueError(f"Method '{method}' not recognized. Must be one of: {valid_methods}")

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
        # print(f"~~~~~~~~~~~~~~~~~~ Logits shape: {logits.shape} ~~~~~~~~~~~~~~~~~~")

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

            return avg_log_prob, logits

        else:
            return selected_log_probs.mean(-1), logits

    def compute_dpo_loss(
            self, 
            model_chosen_logprobs,       # Policy model log P(chosen)
            model_rejected_logprobs,     # Policy model log P(rejected)
            reference_chosen_logprobs,   # Reference model log P(chosen)
            reference_rejected_logprobs, # Reference model log P(rejected)
            model_chosen_logits,         # Policy model logits
            model_rejected_logits,       # Policy model logits
            reference_chosen_logits,     # Reference model logits
            reference_rejected_logits    # Reference model logits
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
        # Compute log probability differences
        pi_diff = model_chosen_logprobs - model_rejected_logprobs
        ref_diff = reference_chosen_logprobs - reference_rejected_logprobs
        logits = pi_diff - ref_diff
        # print(f"Logits: {logits.mean().item():.4f}")
        # logits = (logits - logits.mean()) / (logits.std() + 1e-8)  # Normalize logits
        
        # ===== Standard DPO Loss =====
        # DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        dpo_loss = -F.logsigmoid(self.beta * logits)
        
        # Apply different methods for calculating the final loss
        if self.method == 'dpo':
            # Standard DPO loss
            losses = dpo_loss
            
        elif self.method == 'dpop':
            # DPOP: DPO with preferred completion likelihood penalty
            # max(0, log(reference_chosen / model_chosen)) or equivalently max(0, reference_chosen_logprobs - model_chosen_logprobs)
            dpop_term = torch.maximum(
                torch.zeros_like(reference_chosen_logprobs),
                reference_chosen_logprobs - model_chosen_logprobs
            )
            losses = dpo_loss + self.lambda_dpop * dpop_term
            
        elif self.method == 'dpokl':
            # KL divergence penalty: prevent drift from reference model
            # Compute KL divergence between model and reference distributions
            model_chosen_logprobs_full = F.log_softmax(model_chosen_logits, dim=-1)
            reference_chosen_probs_full = F.softmax(reference_chosen_logits, dim=-1)
            model_rejected_logprobs_full = F.log_softmax(model_rejected_logits, dim=-1)
            reference_rejected_probs_full = F.softmax(reference_rejected_logits, dim=-1)

            # F.kl_div(input=logP, target=Q) computes D_KL(Q || P)
            kl_penalty_chosen = F.kl_div(
                input=model_chosen_logprobs_full,     # log P(model) 
                target=reference_chosen_probs_full,   # Q(reference)
                reduction="batchmean",
                log_target=False
            )
            
            kl_penalty_rejected = F.kl_div(
                input=model_rejected_logprobs_full,
                target=reference_rejected_probs_full,
                reduction="batchmean",
                log_target=False
            )

            # Combine both KL penalties
            kl_penalty = kl_penalty_chosen + kl_penalty_rejected
            losses = dpo_loss + self.lambda_kl * kl_penalty
        elif self.method == 'dpopkl':
            # 1. DPOP term: penalize when preferred completion likelihood is lower than reference
            dpop_term = torch.maximum(
                torch.zeros_like(reference_chosen_logprobs),
                reference_chosen_logprobs - model_chosen_logprobs
            )
            
            # 2. KL divergence penalty: prevent drift from reference model
            # Compute KL divergence between model and reference distributions
            model_chosen_logprobs_full = F.log_softmax(model_chosen_logits, dim=-1)
            reference_chosen_probs_full = F.softmax(reference_chosen_logits, dim=-1)
            model_rejected_logprobs_full = F.log_softmax(model_rejected_logits, dim=-1)
            reference_rejected_probs_full = F.softmax(reference_rejected_logits, dim=-1)
            
            # F.kl_div(input=logP, target=Q) computes D_KL(Q || P)
            kl_penalty_chosen = F.kl_div(
                input=model_chosen_logprobs_full,     # log P(model) 
                target=reference_chosen_probs_full,   # Q(reference)
                reduction="batchmean",
                log_target=False
            )
            
            kl_penalty_rejected = F.kl_div(
                input=model_rejected_logprobs_full,
                target=reference_rejected_probs_full,
                reduction="batchmean",
                log_target=False
            )
            
            kl_penalty = kl_penalty_chosen + kl_penalty_rejected
            
            # 3. Combine all terms: DPO loss + DPOP term + KL penalty
            losses = dpo_loss + self.lambda_dpop * dpop_term + self.lambda_kl * kl_penalty

        elif self.method == 'dpocontrast':
            # DPOLoss with contrastive term
            model_chosen_logprobs_full = F.log_softmax(model_chosen_logits, dim=-1)
            model_rejected_logprobs_full = F.log_softmax(model_rejected_logits, dim=-1)

            kl_chosen_rejected = F.kl_div(
                input=model_chosen_logprobs_full,      # log P(model) 
                target=model_rejected_logprobs_full.exp(),  # P(model) 
                reduction="batchmean",
                log_target=False
            )

            contrast_loss = - kl_chosen_rejected # maximize the KL divergence between chosen and rejected with negative sign

            losses = dpo_loss + self.lambda_contrast * contrast_loss

        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Track progress during training with reward metrics
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
        policy_chosen_log_probas, policy_chosen_logits = self.compute_logprobs(
            logits=policy_model(batch["chosen"]),
            labels=batch["chosen"],
            selection_mask=batch["chosen_mask"]
        )
        policy_rejected_log_probas, policy_rejected_logits = self.compute_logprobs(
            logits=policy_model(batch["rejected"]),
            labels=batch["rejected"],
            selection_mask=batch["rejected_mask"]
        )

        # Compute log probabilities for reference model
        ref_chosen_log_probas, ref_chosen_logits = self.compute_logprobs(
            logits=reference_model(batch["chosen"]),
            labels=batch["chosen"],
            selection_mask=batch["chosen_mask"]
        )
        ref_rejected_log_probas, ref_rejected_logits = self.compute_logprobs(
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
            reference_rejected_logprobs=ref_rejected_log_probas,
            model_chosen_logits=policy_chosen_logits,
            model_rejected_logits=policy_rejected_logits,
            reference_chosen_logits=ref_chosen_logits,
            reference_rejected_logits=ref_rejected_logits
        )
        # print(f"Logits mean: {((policy_chosen_log_probas - policy_rejected_log_probas) - (ref_chosen_log_probas - ref_rejected_log_probas)).mean().item():.4f}")
        return loss, chosen_rewards, rejected_rewards

    # def compute_dpo_loss_loader(self, data_loader, policy_model, reference_model, num_batches=None):
    #     """Apply compute_dpo_loss_batch to a whole data loader"""

    #     total_loss, total_chosen_rewards, total_rejected_rewards = 0., 0., 0.
    #     if len(data_loader) == 0:
    #         return float("nan")

    #     elif num_batches is None:
    #         num_batches = len(data_loader)
    #     else:
    #         # Reduce the number of batches to match the total number of batches in the data loader
    #         # if num_batches exceeds the number of batches in the data loader
    #         num_batches = min(num_batches, len(data_loader))
    #     for i, batch in enumerate(data_loader):
    #         if i < num_batches:
    #             loss, chosen_rewards, rejected_rewards = self.compute_dpo_loss_batch(
    #                 batch=batch,
    #                 policy_model=policy_model,
    #                 reference_model=reference_model,
    #             )
    #             total_loss += loss.item()
    #             total_chosen_rewards += chosen_rewards.item()
    #             total_rejected_rewards += rejected_rewards.item()

    #         else:
    #             break

    #     # calculate average
    #     total_loss /= num_batches
    #     total_chosen_rewards /= num_batches
    #     total_rejected_rewards /= num_batches
    #     return total_loss, total_chosen_rewards, total_rejected_rewards
    
    def compute_dpo_loss_loader_with_components(self, data_loader, policy_model, reference_model, num_batches=None):
        """
        Apply compute_dpo_loss_batch to a whole data loader and track all loss components.
        Returns a dictionary with all metrics including method-specific components.
        """
        metrics = {
            "loss": 0.0, 
            "chosen_reward": 0.0, 
            "rejected_reward": 0.0
        }
        
        # Initialize method-specific metrics
        if self.method in ['dpop', 'dpopkl']:
            metrics["dpop_term"] = 0.0
        
        if self.method in ['dpokl', 'dpopkl']:
            metrics["kl_penalty"] = 0.0
        
        if len(data_loader) == 0:
            return {k: float("nan") for k in metrics.keys()}

        # Determine number of batches to process
        if num_batches is None:
            num_batches = len(data_loader)
        else:
            num_batches = min(num_batches, len(data_loader))
        
        # Process batches
        for i, batch in enumerate(data_loader):
            if i < num_batches:
                # Get basic loss and rewards
                loss, chosen_rewards, rejected_rewards = self.compute_dpo_loss_batch(
                    batch=batch,
                    policy_model=policy_model,
                    reference_model=reference_model,
                )
                
                # Add to metrics
                metrics["loss"] += loss.item()
                metrics["chosen_reward"] += chosen_rewards.item()
                metrics["rejected_reward"] += rejected_rewards.item()
                
                # For component-specific tracking, we need to run parts of the loss calculation again
                if self.method in ['dpop', 'dpopkl'] or self.method in ['dpokl', 'dpopkl']:
                    # This requires computing log probabilities again and extracting components
                    # Similar to what's in compute_dpo_loss_batch but without gradients
                    
                    # Extract log probabilities
                    policy_chosen_log_probas, policy_chosen_logits = self.compute_logprobs(
                        logits=policy_model(batch["chosen"]),
                        labels=batch["chosen"],
                        selection_mask=batch["chosen_mask"]
                    )
                    
                    reference_chosen_log_probas, reference_chosen_logits = self.compute_logprobs(
                        logits=reference_model(batch["chosen"]),
                        labels=batch["chosen"],
                        selection_mask=batch["chosen_mask"]
                    )
                    
                    # Calculate specific components as needed
                    if self.method in ['dpop', 'dpopkl']:
                        dpop_term = torch.maximum(
                            torch.zeros_like(reference_chosen_log_probas),
                            reference_chosen_log_probas - policy_chosen_log_probas
                        ).mean().item()
                        metrics["dpop_term"] += dpop_term
                    
                    if self.method in ['dpokl', 'dpopkl']:
                        # Calculate KL divergence
                        reference_chosen_probs = reference_chosen_log_probas.exp()
                        
                        kl_penalty = F.kl_div(
                            input=policy_chosen_log_probas,
                            target=reference_chosen_probs,
                            reduction="batchmean",
                            log_target=False
                        ).item()
                        
                        metrics["kl_penalty"] += kl_penalty
            else:
                break

        # Calculate averages
        for key in metrics:
            metrics[key] /= num_batches
        
        return metrics
    
    def evaluate_dpo_loss_loader(self, policy_model, reference_model, train_loader=None, val_loader=None, eval_iter=5):
        """
        Compute the DPO loss for the training and/or validation dataset based on provided loaders.
        For advanced methods (dpop, dpokl, dpopkl), also track the individual loss components.
        """
        
        policy_model.eval()
        # Base metrics present for all methods
        res = {
            "train_loss": float("nan"),
            "train_chosen_reward": float("nan"),
            "train_rejected_reward": float("nan"),
            "val_loss": float("nan"),
            "val_chosen_reward": float("nan"),
            "val_rejected_reward": float("nan")
        }
        
        # Initialize method-specific metrics if needed
        if self.method in ['dpop', 'dpopkl']:
            res["train_dpop_term"] = float("nan")
            res["val_dpop_term"] = float("nan")
        
        if self.method in ['dpokl', 'dpopkl']:
            res["train_kl_penalty"] = float("nan")
            res["val_kl_penalty"] = float("nan")
        
        with torch.no_grad():
            # Calculate the loss and rewards for the training dataset if train_loader is provided
            if train_loader is not None:
                # You need to modify compute_dpo_loss_loader to return additional metrics
                train_metrics = self.compute_dpo_loss_loader_with_components(
                    data_loader=train_loader,
                    policy_model=policy_model,
                    reference_model=reference_model,
                    num_batches=eval_iter
                )
                
                # Unpack the basic metrics (loss and rewards)
                res["train_loss"] = train_metrics["loss"]
                res["train_chosen_reward"] = train_metrics["chosen_reward"]
                res["train_rejected_reward"] = train_metrics["rejected_reward"]
                
                # Add method-specific metrics if they exist
                if "dpop_term" in train_metrics and self.method in ['dpop', 'dpopkl']:
                    res["train_dpop_term"] = train_metrics["dpop_term"]
                
                if "kl_penalty" in train_metrics and self.method in ['dpokl', 'dpopkl']:
                    res["train_kl_penalty"] = train_metrics["kl_penalty"]

            # Compute the loss and rewards for the validation dataset if val_loader is provided
            if val_loader is not None:
                val_metrics = self.compute_dpo_loss_loader_with_components(
                    data_loader=val_loader,
                    policy_model=policy_model,
                    reference_model=reference_model,
                    num_batches=eval_iter
                )
                
                # Unpack the basic metrics
                res["val_loss"] = val_metrics["loss"]
                res["val_chosen_reward"] = val_metrics["chosen_reward"]
                res["val_rejected_reward"] = val_metrics["rejected_reward"]
                
                # Add method-specific metrics if they exist
                if "dpop_term" in val_metrics and self.method in ['dpop', 'dpopkl']:
                    res["val_dpop_term"] = val_metrics["dpop_term"]
                
                if "kl_penalty" in val_metrics and self.method in ['dpokl', 'dpopkl']:
                    res["val_kl_penalty"] = val_metrics["kl_penalty"]

        policy_model.train()
        return res