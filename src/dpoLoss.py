import torch
import torch.nn as nn
import torch.nn.functional as F

class DPOLoss(nn.Module):
    def __init__(self, beta, method, lambda_dpop=0.0, lambda_shift=0.0):
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
        self.lambda_shift = lambda_shift
        
        # Validate method selection
        valid_methods = ['dpo', 'dpop', 'dposhift', 'dpopshift']
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
            logits = logits.logits # (B, L, V)

        # Labels are the inputs shifted by one
        labels = labels[:, 1:].clone()

        # Truncate logits to match the labels num_tokens
        logits = logits[:, :-1, :] # (B, L-1, V)

        # Ensure logits are of shape (batch_size, num_tokens, vocab_size) and not nan/inf
        if logits.shape[1] != labels.shape[1]:
            print(f"Shape mismatch: logits={logits.shape}, labels={labels.shape}")
            raise ValueError(f"Logits shape {logits.shape} does not match labels shape {labels.shape}")
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"NaN/Inf detected in logits")
            raise ValueError("Logits contain NaN or Inf values.")

        # token level log probabilities
        log_probs = F.log_softmax(logits, dim=-1) # (B, L-1, V)

        # Gather the log probabilities for the actual labels
        selected_log_probs = torch.gather(
            input=log_probs,
            dim=-1,
            index=labels.unsqueeze(-1)
        ).squeeze(-1) # (B, L-1)

        if selection_mask is not None:
            mask = selection_mask[:, 1:].clone()

            # Apply the mask to filter out padding tokens
            selected_log_probs = selected_log_probs * mask

            if mask.sum(-1).eq(0).any():
                raise ValueError("Mask contains all zeros, which may lead to division by zero.")
            # Calculate the average log probability excluding padding tokens
            # This averages over the tokens, so the shape is (batch_size, num_tokens)
            mean_log_prob = selected_log_probs.sum(-1) / mask.sum(-1)
            return mean_log_prob

        else:
            return selected_log_probs.mean(-1) # mean over the tokens

    def compute_dpo_loss(
            self, 
            model_chosen_logprobs,       # Policy model log P(chosen)
            model_rejected_logprobs,     # Policy model log P(rejected)
            reference_chosen_logprobs,   # Reference model log P(chosen)
            reference_rejected_logprobs, # Reference model log P(rejected)
    ):
        """
        Computes the DPO loss.

        Args:
            model_chosen_logprobs: Log probabilities of the policy model for chosen responses.
            model_rejected_logprobs: Log probabilities of the policy model for rejected responses.
            reference_chosen_logprobs: Log probabilities of the reference model for chosen responses.
            reference_rejected_logprobs: Log probabilities of the reference model for rejected responses.

        Returns:
            loss (Tensor): Scalar, the mean loss over the batch.
            chosen_rewards (Tensor): Scalar, average log‑prob ratio (model vs. reference) on chosen responses.
            rejected_rewards (Tensor): Scalar, average log‑prob ratio (model vs. reference) on rejected responses.
        
        Supports:
            - dpo:     Direct Preference Optimization
            - dpop:    DPO-Positive (adds penalty to maintain preferred-likelihood)
            - dposhift:   Shifted variant of DPO
            - dpopshift:  Shifted + positive‑penalty variant
        """
        # Compute log probability differences
        pi_diff = model_chosen_logprobs - model_rejected_logprobs
        ref_diff = reference_chosen_logprobs - reference_rejected_logprobs
        logits = pi_diff - ref_diff
        # print(f"Logits: {logits.mean().item():.4f}")
        # logits = (logits - logits.mean()) / (logits.std() + 1e-8)  # Normalize logits
        
        # Apply different methods for calculating the final loss
        if self.method == 'dpo':
            # Standard DPO loss
            losses = -F.logsigmoid(self.beta * logits)
        
        # DPO-Positive: DPO with preferred completion likelihood penalty
        elif self.method == 'dpop':
            # max(0, log(reference_chosen / model_chosen)) or equivalently max(0, reference_chosen_logprobs - model_chosen_logprobs)
            dpop_term = torch.maximum(
                torch.zeros_like(reference_chosen_logprobs),
                reference_chosen_logprobs - model_chosen_logprobs
            )
            modified_logits = logits - self.lambda_dpop * dpop_term

            losses = -F.logsigmoid(self.beta * modified_logits)
        
        elif self.method == 'dposhift':
            phi_w = model_chosen_logprobs - reference_chosen_logprobs
            phi_l = model_rejected_logprobs - reference_rejected_logprobs
            logits_shift = phi_w - self.lambda_shift * phi_l

            losses = -F.logsigmoid(self.beta * logits_shift)

        elif self.method == 'dpopshift':
            phi_w = model_chosen_logprobs - reference_chosen_logprobs
            phi_l = model_rejected_logprobs - reference_rejected_logprobs
            logits_shift = phi_w - self.lambda_shift * phi_l

            dpop_term = torch.maximum(
                torch.zeros_like(reference_chosen_logprobs),
                reference_chosen_logprobs - model_chosen_logprobs
            )
            modified_logits = logits_shift - self.lambda_dpop * dpop_term

            losses = -F.logsigmoid(self.beta * modified_logits)
        
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

        # Compute the DPO loss
        loss, chosen_rewards, rejected_rewards = self.compute_dpo_loss(
            model_chosen_logprobs=policy_chosen_log_probas,
            model_rejected_logprobs=policy_rejected_log_probas,
            reference_chosen_logprobs=ref_chosen_log_probas,
            reference_rejected_logprobs=ref_rejected_log_probas
        )
        # print(f"Logits mean: {((policy_chosen_log_probas - policy_rejected_log_probas) - (ref_chosen_log_probas - ref_rejected_log_probas)).mean().item():.4f}")
        return loss, chosen_rewards, rejected_rewards
    
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
        if self.method in ['dpop', 'dpopshift']:
            metrics["dpop_term"] = 0.0
        
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
                if self.method in ['dpop', 'dpopshift']:
                    # This requires computing log probabilities again and extracting components
                    # Similar to what's in compute_dpo_loss_batch but without gradients
                    
                    # Extract log probabilities
                    policy_chosen_log_probas = self.compute_logprobs(
                        logits=policy_model(batch["chosen"]),
                        labels=batch["chosen"],
                        selection_mask=batch["chosen_mask"]
                    )
                    reference_chosen_log_probas = self.compute_logprobs(
                        logits=reference_model(batch["chosen"]),
                        labels=batch["chosen"],
                        selection_mask=batch["chosen_mask"]
                    )
                    
                    # Calculate specific components as needed
                    if self.method in ['dpop', 'dpopshift']:
                        dpop_term = torch.maximum(
                            torch.zeros_like(reference_chosen_log_probas),
                            reference_chosen_log_probas - policy_chosen_log_probas
                        ).mean().item()
                        metrics["dpop_term"] += dpop_term
                    
            else:
                break

        # Calculate averages
        for key in metrics:
            metrics[key] /= num_batches
        
        return metrics
    
    def evaluate_dpo_loss_loader(self, policy_model, reference_model, train_loader=None, val_loader=None, eval_iter=5):
        """
        Compute the DPO loss for the training and/or validation dataset based on provided loaders.
        For advanced methods (dpop), also track the individual loss components.
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
        if self.method in ['dpop', 'dpopshift']:
            res["train_dpop_term"] = float("nan")
            res["val_dpop_term"] = float("nan")
        
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
                if "dpop_term" in train_metrics and self.method in ['dpop', 'dpopshift']:
                    res["train_dpop_term"] = train_metrics["dpop_term"]

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
                if "dpop_term" in val_metrics and self.method in ['dpop', 'dpopshift']:
                    res["val_dpop_term"] = val_metrics["dpop_term"]

        policy_model.train()
        return res