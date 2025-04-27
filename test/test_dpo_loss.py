import pytest
import torch
from src.dpoLoss import DPOLoss


# ---------- basic tests ----------------------------------------
# Test DPOLoss with basic inputs, all correct
def test_reward_accuracy_all_correct():
    mc = torch.tensor([1.0, 2.0, 3.0])
    mr = torch.tensor([0.5, 1.5, 2.5])
    rc = torch.zeros_like(mc)
    rr = torch.zeros_like(mr)

    loss_fn = DPOLoss(beta=1.0, method="dpo")
    _, _, _, acc = loss_fn.compute_dpo_loss(mc, mr, rc, rr)
    assert pytest.approx(acc.item(), rel=1e-6) == 1.0


# Test DPOLoss with basic inputs, all incorrect
def test_reward_accuracy_all_incorrect():
    mc = torch.tensor([0.1, 0.2, 0.3])
    mr = torch.tensor([0.5, 0.6, 0.7])
    rc = torch.zeros_like(mc)
    rr = torch.zeros_like(mr)

    loss_fn = DPOLoss(beta=1.0, method="dpo")
    _, _, _, acc = loss_fn.compute_dpo_loss(mc, mr, rc, rr)
    assert pytest.approx(acc.item(), rel=1e-6) == 0.0


# ---------- test compute_dpo_loss_batch --------------------------------
def test_compute_dpo_loss_batch_monkeypatch():
    """
    Fake compute_logprobs:
      call #1 → 0.9   (policy-chosen)
      call #2 → 0.1   (policy-rejected)
      call #3,4 → 0   (reference)
    reward_accuracy = β*(0.9−0) > β*(0.1−0)
    0.9 > 0.1
    accuracy should be = 1.0
    """
    class DummyModel:
        def __call__(self, x):
            return None

    batch = {
        "chosen": torch.tensor([[1], [2]]),
        "rejected": torch.tensor([[3], [4]]),
        "chosen_mask": torch.ones(2, 1, dtype=torch.bool),
        "rejected_mask": torch.ones(2, 1, dtype=torch.bool),
    }

    # through a closure to count the number of calls
    # to compute_logprobs, and return the corresponding values
    # for policy-chosen, policy-rejected, reference-chosen, reference-rejected
    # 1) policy_chosen    = [0.9, 0.9]
    # 2) policy_rejected  = [0.1, 0.1]
    # 3) reference_chosen = [0, 0]
    # 4) reference_rejected=[0, 0]
    def fake_compute_logprobs_factory():
        counter = {"i": 0}

        def fake_compute_logprobs(self, model_output, labels, mask):
            counter["i"] += 1
            if counter["i"] == 1:          # policy-chosen
                return torch.tensor([0.9, 0.9])
            elif counter["i"] == 2:        # policy-rejected
                return torch.tensor([0.1, 0.1])
            else:                          # reference-chosen / reference-rejected
                return torch.zeros(labels.shape[0])

        return fake_compute_logprobs

    DPOLoss.compute_logprobs = fake_compute_logprobs_factory()  # monkey-patch
    loss_fn = DPOLoss(beta=1.0, method="dpo")  # β=1 for simplicity

    _, _, _, acc = loss_fn.compute_dpo_loss_batch(
        batch, policy_model=DummyModel(), reference_model=DummyModel()
    )
    assert pytest.approx(acc.item(), rel=1e-6) == 1.0


def test_compute_dpo_loss_batch_with_nonzero_reference():
    """
    Simulate four calls to compute_logprobs:
      1) policy_chosen    = [0.9, 0.9]
      2) policy_rejected  = [0.2, 0.2]
      3) reference_chosen = [0.5, 0.5]
      4) reference_rejected=[0.1, 0.1]
    then compute rewards:
      chosen_reward   = β*(0.9−0.5) = β*0.4
      rejected_reward = β*(0.2−0.1) = β*0.1
    Both samples have 0.4>0.1, accuracy should be 1.0
    """
    class DummyModel:
        def __call__(self, batch):
            return None

    batch = {
        "chosen": torch.tensor([[1], [2]]),
        "rejected": torch.tensor([[3], [4]]),
        "chosen_mask": torch.ones(2, 1, dtype=torch.bool),
        "rejected_mask": torch.ones(2, 1, dtype=torch.bool),
    }

    # through a closure to count the number of calls
    # to compute_logprobs, and return the corresponding values
    # for policy-chosen, policy-rejected, reference-chosen, reference-rejected
    # 1) policy_chosen    = [0.9, 0.9]
    # 2) policy_rejected  = [0.2, 0.2]
    # 3) reference_chosen = [0.5, 0.5]
    # 4) reference_rejected=[0.1, 0.1]
    # reward_accuracy = β*(0.9−0.5) > β*(0.2−0.1)
    # 0.4 > 0.1
    # accuracy should be = 1.0
    def fake_compute_logprobs_factory():
        counter = {"i": 0}

        def fake_compute_logprobs(self, model_output, labels, mask):
            counter["i"] += 1
            if counter["i"] == 1:      # policy_chosen
                return torch.tensor([0.9, 0.9])
            elif counter["i"] == 2:    # policy_rejected
                return torch.tensor([0.2, 0.2])
            elif counter["i"] == 3:    # reference_chosen
                return torch.tensor([0.5, 0.5])
            else:                      # reference_rejected
                return torch.tensor([0.1, 0.1])
        return fake_compute_logprobs

    # Monkey-patch
    DPOLoss.compute_logprobs = fake_compute_logprobs_factory()
    loss_fn = DPOLoss(beta=0.3, method="dpo")

    _, chosen_r, rejected_r, acc = loss_fn.compute_dpo_loss_batch(
        batch, policy_model=DummyModel(), reference_model=DummyModel()
    )

    # Check if intermediate values are as expected
    assert torch.allclose(chosen_r, torch.tensor([0.12, 0.12]))  # 0.3 * 0.4 = 0.12
    assert torch.allclose(rejected_r, torch.tensor([0.03, 0.03]))  # 0.3 * 0.1 = 0.03
    # The accuracy should be 1.0 because 0.12 > 0.03 for both samples
    assert pytest.approx(acc.item(), rel=1e-6) == 1.0
