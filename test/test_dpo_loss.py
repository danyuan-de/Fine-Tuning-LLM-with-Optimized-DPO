import pytest
import torch
import torch.nn.functional as F
from src.dpoLoss import DPOLoss # Assuming your DPOLoss class is at this path

# ---------- basic tests ----------------------------------------
# Test DPOLoss with basic inputs, all correct
def test_reward_accuracy_all_correct():
    mc = torch.tensor([1.0, 2.0, 3.0])
    mr = torch.tensor([0.5, 1.5, 2.5])
    rc = torch.zeros_like(mc)
    rr = torch.zeros_like(mr)

    # Instantiate DPOLoss to call compute_dpo_loss
    loss_fn = DPOLoss(beta=1.0, method="dpo")
    _, _, _, acc = loss_fn.compute_dpo_loss(mc, mr, rc, rr)
    assert pytest.approx(acc.item(), rel=1e-6) == 1.0


# Test DPOLoss with basic inputs, all incorrect
def test_reward_accuracy_all_incorrect():
    mc = torch.tensor([0.1, 0.2, 0.3])
    mr = torch.tensor([0.5, 0.6, 0.7])
    rc = torch.zeros_like(mc)
    rr = torch.zeros_like(mr)

    # Instantiate DPOLoss to call compute_dpo_loss
    loss_fn = DPOLoss(beta=1.0, method="dpo")
    _, _, _, acc = loss_fn.compute_dpo_loss(mc, mr, rc, rr)
    assert pytest.approx(acc.item(), rel=1e-6) == 0.0


# ---------- test compute_dpo_loss_batch --------------------------------
# To avoid conflicts with the tests below that directly test compute_logprobs,
# the monkey patching here needs to ensure it only affects these two test functions.
# Using pytest's monkeypatch fixture is safer.

def test_compute_dpo_loss_batch_monkeypatch(monkeypatch):
    """
    Use the monkeypatch fixture to locally replace compute_logprobs.
    Fake compute_logprobs:
      call #1 → 0.9   (policy-chosen)
      call #2 → 0.1   (policy-rejected)
      call #3,4 → 0   (reference)
    reward_accuracy = β*(0.9−0) > β*(0.1−0) -> 0.9 > 0.1 -> True
    accuracy should be = 1.0
    """
    class DummyModel:
        def __call__(self, x):
            # Return a fake output compatible with logits, e.g., containing .logits attribute
            # Or ensure fake_compute_logprobs doesn't depend on model_output
            return torch.randn(2, 1, 10) # Assume B=2, L=1, V=10

    batch = {
        "chosen": torch.tensor([[1], [2]]),
        "rejected": torch.tensor([[3], [4]]),
        "chosen_mask": torch.ones(2, 1, dtype=torch.bool),
        "rejected_mask": torch.ones(2, 1, dtype=torch.bool),
    }

    call_counter = {"i": 0}
    # Match the new function signature
    def fake_compute_logprobs(self, logits, labels, selection_mask=None, average_log_probs=False):
        call_counter["i"] += 1
        if call_counter["i"] == 1:          # policy-chosen
            return torch.tensor([0.9, 0.9])
        elif call_counter["i"] == 2:        # policy-rejected
            return torch.tensor([0.1, 0.1])
        else:                          # reference-chosen / reference-rejected
            return torch.zeros(labels.shape[0]) # Return a tensor matching batch size

    # Use pytest's monkeypatch fixture
    monkeypatch.setattr(DPOLoss, "compute_logprobs", fake_compute_logprobs)

    loss_fn = DPOLoss(beta=1.0, method="dpo")  # β=1 for simplicity

    (_, _, _, acc, _, _, _, _) = loss_fn.compute_dpo_loss_batch(
        batch, policy_model=DummyModel(), reference_model=DummyModel()
    )
    assert pytest.approx(acc.item(), rel=1e-6) == 1.0


def test_compute_dpo_loss_batch_with_nonzero_reference(monkeypatch):
    """
    Use the monkeypatch fixture.
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
         def __call__(self, x):
            return torch.randn(2, 1, 10) # Assume B=2, L=1, V=10

    batch = {
        "chosen": torch.tensor([[1], [2]]),
        "rejected": torch.tensor([[3], [4]]),
        "chosen_mask": torch.ones(2, 1, dtype=torch.bool),
        "rejected_mask": torch.ones(2, 1, dtype=torch.bool),
    }

    call_counter = {"i": 0}
    # Match the new function signature
    def fake_compute_logprobs(self, logits, labels, selection_mask=None, average_log_probs=False):
        call_counter["i"] += 1
        if call_counter["i"] == 1:      # policy_chosen
            return torch.tensor([0.9, 0.9])
        elif call_counter["i"] == 2:    # policy_rejected
            return torch.tensor([0.2, 0.2])
        elif call_counter["i"] == 3:    # reference_chosen
            return torch.tensor([0.5, 0.5])
        else:                      # reference_rejected
            return torch.tensor([0.1, 0.1])

    # Use pytest's monkeypatch fixture
    monkeypatch.setattr(DPOLoss, "compute_logprobs", fake_compute_logprobs)

    loss_fn = DPOLoss(beta=0.3, method="dpo")
    (_, chosen_r, rejected_r, acc, _, _, _, _)  = loss_fn.compute_dpo_loss_batch(
        batch, policy_model=DummyModel(), reference_model=DummyModel()
    )

    # Check if intermediate values are as expected
    # Note: Comparison here needs torch.allclose or pytest.approx
    assert torch.allclose(chosen_r, torch.tensor(0.3 * 0.4)) # β * (0.9 - 0.5)
    assert torch.allclose(rejected_r, torch.tensor(0.3 * 0.1)) # β * (0.2 - 0.1)
    # The accuracy should be 1.0 because 0.12 > 0.03 for both samples
    assert pytest.approx(acc.item(), rel=1e-6) == 1.0


# ---------- New Tests (Directly testing compute_logprobs calculation) ----------
@pytest.fixture(scope="module")
def logprobs_test_data():
    """Prepare common data needed for compute_logprobs tests."""
    batch_size = 2
    seq_len = 5
    vocab_size = 10
    # Use predictable logits, higher index means higher logit value
    # Logits shape: (B, L, V) = (2, 5, 10)
    logits = torch.arange(batch_size * seq_len * vocab_size, dtype=torch.float32).view(
        batch_size, seq_len, vocab_size
    )
    # Make logits for correct tokens slightly higher to ensure stable log_softmax
    logits[0, 0, 1] += 5 # Item 0, Seq 1 -> label 1
    logits[0, 1, 2] += 5 # Item 0, Seq 2 -> label 2
    logits[0, 2, 3] += 5 # Item 0, Seq 3 -> label 3
    logits[0, 3, 4] += 5 # Item 0, Seq 4 -> label 4 (padding in masked test)
    logits[1, 0, 5] += 5 # Item 1, Seq 1 -> label 5
    logits[1, 1, 6] += 5 # Item 1, Seq 2 -> label 6
    logits[1, 2, 7] += 5 # Item 1, Seq 3 -> label 7
    logits[1, 3, 0] += 5 # Item 1, Seq 4 -> label 0 (padding)

    # Labels shape: (B, L) = (2, 5)
    # Token 0 is used as padding token
    labels_with_padding = torch.tensor([
        [10, 1, 2, 3, 4],  # No padding
        [10, 5, 6, 7, 0]   # Last token is padding
    ], dtype=torch.long)
    mask_with_padding = torch.tensor([
        [1, 1, 1, 1, 1],   # All valid
        [1, 1, 1, 1, 0]    # Last token is masked (padding)
    ], dtype=torch.bool)

    labels_no_padding = torch.tensor([
        [10, 1, 2, 3, 4],
        [10, 5, 6, 7, 8]
    ], dtype=torch.long)

    # Manually calculate expected log probs (after slicing)
    # Logits used: logits[:, :-1, :] -> shape (2, 4, 10)
    # Labels used: labels_...[:, 1:] -> shape (2, 4)
    sliced_logits = logits[:, :-1, :]
    log_probs_all = F.log_softmax(sliced_logits, dim=-1)

    # For labels_with_padding
    expected_logp_pad_item0 = log_probs_all[0, 0, 1] + log_probs_all[0, 1, 2] + log_probs_all[0, 2, 3] + log_probs_all[0, 3, 4]
    expected_logp_pad_item1 = log_probs_all[1, 0, 5] + log_probs_all[1, 1, 6] + log_probs_all[1, 2, 7] # Token 0 is masked

    # For labels_no_padding
    expected_logp_nopad_item0 = log_probs_all[0, 0, 1] + log_probs_all[0, 1, 2] + log_probs_all[0, 2, 3] + log_probs_all[0, 3, 4]
    expected_logp_nopad_item1 = log_probs_all[1, 0, 5] + log_probs_all[1, 1, 6] + log_probs_all[1, 2, 7] + log_probs_all[1, 3, 8]

    # Create a DPOLoss instance to call compute_logprobs
    # beta and method here don't affect the compute_logprobs test
    loss_module = DPOLoss(beta=0.1, method="dpo")

    return {
        "loss_module": loss_module,
        "logits": logits,
        "labels_with_padding": labels_with_padding,
        "mask_with_padding": mask_with_padding,
        "labels_no_padding": labels_no_padding,
        "expected_logp_pad_item0": expected_logp_pad_item0,
        "expected_logp_pad_item1": expected_logp_pad_item1,
        "expected_logp_nopad_item0": expected_logp_nopad_item0,
        "expected_logp_nopad_item1": expected_logp_nopad_item1,
        "batch_size": batch_size
    }


def test_compute_logprobs_sum_with_mask(logprobs_test_data):
    """Test sum calculation with padding mask (default behavior)."""
    data = logprobs_test_data
    log_prob = data["loss_module"].compute_logprobs(
        data["logits"], data["labels_with_padding"], data["mask_with_padding"], average_log_probs=False
    )
    assert log_prob.shape == (data["batch_size"],)
    assert pytest.approx(log_prob[0].item(), rel=1e-5) == data["expected_logp_pad_item0"].item()
    assert pytest.approx(log_prob[1].item(), rel=1e-5) == data["expected_logp_pad_item1"].item()

def test_compute_logprobs_average_with_mask(logprobs_test_data):
    """Test average calculation with padding mask."""
    data = logprobs_test_data
    log_prob = data["loss_module"].compute_logprobs(
        data["logits"], data["labels_with_padding"], data["mask_with_padding"], average_log_probs=True
    )
    assert log_prob.shape == (data["batch_size"],)
    assert pytest.approx(log_prob[0].item(), rel=1e-5) == data["expected_logp_pad_item0"].item() / 4.0 # Item 0 has 4 valid tokens
    assert pytest.approx(log_prob[1].item(), rel=1e-5) == data["expected_logp_pad_item1"].item() / 3.0 # Item 1 has 3 valid tokens

def test_compute_logprobs_sum_without_mask(logprobs_test_data):
    """Test sum calculation without a mask."""
    data = logprobs_test_data
    log_prob = data["loss_module"].compute_logprobs(
        data["logits"], data["labels_no_padding"], selection_mask=None, average_log_probs=False
    )
    assert log_prob.shape == (data["batch_size"],)
    assert pytest.approx(log_prob[0].item(), rel=1e-5) == data["expected_logp_nopad_item0"].item()
    assert pytest.approx(log_prob[1].item(), rel=1e-5) == data["expected_logp_nopad_item1"].item()

def test_compute_logprobs_average_without_mask(logprobs_test_data):
    """Test average calculation without a mask."""
    data = logprobs_test_data
    log_prob = data["loss_module"].compute_logprobs(
        data["logits"], data["labels_no_padding"], selection_mask=None, average_log_probs=True
    )
    assert log_prob.shape == (data["batch_size"],)
    assert pytest.approx(log_prob[0].item(), rel=1e-5) == data["expected_logp_nopad_item0"].item() / 4.0 # Both have 4 valid tokens
    assert pytest.approx(log_prob[1].item(), rel=1e-5) == data["expected_logp_nopad_item1"].item() / 4.0

def test_compute_logprobs_average_with_all_masked(logprobs_test_data):
    """Test average calculation when all tokens are masked."""
    data = logprobs_test_data
    labels_all_pad = torch.tensor([
        [10, 0, 0, 0, 0],
        [10, 0, 0, 0, 0]
    ], dtype=torch.long)
    mask_all_pad = torch.tensor([
        [1, 0, 0, 0, 0], # Only first token (prompt) is "valid" initially
        [1, 0, 0, 0, 0]
    ], dtype=torch.bool)

    log_prob = data["loss_module"].compute_logprobs(
        data["logits"], labels_all_pad, mask_all_pad, average_log_probs=True
    )
    # Sliced mask becomes all zeros. Sum is 0. mask_sum is 0. Should return 0 / epsilon approx 0.
    assert log_prob.shape == (data["batch_size"],)
    assert pytest.approx(log_prob[0].item(), abs=1e-5) == 0.0
    assert pytest.approx(log_prob[1].item(), abs=1e-5) == 0.0

def test_compute_logprobs_sum_with_all_masked(logprobs_test_data):
    """Test sum calculation when all tokens are masked."""
    data = logprobs_test_data
    labels_all_pad = torch.tensor([
        [10, 0, 0, 0, 0],
        [10, 0, 0, 0, 0]
    ], dtype=torch.long)
    mask_all_pad = torch.tensor([
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0]
    ], dtype=torch.bool)

    log_prob = data["loss_module"].compute_logprobs(
        data["logits"], labels_all_pad, mask_all_pad, average_log_probs=False
    )
    # Sliced mask becomes all zeros. masked_log_probs becomes all zeros. Sum is 0.
    assert log_prob.shape == (data["batch_size"],)
    assert pytest.approx(log_prob[0].item(), abs=1e-5) == 0.0
    assert pytest.approx(log_prob[1].item(), abs=1e-5) == 0.0
