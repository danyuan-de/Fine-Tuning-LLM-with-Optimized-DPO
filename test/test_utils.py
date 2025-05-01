import os
import csv
import tempfile
import src.utils as utils

def test_get_output_filename():
    # Given input
    method = "dpo"
    file = "sample_html_dataset.json"
    model = "meta-llama/Llama-3.1-8B-Instruct"
    label = "batch_records"
    learning_rate = 5e-7
    beta = 0.3

    # When
    filename = utils.get_output_filename(
        method=method,
        file=file,
        model=model,
        label=label,
        learning_rate=learning_rate,
        beta=beta,
        typename="csv"
    )

    # Then
    assert filename.endswith(".csv"), "Filename should end with .csv"
    assert "Llama-3.1-8B-Instruct_DPO_html_batch_records" in filename, "Prefix should match"
    assert "lr5.0e-07" in filename, "Learning rate should be embedded"
    assert "b0.30" in filename, "Beta should be embedded"

def test_log_result_csv():
    # Create a temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        test_filename = os.path.join(tmpdir, "test_log.csv")

        # Given one row of metrics
        metrics = {
            "epoch_frac": 0.5,
            "step": 100,
            "train_loss": 0.123,
            "val_loss": 0.456,
            "train_reward_margin": 1.23,
            "val_reward_margin": 1.56,
            "train_reward_accuracy": 0.87,
            "val_reward_accuracy": 0.89,
        }

        # When
        utils.log_result_csv(test_filename, **metrics)

        # Then
        assert os.path.exists(test_filename), "CSV file should be created"

        with open(test_filename, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)

            assert len(rows) == 1, "There should be exactly one row"
            assert rows[0]["epoch_frac"] == str(metrics["epoch_frac"]), "epoch_frac should match"
            assert rows[0]["train_loss"] == str(metrics["train_loss"]), "train_loss should match"
            assert rows[0]["val_reward_accuracy"] == str(metrics["val_reward_accuracy"]), "val_reward_accuracy should match"

