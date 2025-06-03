import json

def append_evaluation_summaries(json_path, output_path=None):
    """
    Read the JSON evaluations file, compute summary statistics for both
    ref_response and policy_response:
      - 'complete': only among is_complete == True
      - 'overall': across all evaluations
    Append them under 'ref_summary' and 'policy_summary'.
    """
    # Load the original JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    evaluations = data.get('evaluations', [])
    total_evals = len(evaluations)

    def compute_summary(key):
        """
        Compute:
          - total: total number of evaluations
          - total_correct_count: count of all where Factual_Accuracy == 5
          - total_accuracy_rate: total_correct_count / total
          - total_complete: count where is_complete == True
          - correct_count: among complete, count where Factual_Accuracy == 5
          - match_count: among complete, count where Alignment_Answer == 4
          - accuracy_rate: correct_count / total_complete
          - match_rate: match_count / total_complete
        """
        # Overall counts
        total_correct = sum(1 for ev in evaluations if ev[key].get('Factual_Accuracy') == 5)
        total_accuracy = round(total_correct / total_evals, 4) if total_evals else 0.0

        # Complete-only counts
        complete = [ev for ev in evaluations if ev[key].get('is_complete', False)]
        total_complete = len(complete)
        correct = sum(1 for ev in complete if ev[key].get('Factual_Accuracy') == 5)
        match = sum(1 for ev in complete if ev[key].get('Alignment_Answer') == 4)
        accuracy = round(correct / total_complete, 4) if total_complete else 0.0
        mrate = round(match / total_complete, 4) if total_complete else 0.0
        matchScore5 = sum(1 for ev in evaluations if ev[key].get('Alignment_Answer') == 5)
        matchScore4 = sum(1 for ev in evaluations if ev[key].get('Alignment_Answer') == 4)
        matchScore3 = sum(1 for ev in evaluations if ev[key].get('Alignment_Answer') == 3)
        matchScore2 = sum(1 for ev in evaluations if ev[key].get('Alignment_Answer') == 2)
        matchScore1 = sum(1 for ev in evaluations if ev[key].get('Alignment_Answer') == 1)

        all_factural_mean = sum(ev[key].get('Factual_Accuracy', 0) for ev in evaluations) / total_evals if total_evals else 0.0
        all_alignment_mean = sum(ev[key].get('Alignment_Answer', 0) for ev in evaluations) / total_evals if total_evals else 0.0
        only_complete_factural_mean = sum(ev[key].get('Factual_Accuracy', 0) for ev in complete) / total_complete if total_complete else 0.0
        only_complete_alignment_mean = sum(ev[key].get('Alignment_Answer', 0) for ev in complete) / total_complete if total_complete else 0.0

        return {
            'total': total_evals,
            'total_correct_count': total_correct,
            'total_accuracy_rate': total_accuracy,
            'total_complete': total_complete,
            'correct_count': correct,
            'match_count': match,
            'accuracy_rate': accuracy,
            'match_rate': mrate,
            'matchScore5': matchScore5,
            'matchScore4': matchScore4,
            'matchScore3': matchScore3,
            'matchScore2': matchScore2,
            'matchScore1': matchScore1,
            'all_factural_mean': all_factural_mean,
            'all_alignment_mean': all_alignment_mean,
            'only_complete_factural_mean': only_complete_factural_mean,
            'only_complete_alignment_mean': only_complete_alignment_mean
        }

    # Compute and append both summaries
    data['ref_summary']    = compute_summary('ref_response')
    data['policy_summary'] = compute_summary('policy_response')

    # Write back (overwrite original unless an output_path is provided)
    out_path = output_path or json_path
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Appended ref_summary and policy_summary to {out_path}")

if __name__ == '__main__':
    append_evaluation_summaries('overall_eval_sum_Llama-3.1-8B-Instruct_DPO_html_generated_output_lr3.0e-06_b0.30.json')
