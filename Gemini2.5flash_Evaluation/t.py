import json

def recalculate_averages(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    evaluations = data.get('evaluations', [])

    # 全部資料的 ref_response.average 與 policy_response.average 總和
    total_ref = 0.0
    total_policy = 0.0

    # 條件篩選 (is_complete_overall == True) 的統計
    cond_total_ref = 0.0
    cond_total_policy = 0.0
    cond_count = 0

    for ev in evaluations:
        ref_avg = ev['ref_response']['average']
        pol_avg = ev['policy_response']['average']

        total_ref += ref_avg
        total_policy += pol_avg

        if ev.get('is_complete_overall', False):
            cond_total_ref += ref_avg
            cond_total_policy += pol_avg
            cond_count += 1

    n = len(evaluations)
    if n == 0:
        raise ValueError("沒有任何評估資料！")

    # 計算平均
    overall_ref_avg = round(total_ref / n, 2)
    overall_policy_avg = round(total_policy / n, 2)

    if cond_count > 0:
        cond_ref_avg = round(cond_total_ref / cond_count, 2)
        cond_policy_avg = round(cond_total_policy / cond_count, 2)
    else:
        cond_ref_avg = cond_policy_avg = None

    return {
        'overall_average': {
            'ref_overall_average': overall_ref_avg,
            'policy_overall_average': overall_policy_avg
        },
        'overall_average_conditional': {
            'ref_overall_average': cond_ref_avg,
            'policy_overall_average': cond_policy_avg,
            'count_included_in_average': cond_count
        }
    }

if __name__ == '__main__':
    filename = 'overall_eval_sum_Llama-3.1-8B-Instruct_DPOP_content_generated_output_lr3.0e-06_b0.30_dp50.0.json'
    result = recalculate_averages(filename)
    print(json.dumps(result, ensure_ascii=False, indent=2))
