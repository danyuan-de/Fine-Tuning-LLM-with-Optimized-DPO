# batch_stats_distances.py
# -*- coding: utf-8 -*-
"""
一次性處理多個 JSON 檔案，計算 chosen 與 rejected 文本間的：
  - Levenshtein 距離 distance
  - 歸一化編輯距離 normalized_distance（除以 max(len(a),len(b))）
  - 加權歸一化距離 weighted_distance（除以 len(a)+len(b)）
並分檔案輸出：
  1. 含 id、distance、normalized_distance、weighted_distance 的 <原檔名>_distances.csv
  2. 各檔案的統計摘要 summary.csv
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_distance(csvfile):
    df = pd.read_csv(csvfile)

    # Filter for the two metrics
    df_mean = df[df['metric'].isin(['normalized_distance', 'weighted_distance'])].copy()

    # Extract short file names (chat, content, html, structure)
    df_mean['short_name'] = df_mean['file'].apply(
        lambda x: os.path.splitext(os.path.basename(x))[0].split('_')[-1]
    )

    # Pivot for plotting
    pivot_df = df_mean.pivot(index='short_name', columns='metric', values='mean')

    # Create the bar chart with a larger figure size
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = pivot_df.plot(kind='bar', ax=ax)

    # Make x-axis labels horizontal
    ax.set_xticklabels(pivot_df.index, rotation=0)

    # Add titles and axis labels
    ax.set_title('Mean Distance Comparison')
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Mean Distance')

    # Move the legend outside the plot area
    ax.legend(title='Metric', loc='upper left', bbox_to_anchor=(1.02, 1))

    # Annotate each bar with its numeric value
    for container in bars.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)

    plt.tight_layout()

    # Save the figure
    plt.savefig('../distance/mean_distance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def levenshtein_distance(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            )
    return dp[n][m]

def normalized_distance(a: str, b: str) -> float:
    d = levenshtein_distance(a, b)
    L = max(len(a), len(b))
    return d/L if L>0 else 0.0

def weighted_distance(a: str, b: str) -> float:
    d = levenshtein_distance(a, b)
    Lsum = len(a) + len(b)
    return d/Lsum if Lsum>0 else 0.0

def main():
    # JSON files
    input_files = [
        "../data/physics_qa_content.json",
        "../data/physics_qa_structure.json",
        "../data/physics_qa_html.json",
        "../data/physics_qa_chat.json",
    ]

    # File save folder
    folder = "../distance"
    if not os.path.exists(folder):
        os.makedirs(folder)

    # collect summary statistics
    summary_rows = []

    for in_path in input_files:
        if not os.path.isfile(in_path):
            print(f"[skip] not found：{in_path}")
            continue

        with open(in_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        
        # make sure the required columns exist
        if not {'chosen','rejected'}.issubset(df.columns):
            print(f"[skip] columns missing：{in_path}")
            continue

        # add id column
        df = df.reset_index().rename(columns={'index':'id'})
        
        # Calculate three distances
        df['distance'] = df.apply(lambda r: levenshtein_distance(r['chosen'], r['rejected']), axis=1)
        df['normalized_distance'] = df.apply(lambda r: normalized_distance(r['chosen'], r['rejected']), axis=1)
        df['weighted_distance'] = df.apply(lambda r: weighted_distance(r['chosen'], r['rejected']), axis=1)

        # Save the distances to a CSV file
        output_filename = in_path.replace('.json', '_distances.csv')
        output_path = os.path.join(folder, os.path.basename(output_filename))
        df[['id','distance','normalized_distance','weighted_distance']].to_csv(
            output_path, index=False, encoding='utf-8-sig'
        )
        print(f"[Finished] {in_path} → {output_path}")

        # Summary statistics
        for col in ['normalized_distance','weighted_distance']:
            stats = df[col].describe()
            summary_rows.append({
                'file': in_path,
                'metric': col,
                'count': int(stats['count']),
                'mean': stats['mean'],
                'median': df[col].median(),
                'min': stats['min'],
                'max': stats['max']
            })

    # Export summary statistics
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_output_path = os.path.join(folder, 'summary.csv')
        summary_df.to_csv(summary_output_path, index=False, encoding='utf-8-sig')
        print("[finished] export summary.csv")
    else:  
        print("[hint] No files processed, no summary statistics to export.")

if __name__ == "__main__":
    main()
    plot_distance(os.path.join("../distance", "summary.csv"))
