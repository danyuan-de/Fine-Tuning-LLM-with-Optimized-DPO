## Training LLMs on domain-specific knowledge base with reinforcement learning based on preference data

### Overview

This project focuses on fine-tuning Large Language Models (LLMs) using reinforcement learning techniques, specifically Direct Preference Optimization (DPO), to adapt them to domain-specific knowledge bases. By leveraging human preferences, we aim to improve the quality and relevance of model responses for specialized applications.

### Features

- Domain-specific fine-tuning of LLMs
- Implementation of DPO (Direct Preference Optimization) and its variants (DPO-Positive, DPO-Shift)
- Preference data processing pipeline
- Evaluation framework for measuring performance improvements

### Installation

```bash
git clone https://github.com/yourusername/Fine-Tuning-LLM-with-DPO.git
cd Fine-Tuning-LLM-with-DPO
```

### Usage
On the first run, use `setup.sh` to handle data download, environment initialization, and any other setup steps. For example:
```bash
./setup.sh --train --run_test --benchmark --method DPO --beta 0.3 --model 8B-Instruct --lr 3e-6 --data content 
```

After the initial setup, you can skip `setup.sh` and run the main module directly. For example:
```bash
python -m src.main --train --run_test --benchmark --method DPO --beta 0.3 --model 8B-Instruct --lr 3e-6 --data content 
```
### Methodology
The fine-tuning process involves:

1. **Model Selection**  
   A SFT LLM checkpoint is selected as the reference policy.

2. **Preference Data Splitting**  
   Preference data are partitioned into disjoint training, validation, and test sets to prevent overlap.

3. **DPO Training**  
   Optimizing the model using the DPO or its variants algorithm to align with human preferences.

4. **Evaluation**  
   - Downstream test set and benchmarks (e.g., win-rate, perplexity) are computed to ensure no degradation.  
   - Reward-margin distributions are compared before and after DPO to confirm that chosen-response likelihood increases without harming overall coherence.

### License

This project is licensed under the MIT License - see the LICENSE file for details.