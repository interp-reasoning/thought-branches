
# Thought Branches 🌳

Most work interpreting reasoning models studies only a single chain-of-thought (CoT), yet these models define distributions over many possible CoTs. We argue that studying a single sample is inadequate for understanding causal influence and the underlying computation. We present case studies using resampling to investigate model decisions. First, when a model states a reason for its action, does that reason actually cause the action? In "agentic misalignment" scenarios, we resample specific sentences to measure their downstream effects. Self-preservation sentences have small causal impact, suggesting they do not meaningfully drive blackmail. Second, are artificial edits to CoT sufficient for steering reasoning? These are common in literature, yet take the model off-policy. Resampling and selecting a completion with the desired property is a principled on-policy alternative. We find off-policy interventions yield small and unstable effects compared to resampling in decision-making tasks. Third, how do we understand the effect of removing a reasoning step when the model may repeat it post-edit? We introduce a resilience metric that repeatedly resamples to prevent similar content from reappearing downstream. Critical planning statements resist removal but have large effects when eliminated. Fourth, since CoT is sometimes "unfaithful", can our methods teach us anything in these settings? Adapting causal mediation analysis, we find that hints that have a causal effect on the output without being explicitly mentioned exert a subtle and cumulative influence on the CoT that persists even if the hint is removed. Overall, studying distributions via resampling enables reliable causal analysis, clearer narratives of model reasoning, and principled CoT interventions.

See more:
* 📄 Paper: https://arxiv.org/abs/2510.27484
* 📊 Datasets: https://huggingface.co/datasets/uzaymacar/blackmail-rollouts and https://huggingface.co/datasets/uzaymacar/whistleblower-rollouts

## Get Started

You can download our [blackmail rollouts dataset](https://huggingface.co/datasets/uzaymacar/blackmail-rollouts) and [whistleblower rollouts dataset](https://huggingface.co/datasets/uzaymacar/whistleblower-rollouts) or resample your own data.

Here's a quick rundown of the main scripts in this repository and what they do:

- `blackmail/generate_blackmail_rollouts` and `whistleblower/generate_whistleblower_rollouts.py` respectively creates base rollouts for the blackmail and whistleblower scenarios. Our datasets were generated with them.
- `blackmail/prompts.py` and `whistleblower/prompts.py` includes the input prompts used and `blackmail/utils.py` and `whistleblower/utils.py` contains helper functions.
- `blackmail/analyze_rollouts.py` and `whistleblower/analyze_rollouts.py` creates the `chunks_labeled.json` files in the respective data folders.
- `blackmail/onpolicy_chain_disruption.py` and `whistleblower/onpolicy_chain_disruption.py` creates on-policy chain-of-thought interventions via resampling.
- `blackmail/measure_determination.py` and `whistleblower/measure_determination.py` creates off-policy chain-of-thought interventions via hand-written edits and same/cross-model insertions.
- `faithfulness/` and `resume_analysis/` folders respectively contains all experiments run in the paper for chain-of-thought faithfulness and resume analysis.

## Citation

Please cite our work if you are using our code or datasets.

```
@misc{macar2025thoughtbranchesinterpretingllm,
      title={Thought Branches: Interpreting LLM Reasoning Requires Resampling}, 
      author={Uzay Macar and Paul C. Bogdan and Senthooran Rajamanoharan and Neel Nanda},
      year={2025},
      eprint={2510.27484},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.27484}, 
}
```

## Contact

For any questions, thoughts, or feedback, please reach out to [uzaymacar@gmail.com](mailto:uzaymacar@gmail.com) and [paulcbogdan@gmail.com](mailto:paulcbogdan@gmail.com).
