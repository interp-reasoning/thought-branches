
# Thought Branches 🪾

- `generate_whistleblower_rollouts.py` creates the base rollouts for the whistleblower scenario
- `prompts.py` includes the input prompts used for the whistleblowing scenario and `utils.py` helper functions
- `analyze_rollouts.py` creates the `chunks_labeled.json` files
- `onpolicy_chain_disruption.py` creates on-policy chain-of-thought interventions via resampling
- `measure_determination.py` creates off-policy chain-of-thought interventions via hand-written edits and same/cross-model insertions