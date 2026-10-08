# RSJEV: inference demo

This directory contains the image-classification inference path only. It does not import the training package. A trained checkpoint is required; base Qwen3.5 weights alone are not a scene-classification checkpoint.

## Files

- `demo.py`: single-image command-line demo.
- `rsjev.py`: prompt construction, answer-slot extraction, and option scoring.
- `classes/aid.json`, `classes/nwpu.json`, `classes/ucm.json`: candidate class lists.
- `prompts.json`: fallback prompt when a checkpoint has no prompt file.

The trained checkpoint must contain `config.json`, model weights, tokenizer and processor files, and `decision_config.json`. When present, its `classes.json` and `prompts.json` are used by default.

## Run

Install a GPU-compatible PyTorch build for your machine and install the remaining dependencies from `requirements.txt`. From this directory:

```bash
HIP_VISIBLE_DEVICES=1 python demo.py \
  --checkpoint /root/private_data/coding/decider-main/runs/aid/Qwen3_5_0_8B_language_model_e10_ls_0.05_brier_0.1_color_0/best \
  --image /root/private_data/DATASET/classification/aid/all_img/airport_81.jpg \
  --json-out prediction.json
```

The command prints the predicted class and top five probabilities. `prediction.json` includes logits and probabilities for every candidate class. Use `--top-k` to change printed length.

By default, classes come from the checkpoint. To score an image against another dataset's candidate list, specify `--dataset nwpu` (or `aid`, `ucm`). A custom class file with the same `{"classes":[{"answer":"..."},...]}` structure can be passed with `--classes`. The option head is expanded to the requested number of candidates without modifying checkpoint weights.

The demo reads `answer_slot`, `prompt_format`, image size, and context limit from `decision_config.json`. Checkpoints without an `answer_slot` field use the original `(` slot; newer `[RSSC]` checkpoints use the saved special token. The model scores option-letter tokens in one forward pass, without generating a text answer.
