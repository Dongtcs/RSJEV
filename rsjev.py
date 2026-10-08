import json
import string
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from transformers import AutoModelForImageTextToText, AutoProcessor


MAX_OPTIONS = 255
NARROW_LETTERS = "ABCDEFGHIJ"
IMAGE_MARKER = "<|vision_start|><|image_pad|><|vision_end|>"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_class_names(path):
    config = read_json(path)
    items = config.get("classes")
    if not isinstance(items, list):
        raise ValueError(f"{path}: expected a classes list")
    names = [item.get("answer") for item in items]
    if not 2 <= len(names) <= MAX_OPTIONS:
        raise ValueError(f"Expected 2..{MAX_OPTIONS} classes")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("Every class needs a nonempty answer")
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError("Class answers must be unique")
    return names


def label_table(tokenizer):
    """Use exactly the same single-token option labels as decider.prompt."""
    labels = []
    alphabet = string.ascii_uppercase
    for name in list(alphabet) + [a + b for a in alphabet for b in alphabet]:
        ids = tokenizer.encode(name, add_special_tokens=False)
        if len(ids) == 1:
            labels.append((name, ids[0]))
        if len(labels) == MAX_OPTIONS:
            break
    if len(labels) != MAX_OPTIONS or len({token for _, token in labels}) != MAX_OPTIONS:
        raise ValueError("Tokenizer cannot supply 255 distinct option tokens")
    for index, letter in enumerate(NARROW_LETTERS):
        if tokenizer.encode(letter, add_special_tokens=False) != [labels[index][1]]:
            raise ValueError(f"Option token for {letter} differs from training")
    return labels


def render_prompt(tokenizer, processor, context, question, classes,
                  answer_slot, prompt_format, max_ctx_tokens, option_tokens):
    """Reproduce the training prompt without importing training code."""
    ids = tokenizer.encode("Context:\n" + context, add_special_tokens=False)[:max_ctx_tokens]
    head = f"\n\nQuestion: {question}\nOptions:"
    tail = f"\nAnswer: {answer_slot}"
    if len(classes) <= len(NARROW_LETTERS):
        option_lines = "".join(f"\n({NARROW_LETTERS[i]}) {name}"
                               for i, name in enumerate(classes))
        ids += tokenizer.encode(head + option_lines + tail, add_special_tokens=False)
    else:
        ids += tokenizer.encode(head, add_special_tokens=False)
        open_ids = tokenizer.encode("\n(", add_special_tokens=False)
        for index, name in enumerate(classes):
            ids += open_ids + [option_tokens[index]]
            ids += tokenizer.encode(f") {name}", add_special_tokens=False)
        ids += tokenizer.encode(tail, add_special_tokens=False)
    text = tokenizer.decode(ids)
    if prompt_format == "plain":
        return IMAGE_MARKER + text
    suffix = tail
    if not text.endswith(suffix):
        raise ValueError("Tokenizer did not preserve the answer prefix")
    content = [{"type": "image"}, {"type": "text", "text": text[:-len(suffix)]}]
    return processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    ) + f"Answer: {answer_slot}"


class RSJEV:
    def __init__(self, checkpoint, max_options, device="auto", dtype="float32"):
        checkpoint = Path(checkpoint).resolve()
        config_path = checkpoint / "decision_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Expected a trained checkpoint: {config_path}")
        self.checkpoint = checkpoint
        self.config = read_json(config_path)
        if not 2 <= max_options <= MAX_OPTIONS:
            raise ValueError(f"max_options must be in 2..{MAX_OPTIONS}")
        self.max_options = max_options
        self.answer_slot = self.config.get("answer_slot", "(")
        self.prompt_format = self.config.get("prompt_format", "chat")
        if self.answer_slot not in ("(", "[RSSC]"):
            raise ValueError(f"Unsupported answer slot: {self.answer_slot}")
        if self.prompt_format not in ("chat", "plain"):
            raise ValueError(f"Unsupported prompt format: {self.prompt_format}")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA/ROCm device is unavailable")
        if self.device.type == "cpu" and dtype != "float32":
            raise ValueError("CPU inference requires float32")
        self.processor = AutoProcessor.from_pretrained(str(checkpoint))
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "right"
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(checkpoint), dtype=getattr(torch, dtype)
        ).to(self.device).eval()
        table = label_table(self.tokenizer)
        self.option_tokens = [token for _, token in table[:max_options]]
        if self.answer_slot == "[RSSC]":
            slot_id = self.tokenizer.convert_tokens_to_ids("[RSSC]")
            if ("[RSSC]" not in self.tokenizer.all_special_tokens
                    or self.tokenizer.encode("[RSSC]", add_special_tokens=False) != [slot_id]
                    or slot_id >= self.model.get_input_embeddings().num_embeddings):
                raise ValueError("Checkpoint does not contain a trained [RSSC] token")
            self.slot_token = slot_id
        else:
            self.slot_token = self.tokenizer.encode(" (", add_special_tokens=False)[-1]

    @torch.no_grad()
    def predict(self, image_path, classes, context, question,
                max_image_side=None, max_ctx_tokens=None):
        if not 2 <= len(classes) <= self.max_options:
            raise ValueError(f"Expected 2..{self.max_options} candidate classes")
        max_image_side = max_image_side or self.config.get("max_image_side", 448)
        max_ctx_tokens = max_ctx_tokens or self.config.get("max_ctx_tokens", 1536)
        if max_image_side < 32 or max_ctx_tokens < 1:
            raise ValueError("Invalid image-side or context-token limit")
        image_path = Path(image_path).resolve()
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((max_image_side, max_image_side), Image.Resampling.LANCZOS)
        prompt = render_prompt(
            self.tokenizer, self.processor, context, question, classes,
            self.answer_slot, self.prompt_format, max_ctx_tokens, self.option_tokens,
        )
        inputs = self.processor(images=[image], text=[prompt],
                                return_tensors="pt", padding=True)
        slot_index = int(inputs["attention_mask"][0].sum()) - 1
        if self.answer_slot == "[RSSC]" and inputs["input_ids"][0, slot_index].item() != self.slot_token:
            raise ValueError("Assistant answer prefix does not end at [RSSC]")
        keys = ("input_ids", "attention_mask", "pixel_values",
                "image_grid_thw", "mm_token_type_ids")
        model_inputs = {key: value.to(self.device)
                        for key, value in inputs.items() if key in keys}
        hidden = self.model.model(**model_inputs, use_cache=False).last_hidden_state
        slot_hidden = hidden[0, slot_index]
        option_ids = torch.tensor(self.option_tokens[:len(classes)],
                                  dtype=torch.long, device=self.device)
        logits = F.linear(slot_hidden, self.model.lm_head.weight[option_ids]).float()
        probabilities = torch.softmax(logits, dim=-1).cpu().tolist()
        raw_logits = logits.cpu().tolist()
        scores = [dict(label=name, probability=float(probability), logit=float(logit))
                  for name, probability, logit in zip(classes, probabilities, raw_logits)]
        scores.sort(key=lambda item: item["probability"], reverse=True)
        return dict(image=str(image_path), prediction=scores[0]["label"],
                    confidence=scores[0]["probability"], answer_slot=self.answer_slot,
                    question=question, scores=scores)
