import argparse
import json
from pathlib import Path

from rsjev import RSJEV, load_class_names, read_json


HERE = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        help="Trained checkpoint directory containing decision_config.json")
    parser.add_argument("--image", required=True, help="Image to classify")
    parser.add_argument("--dataset", choices=("aid", "nwpu", "ucm"),
                        help="Use the bundled candidate classes, including cross-dataset inference")
    parser.add_argument("--classes", help="Custom classes.json; overrides --dataset")
    parser.add_argument("--question", help="Override the checkpoint's evaluation question")
    parser.add_argument("--context", help="Override the checkpoint's context")
    parser.add_argument("--max-image-side", type=int)
    parser.add_argument("--max-ctx-tokens", type=int)
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json-out", help="Save every candidate's logit and probability")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    return args


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    config = read_json(checkpoint / "decision_config.json")
    if args.classes:
        classes_path = Path(args.classes)
    elif args.dataset:
        classes_path = HERE / "classes" / f"{args.dataset}.json"
    else:
        classes_path = checkpoint / "classes.json"
        if not classes_path.is_file():
            dataset = config.get("dataset")
            classes_path = HERE / "classes" / f"{dataset}.json"
    classes = load_class_names(classes_path)
    prompts_path = checkpoint / "prompts.json"
    prompts = read_json(prompts_path if prompts_path.is_file() else HERE / "prompts.json")
    question = args.question or prompts["questions"][prompts.get("eval_prompt_index", 0)]
    context = args.context if args.context is not None else prompts.get("context", "")
    max_options = max(config.get("max_options", len(classes)), len(classes))
    decider = RSJEV(checkpoint, max_options=max_options,
                             device=args.device, dtype=args.dtype)
    result = decider.predict(args.image, classes, context, question,
                             max_image_side=args.max_image_side,
                             max_ctx_tokens=args.max_ctx_tokens)
    result["dataset"] = args.dataset or (None if args.classes else config.get("dataset"))
    print(f"Prediction: {result['prediction']} ({result['confidence']:.4%})")
    for rank, score in enumerate(result["scores"][:args.top_k], 1):
        print(f"{rank:>2}. {score['label']:<26} {score['probability']:.4%}")
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False,
                                     allow_nan=False), encoding="utf-8")
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
