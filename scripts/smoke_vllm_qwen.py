#!/usr/bin/env python3

import argparse

from vllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt",
                        default="Write one short sentence proving vLLM is serving on Intel Arc.")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="half")
    parser.add_argument("--attention-backend")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    attention_config = None
    if args.attention_backend:
        attention_config = {"backend": args.attention_backend}

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        attention_config=attention_config,
        enforce_eager=True,
    )
    outputs = llm.generate(
        [args.prompt],
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
    )
    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()
