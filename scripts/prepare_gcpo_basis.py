#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import re
from typing import Iterable

import torch
from torch import nn
from transformers import AutoModelForCausalLM


def _parse_target_modules(value: str) -> str | set[str]:
    if value == "all-linear":
        return value
    return {item.strip() for item in value.split(",") if item.strip()}


def _matches_target(name: str, module: nn.Module, target_modules: str | set[str], exclude_regex: str | None) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    if exclude_regex and re.search(exclude_regex, name):
        return False
    if name.endswith("lm_head"):
        return False
    if target_modules == "all-linear":
        return True
    leaf_name = name.rsplit(".", 1)[-1]
    return leaf_name in target_modules or name in target_modules


def _iter_target_linear_modules(
    model: nn.Module,
    target_modules: str | set[str],
    exclude_regex: str | None,
) -> Iterable[tuple[str, nn.Linear]]:
    for name, module in model.named_modules():
        if _matches_target(name, module, target_modules, exclude_regex):
            yield name, module


def _svd_basis(weight: torch.Tensor, topk: int, oversample: int, niter: int) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = weight.detach().float()
    max_rank = min(matrix.shape)
    q = min(max_rank, topk + oversample)
    if q >= max_rank:
        u, _, vh = torch.linalg.svd(matrix, full_matrices=False)
        return u[:, :topk].cpu(), vh[:topk].transpose(0, 1).contiguous().cpu()
    u, _, v = torch.svd_lowrank(matrix, q=q, niter=niter)
    return u[:, :topk].contiguous().cpu(), v[:, :topk].contiguous().cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute base-weight SVD bases for GCPO.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--target-modules", default="all-linear")
    parser.add_argument("--exclude-regex", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--oversample", type=int, default=8)
    parser.add_argument("--niter", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    target_modules = _parse_target_modules(args.target_modules)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    model.to(args.device)

    basis = {}
    with torch.no_grad():
        for name, module in _iter_target_linear_modules(model, target_modules, args.exclude_regex):
            weight = module.weight.to(args.device)
            topk = min(args.topk, min(weight.shape))
            u, v = _svd_basis(weight, topk=topk, oversample=args.oversample, niter=args.niter)
            basis[name] = {"u": u.to(torch.float16), "v": v.to(torch.float16)}
            print(f"[gcpo-basis] {name}: weight={tuple(weight.shape)} topk={topk}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(
        {
            "model_path": args.model_path,
            "topk": args.topk,
            "target_modules": args.target_modules,
            "exclude_regex": args.exclude_regex,
            "basis": basis,
        },
        args.output,
    )
    print(f"[gcpo-basis] saved {len(basis)} module bases to {args.output}")


if __name__ == "__main__":
    main()
