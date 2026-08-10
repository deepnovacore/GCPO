# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GCPO parameterization utilities.

GCPO projects both trainable LoRA factors into the complements of frozen
principal left/right singular subspaces. The projection is applied in the
factor forwards used for training and when exporting LoRA weights to rollout
workers.
"""

from __future__ import annotations

from collections import OrderedDict
from types import MethodType

import torch
import torch.nn.functional as F


GCPO_U_PREFIX = "_gcpo_u_"
GCPO_V_PREFIX = "_gcpo_v_"
GCPO_FACTOR_U = "_gcpo_u"
GCPO_FACTOR_V = "_gcpo_v"
GCPO_FACTOR_SIDE = "_gcpo_side"


def _adapter_key(adapter_name: str) -> str:
    return str(adapter_name).replace(".", "_")


def _strip_known_prefixes(name: str) -> str:
    prefixes = (
        "_fsdp_wrapped_module.",
        "base_model.model.",
        "model.",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix) :]
                changed = True
    return name


def _find_basis_key(module_name: str, basis: dict[str, dict[str, torch.Tensor]]) -> str | None:
    candidates = [module_name, _strip_known_prefixes(module_name)]
    for candidate in list(candidates):
        candidates.append(f"model.{candidate}")
        candidates.append(f"base_model.model.{candidate}")

    for candidate in candidates:
        if candidate in basis:
            return candidate

    stripped_module_name = _strip_known_prefixes(module_name)
    for key in basis:
        stripped_key = _strip_known_prefixes(key)
        if stripped_module_name == stripped_key or stripped_module_name.endswith(f".{stripped_key}"):
            return key
        if stripped_key.endswith(f".{stripped_module_name}"):
            return key
    return None


def load_gcpo_basis(path: str, topk: int | None = None) -> dict[str, dict[str, torch.Tensor]]:
    raw_basis = torch.load(path, map_location="cpu")
    if "basis" in raw_basis and isinstance(raw_basis["basis"], dict):
        raw_basis = raw_basis["basis"]

    basis: dict[str, dict[str, torch.Tensor]] = {}
    for name, entry in raw_basis.items():
        u = entry.get("u", entry.get("U", None))
        v = entry.get("v", entry.get("V", None))
        if u is None or v is None:
            continue
        if topk is not None:
            u = u[:, :topk]
            v = v[:, :topk]
        basis[name] = {"u": u.contiguous(), "v": v.contiguous()}
    return basis


def _project_lora_b(weight: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    compute_dtype = torch.float32 if weight.dtype in (torch.float16, torch.bfloat16) else weight.dtype
    weight_for_projection = weight.to(dtype=compute_dtype)
    u = u.to(device=weight.device, dtype=compute_dtype)
    with torch.autocast(device_type=weight.device.type, enabled=False):
        projected = weight_for_projection - u @ (u.transpose(0, 1) @ weight_for_projection)
    return projected.to(dtype=weight.dtype)


def _project_lora_a(weight: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    compute_dtype = torch.float32 if weight.dtype in (torch.float16, torch.bfloat16) else weight.dtype
    weight_for_projection = weight.to(dtype=compute_dtype)
    v = v.to(device=weight.device, dtype=compute_dtype)
    with torch.autocast(device_type=weight.device.type, enabled=False):
        projected = weight_for_projection - (weight_for_projection @ v) @ v.transpose(0, 1)
    return projected.to(dtype=weight.dtype)


def _get_module_basis(module: torch.nn.Module, adapter_name: str) -> tuple[torch.Tensor, torch.Tensor] | None:
    adapter_key = _adapter_key(adapter_name)
    u = getattr(module, f"{GCPO_U_PREFIX}{adapter_key}", None)
    v = getattr(module, f"{GCPO_V_PREFIX}{adapter_key}", None)
    if u is None or v is None:
        return None
    return u, v


def _gcpo_factor_forward(self: torch.nn.Linear, input: torch.Tensor) -> torch.Tensor:
    side = getattr(self, GCPO_FACTOR_SIDE, None)
    if side == "A":
        v = getattr(self, GCPO_FACTOR_V)
        weight = _project_lora_a(self.weight, v)
    elif side == "B":
        u = getattr(self, GCPO_FACTOR_U)
        weight = _project_lora_b(self.weight, u)
    else:
        weight = self.weight
    return F.linear(input, weight, self.bias)


def _patch_lora_factor_forward(module: torch.nn.Linear, side: str) -> None:
    setattr(module, GCPO_FACTOR_SIDE, side)
    if getattr(module, "_gcpo_factor_forward_patched", False):
        return
    module.forward = MethodType(_gcpo_factor_forward, module)
    setattr(module, "_gcpo_factor_forward_patched", True)


def attach_gcpo_basis(
    model: torch.nn.Module,
    basis: dict[str, dict[str, torch.Tensor]],
    adapter_name: str = "default",
) -> int:
    attached = 0
    adapter_key = _adapter_key(adapter_name)
    for module_name, module in model.named_modules():
        if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
            continue
        if adapter_name not in module.lora_A or adapter_name not in module.lora_B:
            continue
        basis_key = _find_basis_key(module_name, basis)
        if basis_key is None:
            continue
        lora_A = module.lora_A[adapter_name]
        lora_B = module.lora_B[adapter_name]
        u = basis[basis_key]["u"]
        v = basis[basis_key]["v"]
        if u.shape[0] != lora_B.weight.shape[0] or v.shape[0] != lora_A.weight.shape[1]:
            continue
        lora_B.register_buffer(
            GCPO_FACTOR_U,
            u.to(device=lora_B.weight.device, dtype=lora_B.weight.dtype),
            persistent=False,
        )
        lora_A.register_buffer(
            GCPO_FACTOR_V,
            v.to(device=lora_A.weight.device, dtype=lora_A.weight.dtype),
            persistent=False,
        )
        _patch_lora_factor_forward(lora_A, "A")
        _patch_lora_factor_forward(lora_B, "B")
        module.register_buffer(
            f"{GCPO_U_PREFIX}{adapter_key}",
            u.to(device=lora_B.weight.device, dtype=lora_B.weight.dtype),
            persistent=False,
        )
        module.register_buffer(
            f"{GCPO_V_PREFIX}{adapter_key}",
            v.to(device=lora_A.weight.device, dtype=lora_A.weight.dtype),
            persistent=False,
        )
        attached += 1
    return attached


def _module_name_from_lora_key(key: str) -> tuple[str, str] | None:
    if ".lora_A." in key:
        return key.split(".lora_A.", 1)[0], "A"
    if ".lora_B." in key:
        return key.split(".lora_B.", 1)[0], "B"
    return None


def project_lora_state_dict_for_rollout(
    state_dict: OrderedDict[str, torch.Tensor] | dict[str, torch.Tensor],
    basis: dict[str, dict[str, torch.Tensor]] | None,
) -> OrderedDict[str, torch.Tensor]:
    if not basis:
        return OrderedDict(state_dict)

    projected: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state_dict.items():
        parsed = _module_name_from_lora_key(key)
        if parsed is None:
            projected[key] = value
            continue
        module_name, side = parsed
        basis_key = _find_basis_key(module_name, basis)
        if basis_key is None:
            projected[key] = value
            continue
        if side == "A":
            v = basis[basis_key]["v"]
            if v.shape[0] == value.shape[1]:
                projected[key] = _project_lora_a(value, v).detach().cpu()
            else:
                projected[key] = value
        else:
            u = basis[basis_key]["u"]
            if u.shape[0] == value.shape[0]:
                projected[key] = _project_lora_b(value, u).detach().cpu()
            else:
                projected[key] = value
    return projected
