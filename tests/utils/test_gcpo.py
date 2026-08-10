from collections import OrderedDict

import torch

from verl.utils.gcpo import (
    _project_lora_a,
    _project_lora_b,
    attach_gcpo_basis,
    project_lora_state_dict_for_rollout,
)


class _FakeLoraLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(5, 3, bias=False)})
        self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(3, 7, bias=False)})


def test_gcpo_projects_lora_factors_into_complement():
    generator = torch.Generator().manual_seed(7)
    u = torch.linalg.qr(torch.randn(7, 2, generator=generator), mode="reduced").Q
    v = torch.linalg.qr(torch.randn(5, 2, generator=generator), mode="reduced").Q
    b = torch.randn(7, 3, generator=generator)
    a = torch.randn(3, 5, generator=generator)

    projected_b = _project_lora_b(b, u)
    projected_a = _project_lora_a(a, v)

    assert torch.allclose(u.T @ projected_b, torch.zeros(2, 3), atol=1e-5)
    assert torch.allclose(projected_a @ v, torch.zeros(3, 2), atol=1e-5)


def test_gcpo_projects_rollout_state_dict():
    generator = torch.Generator().manual_seed(11)
    u = torch.linalg.qr(torch.randn(7, 2, generator=generator), mode="reduced").Q
    v = torch.linalg.qr(torch.randn(5, 2, generator=generator), mode="reduced").Q
    prefix = "base_model.model.layers.0.self_attn.q_proj"
    state = OrderedDict(
        {
            f"{prefix}.lora_A.default.weight": torch.randn(3, 5, generator=generator),
            f"{prefix}.lora_B.default.weight": torch.randn(7, 3, generator=generator),
        }
    )

    projected = project_lora_state_dict_for_rollout(
        state,
        {"layers.0.self_attn.q_proj": {"u": u, "v": v}},
    )

    assert torch.allclose(projected[f"{prefix}.lora_A.default.weight"] @ v, torch.zeros(3, 2), atol=1e-5)
    assert torch.allclose(u.T @ projected[f"{prefix}.lora_B.default.weight"], torch.zeros(2, 3), atol=1e-5)


def test_gcpo_attaches_projection_to_lora_forwards():
    generator = torch.Generator().manual_seed(13)
    u = torch.linalg.qr(torch.randn(7, 2, generator=generator), mode="reduced").Q
    v = torch.linalg.qr(torch.randn(5, 2, generator=generator), mode="reduced").Q
    model = torch.nn.Module()
    model.q_proj = _FakeLoraLayer()

    attached = attach_gcpo_basis(model, {"q_proj": {"u": u, "v": v}})
    effective_a = model.q_proj.lora_A["default"](torch.eye(5)).T
    effective_b = model.q_proj.lora_B["default"](torch.eye(3)).T

    assert attached == 1
    assert torch.allclose(effective_a @ v, torch.zeros(3, 2), atol=1e-5)
    assert torch.allclose(u.T @ effective_b, torch.zeros(2, 3), atol=1e-5)
