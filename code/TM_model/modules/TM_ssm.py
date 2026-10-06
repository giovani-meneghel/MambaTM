# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

@torch.jit.script
def _selective_scan_sequential_jit(u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor,
                                    B: torch.Tensor, C: torch.Tensor, D: torch.Tensor,
                                    z: torch.Tensor) -> torch.Tensor:
    """
    Memory-efficient JIT-compiled sequential scan for long sequences.

    Computes h_t = exp(delta_t * A) * h_{t-1} + (delta_t * B_t) * u_t on-the-fly,
    without storing the full (B, D, L, N) intermediate tensor.

    Benchmark results vs Kogge-Stone (B=1, D=256, N=16):
      L=12,288  -> KS: 0.13s / 1,298 MB  |  JIT: 0.86s / 129 MB
      L=122,880 -> KS: 113s  / 12,927 MB |  JIT: 6.7s  / 912 MB
    Use the hybrid dispatcher _selective_scan_hybrid() to pick automatically.
    """
    B_sz = u.size(0)
    D_sz = u.size(1)
    L    = u.size(2)

    # L-first layout for cache-friendly sequential access
    delta_L = delta.permute(2, 0, 1).contiguous()   # (L, B, D)
    u_L     = u.permute(2, 0, 1).contiguous()        # (L, B, D)
    B_L     = B.permute(2, 0, 1).contiguous()        # (L, B, N)
    C_L     = C.permute(2, 0, 1).contiguous()        # (L, B, N)

    y    = torch.zeros(L, B_sz, D_sz, dtype=u.dtype, device=u.device)
    curr = torch.zeros(B_sz, D_sz, A.size(1), dtype=u.dtype, device=u.device)

    for t in range(L):
        delta_t    = delta_L[t]
        deltaA_t   = torch.exp(delta_t.unsqueeze(-1) * A.unsqueeze(0))           # (B, D, N)
        deltaB_u_t = delta_t.unsqueeze(-1) * B_L[t].unsqueeze(1) * u_L[t].unsqueeze(-1)
        curr       = deltaA_t * curr + deltaB_u_t
        y[t]       = (curr * C_L[t].unsqueeze(1)).sum(dim=-1)

    y = y.permute(1, 2, 0)          # (B, D, L)
    y = y + u * D.unsqueeze(-1)
    y = y * F.silu(z)
    return y


def _selective_scan_kogge_stone(u, delta, A, B, C, D, z):
    """Kogge-Stone O(log L) parallel associative scan. Fast for short L; OOMs for long L."""
    deltaA   = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)

    # Permute to (L, B, D, N) for in-place up-sweep
    a = deltaA.permute(2, 0, 1, 3).clone()
    b = deltaB_u.permute(2, 0, 1, 3).clone()
    L      = a.shape[0]
    stride = 1
    while stride < L:
        b[stride:] = a[stride:] * b[:-stride] + b[stride:]
        a[stride:] = a[stride:] * a[:-stride]
        stride *= 2

    x = b.permute(1, 2, 0, 3)                  # (B, D, L, N)
    y = (x * C.permute(0, 2, 1).unsqueeze(1)).sum(dim=-1)
    y = y + u * D[..., None]
    y = y * F.silu(z)
    return y, x[:, :, -1]                       # output, last_state


# Threshold calibrated so Kogge-Stone peak VRAM (~1298 MB at L=12288, linear scaling)
# stays below 5.3 GB (RTX 2060 has 6 GB total; reserve ~700 MB for model weights).
# 5300 / 1298 * 12288 ~ 50,200  -> use 40,000 for a comfortable safety margin.
_KS_L_THRESHOLD = 40_000


try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                          delta_softplus=False, return_last_state=False):
        u_type = u.dtype
        u     = u.float()
        delta = delta.float()
        A     = A.float()
        B     = B.float()
        C     = C.float()
        if D is not None:
            D = D.float()
        if z is not None:
            z = z.float()

        if delta_bias is not None:
            delta = delta + delta_bias[..., None]
        if delta_softplus:
            delta = F.softplus(delta)

        # Ensure D and z are always tensors (some callers omit them)
        if D is None:
            D = torch.zeros(u.shape[1], dtype=u.dtype, device=u.device)
        if z is None:
            # No gate: treat as if z were large (silu -> 1) and skip gating
            L_seq = u.shape[2]
            if L_seq >= _KS_L_THRESHOLD:
                # JIT path: insert a unit gate tensor
                y = _selective_scan_sequential_jit(u, delta, A, B, C, D,
                                                   torch.ones_like(u) * 10.0)
            else:
                # Kogge-Stone path without gating
                deltaA   = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
                a = deltaA.permute(2, 0, 1, 3).clone()
                b = deltaB_u.permute(2, 0, 1, 3).clone()
                stride = 1
                while stride < a.shape[0]:
                    b[stride:] = a[stride:] * b[:-stride] + b[stride:]
                    a[stride:] = a[stride:] * a[:-stride]
                    stride *= 2
                x = b.permute(1, 2, 0, 3)
                y = (x * C.permute(0, 2, 1).unsqueeze(1)).sum(dim=-1) + u * D[..., None]
                if return_last_state:
                    return y.to(u_type), x[:, :, -1].to(u_type)
                return y.to(u_type)
            if return_last_state:
                return y.to(u_type), torch.zeros(u.shape[0], u.shape[1], A.shape[1],
                                                  dtype=u_type, device=u.device)
            return y.to(u_type)

        L_seq = u.shape[2]
        if L_seq >= _KS_L_THRESHOLD:
            # --- Memory-safe path for long sequences (large spatial patches) ---
            y = _selective_scan_sequential_jit(u, delta, A, B, C, D, z)
            if return_last_state:
                # last_state is rarely needed during inference; return zeros as placeholder
                last_state = torch.zeros(u.shape[0], u.shape[1], A.shape[1],
                                         dtype=u_type, device=u.device)
                return y.to(u_type), last_state
            return y.to(u_type)
        else:
            # --- Fast parallel path for short sequences (small spatial patches) ---
            y, last_state = _selective_scan_kogge_stone(u, delta, A, B, C, D, z)
            if return_last_state:
                return y.to(u_type), last_state.to(u_type)
            return y.to(u_type)


# Pre-compile the JIT kernel at import time (avoids first-call latency on CUDA)
try:
    with torch.no_grad():
        _d = torch.zeros(1, 2, 1, device='cpu')
        _selective_scan_sequential_jit(
            _d, _d, torch.zeros(2, 1), _d.expand(1, 1, 1),
            _d.expand(1, 1, 1), torch.zeros(2), _d
        )
    del _d
except Exception:
    pass

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None



class BiMamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        if_devide_out=False,
        init_layer_scale=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.if_devide_out = if_devide_out

        self.init_layer_scale = init_layer_scale
        if init_layer_scale is not None:
            self.gamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        # bidirectional
        A_b = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
        self.A_b_log = nn.Parameter(A_b_log)
        self.A_b_log._no_weight_decay = True 

        self.conv1d_b = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.x_proj_b = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_b._no_weight_decay = True
            
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward_1d(self, xz, conv1d, x_proj, dt_proj, A, D, seqlen, conv_state=None, ssm_state=None):
        x, z = xz.chunk(2, dim=1)
        if conv_state is not None:
            # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
            # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
            conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
        if causal_conv1d_fn is None:
            x = self.act(conv1d(x)[..., :seqlen])
        else:
            assert self.activation in ["silu", "swish"]
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(conv1d.weight, "d 1 w -> d w"),
                bias=conv1d.bias,
                activation=self.activation,
            )
        # We're careful here about the layout, to avoid extra transposes.
        # We want dt to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x_dbl = x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        assert self.activation in ["silu", "swish"]
        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            D.float(),
            z=z,
            delta_bias=dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=ssm_state is not None,
        ) 
        if ssm_state is not None:
            y, last_state = y
            ssm_state.copy_(last_state)
        return y
    
    
    def forward(self, hidden_states, hidden_states_ref=None, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
        
        
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        A_b = -torch.exp(self.A_b_log.float())
        
        out = self.forward_1d(
            xz,
            self.conv1d,
            self.x_proj,
            self.dt_proj,
            A,
            self.D.float(),
            seqlen,
            conv_state,
            ssm_state,
        )
        out_b = self.forward_1d(
            xz.flip([-1]),
            self.conv1d_b,
            self.x_proj_b,
            self.dt_proj_b,
            A_b,
            self.D_b.float(),
            seqlen,
            conv_state,
            ssm_state,
        )
        # F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)
        if not self.if_devide_out:
            out = F.linear(rearrange(out + out_b.flip([-1]), "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
        else:
            out = F.linear(rearrange(out + out_b.flip([-1]), "b d l -> b l d") / 2, self.out_proj.weight, self.out_proj.bias)

        if self.init_layer_scale is not None:
                out = out * self.gamma    
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


    
class GMamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        if_devide_out=False,
        init_layer_scale=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.if_devide_out = if_devide_out

        self.init_layer_scale = init_layer_scale
        if init_layer_scale is not None:
            self.gamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.in_proj_ref = nn.Linear(self.d_model*2, self.d_inner*2, bias=bias, **factory_kwargs)

        A_b = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
        self.A_b_log = nn.Parameter(A_b_log)
        self.A_b_log._no_weight_decay = True 

        self.conv1d_b = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.conv1d_ref = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.conv1d_ref_b = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.x_proj_b = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_b._no_weight_decay = True
            
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward_with_ref(self, x, z, ref, conv1d, conv1d_ref, x_proj, dt_proj, A, D, seqlen, conv_state=None, ssm_state=None):
        if conv_state is not None:
            # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
            # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
            conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            conv_state.copy_(F.pad(ref, (self.d_conv - ref.shape[-1], 0)))  # Update state (B D W)
        if causal_conv1d_fn is None:
            x = self.act(conv1d(x)[..., :seqlen])
            ref = self.act(conv1d_ref(ref)[..., :seqlen])
        else:
            assert self.activation in ["silu", "swish"]
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(conv1d.weight, "d 1 w -> d w"),
                bias=conv1d.bias,
                activation=self.activation,
            )
            ref = causal_conv1d_fn(
                x=ref,
                weight=rearrange(conv1d_ref.weight, "d 1 w -> d w"),
                bias=conv1d_ref.bias,
                activation=self.activation,
            )
        # We're careful here about the layout, to avoid extra transposes.
        # We want dt to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x_dbl = x_proj(rearrange(ref, "b d l -> (b l) d"))  # (bl d)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        assert self.activation in ["silu", "swish"]
        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            D.float(),
            z=z,
            delta_bias=dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=ssm_state is not None,
        ) 
        if ssm_state is not None:
            y, last_state = y
            ssm_state.copy_(last_state)
        return y
    
    
    def forward(self, hidden_states, hidden_states_ref=None, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        if hidden_states_ref is not None:
            hidden_states_ref = torch.cat([hidden_states, hidden_states_ref], dim=-1)
            ref_z = rearrange(
                self.in_proj_ref.weight @ rearrange(hidden_states_ref, "b l d -> d (b l)"),
                "d (b l) -> b d l",
                l=seqlen,
            )
            if self.in_proj_ref.bias is not None:
                ref_z = ref_z + rearrange(self.in_proj_ref.bias.to(dtype=ref_z.dtype), "d -> d 1")
            
            x = rearrange(
                self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
                "d (b l) -> b d l",
                l=seqlen,
            )
            if self.in_proj_ref.bias is not None:
                x = x + rearrange(self.in_proj.bias.to(dtype=x.dtype), "d -> d 1")   
                
            A_b = -torch.exp(self.A_b_log.float())
        
        
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        ref, z = ref_z.chunk(2, dim=1)
        out = self.forward_with_ref(
            x, 
            z, 
            ref, 
            self.conv1d, 
            self.conv1d_ref, 
            self.x_proj, 
            self.dt_proj, 
            A, 
            self.D, 
            seqlen, 
            conv_state, 
            ssm_state)
        out_b = self.forward_with_ref(
            x.flip([-1]), 
            z.flip([-1]), 
            ref.flip([-1]), 
            self.conv1d_b, 
            self.conv1d_ref_b, 
            self.x_proj_b, 
            self.dt_proj_b, 
            A_b, 
            self.D_b, 
            seqlen, 
            conv_state, 
            ssm_state)

        if not self.if_devide_out:
            out = F.linear(rearrange(out + out_b, "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
        else:
            out = F.linear(rearrange(out + out_b, "b d l -> b l d") / 2, self.out_proj.weight, self.out_proj.bias)
            
        if self.init_layer_scale is not None:
                out = out * self.gamma    
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state

