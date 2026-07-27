
# AOT ID: ['7_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /home/t-jialianggu/work/EndtoEnd-auto-optimization/results/lfm_fusion/fx/inductor_cache/md/cmdruwha3txzuelb6yc5ckonrxiths4uhbuaycalpiz6nigetsxb.py
# Topologically Sorted Source Nodes: [chunk, Bx], Original ATen: [aten.split, aten.mul]
# Source node to ATen node mapping:
#   Bx => mul
#   chunk => split
# Graph fragment:
#   %mm : Tensor "bf16[8, 6144][6144, 1]cuda:0" = PlaceHolder[target=mm]
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%mm, 2048, -1), kwargs = {})
#   %mul : Tensor "bf16[8, 2048][2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_3, %getitem_5), kwargs = {})
#   return %mul
triton_poi_fused_mul_split_0 = async_compile.triton('triton_poi_fused_mul_split_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_split_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 131072}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_split_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 2048)
    x1 = xindex // 2048
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 6144*x1), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (4096 + x0 + 6144*x1), None).to(tl.float32)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: /home/t-jialianggu/work/EndtoEnd-auto-optimization/results/lfm_fusion/fx/inductor_cache/ts/ctsreay6o3szffgd26qjpiyivazeqckq452pumpg4ysqgm2or4iq.py
# Topologically Sorted Source Nodes: [to, causal_conv1d_update], Original ATen: [aten._to_copy, aten.as_strided, sgl_kernel.causal_conv1d_update]
# Source node to ATen node mapping:
#   causal_conv1d_update => as_strided_default, causal_conv1d_update_default
#   to => convert_element_type_2
# Graph fragment:
#   %arg5_1 : Tensor "i64[8][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %convert_element_type_2 : Tensor "i32[8][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg5_1, torch.int32), kwargs = {})
#   %as_strided_default : Tensor "bf16[8, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.as_strided.default](args = (%mul, [8, 2048, 1], [2048, 1, 1], 0), kwargs = {})
#   %causal_conv1d_update_default : [num_users=0] = call_function[target=torch.ops.sgl_kernel.causal_conv1d_update.default](args = (%as_strided_default, %arg6_1, %arg4_1, None, False, None, %convert_element_type_2, -1), kwargs = {})
#   return %buf5
triton_poi_fused__to_copy_as_strided_causal_conv1d_update_1 = async_compile.triton('triton_poi_fused__to_copy_as_strided_causal_conv1d_update_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_as_strided_causal_conv1d_update_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 128}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_as_strided_causal_conv1d_update_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /home/t-jialianggu/work/EndtoEnd-auto-optimization/results/lfm_fusion/fx/inductor_cache/t2/ct2ti67u6nednozp7usvsmnazbkjr3xmeec6oopggszwyv3mxbyi.py
# Topologically Sorted Source Nodes: [chunk, mul_1], Original ATen: [aten.split, aten.unsqueeze, aten.squeeze, aten.mul]
# Source node to ATen node mapping:
#   chunk => split
#   mul_1 => mul_1, squeeze_1, unsqueeze_1
# Graph fragment:
#   %mm : Tensor "bf16[8, 6144][6144, 1]cuda:0" = PlaceHolder[target=mm]
#   %buf7 : Tensor  = PlaceHolder[target=buf7]
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%mm, 2048, -1), kwargs = {})
#   %unsqueeze_1 : Tensor "bf16[8, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul, -1), kwargs = {})
#   %squeeze_1 : Tensor "bf16[8, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%unsqueeze_1, -1), kwargs = {})
#   %mul_1 : Tensor "bf16[8, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_4, %squeeze_1), kwargs = {})
#   return %mul_1
triton_poi_fused_mul_split_squeeze_unsqueeze_2 = async_compile.triton('triton_poi_fused_mul_split_squeeze_unsqueeze_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_split_squeeze_unsqueeze_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 98304}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_split_squeeze_unsqueeze_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 2048)
    x1 = xindex // 2048
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2048 + x0 + 6144*x1), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: /home/t-jialianggu/work/EndtoEnd-auto-optimization/results/lfm_fusion/fx/inductor_cache/ec/cecne64i55ed5fexppvsikvweqzqqbjgibqrgy3moamuoyxcuyta.py
# Topologically Sorted Source Nodes: [float_1, scores, scores_for_choice], Original ATen: [aten._to_copy, aten.sigmoid, aten.add]
# Source node to ATen node mapping:
#   float_1 => convert_element_type_7
#   scores => sigmoid
#   scores_for_choice => add
# Graph fragment:
#   %mm_2 : Tensor "bf16[8, 32][32, 1]cuda:0" = PlaceHolder[target=mm_2]
#   %arg10_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg10_1]
#   %convert_element_type_7 : Tensor "f32[8, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_2, torch.float32), kwargs = {})
#   %sigmoid : Tensor "f32[8, 32][32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_7,), kwargs = {})
#   %add : Tensor "f32[8, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sigmoid, %arg10_1), kwargs = {})
#   return %add
triton_poi_fused__to_copy_add_sigmoid_3 = async_compile.triton('triton_poi_fused__to_copy_add_sigmoid_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_sigmoid_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2688}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_sigmoid_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 32)
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /home/t-jialianggu/work/EndtoEnd-auto-optimization/results/lfm_fusion/fx/inductor_cache/wv/cwvhjjdueprgtagfvhlstkkcza7c5r6jktrqvmnrve5vxl7xtyhc.py
# Topologically Sorted Source Nodes: [getitem_9, getitem_10], Original ATen: [aten.select, aten.index]
# Source node to ATen node mapping:
#   getitem_10 => index
#   getitem_9 => select
# Graph fragment:
#   %getitem_14 : Tensor "i64[8, 4][4, 1]cuda:0" = PlaceHolder[target=getitem_14]
#   %arg11_1 : Tensor "bf16[32, 3584, 2048][7340032, 2048, 1]cuda:0" = PlaceHolder[target=arg11_1]
#   %select : Tensor "i64[4][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%getitem_14, 0, 0), kwargs = {})
#   %index : Tensor "bf16[4, 3584, 2048][7340032, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg11_1, [%select]), kwargs = {})
#   return %index
triton_poi_fused_index_select_4 = async_compile.triton('triton_poi_fused_index_select_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_select_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_index_select_4(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 29360128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = xindex // 7340032
    x0 = (xindex % 7340032)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 32, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 32), "index out of bounds: 0 <= tmp4 < 32")
    tmp6 = tl.load(in_ptr1 + (x0 + 7340032*tmp4), None).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: /home/t-jialianggu/work/EndtoEnd-auto-optimization/results/lfm_fusion/fx/inductor_cache/7d/c7dfcojsfgpg4ej6tommjoz5jxnoxbrzkr2bppj6mmnluhm3ewvk.py
# Topologically Sorted Source Nodes: [gate_up, chunk_1, silu, act], Original ATen: [aten.bmm, aten.view, aten.permute, aten.split, aten.silu, aten.mul]
# Source node to ATen node mapping:
#   act => mul_3
#   chunk_1 => split_1
#   gate_up => permute_9, unsqueeze_default, view_3, view_4
#   silu => convert_element_type_11, convert_element_type_12, mul_2, sigmoid_1
# Graph fragment:
#   %mm_default : Tensor "bf16[8, 14336][14336, 1]cuda:0" = PlaceHolder[target=mm_default]
#   %unsqueeze_default : Tensor "bf16[1, 8, 14336][114688, 14336, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mm_default, 0), kwargs = {})
#   %view_3 : Tensor "bf16[8, 1, 4, 3584][14336, 14336, 3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%unsqueeze_default, [8, 1, 4, 3584]), kwargs = {})
#   %permute_9 : Tensor "bf16[8, 4, 3584, 1][14336, 3584, 1, 14336]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_3, [0, 2, 3, 1]), kwargs = {})
#   %view_4 : Tensor "bf16[8, 4, 3584][14336, 3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_9, [8, 4, 3584]), kwargs = {})
#   %split_1 : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_4, 1792, -1), kwargs = {})
#   %convert_element_type_11 : Tensor "f32[8, 4, 1792][7168, 1792, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_15, torch.float32), kwargs = {})
#   %sigmoid_1 : Tensor "f32[8, 4, 1792][7168, 1792, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_11,), kwargs = {})
#   %mul_2 : Tensor "f32[8, 4, 1792][7168, 1792, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_11, %sigmoid_1), kwargs = {})
#   %convert_element_type_12 : Tensor "bf16[8, 4, 1792][7168, 1792, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2, torch.bfloat16), kwargs = {})
#   %mul_3 : Tensor "bf16[8, 4, 1792][7168, 1792, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_12, %getitem_16), kwargs = {})
#   return %mul_3
triton_poi_fused_bmm_mul_permute_silu_split_view_5 = async_compile.triton('triton_poi_fused_bmm_mul_permute_silu_split_view_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bmm_mul_permute_silu_split_view_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 458752}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_bmm_mul_permute_silu_split_view_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 57344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 1792)
    x1 = xindex // 1792
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 3584*x1), None).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (1792 + x0 + 3584*x1), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr0 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: /home/t-jialianggu/work/EndtoEnd-auto-optimization/results/lfm_fusion/fx/inductor_cache/dj/cdjchwgn5isbeiljeri76yzezsw3d5hgajxqz5tpwbdhu46gn7ju.py
# Topologically Sorted Source Nodes: [getitem_13, getitem_14], Original ATen: [aten.select, aten.index]
# Source node to ATen node mapping:
#   getitem_13 => select_1
#   getitem_14 => index_1
# Graph fragment:
#   %getitem_14 : Tensor "i64[8, 4][4, 1]cuda:0" = PlaceHolder[target=getitem_14]
#   %arg12_1 : Tensor "bf16[32, 2048, 1792][3670016, 1792, 1]cuda:0" = PlaceHolder[target=arg12_1]
#   %select_1 : Tensor "i64[4][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%getitem_14, 0, 0), kwargs = {})
#   %index_1 : Tensor "bf16[4, 2048, 1792][3670016, 1792, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg12_1, [%select_1]), kwargs = {})
#   return %index_1
triton_poi_fused_index_select_6 = async_compile.triton('triton_poi_fused_index_select_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_select_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_index_select_6(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 14680064
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = xindex // 3670016
    x0 = (xindex % 3670016)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 32, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 32), "index out of bounds: 0 <= tmp4 < 32")
    tmp6 = tl.load(in_ptr1 + (x0 + 3670016*tmp4), None).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: /home/t-jialianggu/work/EndtoEnd-auto-optimization/results/lfm_fusion/fx/inductor_cache/mf/cmfkcq653j622moslqrjcyiexocyabsle5vpirfvytqvfqenybpy.py
# Topologically Sorted Source Nodes: [float_1, scores, topk_w, sum_1], Original ATen: [aten._to_copy, aten.sigmoid, aten.gather, aten.sum]
# Source node to ATen node mapping:
#   float_1 => convert_element_type_7
#   scores => sigmoid
#   sum_1 => sum_1
#   topk_w => gather
# Graph fragment:
#   %getitem_14 : Tensor "i64[8, 4][4, 1]cuda:0" = PlaceHolder[target=getitem_14]
#   %mm_2 : Tensor "bf16[8, 32][32, 1]cuda:0" = PlaceHolder[target=mm_2]
#   %convert_element_type_7 : Tensor "f32[8, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_2, torch.float32), kwargs = {})
#   %sigmoid : Tensor "f32[8, 32][32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_7,), kwargs = {})
#   %gather : Tensor "f32[8, 4][4, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.gather.default](args = (%sigmoid, 1, %getitem_14), kwargs = {})
#   %sum_1 : Tensor "f32[8, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%gather, [-1], True), kwargs = {})
#   return %sum_1
triton_poi_fused__to_copy_gather_sigmoid_sum_7 = async_compile.triton('triton_poi_fused__to_copy_gather_sigmoid_sum_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_gather_sigmoid_sum_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_gather_sigmoid_sum_7(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 32, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 32)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 32")
    tmp6 = tl.load(in_ptr1 + (tmp4 + 32*x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tl.sigmoid(tmp7)
    tmp10 = tmp9 + tmp1
    tmp11 = tmp9 < 0
    tmp12 = tl.where(tmp11, tmp10, tmp9)
    tl.device_assert(((0 <= tmp12) & (tmp12 < 32)) | ~(xmask), "index out of bounds: 0 <= tmp12 < 32")
    tmp14 = tl.load(in_ptr1 + (tmp12 + 32*x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp8 + tmp16
    tmp19 = tmp18 + tmp1
    tmp20 = tmp18 < 0
    tmp21 = tl.where(tmp20, tmp19, tmp18)
    tl.device_assert(((0 <= tmp21) & (tmp21 < 32)) | ~(xmask), "index out of bounds: 0 <= tmp21 < 32")
    tmp23 = tl.load(in_ptr1 + (tmp21 + 32*x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tl.sigmoid(tmp24)
    tmp26 = tmp17 + tmp25
    tmp28 = tmp27 + tmp1
    tmp29 = tmp27 < 0
    tmp30 = tl.where(tmp29, tmp28, tmp27)
    tl.device_assert(((0 <= tmp30) & (tmp30 < 32)) | ~(xmask), "index out of bounds: 0 <= tmp30 < 32")
    tmp32 = tl.load(in_ptr1 + (tmp30 + 32*x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tl.sigmoid(tmp33)
    tmp35 = tmp26 + tmp34
    tl.store(out_ptr0 + (x0), tmp35, xmask)
''', device_str='cuda')


# kernel path: /home/t-jialianggu/work/EndtoEnd-auto-optimization/results/lfm_fusion/fx/inductor_cache/wb/cwbqsainuj5ul2brldixqgxdry3sa2gr22foj5veykzqq5rkuwzx.py
# Topologically Sorted Source Nodes: [float_1, scores, per_expert, topk_w, sum_1, topk_w_1, topk_w_2, unsqueeze_1, mul_3, out], Original ATen: [aten._to_copy, aten.sigmoid, aten.view, aten.permute, aten.gather, aten.sum, aten.div, aten.unsqueeze, aten.mul]
# Source node to ATen node mapping:
#   float_1 => convert_element_type_7
#   mul_3 => mul_4
#   out => sum_2
#   per_expert => permute_14, view_7, view_8
#   scores => sigmoid
#   sum_1 => sum_1
#   topk_w => gather
#   topk_w_1 => div
#   topk_w_2 => convert_element_type_8
#   unsqueeze_1 => unsqueeze_9
# Graph fragment:
#   %bmm_1 : Tensor "bf16[4, 8, 2048][16384, 2048, 1]cuda:0" = PlaceHolder[target=bmm_1]
#   %getitem_14 : Tensor "i64[8, 4][4, 1]cuda:0" = PlaceHolder[target=getitem_14]
#   %mm_2 : Tensor "bf16[8, 32][32, 1]cuda:0" = PlaceHolder[target=mm_2]
#   %sum_1 : Tensor "f32[8, 1][1, 8]cuda:0" = PlaceHolder[target=sum_1]
#   %convert_element_type_7 : Tensor "f32[8, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_2, torch.float32), kwargs = {})
#   %sigmoid : Tensor "f32[8, 32][32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_7,), kwargs = {})
#   %view_7 : Tensor "bf16[4, 8, 1, 2048][16384, 2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_1, [4, 8, 1, 2048]), kwargs = {})
#   %permute_14 : Tensor "bf16[8, 4, 2048, 1][2048, 16384, 1, 2048]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_7, [1, 0, 3, 2]), kwargs = {})
#   %view_8 : Tensor "bf16[8, 4, 2048][2048, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_14, [8, 4, 2048]), kwargs = {})
#   %gather : Tensor "f32[8, 4][4, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.gather.default](args = (%sigmoid, 1, %getitem_14), kwargs = {})
#   %sum_1 : Tensor "f32[8, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%gather, [-1], True), kwargs = {})
#   %div : Tensor "f32[8, 4][4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%gather, %sum_1), kwargs = {})
#   %convert_element_type_8 : Tensor "bf16[8, 4][4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.bfloat16), kwargs = {})
#   %unsqueeze_9 : Tensor "bf16[8, 4, 1][4, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_8, -1), kwargs = {})
#   %mul_4 : Tensor "bf16[8, 4, 2048][2048, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_8, %unsqueeze_9), kwargs = {})
#   %sum_2 : Tensor "bf16[8, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_4, [1]), kwargs = {})
#   return %sum_2
triton_poi_fused__to_copy_div_gather_mul_permute_sigmoid_sum_unsqueeze_view_8 = async_compile.triton('triton_poi_fused__to_copy_div_gather_mul_permute_sigmoid_sum_unsqueeze_view_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i64', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_div_gather_mul_permute_sigmoid_sum_unsqueeze_view_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_div_gather_mul_permute_sigmoid_sum_unsqueeze_view_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 2048
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (4*x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (16384 + x2), None).to(tl.float32)
    tmp15 = tl.load(in_ptr1 + (1 + 4*x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr0 + (32768 + x2), None).to(tl.float32)
    tmp28 = tl.load(in_ptr1 + (2 + 4*x1), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr0 + (49152 + x2), None).to(tl.float32)
    tmp41 = tl.load(in_ptr1 + (3 + 4*x1), None, eviction_policy='evict_last')
    tmp2 = tl.full([XBLOCK], 32, tl.int32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 < 0
    tmp5 = tl.where(tmp4, tmp3, tmp1)
    tl.device_assert((0 <= tmp5) & (tmp5 < 32), "index out of bounds: 0 <= tmp5 < 32")
    tmp7 = tl.load(in_ptr2 + (tmp5 + 32*x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tl.sigmoid(tmp8)
    tmp11 = (tmp9 / tmp10)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp0 * tmp12
    tmp16 = tmp15 + tmp2
    tmp17 = tmp15 < 0
    tmp18 = tl.where(tmp17, tmp16, tmp15)
    tl.device_assert((0 <= tmp18) & (tmp18 < 32), "index out of bounds: 0 <= tmp18 < 32")
    tmp20 = tl.load(in_ptr2 + (tmp18 + 32*x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tl.sigmoid(tmp21)
    tmp23 = (tmp22 / tmp10)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp14 * tmp24
    tmp26 = tmp13 + tmp25
    tmp29 = tmp28 + tmp2
    tmp30 = tmp28 < 0
    tmp31 = tl.where(tmp30, tmp29, tmp28)
    tl.device_assert((0 <= tmp31) & (tmp31 < 32), "index out of bounds: 0 <= tmp31 < 32")
    tmp33 = tl.load(in_ptr2 + (tmp31 + 32*x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tl.sigmoid(tmp34)
    tmp36 = (tmp35 / tmp10)
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tmp27 * tmp37
    tmp39 = tmp26 + tmp38
    tmp42 = tmp41 + tmp2
    tmp43 = tmp41 < 0
    tmp44 = tl.where(tmp43, tmp42, tmp41)
    tl.device_assert((0 <= tmp44) & (tmp44 < 32), "index out of bounds: 0 <= tmp44 < 32")
    tmp46 = tl.load(in_ptr2 + (tmp44 + 32*x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp47 = tmp46.to(tl.float32)
    tmp48 = tl.sigmoid(tmp47)
    tmp49 = (tmp48 / tmp10)
    tmp50 = tmp49.to(tl.float32)
    tmp51 = tmp40 * tmp50
    tmp52 = tmp39 + tmp51
    tl.store(out_ptr0 + (x2), tmp52, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1 = args
        args.clear()
        assert_size_stride(arg0_1, (8, 2048), (2048, 1))
        assert_size_stride(arg1_1, (8, 2048), (2048, 1))
        assert_size_stride(arg2_1, (2048, ), (1, ))
        assert_size_stride(arg3_1, (6144, 2048), (2048, 1))
        assert_size_stride(arg4_1, (2048, 3), (3, 1))
        assert_size_stride(arg5_1, (8, ), (1, ))
        assert_size_stride(arg6_1, (64, 2048, 3), (6144, 3, 1))
        assert_size_stride(arg7_1, (2048, 2048), (2048, 1))
        assert_size_stride(arg8_1, (2048, ), (1, ))
        assert_size_stride(arg9_1, (32, 2048), (2048, 1))
        assert_size_stride(arg10_1, (32, ), (1, ))
        assert_size_stride(arg11_1, (32, 3584, 2048), (7340032, 2048, 1))
        assert_size_stride(arg12_1, (32, 2048, 1792), (3670016, 1792, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            # Topologically Sorted Source Nodes: [fused_add_rmsnorm_default], Original ATen: [sgl_kernel.fused_add_rmsnorm]
            torch.ops.sgl_kernel.fused_add_rmsnorm.default(arg1_1, arg0_1, arg2_1, 1e-05, True)
            del arg2_1
            buf3 = empty_strided_cuda((8, 6144), (6144, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [proj], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(arg1_1, reinterpret_tensor(arg3_1, (2048, 6144), (1, 2048), 0), out=buf3)
            del arg1_1
            del arg3_1
            buf4 = empty_strided_cuda((8, 2048), (2048, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [chunk, Bx], Original ATen: [aten.split, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_split_0.run(buf3, buf4, 16384, stream=stream0)
            buf5 = empty_strided_cuda((8, ), (1, ), torch.int32)
            # Topologically Sorted Source Nodes: [to, causal_conv1d_update], Original ATen: [aten._to_copy, aten.as_strided, sgl_kernel.causal_conv1d_update]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_as_strided_causal_conv1d_update_1.run(arg5_1, buf5, 8, stream=stream0)
            del arg5_1
            # Topologically Sorted Source Nodes: [to, causal_conv1d_update], Original ATen: [aten._to_copy, aten.as_strided, sgl_kernel.causal_conv1d_update]
            torch.ops.sgl_kernel.causal_conv1d_update.default(reinterpret_tensor(buf4, (8, 2048, 1), (2048, 1, 1), 0), arg6_1, arg4_1, None, False, None, buf5, -1)
            del arg4_1
            del arg6_1
            del buf5
            buf10 = empty_strided_cuda((8, 2048), (2048, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [chunk, mul_1], Original ATen: [aten.split, aten.unsqueeze, aten.squeeze, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_split_squeeze_unsqueeze_2.run(buf3, buf4, buf10, 16384, stream=stream0)
            del buf3
            buf11 = buf4; del buf4  # reuse
            # Topologically Sorted Source Nodes: [chunk, mul_1, hidden_states], Original ATen: [aten.split, aten.unsqueeze, aten.squeeze, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(buf10, reinterpret_tensor(arg7_1, (2048, 2048), (1, 2048), 0), out=buf11)
            del arg7_1
            del buf10
            # Topologically Sorted Source Nodes: [fused_add_rmsnorm_default_1], Original ATen: [sgl_kernel.fused_add_rmsnorm]
            torch.ops.sgl_kernel.fused_add_rmsnorm.default(buf11, arg0_1, arg8_1, 1e-05, True)
            del arg0_1
            del arg8_1
            buf15 = empty_strided_cuda((8, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [router_logits], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf11, reinterpret_tensor(arg9_1, (2048, 32), (1, 2048), 0), out=buf15)
            del arg9_1
            buf16 = empty_strided_cuda((8, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [float_1, scores, scores_for_choice], Original ATen: [aten._to_copy, aten.sigmoid, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_sigmoid_3.run(buf15, arg10_1, buf16, 256, stream=stream0)
            del arg10_1
            # Topologically Sorted Source Nodes: [float_1, scores, scores_for_choice, topk], Original ATen: [aten._to_copy, aten.sigmoid, aten.add, aten.topk]
            buf17 = torch.ops.aten.topk.default(buf16, 4)
            del buf16
            buf19 = buf17[1]
            assert_size_stride(buf19, (8, 4), (4, 1), 'torch.ops.aten.topk.default')
            assert_alignment(buf19, 16, 'torch.ops.aten.topk.default')
            del buf17
            buf20 = empty_strided_cuda((4, 3584, 2048), (7340032, 2048, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_9, getitem_10], Original ATen: [aten.select, aten.index]
            stream0 = get_raw_stream(0)
            triton_poi_fused_index_select_4.run(buf19, arg11_1, buf20, 29360128, stream=stream0)
            del arg11_1
            buf21 = empty_strided_cuda((8, 14336), (14336, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [gate_up, getitem_9, getitem_10], Original ATen: [aten.unsqueeze, aten.view, aten.bmm, aten.select, aten.index, aten.permute]
            extern_kernels.mm(buf11, reinterpret_tensor(buf20, (2048, 14336), (1, 2048), 0), out=buf21)
            del buf20
            buf22 = empty_strided_cuda((8, 4, 1792), (7168, 1792, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [gate_up, chunk_1, silu, act], Original ATen: [aten.bmm, aten.view, aten.permute, aten.split, aten.silu, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_bmm_mul_permute_silu_split_view_5.run(buf21, buf22, 57344, stream=stream0)
            del buf21
            buf23 = empty_strided_cuda((4, 2048, 1792), (3670016, 1792, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_13, getitem_14], Original ATen: [aten.select, aten.index]
            stream0 = get_raw_stream(0)
            triton_poi_fused_index_select_6.run(buf19, arg12_1, buf23, 14680064, stream=stream0)
            del arg12_1
            buf24 = empty_strided_cuda((4, 8, 2048), (16384, 2048, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [gate_up, chunk_1, silu, act, per_expert, getitem_13, getitem_14], Original ATen: [aten.bmm, aten.view, aten.permute, aten.split, aten.silu, aten.mul, aten.unsqueeze, aten.select, aten.index]
            extern_kernels.bmm(reinterpret_tensor(buf22, (4, 8, 1792), (1792, 7168, 1), 0), reinterpret_tensor(buf23, (4, 1792, 2048), (3670016, 1, 1792), 0), out=buf24)
            del buf22
            del buf23
            buf25 = empty_strided_cuda((8, 1), (1, 8), torch.float32)
            # Topologically Sorted Source Nodes: [float_1, scores, topk_w, sum_1], Original ATen: [aten._to_copy, aten.sigmoid, aten.gather, aten.sum]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_gather_sigmoid_sum_7.run(buf19, buf15, buf25, 8, stream=stream0)
            buf26 = buf11; del buf11  # reuse
            # Topologically Sorted Source Nodes: [float_1, scores, per_expert, topk_w, sum_1, topk_w_1, topk_w_2, unsqueeze_1, mul_3, out], Original ATen: [aten._to_copy, aten.sigmoid, aten.view, aten.permute, aten.gather, aten.sum, aten.div, aten.unsqueeze, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_div_gather_mul_permute_sigmoid_sum_unsqueeze_view_8.run(buf24, buf19, buf15, buf25, buf26, 16384, stream=stream0)
            del buf15
            del buf19
            del buf24
            del buf25
        return (buf26, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((8, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((8, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((2048, 3), (3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg5_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg6_1 = rand_strided((64, 2048, 3), (6144, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg7_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg8_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg9_1 = rand_strided((32, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg10_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((32, 3584, 2048), (7340032, 2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg12_1 = rand_strided((32, 2048, 1792), (3670016, 1792, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
