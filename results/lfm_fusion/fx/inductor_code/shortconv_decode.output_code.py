
# AOT ID: ['1_inference']
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
#   %mul : Tensor "bf16[8, 2048][2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem, %getitem_2), kwargs = {})
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
#   %arg3_1 : Tensor "i64[8][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %convert_element_type_2 : Tensor "i32[8][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg3_1, torch.int32), kwargs = {})
#   %as_strided_default : Tensor "bf16[8, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.as_strided.default](args = (%mul, [8, 2048, 1], [2048, 1, 1], 0), kwargs = {})
#   %causal_conv1d_update_default : [num_users=0] = call_function[target=torch.ops.sgl_kernel.causal_conv1d_update.default](args = (%as_strided_default, %arg4_1, %arg2_1, None, False, None, %convert_element_type_2, -1), kwargs = {})
#   return %buf2
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
#   %buf4 : Tensor  = PlaceHolder[target=buf4]
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%mm, 2048, -1), kwargs = {})
#   %unsqueeze_1 : Tensor "bf16[8, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul, -1), kwargs = {})
#   %squeeze_1 : Tensor "bf16[8, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%unsqueeze_1, -1), kwargs = {})
#   %mul_1 : Tensor "bf16[8, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_1, %squeeze_1), kwargs = {})
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = args
        args.clear()
        assert_size_stride(arg0_1, (6144, 2048), (2048, 1))
        assert_size_stride(arg1_1, (8, 2048), (2048, 1))
        assert_size_stride(arg2_1, (2048, 3), (3, 1))
        assert_size_stride(arg3_1, (8, ), (1, ))
        assert_size_stride(arg4_1, (64, 2048, 3), (6144, 3, 1))
        assert_size_stride(arg5_1, (2048, 2048), (2048, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((8, 6144), (6144, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [proj], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(arg1_1, reinterpret_tensor(arg0_1, (2048, 6144), (1, 2048), 0), out=buf0)
            del arg0_1
            del arg1_1
            buf1 = empty_strided_cuda((8, 2048), (2048, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [chunk, Bx], Original ATen: [aten.split, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_split_0.run(buf0, buf1, 16384, stream=stream0)
            buf2 = empty_strided_cuda((8, ), (1, ), torch.int32)
            # Topologically Sorted Source Nodes: [to, causal_conv1d_update], Original ATen: [aten._to_copy, aten.as_strided, sgl_kernel.causal_conv1d_update]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_as_strided_causal_conv1d_update_1.run(arg3_1, buf2, 8, stream=stream0)
            del arg3_1
            # Topologically Sorted Source Nodes: [to, causal_conv1d_update], Original ATen: [aten._to_copy, aten.as_strided, sgl_kernel.causal_conv1d_update]
            torch.ops.sgl_kernel.causal_conv1d_update.default(reinterpret_tensor(buf1, (8, 2048, 1), (2048, 1, 1), 0), arg4_1, arg2_1, None, False, None, buf2, -1)
            del arg2_1
            del arg4_1
            del buf2
            buf7 = empty_strided_cuda((8, 2048), (2048, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [chunk, mul_1], Original ATen: [aten.split, aten.unsqueeze, aten.squeeze, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_split_squeeze_unsqueeze_2.run(buf0, buf1, buf7, 16384, stream=stream0)
            del buf0
            buf8 = buf1; del buf1  # reuse
            # Topologically Sorted Source Nodes: [chunk, mul_1, linear_1], Original ATen: [aten.split, aten.unsqueeze, aten.squeeze, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(buf7, reinterpret_tensor(arg5_1, (2048, 2048), (1, 2048), 0), out=buf8)
            del arg5_1
            del buf7
        return (buf8, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((8, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((2048, 3), (3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg4_1 = rand_strided((64, 2048, 3), (6144, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg5_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
