"""Small resource and latency checks used by automatic device selection."""
import statistics
import time
import torch
from torch.nn import functional as F


def memory_budgets(core_mib,expert_mib,requested_cache_mib,requested_host_mib,
                   ram_mib,gpu_free_mib):
    cpu=max(0,min(requested_cache_mib,ram_mib-core_mib-2048))
    gpu=max(0,min(requested_cache_mib,gpu_free_mib-core_mib-768,expert_mib))
    host=max(0,min(requested_host_mib,expert_mib,ram_mib-core_mib-4096))
    return dict(cpu_cache_mib=int(cpu),gpu_cache_mib=int(gpu),host_cache_mib=int(host))


def preferred_device(requested,budgets,timings=None):
    if requested=='cpu':return 'cpu','CPU requested'
    if budgets['gpu_cache_mib']<384:return 'cpu','Insufficient free GPU memory'
    if requested=='cuda':return 'cuda','CUDA requested; memory available'
    if timings and timings['cpu_ms']<timings['estimated_gpu_ms']*0.85:
        return 'cpu','CPU faster in the local calibration, including GPU transfers'
    return 'cuda','GPU memory available and local calibration favors CUDA'


def calibrate(projection_factory,quantized,gpu_cache_mib,expert_mib):
    """Measure a representative layer, including transfers for cache misses.

    This predicts token decoding, not SQL loading or prompt processing.
    """
    def measure(operation,cuda=False):
        operation()
        if cuda:torch.cuda.synchronize()
        samples=[]
        for _ in range(3):
            started=time.perf_counter();operation()
            if cuda:torch.cuda.synchronize()
            samples.append((time.perf_counter()-started)*1000)
        return statistics.median(samples)
    with torch.inference_mode(),torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        win=torch.randn(1024,1536)*0.01;wout=torch.randn(1536,512)*0.01
        core=torch.randn(1536,1536)*0.01;core2=core.clone();x=torch.randn(1,1536)
        cpu_weights=[(projection_factory(win),projection_factory(wout)) if quantized else
            (win.bfloat16(),wout.bfloat16()) for _ in range(8)]
        def cpu():
            F.linear(F.linear(x,core),core2)
            for wi,wo in cpu_weights:
                p=wi(x) if quantized else F.linear(x,wi.float())
                a,b=p.chunk(2,dim=-1)
                if quantized:wo(F.silu(a)*b)
                else:F.linear(F.silu(a)*b,wo.float())
        cpu_ms=measure(cpu)
        host=(win.bfloat16().pin_memory(),wout.bfloat16().pin_memory())
        gx=x.cuda();gc=core.cuda();gc2=core2.cuda()
        def upload():return [(a.to('cuda',non_blocking=True),b.to('cuda',non_blocking=True)) for a,b in [host]*8]
        weights=upload()
        def gpu(matrices):
            F.linear(F.linear(gx,gc),gc2)
            wi=torch.stack([v[0] for v in matrices]).float()
            wo=torch.stack([v[1] for v in matrices]).float()
            p=torch.bmm(wi,gx.view(1,-1,1).expand(8,-1,-1)).squeeze(-1)
            a,b=p.chunk(2,dim=-1)
            torch.bmm(wo,(F.silu(a)*b).unsqueeze(-1))
        hot=measure(lambda:gpu(weights),True)
        cold=measure(lambda:gpu(upload()),True)
        misses=max(0.,1-gpu_cache_mib/max(expert_mib,1))
        estimated=hot+max(0,cold-hot)*misses
        return dict(cpu_ms=cpu_ms,gpu_resident_ms=hot,gpu_transfer_ms=cold,
            estimated_gpu_ms=estimated,estimated_gpu_miss_fraction=misses)
