import unittest,warnings
import torch
from torch.ao.nn.quantized.dynamic import Linear
from runtime import Int8Projection


class Int8Tests(unittest.TestCase):
    def test_direct_packing_matches_dynamic_linear_exactly(self):
        torch.backends.quantized.engine='fbgemm'
        generator=torch.Generator().manual_seed(42)
        weight=torch.randn(10,12,generator=generator)
        scales=weight.abs().amax(dim=1).clamp_min(1e-12)/127
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            q=torch.quantize_per_channel(weight,scales,torch.zeros(10,dtype=torch.int64),0,torch.qint8)
        reference=Linear(12,10,bias_=False);reference.set_weight_bias(q,None)
        actual=Int8Projection(weight)
        for shape in ((1,12),(8,12),(2,3,12)):
            inputs=torch.randn(*shape,generator=generator)
            self.assertTrue(torch.equal(actual(inputs),reference(inputs)))


if __name__=='__main__':unittest.main()
