"""Exercise expert reuse, eviction and coefficient preservation."""
import unittest
from concurrent.futures import ThreadPoolExecutor
import threading
from unittest.mock import Mock
from unittest.mock import patch
import torch
from runtime import ExpertCache


class HostCacheTests(unittest.TestCase):
    def cache(self,host_bytes=32):
        parameters={i:{'id':i,'shape':'[2,2]'} for i in range(6)}
        experts={(0,i):{'input_parameter':2*i,'output_parameter':2*i+1} for i in range(3)}
        cache=ExpertCache(None,parameters,experts,12,host_limit_mib=host_bytes/1024**2)
        # One expert fits in the execution cache, two in host RAM.
        cache.layer_limit=16
        return cache

    def test_execution_eviction_reuses_host_without_database_reads(self):
        cache=self.cache()
        with patch('runtime.load_parameter',return_value=torch.ones(2,2)) as load:
            first=cache.get(0,0)
            cache.get(0,1)
            again=cache.get(0,0)
        self.assertEqual(load.call_count,4)
        self.assertTrue(torch.equal(first[0],again[0]))
        self.assertEqual(cache.host_hits,1)
        self.assertEqual(cache.database_loads,2)
        self.assertLessEqual(cache.bytes,cache.layer_limit)
        self.assertLessEqual(cache.host_bytes,cache.host_limit)

    def test_host_eviction_is_bounded_and_keeps_recently_used_expert(self):
        cache=self.cache()
        with patch('runtime.load_parameter',return_value=torch.ones(2,2)) as load:
            for expert in (0,1,0,2):cache.get(0,expert)
            self.assertNotIn((0,1),cache.host_items)
            self.assertIn((0,0),cache.host_items)
            cache.get(0,0)
        self.assertEqual(load.call_count,6)
        self.assertLessEqual(cache.host_bytes,cache.host_limit)

    def test_non_bf16_coefficients_remain_exact(self):
        cache=self.cache(host_bytes=64)
        cache.layer_limit=32
        weight=torch.full((2,2),0.1)
        with patch('runtime.load_parameter',return_value=weight):
            result=cache.get(0,0)
        self.assertEqual(result[0].dtype,torch.float32)
        self.assertTrue(torch.equal(result[0],weight))

    def test_disabled_host_cache_reloads_evicted_experts(self):
        cache=self.cache(host_bytes=0)
        with patch('runtime.load_parameter',return_value=torch.ones(2,2)) as load:
            for expert in (0,1,0):cache.get(0,expert)
        self.assertEqual(load.call_count,6)
        self.assertEqual(cache.host_bytes,0)

    def test_execution_cache_clear_preserves_host_for_cpu_fallback(self):
        cache=self.cache()
        with patch('runtime.load_parameter',return_value=torch.ones(2,2)) as load:
            cache.get(0,0)
            cache.items.clear();cache.bytes=0;cache.layer_bytes={0:0}
            cache.get(0,0)
        self.assertEqual(load.call_count,2)
        self.assertEqual(cache.host_hits,1)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA is unavailable')
    def test_gpu_eviction_reuploads_host_tensors_without_sql(self):
        cache=self.cache();cache.device=torch.device('cuda')
        with patch('runtime.load_parameter',return_value=torch.ones(2,2)) as load:
            cache.get(0,0)
            cache.get(0,1)
            result=cache.get(0,0)
        self.assertEqual(load.call_count,4)
        self.assertEqual(result[0].device.type,'cuda')
        self.assertEqual(cache.host_items[(0,0)][0].device.type,'cpu')
        self.assertTrue(torch.equal(result[0].cpu(),cache.host_items[(0,0)][0]))

    def test_prefetch_uses_separate_connections_and_closes_them(self):
        cache=self.cache();cache.executor=ThreadPoolExecutor(max_workers=2)
        barrier=threading.Barrier(2)
        connections=[Mock(),Mock()]
        def load(connection,metadata):
            self.assertIn(connection,connections)
            if metadata['id']%2==0:barrier.wait(timeout=5)
            return torch.ones(2,2)
        try:
            with patch('runtime.connect',side_effect=connections) as connect, \
                 patch('runtime.load_parameter',side_effect=load):
                cache.prepare(0,[0,1])
                cache.get(0,0);cache.get(0,1)
                self.assertEqual(connect.call_count,2)
        finally:cache.close()
        for connection in connections:connection.close.assert_called_once()

    def test_prefetch_errors_are_reported_to_the_caller(self):
        cache=self.cache();cache.executor=ThreadPoolExecutor(max_workers=1)
        try:
            with patch('runtime.connect',return_value=Mock()), \
                 patch('runtime.load_parameter',side_effect=ValueError('checksum failed')):
                cache.prepare(0,[0])
                with self.assertRaisesRegex(ValueError,'checksum failed'):cache.get(0,0)
        finally:cache.close()

    def test_ram_pressure_evicts_old_host_entries(self):
        cache=self.cache()
        with patch('runtime.load_parameter',return_value=torch.ones(2,2)):
            cache.get(0,0);cache.get(0,1)
        cache.trim_host(16/1024**2)
        self.assertEqual(cache.host_bytes,16)
        self.assertEqual(list(cache.host_items),[(0,1)])


if __name__=='__main__':
    unittest.main()
