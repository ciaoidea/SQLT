import unittest
from resource_policy import memory_budgets,preferred_device


class ResourceTests(unittest.TestCase):
    def test_gpu_budget_reserves_model_and_workspace(self):
        plan=memory_budgets(1064,5760,3584,6144,16000,4096)
        self.assertEqual(plan['gpu_cache_mib'],2264)
        self.assertEqual(plan['host_cache_mib'],5760)

    def test_busy_gpu_uses_cpu(self):
        plan=memory_budgets(1064,5760,3584,6144,16000,900)
        self.assertEqual(preferred_device('auto',plan)[0],'cpu')
        self.assertEqual(preferred_device('cuda',plan)[0],'cpu')

    def test_cpu_wins_when_gpu_transfers_cost_more(self):
        plan={'gpu_cache_mib':2000}
        self.assertEqual(preferred_device('auto',plan,{'cpu_ms':4,'estimated_gpu_ms':8})[0],'cpu')

    def test_fast_gpu_is_selected(self):
        plan={'gpu_cache_mib':2000}
        self.assertEqual(preferred_device('auto',plan,{'cpu_ms':8,'estimated_gpu_ms':4})[0],'cuda')

    def test_forced_cpu_does_not_use_gpu(self):
        self.assertEqual(preferred_device('cpu',{'gpu_cache_mib':6000})[0],'cpu')

    def test_ram_pressure_reduces_cache(self):
        plan=memory_budgets(1064,5760,3584,6144,4000,4096)
        self.assertEqual(plan['cpu_cache_mib'],888)
        self.assertEqual(plan['host_cache_mib'],0)

    def test_budgets_are_never_negative(self):
        plan=memory_budgets(1064,5760,3584,6144,500,100)
        self.assertEqual(plan,{'cpu_cache_mib':0,'gpu_cache_mib':0,'host_cache_mib':0})


if __name__=='__main__':unittest.main()
