"""Device selection and generation fallback tests without a database."""
import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import Mock, patch
import torch
from runtime import DatabaseMoE, execution_device


class DeviceTests(unittest.TestCase):
    def test_cpu_does_not_probe_cuda(self):
        with patch('torch.cuda.is_available', side_effect=AssertionError):
            self.assertEqual(execution_device('cpu').type, 'cpu')

    def test_auto_uses_cpu_without_cuda(self):
        with patch('torch.cuda.is_available', return_value=False):
            self.assertEqual(execution_device('auto').type, 'cpu')

    def test_failed_cuda_probe_uses_cpu(self):
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.empty', side_effect=RuntimeError('driver unavailable')), \
             warnings.catch_warnings(record=True) as notices:
            self.assertEqual(execution_device('auto').type, 'cpu')
            self.assertTrue(notices)

    def test_gpu_out_of_memory_retries_response_on_cpu(self):
        messages=[{'role':'user','content':'Ciao'}]
        callback=Mock()
        model=SimpleNamespace(device=torch.device('cuda'),
            _generate=Mock(side_effect=[torch.cuda.OutOfMemoryError(), {'answer':'Ciao'}]),
            check_resources=Mock())
        model.fallback_to_cpu=Mock(side_effect=lambda: setattr(model,'device',torch.device('cpu')))
        with warnings.catch_warnings(record=True):
            result=DatabaseMoE.generate(model,messages,2,callback)
        self.assertEqual(result, {'answer':'Ciao'})
        self.assertEqual(model._generate.call_count,2)
        model.fallback_to_cpu.assert_called_once()
        callback.assert_called_once_with('')

    def test_cpu_errors_are_not_retried(self):
        model=SimpleNamespace(device=torch.device('cpu'),
            _generate=Mock(side_effect=torch.cuda.OutOfMemoryError()),
            fallback_to_cpu=Mock(),check_resources=Mock())
        with self.assertRaises(torch.cuda.OutOfMemoryError):
            DatabaseMoE.generate(model,[],1)
        model.fallback_to_cpu.assert_not_called()

    def test_new_gpu_pressure_switches_to_cpu_before_generation(self):
        model=SimpleNamespace(device=torch.device('cuda'),
            cache=SimpleNamespace(limit=384*1024**2,host_bytes=0),fallback_to_cpu=Mock())
        with patch('runtime.available_ram_mib',return_value=10000), \
             patch('torch.cuda.mem_get_info',return_value=(10*1024**2,4096*1024**2)), \
             patch('torch.cuda.memory_reserved',return_value=0), \
             patch('torch.cuda.memory_allocated',return_value=0),warnings.catch_warnings(record=True):
            DatabaseMoE.check_resources(model)
        model.fallback_to_cpu.assert_called_once()


if __name__=='__main__':
    unittest.main()
