import unittest
from unittest.mock import Mock,patch
from model_tools import decode_call,decode_decision,generate_with_tools

class ToolTests(unittest.TestCase):
    def test_model_decision_uses_real_result_without_second_generation(self):
        model=Mock()
        model.generate.return_value={'answer':'{"action":"calculate","expression":"x^2+5*x-6=0"}','tokens':25}
        with patch('model_tools.calculate',return_value={'answer':'Verified: -6 and 1','seconds':0.1}) as tool:
            result=generate_with_tools(model,[{'role':'user','content':'Which values make this zero?'}])
        tool.assert_called_once_with(expression='x^2+5*x-6=0')
        self.assertEqual(model.generate.call_count,1)
        self.assertEqual(result['answer'],'Verified: -6 and 1')

    def test_bad_operation_is_corrected_by_model(self):
        model=Mock()
        model.generate.side_effect=[
            {'answer':'{"action":"calculate","expression":"sin(x)","operation":"primitive"}'},
            {'answer':'{"action":"calculate","expression":"sin(x)","operation":"integral"}'}]
        with patch('model_tools.calculate',return_value={'answer':'-cos(x)+C','seconds':0.1}) as tool:
            result=generate_with_tools(model,[{'role':'user','content':'Find an antiderivative.'}])
        tool.assert_called_once_with(expression='sin(x)',operation='integral')
        self.assertEqual(result['tool_operation'],'integral')

    def test_no_keyword_dispatch_and_single_pass_streaming(self):
        model=Mock(); stream=Mock()
        def generate(messages,**kwargs):
            kwargs['callback']('Una risposta')
            kwargs['callback']('Una risposta normale.')
            return {'answer':'Una risposta normale.'}
        model.generate.side_effect=generate
        with patch('model_tools.calculate') as tool:
            result=generate_with_tools(model,[{'role':'user','content':'risolvi x^2+5x-6=0'}],callback=stream)
        self.assertEqual(model.generate.call_count,1)
        tool.assert_not_called()
        self.assertEqual(result['answer'],'Una risposta normale.')
        self.assertEqual(stream.call_args_list[0].args[0],'Una risposta')

    def test_invalid_protocol_is_hidden_and_bounded(self):
        model=Mock(); stream=Mock()
        def generate(messages,**kwargs):
            for text in ('t','too','tools','tools_response {"result":1}'):
                kwargs['callback'](text)
            return {'answer':'tools_response {"result":1}'}
        model.generate.side_effect=generate
        with patch('model_tools.calculate') as tool:
            result=generate_with_tools(model,[{'role':'user','content':'Compute'}],callback=stream)
        self.assertEqual(model.generate.call_count,2)
        tool.assert_not_called()
        stream.assert_called_once_with(result['answer'])
        self.assertNotIn('tools_response',result['answer'])
        self.assertEqual(result['finish_reason'],'tool_error')

    def test_short_prose_and_flags_restored(self):
        model=Mock(tools_enabled=False,planning_mode=False)
        model.generate.return_value={'answer':'Ciao!'}
        stream=Mock()
        generate_with_tools(model,[{'role':'user','content':'Ciao'}],callback=stream)
        stream.assert_called_once_with('Ciao!')
        self.assertFalse(model.tools_enabled)
        self.assertFalse(model.planning_mode)

    def test_protocol_validation(self):
        self.assertEqual(decode_decision('{"action":"answer"}'),{'action':'answer'})
        self.assertIsNone(decode_decision('tools [{"name":"calculate"}]'))
        self.assertIsNone(decode_call('[{"name":"shell","arguments":{"expression":"x"}}]'))

if __name__=='__main__':unittest.main()
