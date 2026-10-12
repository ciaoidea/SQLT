import unittest
from unittest.mock import Mock,patch
from model_tools import decode_call,generate_with_tools

class ToolTests(unittest.TestCase):
    def test_neural_call_then_tool_result_returned_to_model(self):
        model=Mock()
        model.generate.side_effect=[{'answer':'[{"name":"calculate","arguments":{"expression":"x^2+5x-6=0"}}]','tokens':30},
            {'answer':'Le soluzioni sono -6 e 1.','tokens':10}]
        with patch('model_tools.calculate',return_value={'answer':'Verified: -6 and 1','seconds':0.1}) as tool:
            answer=generate_with_tools(model,[{'role':'user','content':'Which values make this zero?'}])
        tool.assert_called_once_with(expression='x^2+5x-6=0')
        followup=model.generate.call_args_list[1].args[0]
        self.assertEqual(followup[-1]['role'],'tool_response')
        self.assertIn('Verified',followup[-1]['content'])
        self.assertEqual(answer['tool_expression'],'x^2+5x-6=0')

    def test_invalid_operation_is_returned_to_model_for_repair(self):
        model=Mock()
        model.generate.side_effect=[
            {'answer':'[{"name":"calculate","arguments":{"expression":"sin(x)","operation":"primitive"}}]'},
            {'answer':'[{"name":"calculate","arguments":{"expression":"sin(x)","operation":"integral"}}]'},
            {'answer':'La primitiva è -cos(x)+C.'}]
        with patch('model_tools.calculate',return_value={'answer':'-cos(x) + C','seconds':0.1}) as tool:
            result=generate_with_tools(model,[{'role':'user','content':'Find an antiderivative.'}])
        tool.assert_called_once_with(expression='sin(x)',operation='integral')
        self.assertIn('Argument error',model.generate.call_args_list[1].args[0][-1]['content'])
        self.assertEqual(result['tool_operation'],'integral')

    def test_no_keyword_routing(self):
        model=Mock()
        model.generate.return_value={'answer':'Una risposta normale.'}
        with patch('model_tools.calculate') as tool:
            result=generate_with_tools(model,[{'role':'user','content':'risolvi x^2+5x-6=0'}])
        model.generate.assert_called_once()
        tool.assert_not_called()
        self.assertEqual(result['answer'],'Una risposta normale.')

    def test_invalid_model_calls_do_not_execute(self):
        for text in ['hello','[]','[{"name":"shell","arguments":{"expression":"x"}}]',
                     '[{"name":"calculate","arguments":{"expression":23}}]']:
            self.assertIsNone(decode_call(text))

if __name__=='__main__':unittest.main()
