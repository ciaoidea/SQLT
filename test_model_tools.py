import unittest
from unittest.mock import Mock,patch
from model_tools import decode_call,decode_decision,generate_with_tools

class ToolTests(unittest.TestCase):
    def test_neural_decision_then_actual_result_explained(self):
        model=Mock()
        model.generate.side_effect=[{'answer':'{"action":"calculate","expression":"x^2+5*x-6=0"}','tokens':25},
            {'answer':'Le soluzioni sono -6 e 1.','tokens':10}]
        with patch('model_tools.calculate',return_value={'answer':'Verified: -6 and 1','seconds':0.1}) as tool:
            answer=generate_with_tools(model,[{'role':'user','content':'Which values make this zero?'}])
        tool.assert_called_once_with(expression='x^2+5*x-6=0')
        final_context=model.generate.call_args_list[1].args[0]
        self.assertIn('Verified',final_context[-1]['content'])
        self.assertEqual(answer['tool_expression'],'x^2+5*x-6=0')

    def test_bad_operation_is_corrected_by_model(self):
        model=Mock()
        model.generate.side_effect=[
            {'answer':'{"action":"calculate","expression":"sin(x)","operation":"primitive"}'},
            {'answer':'{"action":"calculate","expression":"sin(x)","operation":"integral"}'},
            {'answer':'La primitiva è -cos(x)+C.'}]
        with patch('model_tools.calculate',return_value={'answer':'-cos(x)+C','seconds':0.1}) as tool:
            result=generate_with_tools(model,[{'role':'user','content':'Find an antiderivative.'}])
        tool.assert_called_once_with(expression='sin(x)',operation='integral')
        self.assertEqual(result['tool_operation'],'integral')

    def test_no_user_keyword_dispatch(self):
        model=Mock()
        model.generate.side_effect=[{'answer':'{"action":"answer"}'},{'answer':'Una risposta normale.'}]
        with patch('model_tools.calculate') as tool:
            result=generate_with_tools(model,[{'role':'user','content':'risolvi x^2+5x-6=0'}])
        self.assertEqual(model.generate.call_count,2)
        tool.assert_not_called()
        self.assertEqual(result['answer'],'Una risposta normale.')

    def test_invalid_protocol_is_hidden_and_bounded(self):
        model=Mock()
        model.generate.return_value={'answer':'tools [{"name":"calculate"}] tools_response {"result":1}'}
        stream=Mock()
        with patch('model_tools.calculate') as tool:
            result=generate_with_tools(model,[{'role':'user','content':'Compute something'}],callback=stream)
        self.assertEqual(model.generate.call_count,2)
        tool.assert_not_called()
        stream.assert_not_called()
        self.assertNotIn('tools_response',result['answer'])
        self.assertEqual(result['finish_reason'],'tool_error')

    def test_fabricated_tool_response_never_reaches_screen(self):
        model=Mock()
        model.generate.side_effect=[{'answer':'{"action":"calculate","expression":"2+2"}'},
            {'answer':'tools_response {"result":1}'}]
        stream=Mock()
        with patch('model_tools.calculate',return_value={'answer':'4','seconds':0.1}):
            result=generate_with_tools(model,[{'role':'user','content':'Compute'}],callback=stream)
        self.assertEqual(result['answer'],'4')
        stream.assert_called_once_with('4')

    def test_protocol_validation(self):
        self.assertEqual(decode_decision('{"action":"answer"}'),{'action':'answer'})
        self.assertIsNone(decode_decision('tools [{"name":"calculate"}]'))
        self.assertIsNone(decode_call('[{"name":"shell","arguments":{"expression":"x"}}]'))

if __name__=='__main__':unittest.main()
