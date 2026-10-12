import unittest
from math_engine import calculate,perform,validate_request
from model_tools import decode_call

class CalculusTests(unittest.TestCase):
    def test_derivative_and_mixed_partial(self):
        self.assertIn('x**3*cos(x) + 3*x**2*sin(x)',perform('x^3*sin(x)','derivative'))
        self.assertIn('12*x**2',perform('x^4','derivative',order=2))
        self.assertIn('6*x*y**2',perform('x^2*y^3','derivative',variables=['x','y']))

    def test_indefinite_integral_verified_with_constant(self):
        answer=perform('x*exp(x^2)','integral')
        self.assertIn('exp(x**2)/2',answer)
        self.assertIn('costante di integrazione C',answer)
        self.assertIn('confermata',answer)
        self.assertIn('atan(x)',perform('1/(1+x^2)','integral'))

    def test_definite_and_improper_integral_worker(self):
        answer=calculate('exp(-x^2)',operation='integral',lower='-oo',upper='oo')
        self.assertIn('sqrt(pi)',answer['answer'])
        self.assertIn('1/3',perform('x^2','integral',lower='0',upper='1'))

    def test_limits_and_taylor(self):
        self.assertIn('`1`',perform('sin(x)/x','limit',point='0'))
        self.assertIn('oo',perform('1/x','limit',point='0',direction='+'))
        with self.assertRaises(ValueError):perform('1/x','limit',point='0')
        answer=perform('sin(x)','series',point='0',order=7)
        self.assertIn('O(x**7)',answer)
        self.assertIn('x**5/120',answer)

    def test_ode_verified(self):
        answer=perform('diff(y(x),x,2)+y(x)=0','ode')
        self.assertIn('sin(x)',answer)
        self.assertIn('cos(x)',answer)
        self.assertIn('confermata',answer)

    def test_unresolved_integral_not_presented_as_explicit(self):
        answer=perform('exp(x^x)','integral')
        self.assertIn('non è una soluzione esplicita',answer)

    def test_unresolved_improper_integral_gets_numeric_fallback(self):
        answer=perform('x^3/(exp(x)-1)','integral',lower='0',upper='oo')
        self.assertIn('6.4939394022668291491',answer)
        self.assertNotIn('non è un valore finito',answer)
        self.assertIn('approssimato',answer)
        self.assertIn('non è una prova di convergenza',answer)

    def test_actual_infinite_result_is_still_reported(self):
        self.assertIn('non è un valore finito',perform('exp(x)','integral',lower='0',upper='oo'))

    def test_model_protocol_accepts_calculus_and_checks_options(self):
        call=decode_call('[{"name":"calculate","arguments":{"expression":"x^2","operation":"integral","lower":"0","upper":"1"}}]')
        self.assertEqual(call['operation'],'integral')
        for options in [dict(operation='integral',lower='0'),dict(operation='derivative',order=100),
                        dict(operation='limit',direction='sideways'),dict(operation='series',lower='0',upper='1')]:
            with self.subTest(options=options),self.assertRaises(ValueError):validate_request('x',**options)

if __name__=='__main__':unittest.main()
