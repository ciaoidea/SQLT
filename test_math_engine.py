import unittest
from math_engine import parse_expression, solve_expression, calculate

class MathTests(unittest.TestCase):
    def test_quadratic_roots(self):
        answer=solve_expression('x^2+5x-6=0')
        self.assertIn('x = -6',answer)
        self.assertIn('x = 1',answer)
        self.assertNotIn('x = -8',answer)
        self.assertIn('x = -6',solve_expression('x^2+5*x-6'))

    def test_exact_arithmetic_and_units_style_expression(self):
        self.assertEqual(str(parse_expression('0.1+0.2')), '3/10')
        self.assertEqual(str(parse_expression('sqrt(3^2+4^2)')), '5')
        result=float(parse_expression('1/(2*pi*sqrt(0.025*0.0000004))'))
        self.assertAlmostEqual(result,1591.54943091895,places=7)

    def test_real_domain_and_implicit_multiplication(self):
        self.assertIn('Nessuna soluzione reale',solve_expression('x^2+1=0'))
        self.assertIn('x = 3',solve_expression('2x=6'))
        self.assertIn('Ogni numero reale',solve_expression('x=x'))
        self.assertIn('Nessuna soluzione',solve_expression('x=x+1'))

    def test_disallowed_code_and_expensive_expressions(self):
        for expression in ['__import__("os")','x.__class__','[1,2]','2**1000000','1/0','sin(x, x)','9'*300]:
            with self.subTest(expression=expression), self.assertRaises((ValueError,SyntaxError)):
                parse_expression(expression)

    def test_worker(self):
        answer=calculate('x^2+5x-6=0')
        self.assertEqual(answer['engine'],'sympy')
        self.assertIn('x = -6',answer['answer'])

if __name__=='__main__':unittest.main()
