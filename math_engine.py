"""Bounded local symbolic arithmetic, independent of neural generation."""
import ast
import json
from pathlib import Path
import re
import subprocess
import sys
import time


def parse_expression(text, calculus=False, ode=False):
    import sympy as sp
    if len(text) > 256:
        raise ValueError('Expression exceeds 256 characters')
    text = text.replace('^','**').replace('−','-').replace('×','*').replace('÷','/')
    text = re.sub(r'(?<=\d),(?=\d)', '.', text)
    text = re.sub(r'(?<=[0-9)])(?=[xyztabck(])', '*', text)
    tree = ast.parse(text.strip(), mode='eval')
    if sum(1 for _ in ast.walk(tree)) > 100:
        raise ValueError('Expression too complex')
    names = {k: sp.Symbol(k, real=True) for k in ('x','y','z','t','a','b','c','k')}
    names.update(pi=sp.pi, e=sp.E)
    functions = dict(sqrt=sp.sqrt, sin=sp.sin, cos=sp.cos, tan=sp.tan,
                     log=sp.log, exp=sp.exp, abs=sp.Abs,
                     asin=sp.asin, acos=sp.acos, atan=sp.atan,
                     sinh=sp.sinh, cosh=sp.cosh, tanh=sp.tanh, erf=sp.erf)
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int,float):
            value = sp.Rational(str(node.value))
            if abs(value) > 10**12:
                raise ValueError('Numeric literal too large')
            return value
        if isinstance(node, ast.Name) and node.id in names:
            return names[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
            value = visit(node.operand)
            return -value if isinstance(node.op,ast.USub) else value
        if isinstance(node, ast.BinOp):
            a,b = visit(node.left),visit(node.right)
            if isinstance(node.op,ast.Add): return a+b
            if isinstance(node.op,ast.Sub): return a-b
            if isinstance(node.op,ast.Mult): return a*b
            if isinstance(node.op,ast.Div):
                if b == 0: raise ValueError('Division by zero')
                if b.free_symbols and not calculus: raise ValueError('Variable denominators are not supported')
                return a/b
            if isinstance(node.op,ast.Pow):
                if (not b.is_number and not calculus) or (b.is_number and abs(b) > 20):
                    raise ValueError('Exponent must be numeric and between -20 and 20')
                if a.free_symbols and b.is_number and b < 0 and not calculus:
                    raise ValueError('Negative powers of unknowns are not supported')
                return a**b
        if ode and isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and not node.keywords:
            if node.func.id=='y' and len(node.args)==1:
                argument=visit(node.args[0])
                if argument not in (names['x'],names['t']): raise ValueError('Use y(x) or y(t)')
                return sp.Function('y')(argument)
            if node.func.id=='diff' and len(node.args) in (2,3):
                expression,variable=visit(node.args[0]),visit(node.args[1])
                order=visit(node.args[2]) if len(node.args)==3 else sp.Integer(1)
                if variable not in (names['x'],names['t']) or not order.is_Integer or not 1<=order<=4:
                    raise ValueError('ODE derivatives support orders 1 to 4 in x or t')
                return sp.diff(expression,variable,int(order))
        if (isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
                and node.func.id in functions and len(node.args)==1 and not node.keywords):
            return functions[node.func.id](visit(node.args[0]))
        raise ValueError('Unsupported expression; use arithmetic, sqrt(), pi and x/y/z')
    result = visit(tree.body)
    if result.has(sp.zoo,sp.nan,sp.oo,-sp.oo):
        raise ValueError('Undefined or infinite result')
    return result


def solve_expression(expression):
    import sympy as sp
    if expression.count('=') > 1:
        raise ValueError('Use one equation at a time')
    if '=' not in expression:
        value = parse_expression(expression)
        if value.free_symbols:
            # Standard solver convention: a symbolic polynomial denotes = 0.
            return solve_expression(expression+'=0')
        return f'**Risultato:** `{value}`\n\nValore numerico: `{sp.N(value,12)}`.'
    left,right = (parse_expression(p) for p in expression.split('='))
    symbols = left.free_symbols | right.free_symbols
    if len(symbols) != 1:
        raise ValueError('Use a polynomial equation with exactly one unknown')
    x = next(iter(symbols))
    difference = left-right
    if not difference.is_polynomial(x) or sp.degree(difference,x) > 4:
        raise ValueError('This prototype solves polynomial equations up to degree 4')
    solutions = sp.solveset(difference,x,domain=sp.S.Reals)
    lines = ['**Soluzioni reali**', '']
    if solutions is sp.S.Reals:
        lines.append('Ogni numero reale soddisfa l’equazione.')
    elif solutions is sp.S.EmptySet:
        lines.append('Nessuna soluzione reale.')
    elif isinstance(solutions,sp.FiniteSet):
        for root in sorted(solutions,key=sp.default_sort_key):
            if sp.simplify(left.subs(x,root)-right.subs(x,root)) != 0:
                raise ValueError('Solution verification failed')
            lines.append(f'- `{x} = {root}`' + (f' ≈ `{sp.N(root,10)}`' if not root.is_Rational else ''))
        lines.extend(['', '**Verifica:** ogni soluzione sostituita nell’equazione rende uguali i due membri.'])
    else:
        raise ValueError('No explicit solution available')
    return '\n'.join(lines)


OPERATIONS={'evaluate','derivative','integral','limit','series','ode'}
VARIABLES={'x','y','z','t'}
OPTION_KEYS={'variable','variables','order','lower','upper','point','direction'}


def validate_request(expression,operation='evaluate',**options):
    if not isinstance(expression,str) or not 1<=len(expression)<=256:
        raise ValueError('Invalid mathematical expression')
    if not isinstance(operation,str) or operation not in OPERATIONS or set(options)-OPTION_KEYS:
        raise ValueError('Unsupported operation or argument')
    allowed={'evaluate':set(),'derivative':{'variable','variables','order'},'integral':{'variable','lower','upper'},
             'limit':{'variable','point','direction'},'series':{'variable','point','order'},'ode':{'variable'}}
    if set(options)-allowed[operation]:raise ValueError('Argument does not apply to this operation')
    if not isinstance(options.get('variable','x'),str) or options.get('variable','x') not in VARIABLES:
        raise ValueError('Variable must be x, y, z or t')
    order=options.get('order',1 if operation!='series' else 6)
    if type(order) is not int or not 1<=order<=10:
        raise ValueError('Order must be an integer from 1 to 10')
    variables=options.get('variables',[])
    if not isinstance(variables,list) or len(variables)>10 or any(not isinstance(v,str) or v not in VARIABLES for v in variables):
        raise ValueError('Invalid derivative variable list')
    if variables and 'order' in options:
        raise ValueError('Use either a mixed derivative variable list or an order, not both')
    if ('lower' in options)!=('upper' in options):
        raise ValueError('Supply both integral bounds')
    for field in ('lower','upper','point'):
        if field in options and (not isinstance(options[field],str) or len(options[field])>80):
            raise ValueError('Bounds and limit points must be short strings')
    if options.get('direction','+-') not in ('+','-','+-'):
        raise ValueError('Limit direction must be +, - or +-')


def numerical_integral(value,variable,lower,upper):
    """Precision comparison is a numerical check, not a convergence proof."""
    import sympy as sp
    import mpmath as mp
    if value.free_symbols-{variable} or lower.free_symbols or upper.free_symbols:
        return None
    function=sp.lambdify(variable,value,modules='mpmath')
    def estimate(precision):
        with mp.workdps(precision):
            def endpoint(number):
                if number==sp.oo:return mp.inf
                if number==-sp.oo:return -mp.inf
                return mp.mpf(str(sp.N(number,precision)))
            lo,hi=endpoint(lower),endpoint(upper)
            if mp.isinf(lo) and mp.isinf(hi):points=[lo,-1,0,1,hi]
            elif mp.isinf(hi):points=[lo,lo+1,hi]
            elif mp.isinf(lo):points=[lo,hi-1,hi]
            else:points=[lo,(lo+hi)/2,hi]
            return mp.quad(function,points)
    try:
        first,second=estimate(30),estimate(45)
        with mp.workdps(45):
            if not mp.isfinite(second) or abs(first-second)>mp.mpf('1e-20')*max(1,abs(second)):
                return None
            return mp.nstr(second,20)
    except (ValueError,ZeroDivisionError,OverflowError,TypeError):
        return None


def perform(expression,operation='evaluate',**options):
    import sympy as sp
    validate_request(expression,operation,**options)
    if operation=='evaluate': return solve_expression(expression)
    variable=sp.Symbol(options.get('variable','x'),real=True)
    order=options.get('order',6 if operation=='series' else 1)
    def endpoint(text):
        if text in ('oo','inf','infinity'):return sp.oo
        if text in ('-oo','-inf','-infinity'):return -sp.oo
        value=parse_expression(text,calculus=True)
        if value.has(variable):raise ValueError('A bound or limit point cannot contain its variable')
        return value
    if operation=='ode':
        if expression.count('=')>1:raise ValueError('Use one ODE')
        parts=expression.split('=')
        left=parse_expression(parts[0],calculus=True,ode=True)
        right=parse_expression(parts[1],calculus=True,ode=True) if len(parts)==2 else sp.S.Zero
        function=sp.Function('y')(variable)
        equation=sp.Eq(left,right)
        result=sp.dsolve(equation,function)
        checks=sp.checkodesol(equation,result,func=function)
        verified=all(c[0] for c in checks) if isinstance(checks,list) else checks[0]
        return f'**Soluzione generale:** `{result}`\n\nVerifica per sostituzione: '+('confermata.' if verified else 'non conclusiva.')
    value=parse_expression(expression,calculus=True)
    note=''
    if operation=='derivative':
        variables=options.get('variables') or [str(variable)]*order
        result=sp.diff(value,*[sp.Symbol(v,real=True) for v in variables])
        title='Derivata'
    elif operation=='integral':
        if 'lower' in options:
            result=sp.integrate(value,(variable,endpoint(options['lower']),endpoint(options['upper'])))
            title='Integrale definito'
            if result.has(sp.Integral):
                approximation=numerical_integral(value,variable,endpoint(options['lower']),endpoint(options['upper']))
                if approximation is not None:
                    return (f'**Valore numerico approssimato dell’integrale definito:** {approximation}.\n\n'
                            'La forma simbolica è rimasta non risolta; ciò non significa che il valore sia infinito. '
                            'Stabile confrontando 30 e 45 cifre di precisione; questo controllo non è una prova di convergenza.')
        else:
            result=sp.integrate(value,variable)
            title='Primitiva'
            if not result.has(sp.Integral):
                verified=sp.simplify(sp.diff(result,variable)-value)==0
                note='\n\nAggiungere la costante di integrazione C.'
                if verified:note+=' Verifica mediante derivazione: confermata.'
    elif operation=='limit':
        if 'point' not in options:raise ValueError('A limit requires point')
        point=endpoint(options['point'])
        direction=options.get('direction','+-')
        if direction=='+-' and point not in (sp.oo,-sp.oo):
            left=sp.limit(value,variable,point,dir='-')
            right=sp.limit(value,variable,point,dir='+')
            if left!=right and sp.simplify(left-right)!=0:
                raise ValueError(f'Il limite bilaterale non esiste: limite sinistro {left}, destro {right}')
            result=right
        else:
            result=sp.limit(value,variable,point,dir=direction)
        title='Limite'
    elif operation=='series':
        result=sp.series(value,variable,endpoint(options.get('point','0')),order)
        title='Serie (ordine del resto indicato da O)'
    if result.has(sp.Integral,sp.Derivative,sp.Limit):
        note+='\n\nIl risultato contiene operazioni simboliche non risolte; non è una soluzione esplicita.'
    if result in (sp.oo,-sp.oo,sp.zoo,sp.nan):
        note+='\n\nIl risultato non è un valore finito; verificare convergenza e dominio.'
    return f'**{title}:** `{result}`'+note


def calculate(expression,operation='evaluate',**options):
    validate_request(expression,operation,**options)
    started=time.perf_counter()
    timeout=5 if operation=='evaluate' else 10
    try:
        worker=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker'],
            input=json.dumps(dict(expression=expression,operation=operation,**options)),
            capture_output=True,text=True,timeout=timeout,check=False)
        if worker.returncode:raise ValueError('Mathematical worker unavailable')
        data=json.loads(worker.stdout)
        answer=data.get('answer') or ('Calcolo non completato: '+data['error'])
    except subprocess.TimeoutExpired:
        answer=f'Calcolo interrotto dopo {timeout} secondi.'
    except (ValueError,OSError,KeyError) as exc:
        answer='Calcolo non completato: '+str(exc)
    return dict(answer=answer,seconds=time.perf_counter()-started,tokens=0,finish_reason='math',engine='sympy')


if __name__=='__main__':
    try:
        print(json.dumps({'answer':perform(**json.loads(sys.stdin.read(4096)))},ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({'error':str(exc)},ensure_ascii=False))
