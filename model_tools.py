"""Model-selected native Granite tool calls; no user-language routing rules."""
import json
import time
from math_engine import calculate,validate_request

TOOLS=[{'type':'function','function':{
    'name':'calculate',
    'description':'Exact arithmetic, real polynomial equations, derivatives, integrals, limits, Taylor series and ordinary differential equations. Convert units before calling. Functions: sin cos tan exp log sqrt abs. ODE syntax: diff(y(x),x,2)+y(x)=0.',
    'parameters':{'type':'object','properties':{
        'expression':{'type':'string','description':'Expression or equation; symbolic polynomial without = means =0'},
        'operation':{'type':'string','enum':['evaluate','derivative','integral','limit','series','ode'],'description':'Default evaluate; select derivative/integral/etc for calculus'},
        'variable':{'type':'string','enum':['x','y','z','t'],'description':'Default x'},
        'order':{'type':'integer','description':'Derivative order (default 1) or Taylor remainder order (default 6); maximum 10'},
        'variables':{'type':'array','items':{'type':'string'},'description':'Optional mixed derivative sequence, e.g. ["x","y"]'},
        'lower':{'type':'string','description':'Definite integral lower bound; omit both bounds for indefinite integral'},
        'upper':{'type':'string','description':'Definite integral upper bound; infinity is oo'},
        'point':{'type':'string','description':'Required limit point; Taylor expansion point defaults to 0'},
        'direction':{'type':'string','enum':['+','-','+-'],'description':'Limit side, default both sides'}},
        'required':['expression'],'additionalProperties':False}}}]
SYSTEM_INSTRUCTIONS = ('Sei un assistente utile. Rispondi in italiano in modo conciso. '
    'You have access to the calculate tool. For calculations, derivatives, integrals, limits, Taylor series and ODEs use the tool. Choose the operation and required arguments. '
    'When a tool is required, respond only with <|tool_call|> followed by a JSON list: '
    '[{"name":"calculate","arguments":{"expression":"expression here"}}]. '
    'For questions that do not require calculation, answer normally. '
    'Use exactly the operation enum names: an antiderivative or primitive uses integral, never primitive. '
    'For example, integrating sin(x) uses arguments {"expression":"sin(x)","operation":"integral","variable":"x"}. '
    'After a successful tool_response explain its result faithfully. If the tool reports an argument error, correct the call once.')


def decode_protocol_call(answer):
    text=answer.strip().removeprefix('<|tool_call|>').strip()
    if text.startswith('```json') and text.endswith('```'):
        text=text[7:-3].strip()
    try: payload,_=json.JSONDecoder().raw_decode(text)
    except (ValueError,TypeError): return None
    if not isinstance(payload,list) or len(payload)!=1: return None
    call=payload[0]
    if not isinstance(call,dict): return None
    call=call.get('function',call)
    if not isinstance(call,dict) or call.get('name')!='calculate': return None
    arguments=call.get('arguments')
    if isinstance(arguments,str):
        try: arguments=json.loads(arguments)
        except ValueError: return None
    if not isinstance(arguments,dict):return None
    if not isinstance(arguments.get('expression'),str) or not 1<=len(arguments['expression'])<=256:return None
    return arguments


def decode_call(answer):
    arguments=decode_protocol_call(answer)
    if arguments is None:return None
    try:validate_request(**arguments)
    except (ValueError,TypeError):return None
    return arguments


def generate_with_tools(model,messages,max_tokens=512,callback=None):
    started=time.perf_counter()
    def initial_stream(value):
        # Only the model's generated protocol is inspected, never user wording.
        clean=value.lstrip()
        if callback and clean and not clean.startswith(('[','<|tool_call|>','{','```')):
            callback(value)
    result=model.generate(messages,max_tokens=max_tokens,callback=initial_stream)
    history=list(messages)
    total_tokens=result.get('tokens',0)
    last_tool=None
    for attempt in range(2):
        arguments=decode_protocol_call(result['answer'])
        if arguments is None:
            if callback:callback(result['answer'])
            if last_tool:
                result.update(tool_expression=last_tool[0]['expression'],
                    tool_operation=last_tool[0].get('operation','evaluate'),
                    tool_result=last_tool[1]['answer'],tool_seconds=last_tool[1]['seconds'],
                    seconds=time.perf_counter()-started,tokens=total_tokens)
            return result
        try:
            validate_request(**arguments)
        except (ValueError,TypeError) as exc:
            tool={'answer':f'Argument error: {exc}. Use operation integral for a primitive/antiderivative. Allowed operations: evaluate, derivative, integral, limit, series, ode. Correct the call.', 'seconds':0}
        else:
            tool=calculate(**arguments)
            last_tool=(arguments,tool)
        history.extend([
            {'role':'assistant','content':'<|tool_call|>'+json.dumps([{'name':'calculate','arguments':arguments}])},
            {'role':'tool_response','content':json.dumps({'name':'calculate','result':tool['answer']},ensure_ascii=False)}])
        if callback:callback('')
        result=model.generate(list(history),max_tokens=max_tokens,callback=initial_stream)
        total_tokens+=result.get('tokens',0)
        if last_tool:break
    if decode_protocol_call(result['answer']) is not None:
        result['answer']=last_tool[1]['answer'] if last_tool else 'La chiamata matematica generata dal modello non è valida.'
    if callback:callback(result['answer'])
    if last_tool:
        arguments,tool=last_tool
        result.update(tool_expression=arguments['expression'],tool_operation=arguments.get('operation','evaluate'),
            tool_result=tool['answer'],tool_seconds=tool['seconds'])
    result.update(seconds=time.perf_counter()-started,tokens=total_tokens)
    return result
