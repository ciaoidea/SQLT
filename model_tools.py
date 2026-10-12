"""Model-selected native Granite tool calls; no user-language routing rules."""
import json
import time
from math_engine import calculate,validate_request

SYSTEM_INSTRUCTIONS = (
    'Sei un assistente utile. Rispondi in italiano in modo conciso. '
    'For mathematical calculations output ONLY a JSON object: '
    '{"action":"calculate","expression":"...","operation":"..."}. '
    'Operations: evaluate, derivative, integral (also antiderivatives), limit, series, ode. '
    'Optional arguments: variable (default x), order, variables (mixed partial list), '
    'lower and upper (STRING bounds, infinity "oo"), point (STRING), direction (+,-,+-). '
    'Expressions use exp, log, sin, cos, sqrt, pi and parentheses. '
    'ODE syntax: diff(y(x),x,2)+y(x)=0. '
    'Otherwise answer normally, without JSON. Never print tool schemas or invent tool results.')


def decode_decision(answer):
    text=answer.strip()
    if text.startswith('```json'):
        text=text[7:].lstrip()
    try:payload,_=json.JSONDecoder().raw_decode(text)
    except (ValueError,TypeError):return None
    if not isinstance(payload,dict):return None
    if payload.get('action')=='answer' and set(payload)=={'action'}:return payload
    if payload.get('action')!='calculate':return None
    if not isinstance(payload.get('expression'),str) or not 1<=len(payload['expression'])<=256:return None
    return payload


def decode_protocol_call(answer):
    # Compatibility for stored native tool turns; never inspect user wording.
    text=answer.strip().removeprefix('<|tool_call|>').strip()
    if text.startswith('```json'):text=text[7:].lstrip()
    try:payload,_=json.JSONDecoder().raw_decode(text)
    except (ValueError,TypeError):return None
    if not isinstance(payload,list) or len(payload)!=1:return None
    call=payload[0]
    if not isinstance(call,dict):return None
    call=call.get('function',call)
    if not isinstance(call,dict) or call.get('name')!='calculate':return None
    arguments=call.get('arguments')
    if isinstance(arguments,str):
        try:arguments=json.loads(arguments)
        except ValueError:return None
    if not isinstance(arguments,dict) or not isinstance(arguments.get('expression'),str):return None
    return arguments


def decode_call(answer):
    arguments=decode_protocol_call(answer)
    if arguments is None:return None
    try:validate_request(**arguments)
    except (ValueError,TypeError):return None
    return arguments


def protocol_output(text):
    """Inspect generated output only, never the user's language."""
    return text.lstrip().startswith(('{', '[', '```', '<|', 'tools', 'tool_response'))


def generate_with_tools(model,messages,max_tokens=512,callback=None):
    started=time.perf_counter()
    previous_tools=getattr(model,'tools_enabled',False)
    previous_planning=getattr(model,'planning_mode',False)
    model.tools_enabled=True
    model.planning_mode=False
    total_tokens=0
    try:
        def stream(text):
            # Hold the short prefix until prose can be distinguished from the
            # model's internal protocol, then restore progressive rendering.
            if callback and len(text.lstrip()) >= 8 and not protocol_output(text):
                callback(text)
        history=list(messages)
        for attempt in range(2):
            result=model.generate(history,max_tokens=max_tokens,callback=stream)
            total_tokens+=result.get('tokens',0)
            decision=decode_decision(result['answer'])
            arguments=({k:v for k,v in decision.items() if k!='action'}
                       if decision and decision['action']=='calculate' else
                       decode_protocol_call(result['answer']))
            if arguments is not None:
                try:validate_request(**arguments)
                except (ValueError,TypeError) as exc:
                    error=f'Invalid calculation arguments: {exc}. Return corrected calculation JSON only.'
                else:
                    tool=calculate(**arguments)
                    # The model already interpreted the question. Show the real
                    # calculated result directly, without a second neural pass.
                    result.update(answer=tool['answer'],tool_expression=arguments['expression'],
                        tool_operation=arguments.get('operation','evaluate'),
                        tool_result=tool['answer'],tool_seconds=tool['seconds'],finish_reason='tool_result')
                    break
            elif not protocol_output(result['answer']):
                break
            else:
                error='Invalid internal protocol. Answer normally, or return only valid calculation JSON. Never output schemas or fabricated tool results.'
            history=[*messages,{'role':'assistant','content':result['answer']},
                     {'role':'user','content':error}]
        else:
            result.update(answer='Il modello ha generato un protocollo interno non valido. Nessun calcolo eseguito.',finish_reason='tool_error')
        if callback:callback(result['answer'])
        result.update(tokens=total_tokens,seconds=time.perf_counter()-started)
        return result
    finally:
        model.tools_enabled=previous_tools
        model.planning_mode=previous_planning
