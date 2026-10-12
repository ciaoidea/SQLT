"""Model-selected native Granite tool calls; no user-language routing rules."""
import json
import time
from math_engine import calculate,validate_request

SYSTEM_INSTRUCTIONS = 'Sei un assistente utile. Rispondi in italiano in modo conciso.'
PLANNER_INSTRUCTIONS = (
    'Decide whether the latest user request needs a mathematical calculation. '
    'Return ONLY one JSON object. For ordinary conversation return {"action":"answer"}. '
    'For math return {"action":"calculate","expression":"...","operation":"..."} with needed arguments. '
    'Operations: evaluate (arithmetic/equations), derivative, integral (also antiderivatives), limit, series, ode. '
    'Optional arguments: variable (default x), order, variables (mixed partial list), lower and upper '
    '(both STRING bounds for definite integrals; infinity "oo"), point (STRING for limits/series), direction (+,-,+-). '
    'Expressions use exp, log, sin, cos, sqrt, pi, ^ and explicit parentheses. '
    'ODE: diff(y(x),x,2)+y(x)=0. Choose the calculation from the meaning of the request. '
    'Never write tool definitions or invent a tool response. Do not compute the answer yourself.')


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


def generate_with_tools(model,messages,max_tokens=512,callback=None):
    started=time.perf_counter()
    previous_tools=getattr(model,'tools_enabled',False)
    previous_planning=getattr(model,'planning_mode',False)
    model.tools_enabled=False
    total_tokens=0
    try:
        # No keyword dispatch, expression extraction or user-language parsing.
        # The neural model itself produces the operation and its arguments.
        context=[dict(m) for m in messages if m['role']!='system'][-3:]
        planning=[{'role':'system','content':PLANNER_INSTRUCTIONS}]+context
        tool=None
        arguments=None
        for attempt in range(2):
            model.planning_mode=True
            decision_result=model.generate(planning,max_tokens=min(max_tokens,144),callback=None)
            total_tokens+=decision_result.get('tokens',0)
            decision=decode_decision(decision_result['answer'])
            if decision is None:
                error='Return only the required JSON decision object; no tool schema or explanation.'
            elif decision['action']=='answer':break
            else:
                arguments={k:v for k,v in decision.items() if k!='action'}
                try:validate_request(**arguments)
                except (ValueError,TypeError) as exc:
                    error=f'Invalid calculation arguments: {exc}. Correct the JSON. For antiderivatives use operation integral.'
                else:
                    tool=calculate(**arguments)
                    break
            planning=[{'role':'system','content':PLANNER_INSTRUCTIONS},
                *context,{'role':'user','content':error}]
        else:
            return dict(answer='Il modello non ha formulato una chiamata matematica valida. Nessun calcolo eseguito.',
                        finish_reason='tool_error',tokens=total_tokens,seconds=time.perf_counter()-started)
        model.planning_mode=False
        if tool is None:
            result=model.generate(messages,max_tokens=max_tokens,callback=None)
            answer=result['answer'].strip()
            if answer.startswith(('tools','tool_response','tools_response','<|')):
                result['answer']='Il modello ha generato un protocollo interno non valido. Nessun risultato è stato verificato.'
                result['finish_reason']='tool_error'
            if callback:callback(result['answer'])
        else:
            # Keep protocol/schema and old failed tool responses out of the
            # explanation context. The result comes from an actual invocation.
            question=next(m['content'] for m in reversed(messages) if m['role']=='user')
            explanation=[{'role':'system','content':
                'Rispondi in italiano in modo conciso alla domanda usando il risultato verificato fornito. '
                'Non inventare calcoli, chiamate o risultati di strumenti. Una forma simbolica non risolta '
                'non implica divergenza. Mantieni esplicita la natura approssimata dei valori numerici.'},
                {'role':'user','content':question+'\n\nRisultato effettivamente calcolato:\n'+tool['answer']}]
            result=model.generate(explanation,max_tokens=max_tokens,callback=None)
            answer=result['answer'].strip()
            # Never publish a fabricated protocol turn. Numeric results remain
            # available verbatim if the model outputs protocol instead of prose.
            if (decode_decision(answer) is not None or decode_protocol_call(answer) is not None
                    or answer.startswith(('tools','tool_response','tools_response','<|'))):
                result['answer']=tool['answer']
            result.update(tool_expression=arguments['expression'],tool_operation=arguments.get('operation','evaluate'),
                          tool_result=tool['answer'],tool_seconds=tool['seconds'])
            if callback:callback(result['answer'])
        result.update(tokens=total_tokens+result.get('tokens',0),seconds=time.perf_counter()-started)
        return result
    finally:
        model.tools_enabled=previous_tools
        model.planning_mode=previous_planning
