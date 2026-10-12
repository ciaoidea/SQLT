"""Offline installation smoke checks; never load or update model weights."""
import json
import torch
from prompt_toolkit import PromptSession
from rich.console import Console
from learn import train_step
from math_engine import perform
from model_tools import decode_decision
from terminal_input import PauseControl
from terminal_ui import TerminalChat


def check():
    weight=torch.tensor(2.0,requires_grad=True)
    ((weight-1)**2).backward()
    if weight.grad.item()!=2.0:raise RuntimeError('Autograd check failed')
    checks={
        'arithmetic':'x = -6' in perform('x^2+5x-6=0'),
        'derivative':'3*x**2' in perform('x^3','derivative'),
        'integral':'exp(x**2)/2' in perform('x*exp(x^2)','integral'),
        'numeric_integral_fallback':'6.493939402266829' in perform('x^3/(exp(x)-1)','integral',lower='0',upper='oo'),
        'limit':'`1`' in perform('sin(x)/x','limit',point='0'),
        'series':'O(x**5)' in perform('sin(x)','series',order=5),
        'ode':'confermata' in perform('diff(y(x),x,2)+y(x)=0','ode'),
        'tool_protocol':decode_decision('{"action":"calculate","expression":"x^2","operation":"derivative"}') is not None,
        'training_import':callable(train_step),
        'input_editor':callable(PromptSession),
        'terminal_ui':callable(Console) and callable(TerminalChat),
        'escape_pause':callable(PauseControl.checkpoint),
    }
    if not all(checks.values()):raise RuntimeError('Feature checks failed: '+str(checks))
    return {'ready':True,'checks':checks}


if __name__=='__main__':print(json.dumps(check(),indent=2))
