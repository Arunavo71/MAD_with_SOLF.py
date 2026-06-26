from qbaf import QBAFramework
from mad.qbaf_utils import *

def generate_debater_prompt(qbaf: QBAFramework, textual_descriptions: str, pro: bool=True):
    return f'''
    You are an excellent and eloquent debater.
    Argue {'for' if pro else 'against'} the following topic: {textual_descriptions['topic0']}.
    Your argument must be brief, not more than a short sentence.
    Do so by either attacking or supporting the following arguments:
    {[{'text': textual_descriptions[arg], 'id': arg} for arg in get_untargeted_args(qbaf)]}.
    Expected output: {{"arg": <argument as string>, "target":  <target argument ID>, "type": <support or attack>}}.
    Produce exactly one argument in the format above, not an array.
    Ensure that the response's 'target' key refers to the target argument's ID, not its text.
    '''

def generate_oracle_prompt(argument: str, attacks: list[str], supports: list[str]):
    return f'''
    An agent has provided the following argument: {argument}
    This argument is supposed to attack the following arguments: {attacks}
    Also, the argument is supposed to support the the following arguments: {supports}
    Assign a score between 0 and 1 to the initial argument.
    The return format should be: {{"score": "<score>"}}
    '''