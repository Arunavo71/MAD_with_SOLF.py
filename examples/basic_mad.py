import logging, os
from argparse import ArgumentParser
from dotenv import load_dotenv
from litellm import completion

from mad.prompt_templates import *
from mad.agents.Agent import Agent
from mad.dialogues.Dialogue import Dialogue

parser = ArgumentParser()
parser.add_argument('-t', '--topic', help='Specifies the topic the agents should debate.')
args = parser.parse_args()
load_dotenv()

API_KEY = os.getenv("MISTRAL_API_KEY", "")
LOG_LEVEL = os.getenv("LOG_LEVEL", "")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "")
MODEL = os.getenv("MODEL", "")
SEMANTICS = os.getenv("SEMANTICS", "")
WAIT_TIME = int(os.getenv("WAIT_FOR_RATELIMIT", ""))

logging.basicConfig()
logging.getLogger().setLevel(LOG_LEVEL)
logger = logging.getLogger(__name__)
  
def prompt(input_prompt: str):
    response = completion(
        model=f'{MODEL_PROVIDER}/{MODEL}',
        messages=[{"role": "user", "content": input_prompt}],
        response_format={ "type": "json_object" })
    return response.choices[0].message.content


pro_agent = Agent('pro', prompt, logger)
con_agent = Agent('con', prompt, logger)
dialogue = Dialogue([args.topic], [pro_agent, con_agent], prompt, logger, {'convergence': 0.01, 'window_size': 3})
dialogue.run_dialogue()