import logging
from mad.prompt_templates import generate_debater_prompt

class Agent:
    """
    Allows instantiating agents participating in argumentation dialogues.
    """

    def __init__(self, stance: str, prompt: function, logger: logging.Logger):
        """Initializes the agent.

        Args:
            stance (str): The stance the agent has on the topic, either 'pro' or 'con'.
            prompt (function): Prompt function for natural language-based inference.
            logger (logging.Logger): Logger, e.g., for debugging or command line output.
        """
        self.stance = stance
        self.prompt = prompt
        self.logger = logger

    def take_turn(self, qbaf: dict, textual_descriptions: list[str]) -> str:
        """Lets the agent take a turn and utter an argument and support/attack.

        Args:
            qbaf (dict): Current QBAF, as dict representation.
            textual_descriptions (list[str]): List containing the natural language content of all arguments.

        Returns:
            str: Stringified JSON object with the following internal format: {arg: <argument as string>, target:  <target argument ID>, type: <support or attack>}
        """
        agent_prompt = generate_debater_prompt(qbaf, textual_descriptions, True if self.stance == 'con' else False)
        self.logger.info(f'{self.stance} prompt {agent_prompt}')
        return self.prompt(agent_prompt)