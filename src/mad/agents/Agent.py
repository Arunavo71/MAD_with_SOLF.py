import logging, random
from qbaf import QBAFramework
from mad.prompt_templates import generate_debater_prompt

class Agent:
    """
    Allows instantiating agents participating in argumentation dialogues.
    """

    def __init__(self, stances: dict, prompt: function, logger: logging.Logger):
        """Initializes the agent.

        Args:
            stances (dict): The stances the agent has on the topics, either 'pro' or 'con', {topicId: stance} dict.
            prompt (function): Prompt function for natural language-based inference.
            logger (logging.Logger): Logger, e.g., for debugging or command line output.
        """
        self.stances = stances
        self.prompt = prompt
        self.logger = logger

    def take_turn(self, qbaf: dict, textual_descriptions: list[str], topics=['topic0']) -> str:
        """Lets the agent take a turn and utter an argument and support/attack.

        Args:
            qbaf (dict): Current QBAF, as dict representation.
            textual_descriptions (list[str]): List containing the natural language content of all arguments.
            topics (list[str]): List containing the IDs of all topic arguments.

        Returns:
            str: Stringified JSON object with the following internal format: {arg: <argument as string>, target:  <target argument ID>, type: <support or attack>}
        """
        deltas = {}
        for topic in topics:
            topic_index = qbaf['arguments'].index(topic)
            if self.stances[topic] == 'pro':
                deltas[topic] = qbaf['final_strengths'][topic_index] - qbaf['initial_strengths'][topic_index]
            else:
                deltas[topic] = qbaf['initial_strengths'][topic_index] - qbaf['final_strengths'][topic_index]
        max_delta = max(deltas.values())
        max_topics = [topic for topic, delta in deltas.items() if delta == max_delta]
        topic = random.choice(max_topics)
        self.logger.info(f'topic: {topic}')
        qpy_qbaf = QBAFramework(qbaf['arguments'], qbaf['final_strengths'], qbaf['attacks'], qbaf['supports'])
        agent_prompt = generate_debater_prompt(qpy_qbaf, textual_descriptions, True if self.stances[topic] == 'con' else False, topic)
        self.logger.info(f'{self.stances[topic]} prompt {agent_prompt}')
        return self.prompt(agent_prompt)