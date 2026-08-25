import json, logging, time

from qbaf import QBAFramework

import matplotlib.pyplot as plt

from mad.prompt_templates import *
from mad.agents.Agent import Agent

from qbaf_solf.fairness_notions import *
from qbaf_solf.safety_oscillations_liveness import *

qbaf_collection = []


def calculate_threshold_excess(qbaf_collection: list[QBAFramework],
                               topic_set: list[str],
                               threshold: float) -> dict:

    """
      Calculates the threshold exceeding instances of the topic set.

      Args:
        qbaf_collection (list[QBAFramework]): The collection of QBAFs.
        topic_set (list[str]): The considered set of arguments.
        threshold (float): The credibility threshold.

      Returns:
        dict: returns the number of threshold exceeding instances.
    """

    instances_of_credibility = dict()

    for topic_arg in topic_set:
      instances_of_credibility.update({topic_arg: len([qbaf for qbaf in qbaf_collection if (qbaf.final_strengths[topic_arg] >= threshold)])})

    return instances_of_credibility

def solf_analysis(qbaf_coll: list[QBAFramework], topic_set: list[str], threshold: float) -> None:
    """ Provides a SOL analysis of the updated QBAF
        
    Args: 
        qbaf_collection (list[QBAFramework]): The updated list of QBAFramework.
        topic_set (list[QBAFramework]): The topic set of the analysis.
        threshold (float): The threshold w.r.t. which SOLF analysis is done.
    """

    print(f'The topic set is {topic_set}')
    print(f'Is the topic set safe? {is_safe(qbaf_coll, topic_set, threshold)}')
    print(f'Is the topic set live? {is_live(qbaf_coll, topic_set, threshold)}')
    print(f'The osciallations shown across {threshold} by the topic set is {number_of_oscillations(qbaf_coll, topic_set, threshold)}')

    return None

def fairness_analysis(qbaf_coll: list[QBAFramework], topic_set: list[str], threshold: float) -> None:
    """ Provides a SOL analysis of the updated QBAF
            
        Args: 
            qbaf_collection (list[QBAFramework]): The updated list of QBAFramework.
            topic_set (list[QBAFramework]): The topic set of the analysis.
            threshold (float): The threshold w.r.t. which SOLF analysis is done.
    """
    
    print(f'The topic set is {topic_set}')
    print(f'Is the topic set ideally_safe? {is_ideal_fair(qbaf_coll, topic_set, threshold)}')
    print(f'Is the topic set live? {is_live_fair(qbaf_coll, topic_set, threshold)}')
    print(f'The osciallations shown across {threshold} by the topic set is {is_cautious_fair(qbaf_coll, topic_set, threshold)}')

    print(f'The Gini fairness score of the topic set is {calculate_gini_fairness(qbaf_coll, topic_set, threshold)}')
    print(f'The Shannon fairness score of the topic set is {calculate_shannon_fairness(qbaf_coll, topic_set, threshold)}')

    return None


def visualise_fairness_gini(topic_set: list[str],
                        qbaf_collection: list[QBAFramework], 
                        threshold: float) -> None:
    """
    Draws the area plot for Gini fairness.
    
    Args:
        topic_set (list[str]): The topic set.
        qbaf_collection (list[QBAFramework]): The dialogue considered
        threshold (float): The credibility threshold.
    
    """

    x_axis = range(len(topic_set)+1)
    sorted_oscillations = sorted(calculate_threshold_excess(qbaf_collection, topic_set, threshold).items(), key=lambda item: item[1])
    safety_curve = [0]
    for x in sorted_oscillations:
      safety_curve.append(safety_curve[-1] + int(x[1]))

    # Calculating the fairness line
    fairness_line = [(safety_curve[-1]/len(topic_set)) * x for x in x_axis]

    plt.xlabel('Topic Arguments')
    plt.ylabel('Credibility Attainment instances')
    plt.xticks(x_axis, ['']+[x[0] for x in sorted_oscillations])
    plt.plot(x_axis, safety_curve, c= 'red', marker='s')
    plt.plot(x_axis, fairness_line, c= 'green', linestyle= 'dashed', marker= 'o')
    plt.legend()
    plt.fill_between(x_axis, safety_curve, fairness_line, alpha=0.4)

    plt.show()

    return None


def visualise_fairness_shannon(topic_set: list[str],
                        qbaf_collection: list[QBAFramework], 
                        threshold: float) -> None:
    """
    Draws the probability plot for Shannon fairness.
    
    Args:
        topic_set (list[str]): The topic set.
        qbaf_collection (list[QBAFramework]): The dialogue considered
        threshold (float): The credibility threshold.
    
    """
    oscillations = calculate_threshold_excess(qbaf_collection, topic_set, threshold)
    if all(oscillations[x] == 0 for x in oscillations.keys()):
      return 1
    sum_of_oscillations = sum([x[1] for x in oscillations.items()])
    oscillation_probability = {x: oscillations[x]/sum_of_oscillations for x in oscillations.keys()}

    topic_set = [x[0] for x in oscillation_probability]
    prob_of_surprise = [x[1] for x in oscillation_probability]

    bars = plt.bar(
        topic_set,
        prob_of_surprise,
        color="skyblue",
        edgecolor="black",
        width=0.6)

    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.show()

    return None


class Dialogue:
    """
    Allows instantiating dialogues that orchestrate MADs according to protocols.

    """

    def __init__(self, topics: list[str], agents: list[Agent], prompt: function, logger: logging.Logger, stop_condition: dict, sleep_time=1, semantics="DFQuAD_model"):
        """Manages the MAD, i.e., turn-taking and argument strengths assessment.

        Args:
            topics (list[str]): List of topics agents engaging in the dialogue should debate.
            agents (list[Agent]): List of Agents engaging in the dialogue.
            prompt (function): Prompt function for natural language-based inference.
            logger (logging.Logger): Logger, e.g., for debugging or command line output.
            stop_condition (dict): Defines when the dialogue should stop (when topic strength converge, or after a number of iterations).
            sleep_time (int, optional): Waiting time (in seconds) before prompts to avoid issues with API limits. Defaults to 1.
            semantics (str, optional): Gradual semantics used for QBAF evaluation. Defaults to "DFQuAD_model".
        """
        self.topics = topics
        self.topic_ids = [f'topic{index}' for index, _ in enumerate(topics)]
        self.stop_condition = stop_condition
        self.agents = agents
        self.logger = logger
        self.logger = logger
        self.sleep_time = sleep_time
        self.qbafs = []
        self.prompt = prompt
        self.textual_descriptions = {}
        self.semantics = semantics
        for index, topic in enumerate(topics):
            self.textual_descriptions[f'topic{index}'] = topic
        args = [] + self.topic_ids
        initial_strengths = [0.5 for _ in args]
        initial_qbaf = QBAFramework(args, initial_strengths, [], [], semantics=semantics)
        self.qbafs.append(initial_qbaf)
        self.qbaf_dicts = []
        self.qbaf_dicts.append({
            'arguments': args,
            'initial_strengths': initial_strengths,
            'attacks': [],
            'supports': [],
            'final_strengths': initial_strengths
        })

    def update_qbaf(self, qbaf: dict, agent_update: str, iteration: int, agentID: int) -> dict:
        """Updates a QBAF given an agent's update proposal.

        Args:
            qbaf (dict): Dictionary representing a QBAF with `arguments`, `initial_strengths`, `attacks`, and `supports`.
            agent_update (str): String containing an agent's proposed QBAF update.
            iteration (int): Current iteration of the dialogue.
            agentID (int): Identifier of the current agent.

        Returns:
            dict: Dictionary representing a QBAF with `arguments`, `initial_strengths`, `attacks`, and `supports`.
        """
        args = qbaf['arguments']
        initial_strengths = qbaf['initial_strengths']
        atts = qbaf['attacks']
        supps = qbaf['supports']
        j_agent_update = json.loads(agent_update)

        if 'arg' in j_agent_update:
            arg = f'A{agentID};I:{iteration}'
            textual_description = j_agent_update['arg']
            self.textual_descriptions[arg] = textual_description
            if 'target' in j_agent_update and 'type' in j_agent_update:
                target = j_agent_update['target']
                type = j_agent_update['type']
                new_atts = []
                new_supps = []
                if target in args:
                    if type == 'attack':
                        atts.append((arg, target))
                        new_atts.append((arg, target))
                    else:
                        supps.append((arg, target))
                        new_supps.append((arg, target))
                    args.append(arg)
                    time.sleep(self.sleep_time)
                    oracle_prompt = generate_oracle_prompt(textual_description, new_atts, new_supps)
                    initial_strength_assessment = self.prompt(oracle_prompt)
                    initial_strength = json.loads(initial_strength_assessment)
                    self.logger.info(f'initial strength: {initial_strength}')
                    initial_strengths.append(initial_strength['score'])
        qpy_qbaf = QBAFramework(args, initial_strengths, atts, supps, semantics=self.semantics)
        qbaf_collection.append(qpy_qbaf)
        topic_set = ['topic'+str(i) for i in range(0, len(self.topics))]
        #print(f"The topic set is {topic_set}")
        solf_analysis(qbaf_collection, topic_set, threshold=0.5)
        final_strengths = [qpy_qbaf.final_strength(arg) for arg in args]
        return {
            'arguments': args,
            'initial_strengths': initial_strengths,
            'attacks': atts,
            'supports': supps,
            'final_strengths': final_strengths
        }


    def run_turn(self, iteration: int):
        """Moves the dialogue forward by one turn.

        Args:
            iteration (int): Current iteration.
        """
        qbaf_dict = self.qbaf_dicts[iteration]
        agent_updates = []
        self.logger.info(f'Iteration: {iteration}')
        for index, agent in enumerate(self.agents):
            time.sleep(self.sleep_time)
            agent_update = agent.take_turn(qbaf_dict, self.textual_descriptions, self.topic_ids)
            agent_updates.append(agent_update)
            self.logger.info(f'Agent {index}: {agent_update}')
            qbaf_dict = self.update_qbaf(qbaf_dict, agent_update, iteration, index)
        self.qbaf_dicts.append(qbaf_dict)
        self.logger.info(qbaf_dict)
        args = qbaf_dict['arguments']
        initial_strengths = qbaf_dict['initial_strengths']
        atts = qbaf_dict['attacks']
        supps = qbaf_dict['supports']
        qbaf = QBAFramework(args, initial_strengths, atts, supps, semantics=self.semantics)
        self.qbafs.append(qbaf)
        self.logger.info(self.topic_ids)
        self.logger.info(f'Current topic strengths: {[qbaf.final_strength(topic) for topic in self.topic_ids]}')
    
    def run_dialogue(self):
        """Runs the dialogue until the stop condition is reached.
        """
        if 'iterations' in self.stop_condition:
            for i in range(self.stop_condition['iterations']):
                self.run_turn(i)
        if 'convergence' in self.stop_condition:
            i = 0
            delta_threshold = self.stop_condition['convergence']
            window_size = 2
            if 'window_size' in self.stop_condition:
                window_size = self.stop_condition['window_size']
            topic_strengths = []
            max_deltas = []
            while len(topic_strengths) < window_size or max(max_deltas) > delta_threshold:
                self.run_turn(i)
                convergence_window = self.qbafs[-window_size:]
                max_deltas = []
                for topic_id in self.topic_ids:
                    topic_strengths = [qbaf.final_strength(topic_id) for qbaf in convergence_window]
                    deltas =  [abs(topic_strength_1 - topic_strength_2) for topic_strength_1 in topic_strengths for topic_strength_2 in topic_strengths]
                    max_deltas.append(max(deltas))
                i += 1

        topic_set = ['topic'+str(i) for i in range(0, len(self.topics))]    
        fairness_analysis(topic_set=topic_set, qbaf_coll=qbaf_collection, threshold=0.5)

        visualise_fairness_gini(topic_set=topic_set, qbaf_collection=qbaf_collection, threshold=0.5)
        visualise_fairness_shannon(topic_set=topic_set, qbaf_collection=qbaf_collection, threshold=0.5)