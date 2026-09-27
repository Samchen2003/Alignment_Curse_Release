import json
import os

from mutation_utils import persuasion_demo_mutation
import pandas as pd
import numpy as _np


POLICY = [
    'Logical appeal',
    'Authority endorsement',
    'Misrepresentation',
    'Evidence-based Persuasion',
    'Expert Endorsement'
]


def get_pap_prompt_list(goal):


    rewritten_list = []     

    for template in POLICY:
        rewritten_prompt = persuasion_demo_mutation(template, goal)
        rewritten_list.append(rewritten_prompt)


    return rewritten_list


