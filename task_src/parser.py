#!/usr/bin/env python

import yaml
import itertools
import re

class Task:
    def __init__(self, src):
        self.name = src['name']
        self.events = src['events']
        self.queries = src['queries']
        self.right_answer_regex = re.compile(src['right_answer'])
        self.wrong_answer_regex = re.compile("|".join(src['wrong_answers']))

    def all_prompts(self):
        for sequence in itertools.permutations(self.events):
            for query in self.queries:
                yield ' '.join(sequence) + " " + query

    def score_reply(self, reply):
        if self.wrong_answer_regex.search(reply):
            return 0.0
        if not self.right_answer_regex.search(reply):
            return 0.0
        return 1.0

def load_task(filename):
    with open(filename, "r") as f:
        return Task(yaml.safe_load(f))

task = load_task("pool.yaml")
for p in task.all_prompts():
    print("*****")
    print(p)
