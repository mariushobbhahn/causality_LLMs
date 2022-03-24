import numpy as np
import itertools
import pandas as pd
import json

class CETwoSentences():

    def __init__(self):

        self.df_ce_two_sentences = self.generate_sequences_df()
        self.df_ce_two_sentences_non_switched = self.df_ce_two_sentences[self.df_ce_two_sentences["switched"] == False]
        self.df_ce_two_sentences_switched = self.df_ce_two_sentences[self.df_ce_two_sentences["switched"] == True]
        self.length = len(self.df_ce_two_sentences_switched)

        # generate sequences and prompts
        self.question_cause = "Which sentence caused the other?"
        self.question_effect = "Which sentence is the effect of the other?"
        self.answer = " Answer by copying the sentence:"

    def generate_sequences_df(self, filepath="data/bigbench_originals/ce_two_sentences.json"):
        
        with open(filepath, "r") as read_file:
            data = json.load(read_file)

        examples = data["examples"]

        sequences = []
        cause_answers = []
        effect_answers = []
        switched = []

        # run through all examples
        for ex in examples:
            target_scores = ex['target_scores']
            keys = list(target_scores.keys())
            values = list(target_scores.values())
            classic_sequence = keys[0] + " " + keys[1]
            reverse_sequence = keys[1] + " " + keys[0]
            sequences.append(classic_sequence)
            sequences.append(reverse_sequence)
            # 2x because we also flipped the order
            cause_answers.append(keys[0])
            cause_answers.append(keys[0])
            effect_answers.append(keys[1])
            effect_answers.append(keys[1])
            switched.append(False)
            switched.append(True)

        ### save as csv
        df_ce_two_sentences = pd.DataFrame({
            "sequence":sequences,
            "answer_cause":cause_answers,
            "answer_effect":effect_answers,
            "switched":switched
        })

        return(df_ce_two_sentences)

    def save_sequences_df(self, filepath="data/bigbench_csvs/ce_two_sentences.csv", index=False):
        self.df_ce_two_sentences.to_csv(filepath, index=False)

    def get_single_sequence(self, n_sequence, switched=False):
        if switched:
            return(self.df_ce_two_sentences_switched["sequence"].iloc[n_sequence])
        else:
            return(self.df_ce_two_sentences_non_switched["sequence"].iloc[n_sequence])

    def generate_single_prompt_zero_shot(self, n_subset=0, switched=False, question='cause'):
        assert(question in ["cause", "effect"])
        sequence = self.get_single_sequence(n_subset, switched)
        if question=='cause':
            return(sequence + " Question: " + self.question_cause + " " + self.answer)
        elif question=='effect':
            return(sequence + " Question: " + self.question_effect + " " + self.answer)

    def generate_all_prompts_zero_shot(self, question='cause'):
        assert(question in ["cause", "effect"])
        prompts = []
        for s in self.df_ce_two_sentences["sequence"]:
            if question=='cause':
                prompts.append(s + " Question: " + self.question_cause + " " + self.answer)
            elif question=='effect':
                prompts.append(s + " Question: " + self.question_effect + " " + self.answer)
        return(prompts)

    def generate_single_prompt_k_shot(self, n_subset=0, k=5, switched=False, switched_shot=False, question='cause'):
        assert(question in ["cause", "effect"])
        choices = np.random.choice(self.length, size=k, replace=False)
        k_shots = ""
        #create k-shots
        for c in choices: 
            if switched_shot:
                row = self.df_ce_two_sentences_switched.iloc[c]
            else:
                row = self.df_ce_two_sentences_non_switched.iloc[c]
            s = row["sequence"]
            if question=='cause':
                k_shots += s + " Question: " + self.question_cause + self.answer + " " + row["answer_cause"] + "\n"
            elif question=='effect':
                k_shots += s + " Question: " + self.question_effect + self.answer + " " + row["answer_effect"] + "\n"

        #add prompt on top
        prompt = self.generate_single_prompt_zero_shot(n_subset, switched, question)
        return(k_shots + prompt)

    def generate_all_prompts_k_shot(self, k=5, question='cause'):
        assert(question in ["cause", "effect"])
        zero_shot_prompts = self.generate_all_prompts_zero_shot(question)
        k_shots_prompts = []
        for i, row in self.df_ce_two_sentences.iterrows():
            choices = np.random.choice(self.length, size=k, replace=False)
            k_shots = ""
            #create k-shots
            for c in choices: 
                if row["switched"]:
                    r = self.df_ce_two_sentences_switched.iloc[c]
                else:
                    r = self.df_ce_two_sentences_non_switched.iloc[c]
                s = r["sequence"]
                if question=='cause':
                    k_shots += s + " Question: " + self.question_cause + self.answer + " " + r["answer_cause"] + "\n"
                elif question=='effect':
                    k_shots += s + " Question: " + self.question_effect + self.answer + " " + r["answer_effect"] + "\n"
            k_shots += zero_shot_prompts[i]
            k_shots_prompts.append(k_shots)
        return(k_shots_prompts)

    def generate_all_prompts_k_shot_switched_shot(self, k=5, question='cause'):
        assert(question in ["cause", "effect"])
        zero_shot_prompts = self.generate_all_prompts_zero_shot(question)
        k_shots_prompts = []
        for i, row in self.df_ce_two_sentences.iterrows():
            choices = np.random.choice(self.length, size=k, replace=False)
            k_shots = ""
            #create k-shots
            for c in choices: 
                if not row["switched"]:
                    r = self.df_ce_two_sentences_switched.iloc[c]
                else:
                    r = self.df_ce_two_sentences_non_switched.iloc[c]
                s = r["sequence"]
                if question=='cause':
                    k_shots += s + " Question: " + self.question_cause + self.answer + " " + r["answer_cause"] + "\n"
                elif question=='effect':
                    k_shots += s + " Question: " + self.question_effect + self.answer + " " + r["answer_effect"] + "\n"
            k_shots += zero_shot_prompts[i]
            k_shots_prompts.append(k_shots)
        return(k_shots_prompts)

    def generate_single_prompt_one_shot(self, n_subset=0, switched=False, switched_shot=False, question='cause'):
        return(self.generate_single_prompt_k_shot(n_subset, k=1, switched=switched, switched_shot=switched_shot, question=question))

    def generate_all_prompts_one_shot(self, question='cause'):
        return(self.generate_all_prompts_k_shot(k=1, question=question))

    def generate_all_prompts_one_shot_switched_shot(self, question='cause'):
        return(self.generate_all_prompts_k_shot_switched_shot(k=1, question=question))







