import numpy as np
import itertools
import pandas as pd
import random

class ToyProblem5Colors():

    def __init__(self):

        self.events = [
            "The {} ball hit the {} ball.",
            "The {} ball hit the {} ball.",
            "The {} ball hit the {} ball.",
            "The {} ball hit the {} ball."
        ]
        self.outro = "The {} ball fell into the hole."
        self.colors = [
            "blue",
            "red",
            "green",
            "brown",
            "purple",
            "black",
            "white"
        ]
        self.all_subsets = np.array(list(itertools.combinations(self.colors, 5))) #only take the first 100 samples
        self.num_subsets = len(self.all_subsets)

        # generate sequences and prompts
        self.question_first = "Which ball started the chain?"
        self.question_second = "Which ball was second in the chain?"
        self.question_third = "Which ball was third in the chain?"
        self.question_fourth = "Which ball was fourth in the chain?"
        self.question_final = "Which ball fell into the hole?"
        self.answer = " Answer in three words:"
        self.sequences_df = self.generate_sequences_df()
        self.sequences_df_shuffled = self.sequences_df[self.sequences_df["shuffled"] == True]
        self.sequences_df_non_shuffled = self.sequences_df[self.sequences_df["shuffled"] == False]
        self.sequences = self.sequences_df["sequence"]


    def get_random_subsets(self, num_random_subsets=1):
        return(self.all_subsets[np.random.choice(self.num_subsets, size=num_random_subsets, replace=False)])

    def generate_single_sequence(self, n_subset=0, shuffled=False):
        c1, c2, c3, c4, c5 = self.all_subsets[n_subset]
        s1, s2, s3, s4 = self.events[0].format(c1, c2), self.events[1].format(c2, c3), self.events[1].format(c3, c4), self.events[1].format(c4, c5)
        o = self.outro.format(c5)
        if shuffled:
            l = random.sample([s1, s2, s3, s4], 4)
            return(l[0] + " " + l[1] + " " + l[2] + " " + l[3] + " "+ o)
        else:
            return(s1 + " " + s2 + " " + s3 + " " + s4 + " "+ o)

    def generate_sequences_df(self):
        sequences = []
        shuffled = []
        first_color = []
        second_color = []
        third_color = []
        fourth_color = []
        final_color = []
        for color_triplets in self.all_subsets:
            c1, c2, c3, c4, c5 = color_triplets
            s1, s2, s3, s4 = self.events[0].format(c1, c2), self.events[1].format(c2, c3), self.events[1].format(c3, c4), self.events[1].format(c4, c5)
            o = self.outro.format(c5)
            # create the prompts
            prompt_in_order = s1 + " " + s2 + " " + s3 + " " + s4 + " "+ o
            l = random.sample([s1, s2, s3, s4], 4)
            prompt_shuffled = l[0] + " " + l[1] + " " + l[2] + " " + l[3] + " "+ o
            sequences.append(prompt_in_order)
            shuffled.append(False)
            sequences.append(prompt_shuffled)
            shuffled.append(True)
            first_color.extend([c1, c1])
            second_color.extend([c2, c2])
            third_color.extend([c3, c3])
            fourth_color.extend([c4, c4])
            final_color.extend([c5, c5])

        df_toy_problem_5c = pd.DataFrame({
            "sequence":sequences,
            "shuffled":shuffled,
            "first_color":first_color,
            "second_color":second_color,
            "third_color":third_color,
            "fourth_color":fourth_color,
            "final_color":final_color
        })
        return(df_toy_problem_5c)

    def save_sequences_df(self, filepath):
        self.sequences_df.to_csv(filepath, index=False)

    def generate_single_prompt_zero_shot(self, n_subset=0, shuffled=False, question='first'):
        assert(question in ["first", "second", "third", "fourth", "final"])
        sequence = self.generate_single_sequence(n_subset, shuffled)
        if question=='first':
            return(sequence + " Question: " + self.question_first + self.answer)
        elif question=='second':
            return(sequence + " Question: " + self.question_second + self.answer)
        elif question=='third':
            return(sequence + " Question: " + self.question_third + self.answer)
        elif question=='fourth':
            return(sequence + " Question: " + self.question_fourth + self.answer)
        elif question=='final':
            return(sequence + " Question: " + self.question_final + self.answer)

    def generate_all_prompts_zero_shot(self, question='first'):
        assert(question in ["first", "second", "third", "fourth", "final"])
        prompts = []
        for s in self.sequences:
            if question=='first':
                prompts.append(s + " Question: " + self.question_first + self.answer)
            elif question=='second':
                prompts.append(s + " Question: " + self.question_second + self.answer)
            elif question=='third':
                prompts.append(s + " Question: " + self.question_third + self.answer)
            elif question=='fourth':
                prompts.append(s + " Question: " + self.question_fourth + self.answer)
            elif question=='final':
                prompts.append(s + " Question: " + self.question_final + self.answer)
        return(prompts)

    def generate_single_prompt_k_shot(self, n_subset=0, k=5, shuffled=False, shuffled_shot=False, question='first'):
        assert(question in ["first", "second", "third", "fourth", "final"])
        choices = np.random.choice(self.num_subsets, size=k, replace=False)
        k_shots = ""
        #create k-shots
        for c in choices: 
            if shuffled_shot:
                row = self.sequences_df_shuffled.iloc[c]
            else:
                row = self.sequences_df_non_shuffled.iloc[c]
            s = row["sequence"]
            if question=='first':
                k_shots += s + " Question: " + self.question_first + self.answer + " The {} ball.\n".format(row["first_color"])
            elif question=='second':
                k_shots += s + " Question: " + self.question_second + self.answer + " The {} ball.\n".format(row["second_color"])
            elif question=='third':
                k_shots += s + " Question: " + self.question_third + self.answer + " The {} ball.\n".format(row["third_color"])
            elif question=='fourth':
                k_shots += s + " Question: " + self.question_fourth + self.answer + " The {} ball.\n".format(row["fourth_color"])
            elif question=='final':
                k_shots += s + " Question: " + self.question_final + self.answer + " The {} ball.\n".format(row["final_color"])

        #add prompt on top
        prompt = self.generate_single_prompt_zero_shot(n_subset, shuffled, question)
        return(k_shots + prompt)

    def generate_all_prompts_k_shot(self, k=5, question='first'):
        assert(question in ["first", "second", "third", "fourth", "final"])
        zero_shot_prompts = self.generate_all_prompts_zero_shot(question)
        k_shots_prompts = []
        for i, row in self.sequences_df.iterrows():
            choices = np.random.choice(self.num_subsets, size=k, replace=False)
            k_shots = ""
            #create k-shots
            for c in choices: 
                if row["shuffled"]:
                    r = self.sequences_df_shuffled.iloc[c]
                else:
                    r = self.sequences_df_non_shuffled.iloc[c]
                s = r["sequence"]
                if question=='first':
                    k_shots += s + " Question: " + self.question_first + self.answer + " The {} ball.\n".format(r["first_color"])
                elif question=='second':
                    k_shots += s + " Question: " + self.question_second + self.answer + " The {} ball.\n".format(r["second_color"])
                elif question=='third':
                    k_shots += s + " Question: " + self.question_third + self.answer + " The {} ball.\n".format(r["third_color"])
                elif question=='fourth':
                    k_shots += s + " Question: " + self.question_fourth + self.answer + " The {} ball.\n".format(r["fourth_color"])
                elif question=='final':
                    k_shots += s + " Question: " + self.question_final + self.answer + " The {} ball.\n".format(r["final_color"])
            k_shots += zero_shot_prompts[i]
            k_shots_prompts.append(k_shots)
        return(k_shots_prompts)

    def generate_all_prompts_k_shot_shuffled_shot(self, k=5, question='first'):
        assert(question in ["first", "second", "third", "fourth", "final"])
        zero_shot_prompts = self.generate_all_prompts_zero_shot(question)
        k_shots_prompts = []
        for i, row in self.sequences_df.iterrows():
            choices = np.random.choice(self.num_subsets, size=k, replace=False)
            k_shots = ""
            #create k-shots
            for c in choices: 
                if not row["shuffled"]:
                    r = self.sequences_df_shuffled.iloc[c]
                else:
                    r = self.sequences_df_non_shuffled.iloc[c]
                s = r["sequence"]
                if question=='first':
                    k_shots += s + " Question: " + self.question_first + self.answer + " The {} ball.\n".format(r["first_color"])
                elif question=='second':
                    k_shots += s + " Question: " + self.question_second + self.answer + " The {} ball.\n".format(r["second_color"])
                elif question=='third':
                    k_shots += s + " Question: " + self.question_third + self.answer + " The {} ball.\n".format(r["third_color"])
                elif question=='fourth':
                    k_shots += s + " Question: " + self.question_fourth + self.answer + " The {} ball.\n".format(r["fourth_color"])
                elif question=='final':
                    k_shots += s + " Question: " + self.question_final + self.answer + " The {} ball.\n".format(r["final_color"])
            k_shots += zero_shot_prompts[i]
            k_shots_prompts.append(k_shots)
        return(k_shots_prompts)

    def generate_single_prompt_one_shot(self, n_subset=0, shuffled=False, shuffled_shot=False, question='first'):
        return(self.generate_single_prompt_k_shot(n_subset, k=1, shuffled=shuffled, shuffled_shot=shuffled_shot, question=question))

    def generate_all_prompts_one_shot(self, question='first'):
        return(self.generate_all_prompts_k_shot(k=1, question=question))

    def generate_all_prompts_one_shot_shuffled_shot(self, question='first'):
        return(self.generate_all_prompts_k_shot_shuffled_shot(k=1, question=question))







