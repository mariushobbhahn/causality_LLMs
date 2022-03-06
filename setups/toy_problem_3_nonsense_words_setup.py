import numpy as np
import itertools
import pandas as pd

class ToyProblem3NonsenseWords():

    def __init__(self):

        self.events = [
            "The {} hit the {}.",
            "The {} hit the {}."
        ]
        self.outro = "The {} fell in the hole."
        self.words= [
            "baz",
            "fuu",
            "schleep",
            "blubb",
            "bla",
            "plomp",
            "dinglebob"
        ]
        self.all_subsets = np.array(list(itertools.combinations(self.words, 3)))
        self.num_subsets = len(self.all_subsets)

        # generate sequences and prompts
        self.question_first = "What started the chain?"
        self.question_second = "What was second in the chain?"
        self.question_final = "What fell into the hole?"
        self.answer = " Answer in two words: "
        self.sequences_df = self.generate_sequences_df()
        self.sequences = self.sequences_df["sequence"]


    def get_random_subsets(self, num_random_subsets=1):
        return(self.all_subsets[np.random.choice(self.num_subsets, size=num_random_subsets, replace=False)])

    def generate_single_sequence(self, n_subset=0, switched=False):
        c1, c2, c3 = self.all_subsets[n_subset]
        s1, s2 = self.events[0].format(c1, c2), self.events[1].format(c2, c3)
        o = self.outro.format(c3)
        if switched:
            return(s2 + " " + s1 + " " + o)
        else:
            return(s1 + " " + s2 + " " + o)

    def generate_sequences_df(self):
        sequences = []
        switched = []
        first_word = []
        second_word = []
        final_word = []
        for word_triplets in self.all_subsets:
            w1, w2, w3 = word_triplets
            s1, s2 = self.events[0].format(w1, w2), self.events[1].format(w2, w3)
            o = self.outro.format(w3)
            # create the prompts
            prompt_in_order = s1 + " " + s2 + " " + o
            prompt_switched = s2 + " " + s1 + " " + o
            sequences.append(prompt_in_order)
            switched.append(False)
            sequences.append(prompt_switched)
            switched.append(True)
            first_word.extend([w1, w1])
            second_word.extend([w2, w2])
            final_word.extend([w3, w3])

        df_toy_problem_3w = pd.DataFrame({
            "sequence":sequences,
            "switched":switched,
            "first_word":first_word,
            "second_word":second_word,
            "final_word":final_word
        })
        return(df_toy_problem_3w)

    def save_sequences_df(self, filepath):
        self.sequences_df.to_csv(filepath, index=False)

    def generate_single_prompt_zero_shot(self, n_subset=0, switched=False, question='first'):
        assert(question in ["first", "second", "final"])
        sequence = self.generate_single_sequence(n_subset, switched)
        if question=='first':
            return(sequence + " Question: " + self.question_first + self.answer)
        elif question=='second':
            return(sequence + " Question: " + self.question_second + self.answer)
        elif question=='final':
            return(sequence + " Question: " + self.question_final + self.answer)

    def generate_all_prompts_zero_shot(self, question='first'):
        assert(question in ["first", "second", "final"])
        prompts = []
        for s in self.sequences:
            if question=='first':
                prompts.append(s + " Question: " + self.question_first + self.answer)
            elif question=='second':
                prompts.append(s + " Question: " + self.question_second + self.answer)
            elif question=='final':
                prompts.append(s + " Question: " + self.question_final + self.answer)
        return(prompts)

    def generate_single_prompt_k_shot(self, n_subset=0, k=5, switched=False, question='first'):
        assert(question in ["first", "second", "final"])
        choices = np.random.choice(len(self.sequences_df), size=k, replace=False)
        k_shots = ""
        #create k-shots
        for c in choices: 
            row = self.sequences_df.iloc[c]
            s = row["sequence"]
            if question=='first':
                k_shots += s + " Question: " + self.question_first + self.answer + "the {}\n".format(row["first_word"])
            elif question=='second':
                k_shots += s + " Question: " + self.question_second + self.answer + "the {}\n".format(row["second_word"])
            elif question=='final':
                k_shots += s + " Question: " + self.question_final + self.answer + "the {}\n".format(row["final_word"])

        #add prompt on top
        prompt = self.generate_single_prompt_zero_shot(n_subset, switched, question)
        return(k_shots + prompt)

    def generate_all_prompts_k_shot(self, k=5, question='first'):
        assert(question in ["first", "second", "final"])
        zero_shot_prompts = self.generate_all_prompts_zero_shot(question)
        k_shots_prompts = []
        for i, row in self.sequences_df.iterrows():
            choices = np.random.choice(len(self.sequences_df), size=k, replace=False)
            k_shots = ""
            #create k-shots
            for c in choices: 
                row = self.sequences_df.iloc[c]
                s = row["sequence"]
                if question=='first':
                    k_shots += s + " Question: " + self.question_first + self.answer + "the {}\n".format(row["first_word"])
                elif question=='second':
                    k_shots += s + " Question: " + self.question_second + self.answer + "the {}\n".format(row["second_word"])
                elif question=='final':
                    k_shots += s + " Question: " + self.question_final + self.answer + "the {}\n".format(row["final_word"])
            k_shots += zero_shot_prompts[i]
            k_shots_prompts.append(k_shots)
        return(k_shots_prompts)

    def generate_single_prompt_one_shot(self, n_subset=0, switched=False, question='first'):
        return(self.generate_single_prompt_k_shot(n_subset, k=1, switched=switched, question=question))

    def generate_all_prompts_one_shot(self, question='first'):
        return(self.generate_all_prompts_k_shot(k=1, question=question))







