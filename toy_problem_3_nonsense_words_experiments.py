import numpy as np
import pandas as pdf
import openai
import os
from tqdm import tqdm
from setups.toy_problem_3_nonsense_words_setup import ToyProblem3NonsenseWords


def run_gpt3_experiments(model, prompts, verbose=False):

    print("evaluate on gpt3 model: ", model)

    responses = []
    for prompt in tqdm(prompts):

        response = openai.Completion.create(engine=model, prompt=prompt, max_tokens=20, temperature=0)
        r = response["choices"][0]["text"]
        if verbose: 
            print(r)
        responses.append(r)

    return(responses)

def run_3_nonsense_words_experiments_zero_shot(setup, model_names, setup_df, filepath, verbose=False):

    prompts_zero_shot_first = setup.generate_all_prompts_zero_shot(question="first")
    prompts_zero_shot_second = setup.generate_all_prompts_zero_shot(question="second")
    prompts_zero_shot_final = setup.generate_all_prompts_zero_shot(question="final")

    for model in model_names:

        responses_first = run_gpt3_experiments(model, prompts_zero_shot_first)
        responses_second = run_gpt3_experiments(model, prompts_zero_shot_second)
        responses_final = run_gpt3_experiments(model, prompts_zero_shot_final)

        df = setup_df.copy()
        df["responses_first"] = responses_first
        df["responses_second"] = responses_second
        df["responses_final"] = responses_final

        #save df
        df.to_csv(filepath.format(model))

def run_3_nonsense_words_experiments_one_shot(setup, model_names, setup_df, filepath, verbose=False):

    prompts_one_shot_first = setup.generate_all_prompts_one_shot(question="first")
    prompts_one_shot_second = setup.generate_all_prompts_one_shot(question="second")
    prompts_one_shot_final = setup.generate_all_prompts_one_shot(question="final")
    prompts_one_shot_first_ss = setup.generate_all_prompts_one_shot_switched_shot(question="first")
    prompts_one_shot_second_ss = setup.generate_all_prompts_one_shot_switched_shot(question="second")
    prompts_one_shot_final_ss = setup.generate_all_prompts_one_shot_switched_shot(question="final")

    for model in model_names:

        responses_first = run_gpt3_experiments(model, prompts_one_shot_first)
        responses_second = run_gpt3_experiments(model, prompts_one_shot_second)
        responses_final = run_gpt3_experiments(model, prompts_one_shot_final)
        responses_first_ss = run_gpt3_experiments(model, prompts_one_shot_first_ss)
        responses_second_ss = run_gpt3_experiments(model, prompts_one_shot_second_ss)
        responses_final_ss = run_gpt3_experiments(model, prompts_one_shot_final_ss)

        df = setup_df.copy()
        df["responses_first"] = responses_first
        df["responses_second"] = responses_second
        df["responses_final"] = responses_final
        df["responses_first_ss"] = responses_first_ss
        df["responses_second_ss"] = responses_second_ss
        df["responses_final_ss"] = responses_final_ss

        #save df
        df.to_csv(filepath.format(model))

def run_3_nonsense_words_experiments_k_shot(setup, k, model_names, setup_df, filepath, verbose=False):

    prompts_k_shot_first = setup.generate_all_prompts_k_shot(k, question="first")
    prompts_k_shot_second = setup.generate_all_prompts_k_shot(k, question="second")
    prompts_k_shot_final = setup.generate_all_prompts_k_shot(k, question="final")
    prompts_k_shot_first_ss = setup.generate_all_prompts_k_shot_switched_shot(k, question="first")
    prompts_k_shot_second_ss = setup.generate_all_prompts_k_shot_switched_shot(k, question="second")
    prompts_k_shot_final_ss = setup.generate_all_prompts_k_shot_switched_shot(k, question="final")

    for model in model_names:

        responses_first = run_gpt3_experiments(model, prompts_k_shot_first)
        responses_second = run_gpt3_experiments(model, prompts_k_shot_second)
        responses_final = run_gpt3_experiments(model, prompts_k_shot_final)
        responses_first_ss = run_gpt3_experiments(model, prompts_k_shot_first_ss)
        responses_second_ss = run_gpt3_experiments(model, prompts_k_shot_second_ss)
        responses_final_ss = run_gpt3_experiments(model, prompts_k_shot_final_ss)

        df = setup_df.copy()
        df["responses_first"] = responses_first
        df["responses_second"] = responses_second
        df["responses_final"] = responses_final
        df["responses_first_ss"] = responses_first_ss
        df["responses_second_ss"] = responses_second_ss
        df["responses_final_ss"] = responses_final_ss

        #save df
        df.to_csv(filepath.format(model))


## run script
if __name__ == "__main__":

    # Load your API key from an environment variable or secret management service
    openai.api_key = os.getenv("OPENAI_KEY")

    gpt3_model_names = [
        "text-ada-001",
        "text-babbage-001",
        "text-curie-001",
        "text-davinci-001"
    ]
    
    #create setup & dataframe
    setup = ToyProblem3NonsenseWords()
    setup_df = setup.generate_sequences_df()
    setup.save_sequences_df("data/toy_problem_3_nonsense_words/toy_problem_3n.csv")

    #run 
    run_3_nonsense_words_experiments_zero_shot(setup=setup, model_names=gpt3_model_names, setup_df=setup_df, filepath="data/toy_problem_3_nonsense_words_results/toy_problem_3n_zero_shot_results_{}.csv")
    run_3_nonsense_words_experiments_one_shot(setup, gpt3_model_names, setup_df, "data/toy_problem_3_nonsense_words_results/toy_problem_3n_one_shot_results_{}.csv")
    run_3_nonsense_words_experiments_k_shot(setup, 5, gpt3_model_names, setup_df, "data/toy_problem_3_nonsense_words_results/toy_problem_3n_k_shot_results_{}.csv")




