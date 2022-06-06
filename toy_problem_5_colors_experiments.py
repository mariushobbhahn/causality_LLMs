import numpy as np
import pandas as pdf
import openai
import os
from tqdm import tqdm
from setups.toy_problem_5_colors_setup import ToyProblem5Colors


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

def run_5_colors_experiments_zero_shot(setup, model_names, setup_df, filepath, verbose=False):

    prompts_zero_shot_first = setup.generate_all_prompts_zero_shot(question="first")
    prompts_zero_shot_second = setup.generate_all_prompts_zero_shot(question="second")
    prompts_zero_shot_third = setup.generate_all_prompts_zero_shot(question="third")
    prompts_zero_shot_fourth = setup.generate_all_prompts_zero_shot(question="fourth")
    prompts_zero_shot_final = setup.generate_all_prompts_zero_shot(question="final")

    for model in model_names:

        responses_first = run_gpt3_experiments(model, prompts_zero_shot_first)
        responses_second = run_gpt3_experiments(model, prompts_zero_shot_second)
        responses_third = run_gpt3_experiments(model, prompts_zero_shot_third)
        responses_fourth = run_gpt3_experiments(model, prompts_zero_shot_fourth)
        responses_final = run_gpt3_experiments(model, prompts_zero_shot_final)

        df = setup_df.copy()
        df["responses_first"] = responses_first
        df["responses_second"] = responses_second
        df["responses_third"] = responses_third
        df["responses_fourth"] = responses_fourth
        df["responses_final"] = responses_final

        #save df
        df.to_csv(filepath.format(model))

def run_5_colors_experiments_one_shot(setup, model_names, setup_df, filepath, verbose=False):

    prompts_one_shot_first = setup.generate_all_prompts_one_shot(question="first")
    prompts_one_shot_second = setup.generate_all_prompts_one_shot(question="second")
    prompts_one_shot_third = setup.generate_all_prompts_one_shot(question="third")
    prompts_one_shot_fourth = setup.generate_all_prompts_one_shot(question="fourth")
    prompts_one_shot_final = setup.generate_all_prompts_one_shot(question="final")

    prompts_one_shot_first_ss = setup.generate_all_prompts_one_shot_shuffled_shot(question="first")
    prompts_one_shot_second_ss = setup.generate_all_prompts_one_shot_shuffled_shot(question="second")
    prompts_one_shot_third_ss = setup.generate_all_prompts_one_shot_shuffled_shot(question="third")
    prompts_one_shot_fourth_ss = setup.generate_all_prompts_one_shot_shuffled_shot(question="fourth")
    prompts_one_shot_final_ss = setup.generate_all_prompts_one_shot_shuffled_shot(question="final")

    for model in model_names:

        responses_first = run_gpt3_experiments(model, prompts_one_shot_first)
        responses_second = run_gpt3_experiments(model, prompts_one_shot_second)
        responses_third = run_gpt3_experiments(model, prompts_one_shot_third)
        responses_fourth = run_gpt3_experiments(model, prompts_one_shot_fourth)
        responses_final = run_gpt3_experiments(model, prompts_one_shot_final)

        responses_first_ss = run_gpt3_experiments(model, prompts_one_shot_first_ss)
        responses_second_ss = run_gpt3_experiments(model, prompts_one_shot_second_ss)
        responses_third_ss = run_gpt3_experiments(model, prompts_one_shot_third_ss)
        responses_fourth_ss = run_gpt3_experiments(model, prompts_one_shot_fourth_ss)
        responses_final_ss = run_gpt3_experiments(model, prompts_one_shot_final_ss)

        df = setup_df.copy()
        df["responses_first"] = responses_first
        df["responses_second"] = responses_second
        df["responses_third"] = responses_third
        df["responses_fourth"] = responses_fourth
        df["responses_final"] = responses_final

        df["responses_first_ss"] = responses_first_ss
        df["responses_second_ss"] = responses_second_ss
        df["responses_third_ss"] = responses_third_ss
        df["responses_fourth_ss"] = responses_fourth_ss
        df["responses_final_ss"] = responses_final_ss

        #save df
        df.to_csv(filepath.format(model))

def run_5_colors_experiments_k_shot(setup, k, model_names, setup_df, filepath, verbose=False):

    prompts_k_shot_first = setup.generate_all_prompts_k_shot(k, question="first")
    prompts_k_shot_second = setup.generate_all_prompts_k_shot(k, question="second")
    prompts_k_shot_third = setup.generate_all_prompts_k_shot(k, question="third")
    prompts_k_shot_fourth = setup.generate_all_prompts_k_shot(k, question="fourth")
    prompts_k_shot_final = setup.generate_all_prompts_k_shot(k, question="final")

    prompts_k_shot_first_ss = setup.generate_all_prompts_k_shot_shuffled_shot(k, question="first")
    prompts_k_shot_second_ss = setup.generate_all_prompts_k_shot_shuffled_shot(k, question="second")
    prompts_k_shot_third_ss = setup.generate_all_prompts_k_shot_shuffled_shot(k, question="third")
    prompts_k_shot_fourth_ss = setup.generate_all_prompts_k_shot_shuffled_shot(k, question="fourth")
    prompts_k_shot_final_ss = setup.generate_all_prompts_k_shot_shuffled_shot(k, question="final")

    for model in model_names:

        responses_first = run_gpt3_experiments(model, prompts_k_shot_first)
        responses_second = run_gpt3_experiments(model, prompts_k_shot_second)
        responses_third = run_gpt3_experiments(model, prompts_k_shot_third)
        responses_fourth = run_gpt3_experiments(model, prompts_k_shot_fourth)
        responses_final = run_gpt3_experiments(model, prompts_k_shot_final)

        responses_first_ss = run_gpt3_experiments(model, prompts_k_shot_first_ss)
        responses_second_ss = run_gpt3_experiments(model, prompts_k_shot_second_ss)
        responses_third_ss = run_gpt3_experiments(model, prompts_k_shot_third_ss)
        responses_fourth_ss = run_gpt3_experiments(model, prompts_k_shot_fourth_ss)
        responses_final_ss = run_gpt3_experiments(model, prompts_k_shot_final_ss)

        df = setup_df.copy()
        df["responses_first"] = responses_first
        df["responses_second"] = responses_second
        df["responses_third"] = responses_third
        df["responses_fourth"] = responses_fourth
        df["responses_final"] = responses_final

        df["responses_first_ss"] = responses_first_ss
        df["responses_second_ss"] = responses_second_ss
        df["responses_third_ss"] = responses_third_ss
        df["responses_fourth_ss"] = responses_fourth_ss
        df["responses_final_ss"] = responses_final_ss

        #save df
        df.to_csv(filepath.format(model))


## run script
if __name__ == "__main__":

    # Load your API key from an environment variable or secret management service
    openai.api_key = os.getenv("OPENAI_KEY")

    gpt3_model_names = [
        #"text-ada-001",
        #"text-babbage-001",
        #"text-curie-001",
        "text-davinci-002"
    ]
    
    #create setup & dataframe
    setup = ToyProblem5Colors()
    setup_df = setup.generate_sequences_df()
    setup.save_sequences_df("data/toy_problem_5_colors/toy_problem_5c.csv")

    #run 
    run_5_colors_experiments_zero_shot(setup=setup, model_names=gpt3_model_names, setup_df=setup_df, filepath="data/toy_problem_5_colors_results/toy_problem_5c_zero_shot_results_{}.csv")
    run_5_colors_experiments_one_shot(setup, gpt3_model_names, setup_df, "data/toy_problem_5_colors_results/toy_problem_5c_one_shot_results_{}.csv")
    run_5_colors_experiments_k_shot(setup, 5, gpt3_model_names, setup_df, "data/toy_problem_5_colors_results/toy_problem_5c_k_shot_results_{}.csv")




