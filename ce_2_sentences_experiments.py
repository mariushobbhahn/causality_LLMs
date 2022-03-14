import numpy as np
import pandas as pdf
import openai
import os
from tqdm import tqdm
from setups.ce_two_sentences_setup import CETwoSentences


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

def run_2_sentences_experiments_zero_shot(setup, model_names, setup_df, filepath, verbose=False):

    prompts_zero_shot_cause = setup.generate_all_prompts_zero_shot(question="cause")
    prompts_zero_shot_effect = setup.generate_all_prompts_zero_shot(question="effect")

    for model in model_names:

        responses_cause = run_gpt3_experiments(model, prompts_zero_shot_cause)
        responses_effect = run_gpt3_experiments(model, prompts_zero_shot_effect)

        df = setup_df.copy()
        df["responses_cause"] = responses_cause
        df["responses_effect"] = responses_effect

        #save df
        df.to_csv(filepath.format(model))

def run_2_sentences_experiments_one_shot(setup, model_names, setup_df, filepath, verbose=False):

    prompts_one_shot_cause = setup.generate_all_prompts_one_shot(question="cause")
    prompts_one_shot_effect = setup.generate_all_prompts_one_shot(question="effect")

    for model in model_names:

        responses_cause = run_gpt3_experiments(model, prompts_one_shot_cause)
        responses_effect = run_gpt3_experiments(model, prompts_one_shot_effect)

        df = setup_df.copy()
        df["responses_cause"] = responses_cause
        df["responses_effect"] = responses_effect

        #save df
        df.to_csv(filepath.format(model))

def run_2_sentences_experiments_k_shot(setup, k, model_names, setup_df, filepath, verbose=False):

    prompts_k_shot_cause = setup.generate_all_prompts_k_shot(k, question="cause")
    prompts_k_shot_effect = setup.generate_all_prompts_k_shot(k, question="effect")

    for model in model_names:

        responses_cause = run_gpt3_experiments(model, prompts_k_shot_cause)
        responses_effect = run_gpt3_experiments(model, prompts_k_shot_effect)

        df = setup_df.copy()
        df["responses_cause"] = responses_cause
        df["responses_effect"] = responses_effect

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
    setup = CETwoSentences()
    setup_df = setup.generate_sequences_df()
    setup.save_sequences_df("data/bigbench_csvs/ce_two_sentences.csv")

    #run 
    run_2_sentences_experiments_zero_shot(setup=setup, model_names=gpt3_model_names, setup_df=setup_df, filepath="data/bigbench_results/ce_two_sentences_zero_shot_results_{}.csv")
    run_2_sentences_experiments_one_shot(setup, gpt3_model_names, setup_df, "data/bigbench_results/ce_two_sentences_one_shot_results_{}.csv")
    run_2_sentences_experiments_k_shot(setup, 5, gpt3_model_names, setup_df, "data/bigbench_results/ce_two_sentences_k_shot_results_{}.csv")




