import argparse
import logging
import os

import torch
from tqdm import tqdm
from unseal.transformers_util import load_from_pretrained

from setups.toy_problem_3_colors_setup import ToyProblem3Colors

def run_gpt2_experiments(model, tokenizer, prompts, verbose=False, output_scores=False, device='cpu'):
    if output_scores:
        logging.warning('output_scores is not implemented yet')
        
    responses = []
    for prompt in tqdm(prompts):
        model_inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        output = model.generate(model_inputs, min_length=0, max_new_tokens=20, temperature=0, output_scores=output_scores, return_dict_in_generate=output_scores, pad_token_id=50256)
        response = tokenizer.decode(output[0], skip_special_tokens=True).lstrip(prompt)
        if verbose: 
            print(response)
        responses.append(response)

    return(responses)

def load_data(setup, k):
    if k == 0:
        prompts_first = setup.generate_all_prompts_zero_shot(question="first")
        prompts_second = setup.generate_all_prompts_zero_shot(question="second")
        prompts_final = setup.generate_all_prompts_zero_shot(question="final")
    elif k == 1:
        prompts_first = setup.generate_all_prompts_one_shot(question="first")
        prompts_second = setup.generate_all_prompts_one_shot(question="second")
        prompts_final = setup.generate_all_prompts_one_shot(question="final")  
    else:
        prompts_first = setup.generate_all_prompts_k_shot(k, question="first")
        prompts_second = setup.generate_all_prompts_k_shot(k, question="second")
        prompts_final = setup.generate_all_prompts_k_shot(k, question="final")
    
    return prompts_first, prompts_second, prompts_final

def get_path(k):
    if k == 0:
        path = './data/toy_problem_3_colors_results/gpt2/toy_problem_3c_zero_shot_results_{}.csv'
    elif k == 1:
        path = './data/toy_problem_3_colors_results/gpt2/toy_problem_3c_one_shot_results_{}.csv'
    else:
        path = './data/toy_problem_3_colors_results/gpt2/toy_problem_3c_k_shot_results_{}.csv'
    return path

def main(
    setup, 
    k,
    model_names,
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    prompts_first, prompts_second, prompts_final = load_data(setup, k)
    path = get_path(k)
    
    os.makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)
    
    for model_name in model_names:
        print(f'\nRunning {model_name}...')
        
        model, tokenizer, config = load_from_pretrained(model_name)
        model.to(device)
        
        responses_first = run_gpt2_experiments(model, tokenizer, prompts_first, device=device)
        responses_second = run_gpt2_experiments(model, tokenizer, prompts_second, device=device)
        responses_final = run_gpt2_experiments(model, tokenizer, prompts_final, device=device)
        
        df = setup_df.copy()
        df["responses_first"] = responses_first
        df["responses_second"] = responses_second
        df["responses_final"] = responses_final

        #save df
        df.to_csv(path.format(model_name))

        

## run script
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=0, help='k-shot')
    args = parser.parse_args()
    
    gpt2_model_names = [
        "gpt2",
        "gpt2-medium",
        "gpt2-large",
        "gpt2-xl",
    ]
    
    # TODO only do this if data doesn't exist yet
    #create setup & dataframe
    setup = ToyProblem3Colors()
    setup_df = setup.generate_sequences_df()
    setup.save_sequences_df("data/toy_problem_3_colors/toy_problem_3c.csv")

    main(setup, args.k, gpt2_model_names)

