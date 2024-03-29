import argparse
import logging
import os

import torch
from tqdm import tqdm
from unseal.transformers_util import load_from_pretrained

from setups.toy_problem_3_colors_setup import ToyProblem3Colors

def run_experiments(model, tokenizer, prompts, verbose=False, output_scores=False, device='cpu'):
    if output_scores:
        logging.warning('output_scores is not implemented yet')
        
    responses = []
    for prompt in tqdm(prompts):
        model_inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        output = model.generate(model_inputs, min_length=0, max_new_tokens=4, temperature=0, output_scores=output_scores, return_dict_in_generate=output_scores, pad_token_id=50256)
        response = tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):]
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

def get_paths(k, model_names):
    paths = []
    for model_name in model_names:
        if k == 0:
            path = f'./data/toy_problem_3_colors_results/' + '-'.join(model_name.split('-')[:-1]) + '/toy_problem_3c_zero_shot_results_{}.csv'
        elif k == 1:
            path = f'./data/toy_problem_3_colors_results/' + '-'.join(model_name.split('-')[:-1]) + '/toy_problem_3c_one_shot_results_{}.csv'
        else:
            path = f'./data/toy_problem_3_colors_results/' + '-'.join(model_name.split('-')[:-1]) + '/toy_problem_3c_k_shot_results_{}.csv'
        paths.append(path)
    return paths

def main(
    setup, 
    k,
    model_names,
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    prompts_first, prompts_second, prompts_final = load_data(setup, k)
    paths = get_paths(k, model_names)
    for path in paths:
        os.makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)
    
    for model_name, path in zip(model_names, paths):
        print(f'\nRunning {model_name}...')
        
        model, tokenizer, config = load_from_pretrained(model_name)
        model.to(device)
        
        responses_first = run_experiments(model, tokenizer, prompts_first, device=device)
        responses_second = run_experiments(model, tokenizer, prompts_second, device=device)
        responses_final = run_experiments(model, tokenizer, prompts_final, device=device)
        
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
    parser.add_argument('--models', type=str, nargs="*", help='model_names')
    args = parser.parse_args()
    
    
    # TODO only do this if data doesn't exist yet
    #create setup & dataframe
    setup = ToyProblem3Colors()
    setup_df = setup.generate_sequences_df()
    setup.save_sequences_df("data/toy_problem_3_colors/toy_problem_3c.csv")

    main(setup, args.k, args.models)

