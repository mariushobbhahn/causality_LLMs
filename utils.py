import numpy as np
import pandas as pd
import definitions
import matplotlib.pyplot as plt

colors = definitions.def_3_colors["colors"]
words = definitions.def_3_nonsense_words["words"]

### helpers for color
def eval_first_color(df):
    fc = df["first_color"].values
    fc_answers = df["responses_first"].values
    correct = []
    for i, c in enumerate(fc):
        color_in_answer = c in fc_answers[i]
        colors_wo_c = [c_ for c_ in colors if c_ != c]
        other_color_in_answer = any([c_ in fc_answers[i] for c_ in colors_wo_c])
        correct.append(color_in_answer and not other_color_in_answer)
        
    return(correct)

def eval_second_color(df):
    sc = df["second_color"].values
    sc_answers = df["responses_second"].values
    correct = []
    for i, c in enumerate(sc):
        color_in_answer = c in sc_answers[i]
        colors_wo_c = [c_ for c_ in colors if c_ != c]
        other_color_in_answer = any([c_ in sc_answers[i] for c_ in colors_wo_c])
        correct.append(color_in_answer and not other_color_in_answer)
        
    return(correct)

def eval_final_color(df):
    fc = df["final_color"].values
    fc_answers = df["responses_final"].values
    correct = []
    for i, c in enumerate(fc):
        color_in_answer = c in fc_answers[i]
        colors_wo_c = [c_ for c_ in colors if c_ != c]
        other_color_in_answer = any([c_ in fc_answers[i] for c_ in colors_wo_c])
        correct.append(color_in_answer and not other_color_in_answer)
        
    return(correct)

def eval_df_colors(df):
    
    df_switched = df[df["switched"] == True]
    df_non_switched = df[df["switched"] == False]
    
    sum_first_color_total = np.sum(eval_first_color(df))
    sum_first_color_non_switched = np.sum(eval_first_color(df_non_switched))
    sum_first_color_switched = np.sum(eval_first_color(df_switched))
    
    sum_second_color_total = np.sum(eval_second_color(df))
    sum_second_color_non_switched = np.sum(eval_second_color(df_non_switched))
    sum_second_color_switched = np.sum(eval_second_color(df_switched))
    
    sum_final_color_total = np.sum(eval_final_color(df))
    sum_final_color_non_switched = np.sum(eval_final_color(df_non_switched))
    sum_final_color_switched = np.sum(eval_final_color(df_switched))
    
    return([[sum_first_color_total,
            sum_first_color_non_switched,
            sum_first_color_switched],

            [sum_second_color_total,
            sum_second_color_non_switched,
            sum_second_color_switched],

            [sum_final_color_total,
            sum_final_color_non_switched,
            sum_final_color_switched]])

def plot_single_model_colors(path_string):
    
    df = pd.read_csv(path_string)
    results = np.array(eval_df_colors(df))
    
    x = np.arange(3)
    c = 0.25
    w = 0.2
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.bar(x - c, results[:,0], width=w, label="combined")
    ax.bar(x , results[:,1], width=w, label="non switched")
    ax.bar(x + c, results[:,2], width=w, label="switched")
    ax.set_xticks(x)
    ax.set_xticklabels(["first color", "second color", "final color"])

    plt.legend()
    plt.grid()
    plt.show();
    
### compare models
def compare_model_sizes_colors(path_string, gpt2=False):
    if gpt2:
        df_xl =  pd.read_csv(path_string.format("gpt2-xl"))
        df_large =  pd.read_csv(path_string.format("gpt2-large"))
        df_medium =  pd.read_csv(path_string.format("gpt2-medium"))
        # df_small =  pd.read_csv(path_string.format("gpt2"))

        results_xl = np.array(eval_df_colors(df_xl))
        results_large = np.array(eval_df_colors(df_large))
        results_medium = np.array(eval_df_colors(df_medium))
        # results_small = np.array(eval_df_colors(df_small))
        return([results_xl, results_large, results_medium])
    
    else:
        df_davinci =  pd.read_csv(path_string.format("text-davinci-001"))
        df_curie =  pd.read_csv(path_string.format("text-curie-001"))
        df_babbage =  pd.read_csv(path_string.format("text-babbage-001"))
        df_ada =  pd.read_csv(path_string.format("text-ada-001"))

        results_davinci = np.array(eval_df_colors(df_davinci))
        results_curie = np.array(eval_df_colors(df_curie))
        results_babbage = np.array(eval_df_colors(df_babbage))
        results_ada = np.array(eval_df_colors(df_ada))
        return([results_davinci, results_curie, results_babbage, results_ada])

def plot_model_sizes_colors(path_string, gpt2=False):
    
    if gpt2:
        labels = ["XL", "Large", "Medium"]
    else:
        labels = ["davinci", "curie", "babbage", "ada"]
    titles = ['combined', 'non switched', 'switched']
    tick_labels = ["first color", "second color", "final color"]
    
    results = compare_model_sizes_colors(path_string, gpt2)
    
    fig, ax = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    x = np.arange(3)
    w = 0.18
    c = 0.2
    for i in range(3):
        
        #compare all answers
        for j, result in enumerate(results):
            ax[i].bar(x + (-1.5 + j) * c, result[:,i], width=w, label=labels[j])
        
        ax[i].set_xticks(x)
        ax[i].set_xticklabels(tick_labels)
        ax[i].grid()
        ax[i].legend()
        ax[i].set_title(titles[i])
    if gpt2:
        plt.suptitle('GPT2')
    else:
        plt.suptitle('GPT3')
    plt.show()

###############################
### helpers for nonsense words
###############################

def eval_first_word(df):
    fw = df["first_word"].values
    fw_answers = df["responses_first"].values
    correct = []
    for i, w in enumerate(fw):
        word_in_answer = w in fw_answers[i]
        words_wo_w = [w_ for w_ in words if w_ != w]
        other_words_in_answer = any([w_ in fw_answers[i] for w_ in words_wo_w])
        correct.append(word_in_answer and not other_words_in_answer)
        
    return(correct)

def eval_second_word(df):
    sw = df["second_word"].values
    sw_answers = df["responses_second"].values
    correct = []
    for i, w in enumerate(sw):
        word_in_answer = w in sw_answers[i]
        words_wo_w = [w_ for w_ in words if w_ != w]
        other_words_in_answer = any([w_ in sw_answers[i] for w_ in words_wo_w])
        correct.append(word_in_answer and not other_words_in_answer)
        
    return(correct)

def eval_final_word(df):
    fw = df["final_word"].values
    fw_answers = df["responses_final"].values
    correct = []
    for i, w in enumerate(fw):
        word_in_answer = w in fw_answers[i]
        words_wo_w = [w_ for w_ in words if w_ != w]
        other_words_in_answer = any([w_ in fw_answers[i] for w_ in words_wo_w])
        correct.append(word_in_answer and not other_words_in_answer)
        
    return(correct)

def eval_df_nonsense_words(df):
    
    df_switched = df[df["switched"] == True]
    df_non_switched = df[df["switched"] == False]
    
    sum_first_word_total = np.sum(eval_first_word(df))
    sum_first_word_non_switched = np.sum(eval_first_word(df_non_switched))
    sum_first_word_switched = np.sum(eval_first_word(df_switched))
    
    sum_second_word_total = np.sum(eval_second_word(df))
    sum_second_word_non_switched = np.sum(eval_second_word(df_non_switched))
    sum_second_word_switched = np.sum(eval_second_word(df_switched))
    
    sum_final_word_total = np.sum(eval_final_word(df))
    sum_final_word_non_switched = np.sum(eval_final_word(df_non_switched))
    sum_final_word_switched = np.sum(eval_final_word(df_switched))
    
    return([[sum_first_word_total,
            sum_first_word_non_switched,
            sum_first_word_switched],

            [sum_second_word_total,
            sum_second_word_non_switched,
            sum_second_word_switched],

            [sum_final_word_total,
            sum_final_word_non_switched,
            sum_final_word_switched]])

def plot_single_model_nonsense_words(path_string):
    
    df = pd.read_csv(path_string)
    results = np.array(eval_df_nonsense_words(df))
    
    x = np.arange(3)
    c = 0.25
    w = 0.2
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.bar(x - c, results[:,0], width=w, label="combined")
    ax.bar(x , results[:,1], width=w, label="non switched")
    ax.bar(x + c, results[:,2], width=w, label="switched")
    ax.set_xticks(x)
    ax.set_xticklabels(["first word", "second word", "final word"])

    plt.legend()
    plt.grid()
    plt.show();
    
### compare models
def compare_model_sizes_nonsense_words(path_string):
    
    df_davinci =  pd.read_csv(path_string.format("text-davinci-001"))
    df_curie =  pd.read_csv(path_string.format("text-curie-001"))
    df_babbage =  pd.read_csv(path_string.format("text-babbage-001"))
    df_ada =  pd.read_csv(path_string.format("text-ada-001"))

    results_davinci = np.array(eval_df_nonsense_words(df_davinci))
    results_curie = np.array(eval_df_nonsense_words(df_curie))
    results_babbage = np.array(eval_df_nonsense_words(df_babbage))
    results_ada = np.array(eval_df_nonsense_words(df_ada))
    
    return([results_davinci, results_curie, results_babbage, results_ada])

def plot_model_sizes_nonsense_words(path_string):
    
    results_davinci, results_curie, results_babbage, results_ada = compare_model_sizes_nonsense_words(path_string)
    
    fig, ax = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    x = np.arange(3)
    w = 0.18
    c = 0.2
    
    #compare all answers
    ax[0].bar(x - 1.5*c, results_ada[:,0], width=w, label="ada")
    ax[0].bar(x - 0.5*c, results_babbage[:,0], width=w, label="babbage")
    ax[0].bar(x + 0.5*c, results_curie[:,0], width=w, label="curie")
    ax[0].bar(x + 1.5*c, results_davinci[:,0], width=w, label="davinci")
    
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(["first word", "second word", "final word"])
    ax[0].grid()
    ax[0].legend()
    ax[0].set_title("combined")
    
    #compare non-switched answers
    ax[1].bar(x - 1.5*c, results_ada[:,1], width=w, label="ada")
    ax[1].bar(x - 0.5*c, results_babbage[:,1], width=w, label="babbage")
    ax[1].bar(x + 0.5*c, results_curie[:,1], width=w, label="curie")
    ax[1].bar(x + 1.5*c, results_davinci[:,1], width=w, label="davinci")
    ax[1].grid()
    ax[1].set_title("non-switched")
    
    #compare switched answers
    ax[2].bar(x - 1.5*c, results_ada[:,2], width=w, label="ada")
    ax[2].bar(x - 0.5*c, results_babbage[:,2], width=w, label="babbage")
    ax[2].bar(x + 0.5*c, results_curie[:,2], width=w, label="curie")
    ax[2].bar(x + 1.5*c, results_davinci[:,2], width=w, label="davinci")
    ax[2].grid()
    ax[2].set_title("switched")
    
    plt.show()





