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
    
    len_df = len(df)
    len_df_switched = len(df_switched)
    len_df_non_switched = len(df_non_switched)
    
    sum_first_color_total = np.sum(eval_first_color(df))
    sum_first_color_non_switched = np.sum(eval_first_color(df_non_switched))
    sum_first_color_switched = np.sum(eval_first_color(df_switched))
    
    sum_second_color_total = np.sum(eval_second_color(df))
    sum_second_color_non_switched = np.sum(eval_second_color(df_non_switched))
    sum_second_color_switched = np.sum(eval_second_color(df_switched))
    
    sum_final_color_total = np.sum(eval_final_color(df))
    sum_final_color_non_switched = np.sum(eval_final_color(df_non_switched))
    sum_final_color_switched = np.sum(eval_final_color(df_switched))
    
    sum_array = np.array([[sum_first_color_total,
                sum_first_color_non_switched,
                sum_first_color_switched],

                [sum_second_color_total,
                sum_second_color_non_switched,
                sum_second_color_switched],

                [sum_final_color_total,
                sum_final_color_non_switched,
                sum_final_color_switched]])
    
    len_array = np.array([len_df, len_df_non_switched, len_df_switched] * 3).reshape(3,3)
    
    return(sum_array, len_array)

def plot_single_model_colors(path_string, percentages=True):
    
    df = pd.read_csv(path_string)
    results, len_array = eval_df_colors(df)
    perc = results / len_array
    
    x = np.arange(3)
    c = 0.25
    w = 0.2
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    if percentages:
        ax.bar(x - c, perc[:,0], width=w, label="combined", color='firebrick')
        ax.bar(x , perc[:,1], width=w, label="non switched", color='cornflowerblue')
        ax.bar(x + c, perc[:,2], width=w, label="switched", color='navy')
        ax.axhline(1/3, linestyle='--', color='black', label="random")
    else:
        ax.bar(x - c, results[:,0], width=w, label="combined", color='firebrick')
        ax.bar(x , results[:,1], width=w, label="non switched", color='cornflowerblue')
        ax.bar(x + c, results[:,2], width=w, label="switched", color='navy')
    ax.set_xticks(x)
    ax.set_ylabel("accuracy")
    ax.set_xticklabels(["first color", "second color", "final color"])

    plt.legend()
    plt.grid()
    plt.show();
    
### compare models
def compare_model_sizes_colors(path_string, gpt2=False):
    if gpt2:
        # df_small =  pd.read_csv(path_string.format("gpt2"))
        df_medium =  pd.read_csv(path_string.format("gpt2-medium"))
        df_large =  pd.read_csv(path_string.format("gpt2-large"))
        df_xl =  pd.read_csv(path_string.format("gpt2-xl"))
        
        # results_small = np.array(eval_df_colors(df_small))
        results_medium, len_medium = np.array(eval_df_colors(df_medium))
        results_large, len_large = np.array(eval_df_colors(df_large))
        results_xl, len_xl = np.array(eval_df_colors(df_xl))
        return([results_medium, results_large, results_xl], [len_medium, len_large, len_xl])
    
    else:
        df_ada =  pd.read_csv(path_string.format("text-ada-001"))
        df_babbage =  pd.read_csv(path_string.format("text-babbage-001"))
        df_curie =  pd.read_csv(path_string.format("text-curie-001"))
        df_davinci =  pd.read_csv(path_string.format("text-davinci-001"))

        results_ada, len_ada = np.array(eval_df_colors(df_ada))
        results_babbage, len_babbage = np.array(eval_df_colors(df_babbage))
        results_curie, len_curie = np.array(eval_df_colors(df_curie))
        results_davinci, len_davinci = np.array(eval_df_colors(df_davinci))
        return([results_ada, results_babbage, results_curie, results_davinci],
               [len_ada, len_babbage, len_curie, len_davinci])

def plot_model_sizes_colors(path_string, gpt2=False, percentages=True):
    
    if gpt2:
        labels = ["Medium", "Large", "XL"]
    else:
        labels = ["ada", "babbage", "curie", "davinci"]
    titles = ['combined', 'non switched', 'switched']
    tick_labels = ["first color", "second color", "final color"]
    colors = ["aqua", "lightskyblue", "violet", "darkviolet"]
    alphas = [1, 0.6, 0.6]
    
    results, lengths = compare_model_sizes_colors(path_string, gpt2)
    
    fig, ax = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    x = np.arange(3)
    w = 0.18
    c = 0.2
    for i in range(3):
        
        #compare all answers
        for j, result in enumerate(results):
            if percentages:
                perc = result / lengths[j]
                ax[i].bar(x + (-1.5 + j) * c, perc[:,i], width=w, label=labels[j], color=colors[j], alpha=alphas[i])
            else:
                ax[i].bar(x + (-1.5 + j) * c, result[:,i], width=w, label=labels[j], color=colors[j], alpha=alphas[i])
        
        ax[i].set_xticks(x)
        if percentages:
            ax[i].axhline(1/3, linestyle='--', color='black', label="random")
        ax[i].set_xticklabels(tick_labels)
        ax[i].grid()
        ax[i].set_title(titles[i])
    if gpt2:
        plt.suptitle('GPT2')
    else:
        plt.suptitle('GPT3')
    ax[0].legend()
    plt.show();

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
        #print("word: ", w, " fw_answer: ", fw_answers[i], "correct: ", word_in_answer and not other_words_in_answer)
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
    
    len_df = len(df)
    len_df_switched = len(df_switched)
    len_df_non_switched = len(df_non_switched)
    
    sum_first_word_total = np.sum(eval_first_word(df))
    sum_first_word_non_switched = np.sum(eval_first_word(df_non_switched))
    sum_first_word_switched = np.sum(eval_first_word(df_switched))
    
    sum_second_word_total = np.sum(eval_second_word(df))
    sum_second_word_non_switched = np.sum(eval_second_word(df_non_switched))
    sum_second_word_switched = np.sum(eval_second_word(df_switched))
    
    sum_final_word_total = np.sum(eval_final_word(df))
    sum_final_word_non_switched = np.sum(eval_final_word(df_non_switched))
    sum_final_word_switched = np.sum(eval_final_word(df_switched))
    
    results_array = np.array([[sum_first_word_total,
            sum_first_word_non_switched,
            sum_first_word_switched],

            [sum_second_word_total,
            sum_second_word_non_switched,
            sum_second_word_switched],

            [sum_final_word_total,
            sum_final_word_non_switched,
            sum_final_word_switched]])
    
    len_array = np.array([len_df, len_df_non_switched, len_df_switched] * 3).reshape(3,3)
    
    return(results_array, len_array)

def plot_single_model_nonsense_words(path_string, percentages=True):
    
    df = pd.read_csv(path_string)
    results, len_array = eval_df_nonsense_words(df)
    perc = results / len_array
    
    x = np.arange(3)
    c = 0.25
    w = 0.2
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    if percentages:
        ax.bar(x - c, perc[:,0], width=w, label="combined", color='firebrick')
        ax.bar(x , perc[:,1], width=w, label="non switched", color='cornflowerblue')
        ax.bar(x + c, perc[:,2], width=w, label="switched", color='navy')
        ax.axhline(1/3, linestyle='--', color='black', label="random")
    else:
        ax.bar(x - c, results[:,0], width=w, label="combined", color='firebrick')
        ax.bar(x , results[:,1], width=w, label="non switched", color='cornflowerblue')
        ax.bar(x + c, results[:,2], width=w, label="switched", color='navy')
    ax.set_xticks(x)
    ax.set_ylabel("accuracy")
    ax.set_xticklabels(["first word", "second word", "final word"])

    plt.legend()
    plt.grid()
    plt.show();
    
### compare models
def compare_model_sizes_nonsense_words(path_string, gpt2=False):
    
    # TODO add option for gpt2
    df_ada =  pd.read_csv(path_string.format("text-ada-001"))    
    df_babbage =  pd.read_csv(path_string.format("text-babbage-001"))
    df_curie =  pd.read_csv(path_string.format("text-curie-001"))
    df_davinci =  pd.read_csv(path_string.format("text-davinci-001"))

   
    results_ada, len_ada = np.array(eval_df_nonsense_words(df_ada))  
    results_babbage, len_babbage = np.array(eval_df_nonsense_words(df_babbage))  
    results_curie, len_curie = np.array(eval_df_nonsense_words(df_curie))
    results_davinci, len_davinci = np.array(eval_df_nonsense_words(df_davinci))
    
    return([results_ada, results_babbage, results_curie, results_davinci],
           [len_ada, len_babbage, len_curie, len_davinci])

def plot_model_sizes_nonsense_words(path_string, gpt2=False, percentages=True):
    
    if gpt2:
        labels = ["Medium", "Large", "XL"]
    else:
        labels = ["ada", "babbage", "curie", "davinci"]
    titles = ['combined', 'non switched', 'switched']
    tick_labels = ["first word", "second word", "final word"]
    colors = ["aqua", "lightskyblue", "violet", "darkviolet"]
    alphas = [1, 0.6, 0.6]
    
    results, lengths = compare_model_sizes_nonsense_words(path_string, gpt2)
    
    fig, ax = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    x = np.arange(3)
    w = 0.18
    c = 0.2
    for i in range(3):
        
        #compare all answers
        for j, result in enumerate(results):
            if percentages:
                perc = result / lengths[j]
                ax[i].bar(x + (-1.5 + j) * c, perc[:,i], width=w, label=labels[j], color=colors[j], alpha=alphas[i])
            else:
                ax[i].bar(x + (-1.5 + j) * c, result[:,i], width=w, label=labels[j], color=colors[j], alpha=alphas[i])
        
        ax[i].set_xticks(x)
        if percentages:
            ax[i].axhline(1/3, linestyle='--', color='black', label="random")
        ax[i].set_xticklabels(tick_labels)
        ax[i].grid()
        ax[i].set_title(titles[i])
    if gpt2:
        plt.suptitle('GPT2')
    else:
        plt.suptitle('GPT3')
    ax[0].legend()
    plt.show();

###############################
### helpers for cause and effect two sentences
###############################

def eval_ce_2_cause(df):
    cause_sentence = df["answer_cause"].values
    cause_sentence_answers = df["responses_cause"].values
    correct = []
    for i, s in enumerate(cause_sentence):
        answer_clean = cause_sentence_answers[i].replace("\n", "")
        c = s == answer_clean 
        #print(answer_clean, s, c)
        correct.append(c)
        
    return(correct)

def eval_ce_2_effect(df):
    effect_sentence = df["answer_effect"].values
    effect_sentence_answers = df["responses_effect"].values
    correct = []
    for i, s in enumerate(effect_sentence):
        answer_clean = effect_sentence_answers[i].replace("\n", "")
        c = s == answer_clean 
        #print(answer_clean, s, c)
        correct.append(c)
        
    return(correct)

def eval_df_ce_2_sentences(df):
    
    df_switched = df[df["switched"] == True]
    df_non_switched = df[df["switched"] == False]
    
    len_df = len(df)
    len_df_switched = len(df_switched)
    len_df_non_switched = len(df_non_switched)
    
    sum_cause_total = np.sum(eval_ce_2_cause(df))
    sum_cause_non_switched = np.sum(eval_ce_2_cause(df_non_switched))
    sum_cause_switched = np.sum(eval_ce_2_cause(df_switched))
    
    sum_effect_total = np.sum(eval_ce_2_effect(df))
    sum_effect_non_switched = np.sum(eval_ce_2_effect(df_non_switched))
    sum_effect_switched = np.sum(eval_ce_2_effect(df_switched))
    
    results_array = np.array([[sum_cause_total,
            sum_cause_non_switched,
            sum_cause_switched],

            [sum_effect_total,
            sum_effect_non_switched,
            sum_effect_switched]])
    
    len_array = np.array([len_df, len_df_non_switched, len_df_switched] * 2).reshape(2,3)
    
    return(results_array, len_array)

def plot_single_model_ce_2_sentences(path_string, percentages=True):
    
    df = pd.read_csv(path_string)
    results, len_array = eval_df_ce_2_sentences(df)
    perc = results / len_array
    
    x = np.arange(2)
    c = 0.25
    w = 0.2
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    if percentages:
        ax.bar(x - c, perc[:,0], width=w, label="combined", color='firebrick')
        ax.bar(x , perc[:,1], width=w, label="non switched", color='cornflowerblue')
        ax.bar(x + c, perc[:,2], width=w, label="switched", color='navy')
        ax.axhline(1/2, linestyle='--', color='black', label="random")
    else:
        ax.bar(x - c, results[:,0], width=w, label="combined", color='firebrick')
        ax.bar(x , results[:,1], width=w, label="non switched", color='cornflowerblue')
        ax.bar(x + c, results[:,2], width=w, label="switched", color='navy')
    ax.set_xticks(x)
    ax.set_ylabel("accuracy")
    ax.set_xticklabels(["cause", "effect"])

    plt.legend()
    plt.grid()
    plt.show();

### compare models
def compare_model_sizes_ce_2_sentences(path_string, gpt2=False):
    
    # TODO add option for gpt2
    df_ada =  pd.read_csv(path_string.format("text-ada-001"))    
    df_babbage =  pd.read_csv(path_string.format("text-babbage-001"))
    df_curie =  pd.read_csv(path_string.format("text-curie-001"))
    df_davinci =  pd.read_csv(path_string.format("text-davinci-001"))

   
    results_ada, len_ada = np.array(eval_df_ce_2_sentences(df_ada))  
    results_babbage, len_babbage = np.array(eval_df_ce_2_sentences(df_babbage))  
    results_curie, len_curie = np.array(eval_df_ce_2_sentences(df_curie))
    results_davinci, len_davinci = np.array(eval_df_ce_2_sentences(df_davinci))
    
    return([results_ada, results_babbage, results_curie, results_davinci],
           [len_ada, len_babbage, len_curie, len_davinci])

def plot_model_sizes_ce_2_sentences(path_string, gpt2=False, percentages=True):
    
    if gpt2:
        labels = ["Medium", "Large", "XL"]
    else:
        labels = ["ada", "babbage", "curie", "davinci"]
    titles = ['combined', 'non switched', 'switched']
    tick_labels = ["cause", "effect"]
    colors = ["aqua", "lightskyblue", "violet", "darkviolet"]
    alphas = [1, 0.6, 0.6]
    
    results, lengths = compare_model_sizes_ce_2_sentences(path_string, gpt2)
    
    fig, ax = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    x = np.arange(2)
    w = 0.18
    c = 0.2
    for i in range(3):
        
        #compare all answers
        for j, result in enumerate(results):
            if percentages:
                perc = result / lengths[j]
                ax[i].bar(x + (-1.5 + j) * c, perc[:,i], width=w, label=labels[j], color=colors[j], alpha=alphas[i])
            else:
                ax[i].bar(x + (-1.5 + j) * c, result[:,i], width=w, label=labels[j], color=colors[j], alpha=alphas[i])
        
        ax[i].set_xticks(x)
        if percentages:
            ax[i].axhline(1/2, linestyle='--', color='black', label="random")
        ax[i].set_xticklabels(tick_labels)
        ax[i].grid()
        ax[i].set_title(titles[i])
    if gpt2:
        plt.suptitle('GPT2')
    else:
        plt.suptitle('GPT3')
    ax[0].legend()
    plt.show();


###############################
### helpers for cause and effect one sentence
###############################

def eval_ce_1(df):
    cause_sentence = df["answer_cause"].values
    cause_sentence_answers = df["responses"].values
    correct = []
    for i, s in enumerate(cause_sentence):
        answer_clean = cause_sentence_answers[i].replace("\n", "")
        c = s == answer_clean 
        #print(answer_clean, s, c)
        correct.append(c)
        
    return(correct)

def eval_df_ce_1_sentence(df):
    
    df_switched = df[df["switched"] == True]
    df_non_switched = df[df["switched"] == False]
    
    len_df = len(df)
    len_df_switched = len(df_switched)
    len_df_non_switched = len(df_non_switched)
    
    sum_total = np.sum(eval_ce_1(df))
    sum_non_switched = np.sum(eval_ce_1(df_non_switched))
    sum_switched = np.sum(eval_ce_1(df_switched))
    
    results_array = np.array([[sum_total,
            sum_non_switched,
            sum_switched]])
    
    len_array = np.array([len_df, len_df_non_switched, len_df_switched]).reshape(1,3)
    
    return(results_array, len_array)

def plot_single_model_ce_1_sentence(path_string, percentages=True):
    
    df = pd.read_csv(path_string)
    results, len_array = eval_df_ce_1_sentence(df)
    perc = results / len_array
    
    x = np.arange(1)
    c = 0.25
    w = 0.2
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    if percentages:
        ax.bar(x - c, perc[:,0], width=w, label="combined", color='firebrick')
        ax.bar(x , perc[:,1], width=w, label="non switched", color='cornflowerblue')
        ax.bar(x + c, perc[:,2], width=w, label="switched", color='navy')
        ax.axhline(1/2, linestyle='--', color='black', label="random")
    else:
        ax.bar(x - c, results[:,0], width=w, label="combined", color='firebrick')
        ax.bar(x , results[:,1], width=w, label="non switched", color='cornflowerblue')
        ax.bar(x + c, results[:,2], width=w, label="switched", color='navy')
    ax.set_xticks(x)
    ax.set_ylabel("accuracy")
    ax.set_xticklabels(["cause & effect"])

    plt.legend()
    plt.grid()
    plt.show();

### compare models
def compare_model_sizes_ce_1_sentence(path_string, gpt2=False):
    
    # TODO add option for gpt2
    df_ada =  pd.read_csv(path_string.format("text-ada-001"))    
    df_babbage =  pd.read_csv(path_string.format("text-babbage-001"))
    df_curie =  pd.read_csv(path_string.format("text-curie-001"))
    df_davinci =  pd.read_csv(path_string.format("text-davinci-001"))

    results_ada, len_ada = np.array(eval_df_ce_1_sentence(df_ada))  
    results_babbage, len_babbage = np.array(eval_df_ce_1_sentence(df_babbage))  
    results_curie, len_curie = np.array(eval_df_ce_1_sentence(df_curie))
    results_davinci, len_davinci = np.array(eval_df_ce_1_sentence(df_davinci))
    
    return([results_ada, results_babbage, results_curie, results_davinci],
           [len_ada, len_babbage, len_curie, len_davinci])

def plot_model_sizes_ce_1_sentence(path_string, gpt2=False, percentages=True):
    
    if gpt2:
        labels = ["Medium", "Large", "XL"]
    else:
        labels = ["ada", "babbage", "curie", "davinci"]
    titles = ['combined', 'non switched', 'switched']
    tick_labels = ["cause & effect"]
    colors = ["aqua", "lightskyblue", "violet", "darkviolet"]
    alphas = [1, 0.6, 0.6]
    
    results, lengths = compare_model_sizes_ce_1_sentence(path_string, gpt2)
    
    fig, ax = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    x = np.arange(1)
    w = 0.18
    c = 0.2
    for i in range(3):
        
        #compare all answers
        for j, result in enumerate(results):
            if percentages:
                perc = result / lengths[j]
                ax[i].bar(x + (-1.5 + j) * c, perc[:,i], width=w, label=labels[j], color=colors[j], alpha=alphas[i])
            else:
                ax[i].bar(x + (-1.5 + j) * c, result[:,i], width=w, label=labels[j], color=colors[j], alpha=alphas[i])
        
        ax[i].set_xticks(x)
        if percentages:
            ax[i].axhline(1/2, linestyle='--', color='black', label="random")
        ax[i].set_xticklabels(tick_labels)
        ax[i].grid()
        ax[i].set_title(titles[i])
    if gpt2:
        plt.suptitle('GPT2')
    else:
        plt.suptitle('GPT3')
    ax[0].legend()
    plt.show();
