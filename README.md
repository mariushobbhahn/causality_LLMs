# causality_LLMs

We play around with different tasks revolving around causality in LLMs like GPT-3. Our goal is to measure the quality of its causal modelling capabilities in real-world tasks, toy problems and adversarial examples

Some of the tasks have been taken from big bench.
TODO: explain which ones. 

The project is currently very experimental. Future versions will include better code and documentation. 

The data cleaning notebook is used to prepare the bigbench problems in a useful format

The create ball toy problems is used to create toy problems with the ball example

The playground gpt3 notebook is used to get gpt3 to answer some questions.


## Interpretability Tooling

We are using [Unseal](https://github.com/TomFrederik/unseal) to streamline our interpretability work.

### Installing Unseal and PySvelte
Clone the repo for [Unseal](https://github.com/TomFrederik/unseal) and [PySvelte](https://github.com/TomFrederik/pysvelte), cd into the respective directories and install them via 
```sh
pip install -e .
```

### Usage of Unseal
If you just want to visualize the attention patterns of GPT-like models from the huggingface transformers library,
the easiest way to do so is

```sh
cd unseal/unseal/interface/plain_interfaces
streamlit run $FILE
```

where $FILE is one of 
- single_layer_single_input.py
- all_layers_single_input.py
- compare_two_inputs.py

Names of the files should be self-explanatory.

If there are any issues with using Unseal, message [Tom Lieberum](mailto:tlieberum@outlook.de).
