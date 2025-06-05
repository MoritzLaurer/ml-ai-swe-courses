I am a machine learning engineer and I want to create an introductory course on PyTorch. It should be designed for people who don't know much about PyTorch and I want you to act as my pytorch teacher. Please create a pytorch curriculum for me. Start with creating a general overview of the main modules for this curriculum.

Each learning module should consist of one well-commented script that illustrates the main concepts and functionalities of the respective learning module. All explanatory content you produce should also be part of the respective script in the form of comments, so that I can directly copy the scripts into my local learning directory in VSCode and have all explanatory content there in learning modules of well structured and formatted scripts.

I'd suggest that we choose one overarching topic that is typical for industry use-cases which connects the different modules. The running topic should be: creating and tuning a small generative LLM like GPT/Llama.

I already know some JAX and I would like you to also add comparisons of PyTorch to JAX in each script, which will make things easier for me to understand. Also please make your first module about a comparison between JAX and PyTorch.

Now please create the first version of the learning curriculum. We might then refine the curriculum a bit and then afterwards we will go through the modules in a turn-based conversation step by step. 

---
Some feedback based on initial outputs:

- please sometimes add linebreaks at the end of a print statement so that the terminal output is easier to read

- I like how you numbered the comments so that the hierarchy/flow between the different sections is visible in the print outputs. Please continue doing this, this makes it easier for me to map the print outputs to the relevant parts in the script.

- please always say the title of the script in the beginning and please number them so that they are nicely sequential files in my local learning repo. (I've called the first one 01_pt_jax_philosophy.py)

- I've installed the relevant libraries and here are the versions, FYI: Using PyTorch version: 2.7.0 and JAX version: 0.6.0 