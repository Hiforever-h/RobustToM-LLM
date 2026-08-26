---
license: cc-by-nc-4.0
task_categories:
- question-answering
language:
- en
tags:
- theory-of-mind
- reasoning
---

# Data sample for *ExploreToM: Program-guided adversarial data generation for theory of mind reasoning*

ExploreToM is the first framework to allow **large-scale generation of diverse and challenging theory of mind data for robust training and evaluation**.
Our approach leverages an A* search over a custom domain-specific language to produce complex story structures and novel, diverse, yet plausible scenarios to stress test the limits of LLMs.


Our A* search procedure aims to find particularly difficult stories for a given model. Here we present a data sample generated adversarially for [Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct). We generated 10 story structures across the 18 settings presented in the paper using a budget of 50 nodes per story structure. We then infill the story structures as described in the paper. A large sample of the resulting data is presented here.

**If your goal is to test a model, we highly recommend running the algorithm using your specific model as ExploreToM works by finding stories adversarially towards a given model.** If this were unfeasible, our experiments show that ExploreToM-generated data using Llama-3.1-70B-Instruct is still challenging for testing other frontier models but please **DO NOT USE THIS DATA AS THE CANONICAL TEST SET FOR EXPLORETOM**.

**If your goal is to use ExploreToM as training data, feel free to generate even more data!** You can adjust the A* search function and action sets allowed depending on your needs, or even completely disable the A* search and overgenerate.


## Clarifications on data fields

- qprop -> question-related property
- sprop -> story-related property
- param -> search parameter (e.g. number of people involved)

`qprop=non_unique_mental_state` is a synonym for checking if a question is interesting. If the question is not theory of mind-related (that is, if `nth_order=-1`, which corresponds to memory or factual questions) then `qprop=non_unique_mental_state=True` by default.

## Code
Code to generate data and analyses is available at: https://github.com/facebookresearch/ExploreToM

## Citation

If you found [the paper](https://openreview.net/forum?id=246rHKUnnf) or data helpful, consider citing it:

```
@inproceedings{sclarexplore,
  title={Explore Theory of Mind: program-guided adversarial data generation for theory of mind reasoning},
  author={Sclar, Melanie and Yu, Jane and Fazel-Zarandi, Maryam and Tsvetkov, Yulia and Bisk, Yonatan and Choi, Yejin and Celikyilmaz, Asli},
  booktitle={The Thirteenth International Conference on Learning Representations}
}
```

For questions, please reach out to the first author of the paper.
