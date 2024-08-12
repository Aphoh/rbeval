---
title: rbeval
emoji: 💩
colorFrom: yellow
colorTo: blue
sdk: streamlit
sdk_version: 1.37.0
app_file: app.py
pinned: false
---

# RoBustEval Dashboard

This dashboard is best viewed at [the huggingface space](https://huggingface.co/spaces/mli-will/rbeval)

### Introduction

LLM MCQA (multiple choice question-answering) benchmarks are measured in the following way:
1. Some number of few shot examples are pulled from the validation set of the MCQA benchmark and formatted as
    > **Quesiton**: What is the capital of France? \
    > (A) Paris \
    > (B) London \
    > (C) Berlin \
    > (D) Madrid \
    > **Answer**: A
2. The target question is then appended, without the answer, and fed into the model as
    > **Quesiton**: What is the capital of France? \
    > (A) Paris \
    > (B) London \
    > (C) Berlin \
    > (D) Madrid \
    > **Answer**:
3. The model then outputs it's predictions for the token that should come directly after **Answer**:
4. The log probabilities $\ln p_i, i \in \{A,B,C,D\}$ for the tokens resulting from tokenizing the strings `"A", "B", "C", "D"` are then recorded
5. A question with correct answer $k$ is marked is correct if $\ln p_k > \ln p_i, i \in \{A,B,C,D\} \setminus \{k\}$
6. The accuracy is reported as the percentage of questions correctly answered

This method for evaluation is reasonable, but leaves behind significant amounts of information about model inference.
For example, consider a question with correct answer $C$. The following model outputs are scored the same:
| Probability | $p_A$   | $p_B$   | $\mathbf{p_C}$   | $p_D$   |
|-------------|-----|-----|-----|-----|
| Model 1     | 0.00 | 0.00 | **1.00** | 0.00 |
| Model 2     | 0.20 | 0.10 | **0.60** | 0.01 |
| Model 3     | 0.25 | 0.25 | **0.26** | 0.24 |
| Model 4     | 0.0 | 0.0 | **0.01** | 0.0 |

In this case, Model 1 is a clear best, with full confidence in the correct answer.
Model 2 is also very good, but not as sure as Model 1.
Model 3 is _almost entirely guessing_, and Model 4 doesn't even understand the format of the question, but since the evaluation method only considers the probability of the ABCD tokens, it still gets marked as correct.

All of these scenarios result in the exact same score, when maybe, they really shouldn't.

**This notebook is an attempt to address this issue by providing a more detailed analysis of model predictions on MCQA benchmarks.**

We will be interested in the following quantities
1. $\Phi = p_{\text{correct}}$, where $k$ is the correct answer
2. $\Delta = p_{\text{correct}} - \max(\{p_i : i \in I\})$ where $I$ is the set of incorrect answers

We will then plot the distribution of these quantities for a given model, and compare them across models.

Here, $\Delta$ is a measure of how much more confident the model is in the correct answer compared to the most confident incorrect answer, while $p_{\text{correct}}$ is a measure of how confident the model is in the correct answer.

An ideal model would have $\Phi = 1$ (and therefore $\Delta=1$) always, while a model that performs random guessing would have $p_i = \Phi = 0.25$ (and therefore $\Delta=0$) always.

### Reading $\Phi$ plots
Let's look at an example: MMLU on Llama-7b and Guanaco-7b, an early example of instruction tuning, in the 5-shot setting.

![Phi Plot](llama1-guanaco-base-phi-plot.png)

The <span style="color:lightblue">**blue line is Llama-7b**</span> and the <span style="color:tomato">**red line is Guanaco-7b**</span>. There are a few things of note.
* The y-axis denotes the percentage of questions which have a $\Phi$ value greater than the corresponding x-axis value. For example, looking at $\Phi=0.25$, we can see ~60% of Llama-7b's questions give correct answers with a probability greater than 0.25, corresponding to random guessing.
* Both models intersect at roughly $\Phi=0.25$ with 60% of samples, which means they both perform random guessing or worse for roughly 40% of the questions.
* Guanaco-7b has a greater number of samples with low $\Phi$ values, indicating that it's more confident of it's incorrectness.
* Similarly, Guanaco-7b has a greater number of samples with *large* $\Phi$ values, indicating that it's more confident of it's correctness as well.

### Reading $\Delta$ plots

![Delta Plot](llama1-guanaco-base-delta-plot.png)

Again, the <span style="color:lightblue">**blue line is Llama-7b**</span> and the <span style="color:tomato">**red line is Guanaco-7b**</span>. From this plot, we can see a few things:
* The 'accuracy' as we defined earlier is the percentage of samples with $\Delta > 0$. We can see this as the intersection of the curves with the vertical line at $\Delta = 0$. We can see that while instruction tuning doesn't seem to have changed the accuracy significantly, but it has vastly altered the distribution of $\Delta$ values.
* Guanaco-7b has a higher percentage of samples with large $\Delta$ values than Llama-7b. For example, in ~12-13% of the samples, Guanaco-7b predicts the correct answer with a probability at least 0.2 greater than the most confident incorrect answer.
* Guanaco-7b also has a higher percentage of samples with very low $\Delta$ values. For example, we can read that ~75% of the samples have $\Delta > -0.2$, meaning that ~25% have $\Delta \leq -0.2$. This means that Guanaco-7b predicts the wrong answer with a probability at least 0.2 greater than the correct answer in 25% of the samples, when compared to Llama-7b which only does that in ~6-7% of the samples.


## How to use this notebook

Now that you know what $\Phi$ and $\Delta$ plots are, below I've provided a simple interface to inspect plots for a wide variety of common models.
Currently, Llama 1,2,3 7/8B and variations of these models are available to compare.
I will be adding more models soon.

**Note**: All figures are *fully interactive*. Click and shift-click on their legends to select individual lines, zoom in and out, and hover over lines to see exact values at each point.
**Note**: Some models show strange behaviour in the $\Delta$ plots around $\Delta=0$. This appears to only be in instruction tuned models, and I'm currently investigating the cause. It could be weird fp16 behaviour, but I'm not sure yet.

**TODO**: Clean up and explain the model comparison tool below the performance plots.
