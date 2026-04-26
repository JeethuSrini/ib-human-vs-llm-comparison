# Memory Through the Lens of Compression  
## An Empirical Information Bottleneck Comparison of Humans and Large Language Models

### Authors
- Jeethu Srinivas Amuthan (NYU, Psychology)
- Dhiwahar Adhithya Kennady (NYU, CDS)

---

# Abstract

Every learning system, biological or artificial, must balance two competing pressures: memorizing specific experiences and extracting generalizable structure. The Information Bottleneck (IB) principle provides a principled framework for this tradeoff by characterizing representations that retain task-relevant information while compressing input data.

In this project, we use the IB framework as a comparative lens to study humans and Large Language Models (LLMs). Rather than attempting to compute the exact IB-optimal bound—which is intractable in real-world settings—we construct empirical rate–distortion profiles using behavioral accuracy and representation-based proxies for information.

Using controlled memory tasks and matched evaluation settings, we measure how each system trades off memorization and generalization. We approximate task-relevant information using accuracy and estimate input information retention using probe-based decoding of hidden representations.

---

# 1. Introduction

Human memory is not a recording device; it is a compression process. Rather than storing experiences verbatim, the cognitive system selectively retains information that is useful for future behavior while discarding irrelevant detail. Classic findings such as schema-driven distortions, gist-based memory, and working memory limits suggest that human cognition operates under strong compression constraints.

The Information Bottleneck (IB) principle formalizes this tradeoff. Given input $begin:math:text$X$end:math:text$, target $begin:math:text$Y$end:math:text$, and representation $begin:math:text$T$end:math:text$, the IB objective minimizes:

I(X;T) - β I(T;Y)

where:
- I(X;T) measures how much information from the input is retained (rate)
- I(T;Y) measures task-relevant information (utility)

Large Language Models (LLMs), trained on massive datasets, face a similar tension between memorization and generalization. Recent work suggests that both humans and LLMs may be understood through an IB-like lens.

However, computing the true IB-optimal curve requires access to full data distributions and is generally intractable. In this project, we adopt an empirical approach using measurable proxies.

---

# 2. Research Objective

We aim to answer:

How do humans and LLMs differ in how much information they retain versus how well they perform?

Specifically:
- Do humans operate at lower effective rates (more compression)?
- Do LLMs retain more input information (more memorization)?
- Do both systems converge under tasks requiring abstraction?

---

# 3. Methodology

## 3.1 Task Design (Controlled Memory Setting)

We construct synthetic tasks with a fixed set of possible facts:

Example fact pool:
- A → B  
- C → D  
- E → F  
- G → H  
- I → J  

Each input X consists of a subset of these facts.

Example input:

Facts:
A → B
E → F
G → H

We then query the system:

Question: What is A?

---

## 3.2 Human Baseline

- Distortion: 1 − accuracy  
- Rate proxy: confidence ratings  

---

## 3.3 LLM Evaluation

For each model:
1. Feed input text  
2. Extract hidden representation T (last layer, last token)  
3. Generate answer and compute accuracy  

---

## 3.4 Measuring Task-Relevant Information

We approximate I(T;Y) using accuracy.

---

## 3.5 Measuring Input Information via Probing

Step 1: Create labels

Example:

Facts:
A → B
E → F
G → H

Label:
[1, 0, 1, 1, 0]

---

Step 2: Extract hidden representation T

---

Step 3: Train probe models

- Input: T  
- Output: 0/1 for each fact  

---

Step 4: Evaluate probe accuracy

- High accuracy → more information retained  
- Low accuracy → more compression  

---

## 3.6 Rate–Distortion Plot

- X-axis: distortion (1 − accuracy)  
- Y-axis: probe accuracy (rate proxy)  

Each system corresponds to a point.

---

# 4. Hypotheses

- H1: Humans operate at lower effective rates (more compression)  
- H2: LLMs retain more information (more memorization)  
- H3: Both converge on reasoning tasks  

---

# 5. Expected Contributions

- Practical IB approximation framework  
- Controlled comparison between humans and LLMs  
- Insights into memory vs generalization tradeoffs  

---

# 6. Limitations

- No exact IB bound  
- Mutual information approximated via proxies  
- Probe captures accessible information only  

---

# 7. Timeline

- Weeks 1–2: Task design  
- Weeks 3–4: Model evaluation  
- Weeks 5–6: Probe training  
- Weeks 7–8: Analysis and writing  

---

# 8. Key Takeaway

We compare systems based on:

- How much information they store  
- How useful that information is  

This provides a unified framework for studying learning systems.

---
