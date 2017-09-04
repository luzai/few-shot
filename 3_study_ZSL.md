---
title: 'A Stduy about Zero Shot Learning'
journal: ''
bibliography: references.bib
linenumbers: false
author:
  - name: intern@mmlab
  - name: _Sep. 4, 2017_
abstract: |
 This study contains **1).** The state of the art of Zero-Shot Learning(_ZSL_). **2).** Benchmark and evaluation metric for ZSL. **3).** Insights and proposals for our stduy.
fignos-cleveref: On
fignos-plus-name: Fig.
header-includes: \usepackage{caption}
toc: true
...

<!-- # TODO
- zero-shot dataset
- evaluation and train method and metric
- the state of the art
    - semantic autoencoder for single-shot
    - zero-shot learning code & data
    - sythetic classifier fot zero-shot [@changpinyo2016synthesized] -->


# Task Formulation
- _Tranfer Learning_/_Domain Adaption_ [@Goodfellow-et-al-2016]: Use learned feature in one setting (i.e., distribution $P_1$)  to improve generalization in another setting (say distribution $P_2$).
- _One-shot Learning_: A extreme form of transfer learning.
- _Zero-shot Larning_: Compare to tradition learning scenario that needs  inputs $\mathbf{x}$ and targets $\mathbf{y}$, zero-shot learning must need **side information** exploited during training, that is the task ${T}$. The  model is trained to estimate the conditional distribution $p(\mathbf{y}|\mathbf{x},T)$.
    - Side information includes: Attributes, WordNet, detailed visual descriptions and its deep representations, human gaze and its embeddings.

## Zero Shot Learning(ZSL)

Gvien $\mathcal{S}=\{(x_n,y_n),n=1...N\}$, learn $f:\mathcal{X} \rightarrow \mathcal{Y}$ by minimizing the regularized empirical risk
$$\frac{1}{N}\sum\limits_{n=1}^{N}L(y_n,f(x_n,\mathbf{W} ))+ \Omega (\mathbf{W})$$
For Zero Shot Learning, $\mathcal{Y}^{tr} \cup \mathcal{Y}^{ts}=\Phi$; for  Generalized  Zero Shot Learning, $\mathcal{Y}^{ts} \subseteq \mathcal{Y}^{tr}$, _i.e._, test image can be labeled with both seen and unseen classes.

![The Method for ZSL. Class in $\mathbf{y}$-space is embeded to another latent space $\mathbf{h}_y$. There is internal structure between classes! (May be hierarchy or overlapping!) ZSL aim to learn word and image representation and the relations between them. The relation may be learned by similarity of structure or cue of co-occurance.](assets/markdown-img-paste-20170904001525722.png){#fig:method width=5in}

# Some Relevant Methods
## Sythetic classifier
Synthesized Classifiers for Zero-Shot Learning [@changpinyo2016synthesized]. This paper makes lots of simplication.

![How to align two representation space? Embedding semantic space into image feature space.](assets/markdown-img-paste-20170904003209472.png){#fig:sythetic width=5in}

The final unified loss function of [@changpinyo2016synthesized] is:

![](assets/markdown-img-paste-20170904003229338.png){width=3in}

$x_n$ is hand-crafted shallow feature in this paper, _e.g._ SIFT. The first two terms are one-versus-other loss. The third term assumes phantom base code $\mathbf{b}_r$ can be made from existed human labeled attributes $\mathbf{a}_c$, that is $b_r=\Sigma^{ \mathsf{S}}_{c=1} \beta_{rc} \mathbf{a}_c$, and this term is just regularizing linear combination weights. And for the constrains contains a similarity $s_{cr}$ is designed as $s_{cr}=\frac{exp(-d(\mathbf{a}_c,\mathbf{b}_r))}{\Sigma^{\mathsf{R}}_{r=1}exp(-d(\mathbf{a}_c,\mathbf{b}_r))}$ and the constrains can be explained as enforce similarity structure in semantic space to feature space. (The author explains that it is analytical solution of Laplacian eigenmaps that minimize distortion error $\min_{\mathbf{w}_c,\mathbf{v}_r} \|\mathbf{w}_c-\Sigma_{r=1}^{\mathsf{R}}s_{cr}\mathbf{v}_r\|_2^2$, but it is strange that when $\mathbf{w}_c=\Sigma_{r=1}^{\mathsf{R}}s_{cr}\mathbf{v}_r$, the error is exact zero!)

In fact, this model is non-convex.


# Benchmark
Zero-Shot Learning - The Good, the Bad and the Ugly [@DBLP:journals/corr/XianSA17].

## Dataset

![Example images from [a-Pascal](http://vision.cs.uiuc.edu/attributes/) (top row) and a-Yahoo (bottom row). Images in a-Pascal and a-Yahoo are from disjoint categories.](assets/markdown-img-paste-20170904203328646.png){#fig:apy width=5in}

![Animals with Attributes (AWA)](assets/markdown-img-paste-20170904204025851.png){#fig:rand width=3in}

<!-- - Caltech-UCSD-Birds 200-2011 (CUB)
- SUN
- ImageNet -->

## Metric

- *Proposed Split*: 101-layerd ResNet pretrained on ImageNet 1K, and this 1K classes should not exist in $\mathcal{Y}^{ts}$.

| Dataset | Classes  $\mathcal{Y}^{tr}$ | Classes   $\mathcal{Y}^{ts}$ |
| ------- | --------------------------- | ---------------------------- |
| SUN     | 580+65                      | 72                           |
|CUB         |100+50                             |50                              |
|AWA   |27+13   |10   |
|aPY   |15+5   |12   |
|ImageNet   | 800+200  | 500/1K/5K[^fn]  |

[^fn]: Make sure to be far away from training classes, _i.e._, 2-hops/3-hops far away. Test classes can be most-populated classes/least populated classes(containing few images in this class).

- *Evaluatioin Critetia*: First calculate top-1 accuracy for each class, then take average on classes.



<!-- CUB      |                               |                                |                                   |
|   |   |   |   |
AWA      |                               |                                |                                   |
aPY      |                               |                                |                                   |
ImageNet |                               |                                |                                   | -->




<!--
Latent Attribute embedding

![](assets/markdown-img-paste-20170903161828726.png)
![](assets/markdown-img-paste-20170903162356559.png)
![](assets/markdown-img-paste-20170903162527609.png)


![](assets/markdown-img-paste-20170903160139847.png)
![](assets/markdown-img-paste-20170903160312111.png)
![](assets/markdown-img-paste-20170903161635840.png)


![](assets/markdown-img-paste-20170903163549781.png)
![](assets/markdown-img-paste-20170903163608233.png)
![](assets/markdown-img-paste-2017090322205947.png)
![](assets/markdown-img-paste-20170903222411866.png)
![](assets/markdown-img-paste-2017090322242065.png) -->
