# ANLS: Average Normalized Levenshtein Similarity

[![CI](https://github.com/shunk031/ANLS/actions/workflows/ci.yaml/badge.svg)](https://github.com/shunk031/ANLS/actions/workflows/ci.yaml)

This script is based on the one provided by [the Robust Reading Competition](https://rrc.cvc.uab.es/?com=introduction#) for evaluation of the InfographicVQA task.

The Average Normalized Levenshtein Similarity (ANLS) proposed by [[Biten+ ICCV'19](https://arxiv.org/abs/1905.13648)] smoothly captures the OCR mistakes applying a slight penalization in case of correct intended responses, but badly recognized. It also makes use of a threshold of value 0.5 that dictates whether the output of the metric will be the ANLS if its value is equal or bigger than 0.5 or 0 otherwise. The key point of this threshold is to determine if the answer has been correctly selected but not properly recognized, or on the contrary, the output is a wrong text selected from the options and given as an answer.

More formally, the ANLS between the net output and the ground *truth answers is given by* equation 1. Where $N$ is the total number of questions, $M$ total number of GT answers per question, $a_{ij}$ the ground truth answers where $i = \{0, ..., N\}$, and $j = \{0, ..., M\}$, and $o_{qi}$ be the network's answer for the ith question $q_i$:

$$
    \mathrm{ANLS} = \frac{1}{N} \sum_{i=0}^{N} \left(\max_{j} s(a_{ij}, o_{qi}) \right),
$$
where $s(\cdot, \cdot)$ is defined as follows:
$$
    s(a_{ij}, o_{qi}) = \begin{cases}
    1 - \mathrm{NL}(a_{ij}, o_{qi}), & \text{if } \mathrm{NL}(a_{ij}, o_{qi}) \lt \tau \\
    0,                               & \text{if } \mathrm{NL}(a_{ij}, o_{qi}) \ge \tau
    \end{cases}
$$

## References

- Biten, Ali Furkan, et al. "Scene text visual question answering." Proceedings of the IEEE/CVF international conference on computer vision. 2019.
