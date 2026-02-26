Code of "HCLAA: Hierarchical Contrastive Learning with Adaptive Attention" (Information Sciences, 2026)

Multi-view clustering

Abstract:

Multi-view clustering integrates complementary features and semantics across modalities to provide comprehensive data representations. However, existing methods struggle with semantic fragmentation and inefficient view integration when processing complex multi-view data. To address these challenges, this paper proposes a Hierarchical Contrastive Learning and Adaptive Attention (HCLAA) fusion framework. HCLAA achieves fine-grained semantic decoupling by constructing multi-level semantic spaces (independent semantic space, cross-view fused semantic space, and global semantic space), and designs a dual-path attention mechanism to adaptively allocate view weights based on global semantics, suppressing interference from low-quality views while enhancing the dominant role of high-contribution views. Additionally, HCLAA jointly optimizes contrastive losses between cross-view fused features and independent features to balance view commonality and specificity, enhancing both discriminability and consistency. Experimental results confirm HCLAA’s superiority over state-of-the-art methods on benchmark datasets. Ablation studies confirm the efficacy of hierarchical semantic modeling and adaptive weighting, while parameter analysis highlights robustness to hyperparameter variations.

## Framework
![HCLAA Framework](https://raw.githubusercontent.com/XiaojianDing/2026-HCLAA/master/Figure/architecture.png)

## Requirements

torch==1.12.0  

tensorflow==2.10.0  

numpy>=1.21.0  

scikit-learn>=0.22.0  

munkres>=1.1.4

## Citation
If you find HCLAA useful in your research, please consider giving us a star and citing it with the following BibTeX entry:

```bibtex
@article{ding2026hclaa,
  title={HCLAA: Hierarchical contrastive learning with adaptive attention},
  author={Ding, Xiaojian and Li, Xian and Zhao, Lin and Zhu, Xiaoying},
  journal={Information Sciences},
  volume={741},
  pages={123249},
  year={2026},
  publisher={Elsevier}
}
