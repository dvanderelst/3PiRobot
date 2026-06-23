# Introduction structure

Given senses encode different features with different levels of ease.
But there is overlap in the set of features different senses encode.

This allows to learn to extract features using one sense by using the other as "ground truth."
Learning this means learning an inverse model.

Bats are a good example of an animal with different senses that encode different features with different ease.

Sonar: features beyond distance need processing and the ability to recover features might be generally limited.
Vision: probably good enough to extract environmental features (object properties, and 3d spatial layouts)
[We already have some info about resolution of bat vision in the paper. Perhaps we should also mention something about 3D vision? Can we extrapolate from other animals?]

Inverse models for sonar are probably necessarily feature based instead of generic 3D reconstructions: sonar does not support rich complex 3d reconstruction [The arguments for this are already partially in the paper. We should also refer to the limited success of the echo to depth map papers but without providing details, we can keep this for this discussion] 

While the combination of sonar and vision might enable learning feature-based inverse models, this has not been demonstrated. [There is work on sonar-based classification of vegetation and scenes - by Yovel, Muller and us - however, these do not show the extraction of action-relevant features. In other words, they don't show that the features can be used in control or complete tasks.]

Refs to check - full texts in zotero:
Yovel, Y., Franz, M. O., Stilz, P. & Schnitzler, H.-U. Complex echo classification by echo-locating bats: a review. J Comp Physiol A 197, 475–490 (2011).
Eliakim, I., Cohen, Z., Kosa, G. & Yovel, Y. A fully autonomous terrestrial bat-like acoustic robot. PLOS Computational Biology 14, e1006406 (2018).
Wang, R., Liu, Y. & Müller, R. Detection of passageways in natural foliage using biomimetic sonar. Bioinspir. Biomim. 17, 056009 (2022).
Vanderelst, D., Steckel, J., Boen, A., Peremans, H. & Holderied, M. W. Place recognition using batlike sonar. eLife 5, e14188 (2016).
Achutha, A. C., Peremans, H., Firzlaff, U. & Vanderelst, D. Efficient encoding of spectrotemporal information for bat echolocation. PLOS Computational Biology 17, e1009052 (2021).




Here we demonstrate training of an inverse model for a small set of environmental features (detection and localization of pole, wall and a very coarse depth profile). The results show that the resulting inverse model is noisy and that extracting these features is not trivial. However, and going beyond existing studies, despite this, we show that the inverse model can be used to complete two tasks:

1) Obstacle avoidance and target approach
2) Following a preset path, integrating path integration and landmark recognition

This study uses a small mobile robot equipped with a bat-like sonar system (including biologically plausible echo processing), which implies realistically complex echoes and noise, which increases the biological plausibility of the model.

In the discussion, we discuss how an inverse model could be used for cross modal and cross model vicarious learning