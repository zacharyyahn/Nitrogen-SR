This is the repository for my MSc Thesis at University College Dublin

<b>Abstract</b>
Nitrogen dioxide is a common air pollutant that can be diffi-
cult to track with traditional ground-based monitoring. While satellites
can provide plentiful data and global coverage, such images are often
captured at low resolutions that prohibit fine-grained predictions. In this
work we investigate the application of super-resolution to the nitrogen
dioxide prediction problem by super-resolving Sentinel-2 images with
several popular models and our own lightweight architecture. Instead
of collecting a second high-resolution dataset for ground-truths, we ex-
plore two methods for super-resolving images with a single dataset, thus
assessing the viability of a greatly simplified super-resolution pipeline.
Furthermore, we investigate the relationship between common super-
resolution metrics such as per-pixel error and perceptual quality with
empirical performance on the downstream nitrogen dioxide prediction
task. We find that our bespoke very small model, NinaCustom, outper-
forms popular deep models on common error benchmarks while offering
inference time speedups of up to 50x. We observe that our approach has
the potential to slot into the pre-processing steps of numerous computer
vision tasks without adding a significant amount of computational over-
head or requiring additional data collection while increasing downstream
performance.

For inquiries please contact me at zachary {dot} yahn {at} ucdonnect {dot} ie
