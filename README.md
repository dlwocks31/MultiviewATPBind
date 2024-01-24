# Residue-Level Multi-View Deep Learning for Accurate ATP Binding Site Prediction and Its Applications in Kinase Drug Binding

Accurate prediction of ATP binding sites is pivotal for understanding complex cellular functions and advancing drug
discovery efforts. ATP, a key player in cellular energy processes and signaling pathways, significantly influences drug
development by regulating kinase activity. Despite its importance, predicting ATP binding sites encounters challenges,
such as the burden of feature extraction in existing methodologies and the necessity for seamless knowledge transfer
to kinase drug binding contexts. Motivated by these challenges, our research presents a novel approach: Multiview-
ATPBind and ResiBoost. Mutliview-ATPBind, an end-to-end deep learning model, integrates 1D and 3D information
into a multi-view architecture, effectively utilizing sequence and 3D coordinate data for swift residue-level pocket-ligand
interaction prediction. This design facilitates seamless knowledge transfer of ligand-pocket interactions from ATP to
kinase inhibitors. We also introduce ResiBoost, a unique residue-level boosting algorithm, to handle challenges associated
with heavily biased data in binding site prediction. On the PATP-TEST benchmark dataset, our model outperformed
the existing state-of-the-art model in balanced metrics. Notably, the implementation of the ResiBoost algorithm brought
a noticeable enhancement in the Mathew’s correlation coefficient metric. Leveraging deep learning’s ability to transfer
generalizable embeddings, preliminary results show promising accuracy in predicting ATP pocket-targeting drugs’ binding
sites, and improving docking simulations with our predictions as informative priors. Case studies on ATP and kinase
inhibitor predictions underscore the commonality between binding sites, demonstrating the applicability of our model in
the broader landscape of drug discovery.