# Learning Reflection Beamforming Codebooks for Arbitrary RIS and Non-Stationary Channels
This is the simulation codes related to the following article: Y. Zhang and A. Alkhateeb, "[Learning Reflection Beamforming Codebooks for Arbitrary RIS and Non-Stationary Channels](https://arxiv.org/abs/2109.14909)," in arXiv preprint arXiv:2109.14909.

# Abstract of the Article
Reconfigurable intelligent surfaces (RIS) are expected to play an important role in future wireless communication systems. These surfaces typically rely on their reflection beamforming codebooks to reflect and focus the signal on the target receivers. Prior work has mainly considered pre-defined RIS beamsteering codebooks that do not adapt to the environment and hardware and lead to large beam training overhead. In this work, a novel deep reinforcement learning based framework is developed to efficiently construct the RIS reflection beam codebook. This framework adopts a multi-level design approach that transfers the learning between the multiple RIS subarrays, which speeds up the learning convergence and highly reduces the computational complexity for large RIS surfaces. The proposed approach is generic for co-located/distributed RIS surfaces with arbitrary array geometries and with stationary/non-stationary channels. Further, the developed solution does not require explicitly channel knowledge and adapts the codebook beams to the surrounding environment, user distribution, and hardware characteristics. Simulation results show that the proposed learning framework can learn optimized interaction codebooks within reasonable iterations. Besides, with only 6 beams, the learned codebook outperforms a 256-beam DFT codebook, which significantly reduces the beam training overhead.

<!---
# How to generate this codebook beam patterns figure?
1. Download all the files of this repository.
2. Run `main.py` in `critic_net_training` directory.
3. After it is finished, there will be a file named `critic_params_trsize_2000_epoch_500_3bit.mat` that will be used in the next step.
4. Run `main.py` in `analog_beam_learning` directory.
5. After it is finished, run `read_beams.py` in the same directory.
6. Copy the generated file, i.e., `ULA_PS_only.mat` to the `td_searching` directory.
7. Run `NFWB_BF_TTD_PS_hybrid_low_complexity_search_algorithm.m` in Matlab, which will generate the figure shown below.

![Figure](https://github.com/YuZhang-GitHub/NFWB_BF/blob/main/N_16.png)

-->

# Simulation setup and structure

If you have any problems with generating the figure, please contact [Yu Zhang](https://www.linkedin.com/in/yu-zhang-391275181/).

# License and Referencing
This code package is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/). If you in any way use this code for research that results in publications, please cite our original article:
> Y. Zhang and A. Alkhateeb, "[Learning Reflection Beamforming Codebooks for Arbitrary RIS and Non-Stationary Channels](https://arxiv.org/abs/2109.14909)," in arXiv preprint arXiv:2109.14909.
 
![Figure](https://github.com/YuZhang-GitHub/RIS_Codebook/blob/master/deep_mimo_O1_60_distributed_LIS.png)
