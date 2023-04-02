# Learning Reflection Beamforming Codebooks for Arbitrary RIS and Non-Stationary Channels
<p align="justify"> This is the simulation codes related to the following article: Y. Zhang and A. Alkhateeb, <a href="https://arxiv.org/abs/2109.14909">"Learning Reflection Beamforming Codebooks for Arbitrary RIS and Non-Stationary Channels,"</a> in arXiv preprint arXiv:2109.14909. </p>

# Abstract of the Article

<p align="justify"> Reconfigurable intelligent surfaces (RIS) are expected to play an important role in future wireless communication systems. These surfaces typically rely on their reflection beamforming codebooks to reflect and focus the signal on the target receivers. Prior work has mainly considered pre-defined RIS beamsteering codebooks that do not adapt to the environment and hardware and lead to large beam training overhead. In this work, a novel deep reinforcement learning based framework is developed to efficiently construct the RIS reflection beam codebook. This framework adopts a multi-level design approach that transfers the learning between the multiple RIS subarrays, which speeds up the learning convergence and highly reduces the computational complexity for large RIS surfaces. The proposed approach is generic for co-located/distributed RIS surfaces with arbitrary array geometries and with stationary/non-stationary channels. Further, the developed solution does not require explicitly channel knowledge and adapts the codebook beams to the surrounding environment, user distribution, and hardware characteristics. Simulation results show that the proposed learning framework can learn optimized interaction codebooks within reasonable iterations. Besides, with only 6 beams, the learned codebook outperforms a 256-beam DFT codebook, which significantly reduces the beam training overhead. </p>

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

# Dataset Structure

<p align="justify"> As shown in the figure below, we leverage <a href="https://deepmimo.net/">DeepMIMO</a> dataset to generate the data used in this paper. To simulate the non-stationarity of the large surface, we create four _geographically distributed_ reflecting surfaces with each of them employing a 64-element ULA. These surfaces maintain a distance of 1 meter between each other, and are aligned along the y-axis in the <a href="https://deepmimo.net/scenarios/o1-scenario/">O1_60</a> scenario. The generated datasets are already included in this repository and can be found in each subfolder. </p>

![Figure](https://github.com/YuZhang-GitHub/RIS_Codebook/blob/master/deep_mimo_O1_60_distributed_LIS.png)

# Simulation Structure

<p align="justify"> In the paper, we propose a multi-level RIS codebook design solution to reduce the design complexity. As shown in the figure below, the distributed RISs consist of four RISs and we further divide each RIS into two subarrays. The design starts from subarray, then the RIS and finally the four RISs. </p>

<!---
![Figure](https://github.com/YuZhang-GitHub/RIS_Codebook/blob/master/RISs.png)
-->
<p align="center">
  <img src="https://github.com/YuZhang-GitHub/RIS_Codebook/blob/master/RISs.png" alt="drawing" width="600"/>
</p>

Corresponding to the designed solution, we have four subfolders named "LIS_x" in this repository. Each of the folder corresponds to one RIS, consisting of two subarrays. A two-level learning is based on each folder:
- Step 1 (In folder `S1`): Learning from scratch of one subarray.
- Step 2 (In folder `S2`): Transfer learning of the second subarray by initializing the network parameters with the trained first subarray's.
- Step 3 (In folder `C1`): The second-level learning that combines the learning results of the two subarrays.
  
After that, the final, i.e., the third-level learning is performed:
- Step 4 (In folder `Comb_net`): The third-level learning that combines the learning results of the four RISs.

Note: At each step mentioned above, you just simply run `main.py` file.

If you have any problems with generating the figure, please contact [Yu Zhang](https://www.linkedin.com/in/yu-zhang-391275181/).

# License and Referencing
This code package is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/). If you in any way use this code for research that results in publications, please cite our original article:
> Y. Zhang and A. Alkhateeb, "[Learning Reflection Beamforming Codebooks for Arbitrary RIS and Non-Stationary Channels](https://arxiv.org/abs/2109.14909)," in arXiv preprint arXiv:2109.14909.
 
