---
datasets:
- nvidia/PhysicalAI-Robotics-GraspGen
language:
- en
---

Project Website: https://graspgen.github.io/ <br>
Code: https://github.com/NVlabs/GraspGen/

Abstract: Grasping is a fundamental robot skill, yet despite significant research advancements, learning-based 6-DOF grasping approaches are still not turnkey and struggle to generalize across different embodiments and in-the-wild settings. We build upon the recent success on modeling the object-centric grasp generation process as an iterative diffusion process. Our proposed framework - GraspGen - consists of a Diffusion-Transformer architecture that enhances grasp generation, paired with an efficient discriminator to score and filter sampled grasps. We introduce a novel and performant on-generator training recipe for the discriminator. To scale GraspGen to both objects and grippers, we release a new simulated dataset consisting of over 53 million grasps. We demonstrate that GraspGen outperforms prior methods in simulations with singulated objects across different grippers, achieves state-of-the-art performance on the FetchBench grasping benchmark, and performs well on a real robot with noisy visual observations.

## Model Architecture: <br> 
**Architecture Type:** Diffusion Model, Point Cloud network. See paper for more details. <br>

## Input: <br>
**Input Type(s):** Object partial point cloud X, Number of grasps to sample (B) <br>
**Input Format(s):** Point Cloud (N X 3) where N is the number of points <br>
**Input Parameters:** 3D <br>
**Other Properties Related to Input:** Point cloud needs to be in the form (N X xyz) where N=2048 is the number of points.<br>

## Output: <br>
**Output Type(s):** Grasp Poses; Corresponding confidence scores <br>
**Output Format:** Homogenous Transformation matrices; score is a scalar value from 0 to 1 <br>
**Output Parameters:** [B, 4, 4] where B is the number of generated grasp poses; [B, 1] confidence score <br>
**Other Properties Related to Output:** <br> 