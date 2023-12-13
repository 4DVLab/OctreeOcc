<p align="center">
  <h1 align="center">OctreeOcc: Efficient and Multi-Granularity Occupancy Prediction Using Octree Queries</h1>
  <p align="center">
    <a href="https://github.com/yuhanglu2000">Yuhang Lu</a>,
    <a href="https://xingezhu.me">Xinge Zhu</a>,
    <a href="https://tai-wang.github.io">Tai Wang</a>,
    <a href="https://yuexinma.me/aboutme.html">Yuexin Ma</a>


  </p>
  <h3 align="center"><a href="https://arxiv.org/abs/2312.03774">Paper</a> <a </h3>
  <div align="center"></div>

## Main Idea

</p>

<p align="center">
  <a href="">
    <img src="assets/teaser.png" alt="Relit" width="75%">
  </a>
</p>
<p align="center">
  OctreeOcc employs octree queries to offer varying granularity for distinct semantic regions, thereby diminishing the requisite number of queries for modeling and mitigating the issue of low information density in 3D space.
</p>

## Architecture overview

<br>

<p align="center">
  <a href="">
    <img src="./assets/pipeline.png" alt="Pipeline" width="99%">
  </a>
</p>
<p align="center">
  Given multi-view images, we extract multi-scale image features utilizing an image backbone. Subsequently, the initial octree structure is derived through image segmentation priors, and the transformation of dense queries into octree
queries is effected. Following this, we concomitantly refine octree queries and rectify the octree structure through the octree encoder. Finally, we decode from the octree query and obtain occupancy prediction outcomes for this frame. For better visualisation, the diagram of Iterative Structure Rectification module shows octree query and mask in 2D form(quadtree).
</p>

<br>

## Performance

<br>

<p align="center">
  <a href="">
    <img src="./assets/exp_1.png" alt="main_res" width="99%">
    <img src="./assets/exp_2.png" alt="eff_res" width="75%">
  </a>
</p>
<p align="center">
  Experiments conducted on the Occ3D-nuScenes dataset demonstrate that our approach enhances performance while substantially decreasing computational overhead (even when compared to 2D modeling approaches).
</p>

<br>

## Visualization

<br>

<p align="center">
  <a href="">
    <img src="./assets/vis.png" alt="vis" width="99%">
  </a>
</p>
<p align="center">
  Qualitative results on Occ3D-nuScenes validation set. The first row displays input multi-view images, while the second row showcases the occupancy prediction results of PanoOcc, FBOCC, our methods, and the ground truth
</p>

<br>

## Next Step

The code will be released after the paper is accepted.

