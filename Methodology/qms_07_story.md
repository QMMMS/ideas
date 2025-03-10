# 论文故事梳理

再好的想法，如果没找准卖的点也可能会被拒

> CVPR 2020 投稿
>
> - 卖点：一个可微分的匹配层，由粗到细来优化，以及用相机位姿做监督
> - 结果：3个borderline，最后被拒，评审人喷这个可微分的匹配层在别的领域已经有人做了，由粗到细来优化也没创新
>
> ECCV 2020 投稿
>
> - 卖点：单纯就卖相机位姿做监督这个最核心的点，之前也没有人做过
> - 结果：接收为 Oral Presentation

如何找准最好的故事角度？核心是回答以下的问题：

- 我们在解决的问题是什么？ 
- 为什么这个问题很重要？ 
- 之前的方法有哪些，他们有什么问题？ 
- 我们方法核心是什么，有什么是只有我们可以做到的？ 
- 我们（将）获得什么新的认知？

记住：在你的项目的每个阶段都要回答这些问题，而不是留到最后

## 案例学习：ResNet

- 我们在解决的问题是什么？ • 深度网络比浅层网络难训练非常多
- 为什么这个问题很重要？ • 越深的网络表现力越强，如果能解决训练问题，太多应用了
- 之前的方法有哪些，他们有什么问题？ • AlexNet, VGG Net, 一旦网络加深就训练不动
- 我们方法核心是什么，有什么是只有我们可以做到的？ • 残差很容易学习，第一次可以把一个152层的网络训练起来
- 我们（将）获得什么新的认知？ • 巨多的实验上全部都有效，巨多的消融实验去分析，牛!

## 案例学习：ConvONet

- 我们在解决的问题是什么？ • 如何获得精细的三维重建
- 为什么这个问题很重要？ • 经典三维视觉和图形学的问题，对虚拟现实，游戏，自动驾驶都有很大意义
- 之前的方法有哪些，他们有什么问题？ • Occupancy Networks: 全局潜在表征 (global latent code) 导致过于平滑的重建，只能物体
- 我们方法核心是什么，有什么是只有我们可以做到的？ • 提出三种局部表征 (local latent code) 的方式，然后对应使用卷积网络去加强表达能力
- 我们（将）获得什么新的认知？ • 不仅大大提升物体重建质量，卷积网络的平移相等性（translation equivariance）也能直接 使得大场景重建成为可能。同时，我们提出的tri-plane 表征能同时降低内存以及提升表达

## 案例学习：KiloNeRF

- 我们在解决的问题是什么？ • 如何在保证高质量渲染的同时，加速神经辐射场 (NeRF) 到实时
- 为什么这个问题很重要？ • 实时渲染可以大大缩短游戏电影制作周期和成本
- 之前的方法有哪些，他们有什么问题？ • NeRF: 单独一个大的网络，每一次渲染过程都要前传整个网络，非常慢
- 我们方法核心是什么，有什么是只有我们可以做到的？ • 把空间分割成一千多个小的区域，每个里面都有一个非常小的网络，直接帮助大大加速
- 我们（将）获得什么新的认知？ • 这样的分治法 (Divide and Conquer) 能大大降低运算成本的同时直接提升运算速度

## Extra Tips

- 主人翁意识：别期待合作者帮你想出故事，你对你的项目是最懂的，积极主动自 己想出你觉得最好的故事 
- 确保核心想法可以清楚的传达到：通过你的题目，摘要, 引言，图等等，不断强化 
- 需要迭代很多次，所以尽早开始梳理你的故事！