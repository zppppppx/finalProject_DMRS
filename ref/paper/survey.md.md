## Introduction
自二十世纪之交以来，无线通信技术以惊人的速度发展。数据传输的速度也大大加快了，无限通信网络的带宽也随之迅速提高。因此，通信系统的质量已成为当今世界的关键因素。对于任何通信系统，信道估计都是至关重要的，因为信道估计的准确性会影响整个系统的质量。
在过去几年中，针对不同类型通信系统的各种传统信道估计算法进行了大量研究[2]。借助于仿真，根据各种参数（如信噪比（SNR）、误码率（BER）、均方误差（MSE）等）比较各种信道估计方案，以确定哪种方案最适合于特定类型的通信系统。对于无线正交频分复用（OFDM）系统，提出了最小均方（MMSE）估计器和最小二乘（LS）估计器[4]，还在时变色散信道中的微蜂窝OFDM上行链路中提出了基于最大似然的信道估计[6]和大型天线系统中的数据辅助信道估计方案[7]。而使用深度学习算法是信道估计系统的一个新趋势。深度学习是学习样本数据的内在规律和表示层次，这些学习过程中获得的信息对诸如文字，图像和声音等数据的解释有很大的帮助。它的最终目标是让机器能够像人一样具有分析学习能力，能够识别文字、图像和声音等数据。 深度学习是一个复杂的机器学习算法，在语音和图像识别方面取得的效果，远远超过先前相关技术。与传统方法相比，基于深度学习的技术的优点之一是其鲁棒性。尤其是当数据量较大时，深度学习具有优于传统方法的描述数据特征的良好表征能力。因此，在我们二十一世纪的无线通信系统中，由于数据速率和带宽日益增长，深度学习算法成为信号识别和信道估计领域的一种具有良好潜力的工具。在信道估计领域，深度学习已经崭露头角，但仍然存在一些问题，
接下来我们将简单介绍信道估计的理论；用于信道估计的不同的深度学习的架构以及相关挑战；最后提出一定的研究展望。

## 信道估计理论
在实际的通信系统中，任何信号在传输过程中均会受到由于不理想信道带来的污染，多种噪声会加在信号本身，给接收端的解调以及检测等工作带来很大的阻碍。为了能够提高通信系统的效率，我们需要尽可能消除不理想信道导致的信号扭曲，而这就需要我们对信道的特征进行描述，也就是信道估计。通常信道估计是通过比较已知信号在收发机两端的不同来获得大致的信道矩阵。整体过程为：首先在发射机端，我们发送一系列已知信号，即导频信号，这些导频信号通常经过一定的设计使其便于在接收端检测得到；随后这些信号通过信道被噪声扭曲；最后我们在接收机端收到相关信号，通过比较收发信号的差异，我们得到其相关关系，从而得到对信道的估计。

## 基于深度学习的信道估计算法概述
在机器学习中，根据训练方式大致可以分为有监督学习、无监督学习与强化学习。其中有监督学习被广泛应用于图像处理领域。在通信领域，对于基于深度学习的信达估计算法，我们不需要进行分类，而是需要获得数据的内在关系，拟合一定的算法，故一般视为无监督学习。
2018：多数论文的深度学习架构基于经典深度学习结构，使用线性层或卷积层的连接来进行信道估计。在论文[**12**]中，作者提出了一种基于深度学习的信道估计算法，主要针对毫米波大规模 MIMO 系统。在该应用场景中，MIMO 系统的天线阵列数量巨大，而射频链路的数量却相对来说非常小，从而使得由收到的信号估计信道这项任务变得具有很大的挑战性。该论文针对该应用场景，提出了基于经典深度学习架构的 LDAMP（learned denoising-based approximate message passing）网络，其中每一层均为相同的结构，通过多层连接，逐步学习得到信道估计。其中的去噪模块，论文使用了计算机视觉领域（CV）的相关工作 DnCNN（denoising convolutional neural network）[DnCNN]来实现。DnCNN 的结构与经典卷积网络 VGG 类似，通过多层卷积层连接，但是最后其学习得到的是残差图像，即为噪声图像。作者通过仿真结果证明了 LDAMP 网络的优越性能，同时比较了 LDAMP 网络、SD 算法、SCAMP 算法以及 DAMP 算法，得出去噪模块的使用使得 LDAMP 网络和 DAMP 算法获得了性能上的提升。
2019：在[Deep CNN-Based Channel Estimation for mmWave Massive MIMO Systems] 中，研究团队针对毫米波大规模 MIMO 系统设计了基于与[12]相同的应用场景。论文逐步提出了三种架构，分别为空间-频率卷积网络（SF-CNN）、空间-频率-时间卷积网络（SFT-CNN）和空间导频减少卷积网络（SPR-CNN），其中 SFT-CNN 比 SF-CNN 能更好地提取到信道时域上的相关性，而 SPR-CNN 能够在一定程度上减少导频的使用，节省频谱资源。数值结果表明，SFT-CNN 能够取得与理想 MMSE 相近的结果，同时大大节省计算量。而 SPR-CNN 能够在减少导频的情况下仍然有与 SF-CNN 相近的性能。
2018：又如[18]，在文中作者提出了针对 MIMO 多用户系统基于深度学习的信道估计算法以及导频设计。他们使用一个两层的神经网络设计导频，以及一个 DNN 用于估计信道。该文提出导频长度可以压缩，虽然会失去正交性，但是利用深度学习的非线性拟合能力可以在一定程度上解决这个问题。模拟实验的数值结果证明在此情形下，提出的深度学习算法取得了远超过 LMMSE 的性能。
2019：部分论文尝试深度学习与经典算法相结合，在论文[Deep Learning-Based Channel Estimation for Doubly Selective Fading Channels]中，作者针对在双选择性衰落信道情形下，难以精确建模的问题，设计了最小二乘（LS）与 DNN 混合的深度学习架构，将 LS 估计得到的信道信息作为特征输入，以已知全信号获得对于信道的估计。获得了逼近线性最小均方误差（LMMSE）的性能。
2019：另外多篇论文将信道矩阵视为二维图像，尝试通过 CV 的方法重建信道矩阵。如在[Deep Learning-Based Channel Estimation]中，一种基于深度学习的信道估计算法被提出，名为 ChannelNet。该论文快衰落 OFDM 信道的时频响应视为一张 2D 图像，其目的是利用导频作为已知信号，获得信道响应的未知值。该论文提出了一种基于深度学习图像处理的方法。导频信号的响应被视为视频响应矩阵的部分采样，通过图片超分辨率重建（Super Resolution, SR）与图像恢复（Image Restoration， IR）来获得完整的时频响应矩阵。首先，图片超分辨率重建模块估计未知位置的时频响应，随后通过图像恢复模块来消除噪声的影响。该论文采用了基于深度卷积网络的深度图像算法分别实现超分辨率模块和图像恢复模块，分别为 SRCNN[SRCNN] 和 DnCNN[DnCNN]。其中 SRCNN 首先通过线性插值大致恢复至原始图像尺寸，随后通过三层卷积网络进行更加精确的重建。论文通过数值实验发现，在信噪比低于 20 dB 时，该算法取得了逼近理想 MMSE 的优良性能，而在高于 23 dB 时，性能有所退化，需要设计、训练新的网络。
2018：此外，还有多篇论文探讨了模型驱动的深度学习架构设计方法，如[CsiNet]中提出了一种基于深度学习的网络的网络，称为CsiNet，以减少大规模MIMO系统中的反馈开销。CsiNet的网络架构是通过模仿CS架构获得的，CS架构可以看作是模型驱动DL的一个特例。CsiNet主要包括一个卷积神经网络（CNN），该网络成功地进行了图像处理，并采用了一种自动编码器架构，该架构包括一个用于压缩感知的编码器和一个用于重建的解码器。每个细化网络单元遵循残余网络的思想，即它将较浅层的输出传输到较深层的输入，以避免DNN中的梯度消失问题。但该网络及其改进网络 CsiNet-LSTM[CsiNet-LSTM] 不适用于实际的时变信道，因为线性全连接网络不适合描述时间相关性。另外该设计也未考虑天线的空间相关性。对于信道估计这一问题，数学模型难以精确建模描述信道的时间相关性与频域相关性，从而使得基于模型驱动的深度学习架构容易忽略数据的相关关系，进而限制模型的性能。


## 研究现存问题
目前已有的基于深度学习的信道估计算法已经有对于 MIMO 系统的信道有良好的性能表现，但是仍然存在部分问题。例如基于导频估计信道的多种深度学习算法，由于需要从少量时频信道值获得整个时频域的信道值，首先需要使用诸如插值等手段重建高精度信道矩阵，在重建过程中会将噪声扩散至其他时频位置，而非线性插值会使得扩散后的噪声非高斯白噪声，从而导致用于处理高斯白噪声的深度卷积网络表现不佳。另外，现有网络存在不够精细的缺点，虽然在信噪比较低的情况下能够取得明显优于传统算法的特点，但是当信噪比较高时，性能出现了一定的退化。传统深度学习，如 DNN 或 CNN，存在一定的精细度的限制，为了提高深度学习在这些情形下对信道的估计能力，需要完成进一步的研究，采用新的结构或者设计更加有效的网络架构。最后，部分设计忽视了实际通信信道中的时间相关性或者频率相关性，难以应用至现实信道。

## 研究展望
近年，深度学习在 CV 以及自然语言处理（NLP）领域取得了引人瞩目的发展。在自然语言处理领域提出了具有重要意义的 Transformer 模型[Transformer]，该模型弥合了 CV 与 NLP 之间的模型差异，且具有应用至多个领域的潜力，目前已有相关尝试[Transformer Empowered CSI Feedback for Massive MIMO Systems]。该模型可以观察到两个维度的信息，如时间的相关性和特征的相关性，在信道估计中可以用于建模时频两个维度的相关性。在 5G 时代，信道的时间与频率选择性衰落效应更加显著，传统算法越来越难以对信道状态进行建模估计，在目前的通信系统架构下，深度学习拥有的非线性拟和能力将会赋能通信。而且硬件的发展使得在基站等终端部署深度学习算法的成本降低，从而开发通信领域的深度学习算法将会成为未来的一个研究重点。另外，深度学习也有助于我们设计更加智能的通信系统，提高通信效率，突破传输速率的限制。

## 参考文献

[2]: D. Neumann, T. Wiese, and W. Utschick, “Learning the MMSE channel estimator,” IEEE Trans. on Signal Process., vol. 66, no. 11, pp. 2905–2917, 2018.
[4]: Z. Du, X. Song, J. Cheng, and N. C. Beaulieu, “Maximum likelihood based channel estimation for macrocellular OFDM uplinks in dispersive time-varying channels,” IEEE Trans. on Wireless Commun., vol. 10, no. 1, pp. 176–187, 2011.
[6]: C. Jiang, H. Zhang, Y. Ren, Z. Han, K.-C. Chen, and L. Hanzo, “Machine learning paradigms for next-generation wireless networks,” IEEE Wireless Commun., vol. 24, no. 2, pp. 98–105, 2017.
[7]: Er, M.J., Zhou, Y.: Theory and novel applications of machine learning. InTech, 2009.
[12]: H. He, C.-K. Wen, S. Jin, and G. Y. Li, “Deep learning-based channel estimation for beamspace mmWave massive MIMO systems,” IEEE Wireless Commun. Lett., vol. 7, no. 5, pp. 852–855, 2018.
[DnCNN]: K. Zhang, W. Zuo, Y. Chen, D. Meng, and L. Zhang, “Beyond a Gaussian denoiser: Residual learning of deep CNN for image denoising,” IEEE Trans. Image Process., vol. 26, no. 7, pp. 3142–3155, 2017.
[Deep CNN-Based Channel Estimation for mmWave Massive MIMO Systems]: P. Dong, H. Zhang, G. Y. Li, I. S. Gaspar, and N. NaderiAlizadeh, “Deep CNN-based channel estimation for mmWave massive MIMO systems,” IEEE J. Sel. Topics in Signal Process., vol. 13, no. 5, pp. 989–1000, 2019.
[18]: C.-J. Chun, J.-M. Kang, and I.-M. Kim, “Deep learning-based joint pilot design and channel estimation for multiuser MIMO channels,” IEEE Commun. Lett., vol. 23, no. 11, pp. 1999–2003, 2019.
[Deep Learning-Based Channel Estimation for Doubly Selective Fading Channels]: Y. Yang, F. Gao, X. Ma, and S. Zhang, “Deep learning-based channel estimation for doubly selective fading channels,” IEEE Access, vol. 7, pp. 36 579–36 589, 2019.
[Deep Learning-Based Channel Estimation]: M. Soltani, V. Pourahmadi, A. Mirzaei, and H. Sheikhzadeh, “Deep learning-based channel estimation,” IEEE Commun. Lett., vol. 23, no. 4, pp. 652–655, 2019.
[SRCNN]: C. Dong, C. C. Loy, K. He, and X. Tang, “Image super-resolution using deep convolutional networks,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 38, no. 2, pp. 295–307, 2016.
[CsiNet]: C. K. Wen, W. T. Shih, and S. Jin, “Deep Learning for Massive MIMO CSI Feedback,” IEEE Wireless Commun. Lett., vol. 7, no. 5, 2018, pp. 748–51.
[CsiNet-LSTM]: T. Wang, C. Wen, S. Jin, G. Y. Li, "Deep Learning-Based CSI Feedback Approach for Time-Varying Massive MIMO Channels," IEEE Wireless Commun. Lett., Vol. 8, no. 2, pp. 416-419, 2019.
[Transformer]: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017.
[Transformer Empowered CSI Feedback for Massive MIMO Systems]: Y. Xu, M. Yuan, and M.-O. Pun, “Transformer empowered CSI feedback for massive MIMO systems,” in Wireless and Optical Commun. Conf., Taipei, Taiwan, Oct. 2021, pp. 157–161.
