# Scene Background Modeling Using Quadtree Decomposition and Motion History Images (MHI) 
## Introduction
Scene background modeling is one of the essential procedures of background
subtraction, which is a method with different proposed algorithms to perform background foreground segmentation. The generated background model will usually be used as the reference image to compare with the video frames, or even be used in neural network training. In other words, the quality of the background model generated will greatly affect the performance of most of the background segmentation method. Therefore, it is not only a fundamental but also an essential topic to investigate.

## Method
A method of scene background modelling using quadtree decomposition
and motion history images (MHI) with a cumulative update mechanism is proposed. The proposed method is inspired by the moving object detection method proposed by Omar ELharrouss,Driss Moujahid, Samah Elkah and Hamid Tairi. Quadtree decomposition can enhance the efficiency of processing the pixels’ value and MHI can provide reliable motion feature of the foreground object in dynamic scenes.

Frames from Dataset of SBMnet and SBI will be used for testing and
evaluating the result of the proposed method. The evaluation of the result would be based on the 6 widely used metrics (AGE, pEPs, pCEPS, MSSSIM, PSNR and CQM).

## Samples with Not Bad Quality
![Image of result]
(bg_img/GT_511.jpg)
