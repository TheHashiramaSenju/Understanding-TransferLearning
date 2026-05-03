
Abstracted topic reads :

1. L1 and L2 regulizers, why used on different formulas and how they are used 
2. pytorch implementations and the actual working of stuffs on how they monitor the val accuracies and what are actually going on inside them actually
3. How they read stuffs like vlaidatoin, how do they get stored inside the pytorch ? and how reading and writing happens ?
4. Any way in which I can boost space complexity using new algortihms in pytorch
5. Implementing a new algorithm and how to make my own algorithm to suit in this pytorch, what are their points or endpoints + how to incorporate inside them?
6. Wavelets and LRs in Jumpings. Know more about these 
7. How detection happens how the network comes to know that this is present and the differentials change
8. 
Regardless of them : Reading the abstracts to be kept for later, let us see the abstracted functions for how they work 

| **Approach**                   | **Manual or Auto?** | **What You Control**                             |
| ------------------------------ | ------------------- | ------------------------------------------------ |
| **No scheduling**              | Manual              | You pick ONE fixed LR for all 30 epochs          |
| **ReduceLROnPlateau**          | Semi-auto           | You set rules, it adapts during training         |
| **Grid search LR**             | Fully manual        | You train 10 times with different LRs, pick best |
| **Neural Architecture Search** | Fully auto          | AI designs network + LR schedule                 |

| **Neural Architecture Search** | Fully auto          | AI designs network + LR schedule                 |
| ------------------------------ | ------------------- | ------------------------------------------------ |

We will try this interesting stuff as a bonus in this Star 1 function ! 

trained from scratch model - TL kooda inaikka mudiuyuma !?

**Why:** Early layers learn general features (edges, colors), don't need big changes. Late layers learn task-specific features, need big changes.

|Strategy|Shape|LR at 50% warmup|Best for|
|---|---|---|---|
|Linear|Straight line|50%|Default, most cases|
|Exponential|Curves up fast|75%|RNNs, LSTMs|
|Polynomial (α=2)|Curves up slow|25%|Large batch (>512)|
|Piecewise|Slow then fast|20%|Huge models (>1B params)|

why ?

Types of decays 

how these NN actually get trained to detect those feature only on that speciifc layers, why not vice versa and however it looks are gets trained 

wow factor - Discriminative Learining rates and Layer Specific Tuning to achieve maximum results
Neural Nets based Ensemble methods ! - each layer each enesemble each Neural netwoerks training to identify with all the parameters that we had actually learnt. ,How to make them actually work and what are the specific challenges they might actually face !


# **Distribution - shift  factors**

Learn about what is this and how this actually is

domain adaptation or retraining on harmonized multi-site data

multi-centre training and deployment.

- Treat distribution shift as a first-class problem and design a principled strategy for multi-centre training and deployment.
    
- Optimize for clinically meaningful metrics (timeliness of alerts, net benefit, calibration, alarm burden), not just AUROC.
    
- Embed explainability and uncertainty in forms that clinicians can actually use and empirically study their interaction.
    
- Follow and showcase best-practice reporting guidelines (for example, TRIPOD-AI, PROBAST) that many current papers neglec

 temperature scaling or isotonic regression

Fourier annealing and close cos annealing
3D 

Training each layer as a neural network, with DLR but comprising ALR  methods inside them 

Why fixed ratios is Discriminative LR 

What are those specific tasks in the middle layer 

mathematically modeling the functions and loss functions. Knowing the gradient and Gradient descent final values and then using the mathematical directly in order to actually find the Global minima and make the neural network escape the local minima 
	1. Fourier stuff (Just what I thought on the go)
	2. Maybe if Fourier overfits we need a more cooler generalization mathematical function 
	3. If at all fourier underfits (MandelBrot) situation, we can use the handling of infinite function and help model the neural network to predict the stuffs more properly and cool !

Using different navigation algorithms for fitting an neural network perfectly on the curve !