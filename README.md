# PAC-Chernoff Bounds: Understanding Generalization in the Interpolation Regime

## Abstract

This paper introduces a distribution-dependent PAC-Chernoff bound that exhibits perfect tightness for interpolators, even within over-parameterized model classes. This bound, which relies on basic principles of Large Deviation Theory, defines a natural measure of the smoothness of a model, characterized by simple real-valued functions. Building upon this bound and the new concept of smoothness, we present an unified theoretical framework revealing why certain interpolators show an exceptional generalization, while others falter. We theoretically show how a wide spectrum of modern learning methodologies, encompassing techniques such as $\ell_2$-norm, distance-from-initialization and input-gradient regularization, in combination with data augmentation, invariant architectures, and over-parameterization, collectively guide the optimizer toward smoother interpolators, which, according to our theoretical framework, are the ones exhibiting superior generalization performance. This study shows that distribution-dependent bounds serve as a powerful tool to understand the complex dynamics behind the generalization capabilities of over-parameterized interpolators.

## Experiments

Regarding the experimental setting of this work, we have mainly used a small InceptionV3 (Szegedy et al.,
2016) used in Zhang et al. (2017), and Cifar10 dataset (Krizhevsky et al., 2009). Before
showing the specific specific architecture of our small Inception, taken from Zhang et al.
(2017), we need to detail some of the modules that compose it:

1. Convolutional module: Convolutional layer, batch-normalization and ReLU activation.
2. Inception module with output channels $o_{1\times 1}$ and $o_{3\times 3}$: Consists on 2 different convolutional layers, one with kernel $1 \times 1$ and $o_{1\times 1}$ output channels and another with kernel $3 \times 3$ and $o_{3\times 3}$ output channels. The output of this layers is then concatenated, so the total number of output channels is $o_{1\times 1} + o_{3\times 3}$.
3. Downsample module: Convolutional module with kernel size $3$, stride $2$ and padding $0$ and MaxPooling with kernel size of $3$ and stride $2$. The outputs of these two layers is concatenated.

With these elements, the architecture of our small InceptionV3 network is
1. Convolutional module with $96$ output channels, kernel size $3$, stride $1$ and padding $0$.
2. Inception Module with $o_{1 \times 1} = 32$ and $o_{3 \times 3} = 32$.
3. Inception Module with $o_{1 \times 1} = 32$ and $o_{3 \times 3} = 48$.
4. DownSample Module with $o_{3 \times 3} = 80$.
5. Inception Module with $o_{1 \times 1} = 112$ and $o_{3 \times 3} = 48$.
6. Inception Module with $o_{1 \times 1} = 96$ and $o_{3 \times 3} = 64$.
7. Inception Module with $o_{1 \times 1} = 80$ and $o_{3 \times 3} = 80$.
8. Inception Module with $o_{1 \times 1} = 48$ and $o_{3 \times 3} = 96$.
9. DownSample Module with $o_{3 \times 3} = 96$.
10. Inception Module with $o_{1 \times 1} = 176$ and $o_{3 \times 3} = 160$.
11. Inception Module with $o_{1 \times 1} = 176$ and $o_{3 \times 3} = 160$.
12. Adaptative Average Pooling layer with kernel $7 \times 7$.
13. Fully connected layer from $16464$ to the number of classes (i.e, $10$).
        
Where the total number of parameters of this model is $1.814.106$.    
    
**Figure 2.** For this experiments, all Inception models where trained using SGD with momentum $0.9$ and learning rate $0.01$ with exponential decay of $0.95$. All models are trained for $30.000$ iterations of batches of size $200$ or until the train loss is under $0.005$. These settings are selected to ensure that the random label model converges to an interpolator. Random cropping is employed using \texttt{RandomResizeCrop} function of \texttt{torchvision} with scale $(0.8, 1.0)$ and ratio $(0.9, 1.1)$. For $\mathbb{E}ll_2$ Regularization, we multiplicative factor is $0.01$.
    
**Figure 3.** For this figure, Standard, L2-Crop and Initial model from Figure~\ref{fig:1} are used. Subsets of size $n=50$ of CIFAR10's test split are used to approximate samples of the data generating distribution and build the histograms.
    
**Figure 6.** The batch size is fixed to $250$ and images are standardize (this was necessary to improve learning in the MLP model). The precise MLP has 3 hidden layers with $512$ units, with a total of $1.735.178$ parameters. All models are trained until the interpolation regime, that is, until the train loss is under $0.015$, which, in the worst case where $20.000$ iterations for the MLP. Inception models are trained using a learning rate of $0.001$ whereas MLP models use $0.1$, both with $0.9$ momentum and $0.95$ exponential decay. Regarding the data, $D_0$ is CIFAR10's test set, $D_1$ is the result of performing random translations of $5\%$ and $D_2$ considers random translations of $5\%$ and rotations of up-to $20\%$. Both transformations are computed using `RandomAffine` function of `
torchvision`.
    
    
**Figure 7.** The model's specification and training setup is the same as in Figure 6. Regarding the data, the random shuffling of the pixels was performed using a random permutation using `Numpy`; the dataset was fully permuted and stores as a new dataset.
    
**Figure 8.** The model's specification and training setup is the same as in Figure 6. $D_1$ considers random translations of $5\%$ and rotations of up-to $20\%$ (the same as $D_2$ in Figure 6). Data-augmentation was produced using the same transformations as those that define $D_1$.
    
**Figure 9.** For this figure, we used a generalization of LeNet5 (three convolutional layers a two fully connected with ReLu activation and average pooling), where the number of channels of the convolutional layers was parameterized by $k$. Precisely, the first layer had $3$ input and $\lfloor 6k \rceil$ output channels; the second layer $\lfloor 6k \rceil$ input and $\lfloor 16k \rceil$ output channels; and the last layer $\lfloor 16k \rceil$ input and $\lfloor 120k \rceil$ output channels. The set of models are created ranging $k$ from $0.2$ to $4.9$ every $0.1$; raising models from $7k$ parameters to models with $1.2M$ parameters. Each of this models is then trained until the train loss is lower than $0.01$ or until the train loss has not lowered in two epochs (this only happens in the smallest models).
    
**Figure 10.** The rate function of a subset of all the models in Figure 9 is computed here.
    


## Estimating the cumulant function and the rate function.
    
From the definition of the cummulant function \(\J\),
$$
    J_{\bm \theta}(\lambda) = \ln \mathbb{E}_{\nu}\left[e^{\lambda (L(\bm \theta)-\mathbb{E}ll(\bm y,\bm x,\bm\theta))}\right] = \ln \mathbb{E}_{\nu}\left[ p(\bm y|\bm x,\bm \theta)^\lambda\right]  - \mathbb{E}_{\nu}[\ln p(\bm y|\bm x, \bm \theta)^\lambda]\,,
$$
it is clear that computing its true value requires access to the true data generation distribution $\nu$. However, in real-world problems, this distribution is unknown and innacesible. 

The Machine Learning community is used to approximate this kind of quantities (such as the expected loss $L(\bm \theta)$) using separate validation datasets $D^{val}$. In fact, due to the large amount of data available in nowadays problems, using this approach is perfectly doable, leading to
$$
    J_{\bm \theta}(\lambda) \approx \ln\left(\frac{1}{M}\sum_{(\bm x, \bm y) \in D^{val}} p(\bm y|\bm x, \bm \theta)^\lambda \right) - \frac{1}{M}\sum_{(\bm x, \bm y) \in D^{val}}\ln p(\bm y|\bm x, \bm \theta)^\lambda\,.
$$
It is important to notice that the above estimator is biased  due to the first term and Jensen's Inequality. In fact,
$$
    \mathbb{E}_{D^{val}}\left[\ln\left(\frac{1}{M}\sum_{(\bm x, \bm y) \in D^{val}} p(\bm y|\bm x, \bm \theta)^\lambda \right)\right] \leq \ln\left( \mathbb{E}_{D^{val}}  \left[\frac{1}{M}\sum_{(\bm x, \bm y) \in D^{val}} p(\bm y|\bm x, \bm \theta)^\lambda \right]\right) = \ln\mathbb{E}_{\nu}\left[ p(\bm y|\bm x,\bm \theta)^\lambda\right]\,.
$$
As a result, if the size of $D^{val}$ is not large enough, we might end understimating the cummulant function. 

In regard of computational stability, computing this estimation can be computationally unstable due to the use of probabilities. We encourage the use of log-probabilities and log-sum-exp operations as,
$$
    J_{\bm \theta}(\lambda) \approx \ln\left(\sum_{(\bm x, \bm y) \in D^{val}} \mathbb{E}xp (\lambda \ln p(\bm y|\bm x, \bm \theta)) \right) - \ln M - \frac{1}{M}\sum_{(\bm x, \bm y) \in D^{val}}\lambda \ln p(\bm y|\bm x, \bm \theta)\,.
$$
From this, it is straight-forward to compute the log-probabilities of the model (for example skipping the softmax layer of a NN), multiply them by $\lambda$ and compute the mean and log-sum-exp of these quantities.


Once the cummulant function has being approximated, computing the rate function relies on computing the optimal value of $\lambda$,
$$
    \mathcal{I}_{\bm \theta}(a) = \sup_{\lambda > 0} \lambda a - J_{\bm \theta}(\lambda)\,.
$$
In this regard, trying to optimize the value of $\lambda$ doing automatic optimization resulted in a very unstable method in our experiments. Thus, the method we are recommending and the one we have employed is using a *binary search* algorithm. Fixed a range in which to optimize lambda $[\lambda_{min}, \lambda_{max}]$, a binary search algorithm has complexity $\mathcal{O}(\log_{2}(\lambda_{max} - \lambda_{min}))$. In fact, if (due to the nature of the problem) the needed value of $\lambda_{max}$ is too large, one might perform the binary search in $[\ln(\lambda_{min}), \ln(\lambda_{max})]$, which has the same complexity but makes it easier to consider larger values of $\lambda$.
