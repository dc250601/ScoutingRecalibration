# Ultra-Low-Latency FPGA-Accelerated Neural Network Inference at 40 MHz at CMS

# Introduction
The Large Hadron Collider (LHC) is one of the most sophisticated experiments ever designed. Its primary purpose is to probe the internal structures of matter and study the underlying physics and interactions of the particles that make up our universe. Particles collide at LHC at a rate of 40 MHz, generating vast amounts of data every second. The collisions are recorded at 4 four interaction points where a sophisticated detector is located. The Compact Muon Solenoid (CMS) Experiment is one such detector.CMS generates data at a theoretical throughput of $\mathcal{O}(40)$ TB/s. It is not possible to store or read out the full event record at such a rate; therefore, a triggering system is required to select interesting events from these vast amounts of data. At CMS, we have a two-level triggering system:


**Level-1 Trigger** The first triggering stage reduced the event rate from 40 MHz to a maximum of 110 kHz, considering the data rate and timing constraints, as well as the hardware constraints of FPGAs. The L1 Trigger can only run a course reconstruction of the incoming physics events, given that it receives only a fraction of the total event-level information from the detector and heavily depends on cut-based algorithms that can introduce significant physics biases within the trigger system.

**High-Level-Trigger** The second stage of the triggering system (HLT) further reduces the event rate from 100 kHz to around 1 kHz, at which point the data can be permanently stored. The HLT receives complete information from the entire detector, enabling a comprehensive physics reconstruction.

Traditionally, physics-based algorithms were used in the triggering system. However, given the massive success of machine learning (ML) and deep learning, it is essential to investigate whether an ML-based algorithm can enhance the triggering system. Since most of the triggering system is based on FPGA-based computational hardware, computational footprint and timing constraints should also be taken into account. This work demonstrates how ML-based triggering algorithms can be designed that are both computationally lighter and have superior performance compared to traditional physics-based algorithms on tasks such as online recalibration of physics objects and classification of dimuon pairs.

---
# Metrics and Definitions
Various metrics are used to compare the performance (in terms of computational cost) and resolution (in terms of prediction accuracy) of different models. The following metrics are defined that will be used extensively later on.

Hardware Values: Hardware values are the data received from the detector and used by the L1-Trigger to make decisions.
Reconstructed Values: Reconstructed values are the corrected values that are calculated after a full physics reconstruction. Reconstructed values are not available on the L1-Trigger.

**Regression**

For regression tasks, the metric employed is the Full-Width-Half-Maximum (FWHM), which reads:

$$\text{FWHM}_Y = \text{FWHM}(\hat{y_i}-y_i)$$
where $\hat{y_i}$ is the predicted value while $y_i$ being the groud truth.

**Classification**
For classification tasks, the Area Under the Receiver Operating Characteristic curve (AUC-ROC) score is chosen as a metric.

$$
\mathrm{AUC} = \int_{0}^{1} \mathrm{TPR}(\mathrm{FPR}) \, d(\mathrm{FPR})
$$

Where:

- $\mathrm{TPR} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FN}}$ is the True Positive Rate  
- $\mathrm{FPR} = \frac{\mathrm{FP}}{\mathrm{FP} + \mathrm{TN}}$ is the False Positive Rate

**Computational and Hardware Cost**

To measure the computational cost of our model on the FPGAs, we will use the following metrics:

**Digital Signal Processors (DSP):** DSP chips are used for processing the relevant signals coming to the FPGA board, particularly additions and multiplications.
**Flip-Flops (FF):** These units help synchronize logic and save logical states between clock cycles within an FPGA circuit.
**Look-up-Tables (LUT):** These units help in implementing arbitrary Boolean logics inside the FPGA.
**textbf{Block RAM (BRAM):** a type of random access memory implemented inside the FPGA for data storage.
    
The number of these resources available is minimal. Hence, models with the least amount of resource consumption are given a higher priority.

---
# $\mu\text{GMT}$ muons recalibration
Recalibrating of $\mu$, $\phi$, and $p_\mathrm{T}$ using the L1 parameters $\mu$, $\phi$, $p_\mathrm{T}$, muon charge, and reconstruction quality is one of the primary focuses of this study. Offline reconstruction data is used as ground truth for training the networks. Knowledge distillation, along with Quantisation-Aware-Training (QAT), is used to achieve a high level of reconstruction resolution by utilizing only a fraction of the resources on the FPGA boards.

## Teacher Model
A four-layer neural network with 256 neurons in each layer is chosen as the teacher model. ReLU non-linearity is used in conjunction with batch normalization after every subsequent layer. The output layer consists of three parameters that predict the recalibrated $\mu$, $\phi$, and $p_\mathrm{T}$. QKeras is used to train the models with QAT precision fixed at [18,6]. Fixed precision notation is used throughout the work, where the leftmost digit signifies the total number of bits used to store the number, while the rightmost digit represents the number of bits used to store the fractional part. The model is trained for 100 epochs using LogCosh loss, and the learning rate is reduced after the model performance plateaus. The Adam optimiser is used with an initial learning rate of $10^{-4}$.

## Student Model
A four-layer neural network with eight neurons in each layer is chosen as the student model. ReLU non-linearity is used in conjunction with batch normalization after every subsequent layer. The output layer consists of three parameters that predict the recalibrated $\mu$, $\phi$ and $p_\mathrm{T}$. QKeras is used to train the model with QAT precision fixed at [18,6]. The model is trained for 500 epochs with LogCosh as the task-agnostic loss and mean square error (MSE) as the distillation loss. The distillation parameter $\alpha$ is set to 0.1, and the learning rate is reduced once the model's performance plateaus. Adam optimiser is used with an initial learning rate of $10^{-2}$.

### Baseline Model
To compare the effectiveness of the proposed method, the above models are evaluated against existing baselines in terms of both reconstruction resolution and computational requirements. The reported model from [1](https://cds.cern.ch/record/2834199) is used as the baseline for this study. The model is a four-hidden-layer neural network with 16 neurons in each layer trained with a similar loss and schedule as the teacher model.



### Table: Comparison of FWHM Across Models

| **FWHM** | **Teacher** | **Student** | **Baseline** |
|:--------:|:------------:|:------------:|:-------------:|
| **$\phi$** | 0.120 | 0.1240 | 0.140 |
| **$\eta$** | 0.061 | 0.063 | 0.0662 |
| **$p_\mathrm{T}$** | 0.413 | 0.417 | 0.433 |

The table above shows the performance of the three models across the three parameters $\phi$, $\eta$, and $p_\mathrm{T}$.  
Although the student model is approximately 1024 times smaller than the teacher, it performs almost as well, outperforming the baseline by a large margin despite being four times lighter.

| **VU9P FPGA**   | **DSP**         | **Flip Flops**     | **Look Up Table** | **BRAMs** |
|----------------|-----------------|--------------------|-------------------|-----------|
| **Available**  | 9024            | 2.6 M              | 1.3 M             | 2160      |
| **Student**    | 72 (0.79%)      | 5677 (0.21%)       | 11.3 K (0.87%)    | 0         |
| **Baseline**   | 238 (2.61%)     | 11.6 K (0.43%)     | 33.2 K (3.38%)    | 0         |

The table above shows the usage of various units of a VU9P FPGA when each model is run on it. The DSP reuse factor is set to 4 for all the experiments. Clearly, the student is much lighter compared to the pre-existing baseline.

## Analysing the performance of the student model
Although the student model outperforms the baseline across all metrics, a deeper understanding of its performance is still required before moving forward. This section examines the recalibration model (student) in detail. The resolution peaks for each kinematic quantity are plotted in the figure below. From the plot, it can be inferred that the model performs well against $\mu\text{GMT}$ across all metrics.


<div style="justify-content: center; gap: 0px;">
  <img src="https://codimd.web.cern.ch/uploads/upload_5421e71cf9fb89d8ae1d81b23dfa7730.png" width="30%"/>
  <img src="https://codimd.web.cern.ch/uploads/upload_c73404ff8748798fa8f03355f0ada84d.png" width="30%"/>
  <img src="https://codimd.web.cern.ch/uploads/upload_408c54468773fd6a81552fb8781b9ab0.png" width="30%"/>
</div>

In the above plot, $\mu\text{GMT}$ represents the default parameter values before recalibration.


Next, the resolution of recalibrated $p_\mathrm{T}$ vs the reconstructed $p_\mathrm{T}$ is plotted in a figure, which shows that the model performs well for low $p_\mathrm{T}$ ranges, but the resolution worsens towards higher $p_\mathrm{T}$ zones.
![](https://codimd.web.cern.ch/uploads/upload_13d80612803706b9b40b2eaadc207f80.png)

Plotting the resolution of recalibrated $\eta$ vs. the reconstructed $\eta$, we do not see a similar trend.
![](https://codimd.web.cern.ch/uploads/upload_f93f1dcebdc10729e5ea188fbe1be363.png)

The reason behind the poor recalibration performance of the ML model at the high $p_T$ region is linked to the following reasons:
1. Presence of exponentially fewer samples in the high $p_T$ region, given that it corresponds to high-energy collisions that occur less frequently.
2. High $p_T$ samples bend significantly less in the magnetic field present within the CMS detector, making their momentum harder to estimate within the muon chamber, making them more error-prone.

### Fixing low resolution at high $p_{\text{T}}$

Although the model performs very well for low to moderate $p_\mathrm{T}$ values, it fails to outperform the $\mu$GMT in the high $p_\mathrm{T}$ range. There are several reasons for the model's failure, which have been discussed in the previous section. In this section, attempts are made to fix this problem. Two primary techniques are proposed and experimented with as described in order to solve the discrepancy:-

-  Loss Scaling: The loss is scaled to incentivise the model to work better for the high $p_\mathrm{T}$ region. The loss is scaled to the inverse of the number of samples in the dataset, in bins of size 1, with similar $p_\mathrm{T}$ values during the distillation step.
-  Two Models: Since the model starts to perform worse for high reco $p_\mathrm{T}$, intuitively, one might think that using two models, one for the low reconstructed $p_\mathrm{T}$ region and the other for the reconstructed high $p_\mathrm{T}$ region, should be sufficient to solve the problem. Hence, a two-model approach is also chosen and experimented with.

### Loss Scaling
Plotting the resolution vs. the reco $p_\mathrm{T}$ for $\mu$GMT, the model with custom loss scaling and the model (Teacher) with no loss scaling yields the figure below.
![](https://codimd.web.cern.ch/uploads/upload_ef059695d1f3d936474d9c2cda89e7e9.png)
The scaling does not significantly affect the performance of muons in the low $p_\mathrm{T}$ zone. Although the scaling is orders of magnitude smaller, the performance in the high $p_\mathrm{T}$ zone further worsens when compared to the non-scaled model. The nature of the scaling factor is given below:
![](https://codimd.web.cern.ch/uploads/upload_e4b0bbb3d3063fd066a6394519dd4ed8.png)


### Two-Model Approach
Next, the two models approach is explored. The figure below shows the resolution vs reco $p_\mathrm{T}$ where a two-model setup is used during the inference. Although this setup solves the problem of low resolution in the high $p_\mathrm{T}$ region, it comes at a cost of degraded performance in the low $p_\mathrm{T}$ region.
![](https://codimd.web.cern.ch/uploads/upload_69bbac397b719e205972df4bcb484624.png)

This is not because the model lacks learning capabilities. The low resolution in the high $p_\mathrm{T}$ region is only seen in the reco $p_\mathrm{T}$ and not in the hardware $p_\mathrm{T}$ plot below; since only the hardware $p_\mathrm{T}$ is visible to the model during evaluation and training, this method fails miserably.
![](https://codimd.web.cern.ch/uploads/upload_fb80921c071ba38319be924802d12c83.png)


# $\mu$GMT di-muon classification
Next, the interest lies in predicting whether a di-muon pair can be identified from a chosen pair of muons using the L1 parameters $\mu$, $\phi$, $p_\mathrm{T}$, muon charge, and reconstruction quality for both the muons. Offline reconstruction data is used as ground truth for training the networks. Knowledge distillation, combined with QAT, is used to achieve a high level of classification performance by utilizing only a fraction of the resources on the FPGA boards.

### Teacher Model
A four-layer neural network with 256 neurons in each layer is chosen as the teacher model. ReLU non-linearity is used in conjunction with batch normalization after every subsequent layer. The output layer consists of a single neuron with linear/no activation. QKeras is used to train the model with QAT precision fixed at [18,6]. The model is trained for 100 epochs using BinaryCrossEntropy (with logits) loss, and the learning rate is reduced after the model performance plateaus. Adam optimiser is used with an initial learning rate of $10^{-3}$.

### Student Model
A four-layer neural network with eight neurons in each layer is used as a student model. ReLU non-linearity is used in conjunction with batch normalization after every subsequent layer. The output layer consists of a single neuron with linear/no activation. QKeras is used to train our model with QAT precision fixed at [18,6]. The model is trained for 500 epochs using BinaryCrossEntropy (with logits) as the task-agnostic loss and KLDivergence as the distillation loss. The distillation parameter $\alpha$ is fixed at 0.1, the temperature is set to 0.1, and the learning rate is reduced once the model's performance plateaus. Adam optimiser is used with an initial learning rate of $10^{-2}$.

### Baseline Model
A model identical to the student model is trained just using BCE Loss (without distillation) as a control. All other parameters are kept identical to those of the student model trained in the previous step.

### **Model Performance Comparison**

| **Metric**     | **Teacher** | **Student** | **Student (Non‑Distilled)** |
|:--------------:|:-----------:|:-----------:|:---------------------------:|
| **AUC‑ROC**    | 0.970       | 0.964       | 0.946                       |
| **AUC‑PR**     | 0.887       | 0.868       | 0.850                       |

The table above shows the performance of the three models. Although the student model is much smaller than the teacher, it performs almost as well while being 1024 times lighter. It also outperforms the non‑distilled version, demonstrating the advantage of knowledge distillation.


### **FPGA Resource Utilization (VU9P)**

| **VU9P FPGA** | **DSP**       | **Flip Flops**     | **Look Up Tables** | **BRAMs** |
|:-------------:|:-------------:|:------------------:|:------------------:|:---------:|
| **Available** | 9024          | 2.6 M              | 1.3 M              | 2160      |
| **Student**   | 78 (0.86 %)   | 6110 (0.23 %)      | 12.2 K (0.94 %)    | 0         |

The table above shows the usage of various units of a VU9P FPGA when each model is deployed.  
The DSP reuse factor is set to 4 for all experiments. The student model is clearly far lighter while maintaining competitive performance.

The PR and ROC curves are plotted in the figure below, where it can be seen that, for both metrics, the distilled model outperforms the non-distilled one by a considerable margin. Moreover, the information loss during the transfer of knowledge from the teacher to the student appears minimal, as the performance difference between the teacher and the distilled student is small.

<div style="display: flex; justify-content: space-between;">
  <img src="https://codimd.web.cern.ch/uploads/upload_3c5208f8e7eddb65773923b93896fd90.png" alt="Image 1" width="48%" />
  <img src="https://codimd.web.cern.ch/uploads/upload_dadfb739fbe91b436ca61faeab230d0f.png" alt="Image 2" width="48%" />
</div>

# Conclusion

In this work, various compression strategies have been explored to implement a lightweight neural network capable of delivering state-of-the-art performance, deployable on FPGA hardware systems in the CMS Level-1 trigger. A technique has been developed employing strategic combinations of quantization-aware training, knowledge distillation, and transfer learning to create a computationally efficient model with negligible loss in performance. The technique is tested on both classification and regression tasks, including $\mu$GMT muon recalibration and $\mu$GMT di-muon classification. The model outperformed the baseline significantly in the muon recalibration task, even after reducing the computational footprint fourfold. However, the model encountered challenges in the high reconstructed $p_\mathrm{T}$ region. Although an increase in the number of samples in the high reconstructed $p_\mathrm{T}$ might help to balance the dataset, it does not mitigate the exponential increase in error in the data itself. This study did not identify a concrete solution to address this issue; however, the analysis suggests that the primary reason behind this phenomenon is the exponentially increasing error in the hardware $p_\mathrm{T}$ compared to the reconstructed $p_\mathrm{T}$. The proposed technique worked well for the di-muon pair classification task. This study demonstrates that neural networks offer competitive performance as triggers; these networks can be made extremely lightweight and completely unbiased from physical assumptions, making them suitable candidates for detecting new physics.

# References
1. https://cds.cern.ch/record/2834199
