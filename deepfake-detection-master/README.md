# Deepfake Detection Tool
Open Source tool for Deepfake detection  
## Solution description
### Summary
The solution consists of three EfficientNet-B7 models, utilizing Noisy Student pre-trained weights. One model processes frame sequences, with a 3D convolution added to each EfficientNet-B7 block. The other two models operate on individual frames and differ in face crop size and training augmentations.

To address overfitting, the mixup technique was applied to aligned real-fake pairs. Additionally, the following augmentations were used: AutoAugment, Random Erasing, Random Crops, Random Flips, and various video compression parameters. Video compression augmentation was performed on-the-fly by saving short cropped tracks (50 frames each) in PNG format, which were then loaded and reencoded with random parameters using ffmpeg during each training iteration.

Due to the mixup strategy, the model's predictions were inherently uncertain. To enhance model confidence during inference, a simple transformation was applied. The final prediction was calculated by averaging the outputs of all models, weighted according to their confidence scores. The total training and preprocessing time was approximately 5 days on a DGX-1 system.
### Key ingredients
#### Mixup on aligned real-fake pairs
One of the main challenges was severe overfitting. Initially, all models began to overfit within 2–3 epochs, as indicated by an increase in validation loss. A key idea that significantly helped mitigate overfitting was training the model on a mix of real and fake faces: for each fake face, the corresponding real face from the original video—using the same bounding box coordinates and frame number—was selected, and a linear combination of the two was performed. In terms of tensor it’s  
```python
input_tensor = (1.0 - target) * real_input_tensor + target * fake_input_tensor
```
where target is drawn from a Beta distribution with parameters alpha=beta=0.5. With these parameters, there is a very 
high probability of picking values close to 0 or 1 (pure real or pure fake face). You can see the examples below:  
![mixup example](images/mixup_example.jpg "Mixup example")  
Due to the fact that real and fake samples are aligned, the background remains almost unchanged on interpolated samples, 
which reduces overfitting and makes the model pay more attention to the face.
#### Video compression augmentation
Augmentations resembling degradations commonly found in real-life video distributions were applied to the test data. Specifically, these included: (1) reducing the video's FPS to 15; (2) lowering the resolution to one-quarter of the original size; and (3) decreasing the overall encoding quality.

To make the model robust to various video compression parameters, augmentations with random video encoding settings were added during training. Applying such augmentations to the original videos on-the-fly during training would be infeasible. Therefore, instead of using the full videos, the approach involved short clips (50 frames each) cropped around the face area (1.5x the bounding box). Each clip was saved as individual PNG frames.. An example of a clip is given below:  
![clip example](images/clip_example.jpg "Clip example")  
 For on-the-fly augmentation, ffmpeg-python was used. At each iteration, the following parameters were randomly sampled 
 (see \[2\]):
- FPS (15 to 30)
- scale (0.25 to 1.0)
- CRF (17 to 40)
- random tuning option
#### Model architecture
As a result of the experiments, the EfficientNet models work better than others (we checked ResNet, 
ResNeXt, SE-ResNeXt). The best model was EfficientNet-B7 with Noisy Student pre-trained weights \[3\]. The size of the 
input image is 224x192 (most of the faces in the training dataset are smaller). The final ensemble consists of three 
models, two of which are frame-by-frame, and the third works on sequence.
##### Frame-by-frame models
Frame-by-frame models work quite well. They differ in the size of the area around the face and augmentations during 
training. Below are examples of input images for each of the models:  
![first and second model inputs](images/first_and_second_model_inputs.jpg "First and second model input examples")  
##### Sequence-based model
Probably, time dependencies can be useful for detecting fakes. Therefore, a 3d convolution was added to each block of the 
EfficientNet model. This model worked slightly better than similar frame-by-frame model. The length of the input 
sequence is 7 frames. The step between frames is 1/15 of a second. An example of an input sequence is given below:  
![third model input](images/third_model_input.jpg "Third model input example")  
#### Image augmentations
To improve model generalization, the following augmentations were used: AutoAugment \[4\], Random Erasing, Random Crops, 
Random Horizontal Flips. Since mixup was used, it was important to augment real-fake pairs the same way (see example). 
For a sequence-based model, it was important to augment frames that belong to the same clip in the same way.  
![augmented mixup](images/augmented_mixup.jpg "Augmented mixup example")
#### Inference post-processing
Due to mixup, the predictions of the models were uncertain, which was not optimal for the logloss. To increase 
confidence, the following transformation was applied:  
![prediction transform](images/pred_transform.jpg "Prediction transformation")  
Due to computational limitations, predictions are made on a subsample of frames. Half of the frames were horizontally 
flipped. The prediction for the video is obtained by averaging all the predictions with weights proportional to the 
confidence (the closer the prediction to 0.5, the lower its weight). Such averaging works like attention, because the 
model gives predictions close to 0.5 on poor quality frames (profile faces, blur, etc.). 
#### References 
\[1\] [https://trac.ffmpeg.org/wiki/Encode/H.264](https://trac.ffmpeg.org/wiki/Encode/H.264)  
\[2\] Qizhe Xie, Minh-Thang Luong, Eduard Hovy, Quoc V. Le, “Self-training with Noisy Student improves ImageNet classification”  
\[3\] Ekin D. Cubuk, Barret Zoph, Dandelion Mane, Vijay Vasudevan, Quoc V. Le, “AutoAugment: Learning Augmentation Policies from Data”
## The hardware used
- CPU: Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz
- GPU: 8x NVIDIA Tesla V100 SXM2 32 GB
- RAM: 512 GB
- SSD: 6 TB
