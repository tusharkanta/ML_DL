# EVA4 Assignment 15

[Github Directory - Assignment -15](https://github.com/tusharkanta/ML_DL/tree/eva/S15)

### Problem

- Assignment description is already shared in the last assignment:
  - Given an image with foreground objects and background image, predict the depth map as well as a mask for the foreground object. 
- It is an open problem and you can solve it any way you want. 
- Let's look at how it can be approached through some examples. 
- Assignment 14 (15A )was given to start preparing you for assignment 15th. 14th (15A) automatically becomes critical to work on the 15th. 
- **The 15th assignment is NOT a group assignment**. You are supposed to submit it along. 
- What happens when you copy? Well, WHAT happens when we copy? WHO knows!
- This assignment is worth 10,000 points. 
- Assignment 15th is THE qualifying assignment. 

#### Solution

​	**Strategy and steps for Data handling/Model Selection/Loss Function**

- Assignment has been executed on a Colab **Pro** GPU machine.

- Input Data sizes are mentioned in the following table:

  | S.No | Type                                  | Count       | Size      | Remarks                             |
  | ---- | ------------------------------------- | ----------- | --------- | ----------------------------------- |
  | 1    | Background (BG) - JPG                 | 100         | 224 x 224 | - Library(50) + Classroom(50).      |
  | 2    | Background - Foreground (BG_FG) - JPG | 200K        | 224 x 224 | Created as part of 15-A assignment. |
  | 3    | Mask                                  | 200k        | 224 x 224 | Ground Truth Image for Mask         |
  | 4    | Depth Map                             | 200k        | 224 x 224 | Ground Truth Image for Depth Map    |
  |      |                                       |             |           |                                     |
  |      |                                       |             |           |                                     |
  |      |                                       |             |           |                                     |
  |      |                                       |             |           |                                     |
  |      |                                       |             |           |                                     |
  |      | **Total No. of Images**               | **600,100** |           |                                     |

  

- 600k images are divided into 20 zip files  of 30k images each. 20 zip files are named from batch_1.zip to batch_20.zip. Each batch file contains the following folder structure (bg_fg_1, bg_fg_mask_1, depthMap). Each of these 3 folders contains 10k images. 

  ![image-20200525165656117](https://github.com/tusharkanta/ML_DL/blob/eva/S15/assets/image-20200525165656117.png)

- All 20 zip files were uploaded to google drive link: https://drive.google.com/drive/folders/1mz9kCU1J1E7400dEPEJb_1gPqPaFWXIb?usp=sharing

- 100 bg images were loaded into google drive same link as **bgimages.zip**

- All 600,100 images were uniquely named so as to prevent any overwriting during extraction of any number of zip files into colab

- autotime module has been used to record timings automatically for each execution block. This did help a lot in measuring the performance.

- After extraction of bgimages.zip and batch_<n>.zip files into colab the folder tree structure appears like:

  - ![image-20200525172734340](https://github.com/tusharkanta/ML_DL/blob/eva/S15/assets/image-20200525172734340.png)

- Before trying big batches of images, various small batches (100, 500, 1000 etc) were loaded to test the memory consumption and speed. Even though I have a colab pro account, i tried these combinations to learn the memory/speed etc

  **Custom TorchVision Data Loader Pipeline**

  

- Prepared a custom torchvision data set with input as bg_fg and bg images, ground truths as depthMap and mask images. Root dir is /content

- Data pipeline flow for bg image was based on splitting bg_fg image name to get the bg image name and then load the corresponding bg image from /content/bg folder

- Full data set was split into 70% and 30% for train and test respectively using the torch random_split technique

- Batch size choice was tricky. After experimenting with batch sizes ranging from 16 to 512, size 16 was chosen for optimised memory consumption. Higher batch size was resulting in CUDA out of the memory 

- Above data pipeline stratgey has been implemented in https://github.com/tusharkanta/ML_DL/blob/eva/S15/dataloader_s15.py file

- Training in the batches and saving the state dictionaries:

  - Trained with various combination of number of images to arrive at a combination of handling 50000 (bg_fg) images (5 zip files from batch_1 to batch_5 and so on) and corresponding bg images, GT images

  - Though I was able to handle 100,000 bg_fg images (and corresponding 200k GT images) simultaneously, it was creating some javascript output displaying issue. I decided to go for 50k bg_fg images for one training/testing run. After each run model state was saved and reloaded in next run

  - Given below is a  summary of training cycles:

    - | Run No | Total Size of bg_fg images | Train Size | Test Size | Batch Size | Epochs |
      | ------ | -------------------------- | ---------- | --------- | ---------- | ------ |
      | 1      | 50,000                     | 35,000     | 15,000    | 16         | 10     |
      | 2      | 50,000                     | 35,000     | 15,000    | 16         | 14     |
      | 3      | 50,000                     | 35,000     | 15,000    | 16         | 14     |
      | 4      | 30,000                     | 21,000     | 9,000     | 16         | 14     |
      | 5      | 20,000                     | 14,000     | 6,000     | 16         | 14     |

**Model Summary, train/test**

​		Various models were tried. U-Net or U-Net like models gave better results. I ran with both U-Net with a Resnet kind of skip connection option on full dataset. I also ran a classic U-net model on a 30k bg-fg, 100 bg, 30k GT images. Models which take into account a global scene detection along with granular features concatenation are good choices. 

**Model Parameters Count:**

Custom Model (depthmax): 

Total Parameters: 12,384,707

Trainable Parameters: 12,384,707



Classic U-Net model:

Total Parameters: 10,882,432

Trainable Parameters: 10,882,432



For model definition please refer to the following link in github:

https://github.com/tusharkanta/ML_DL/blob/eva/S15/model_def_s15_1.py



**Loss function**

BCE loss with logits (segmoid handled internally) is stable has been used for both image and text labels ground truth. For image pixel based tasks, SSIM and even Dice loss can be considered. I tried all 3 (BCE with logits), SSIM (with kernel size 3 and reduction method as mean) and Dice loss. BCE loss seems to be giving better result. SSIM loss produced blank depth image on U-Net and Dice Loss I could not configure as it was expecting int64 conversion of images which i tried but could not achieve.

**Training and Test**

Train was done on 70% of total datasets. While on custom depthmax model, total loss was fixed (2*mask loss + depth loss), I tried a different approach on U-Net model. On U-Net model I took total loss as only mask loss (ignored depth loss) for first few epochs and then took total as only depth loss (ignored mask loss) for subsequent epochs. This produced better result compared to a loss function having mixture of both. In test method, I saved the plot outputs in equal batch intervals in every epoch. Model state was stored in google drive for each run cycle. For test prediction accuracy, I calculated SSIM Index metrics from skimage structural similarity module. Please refer to output presentation below for the same.

For training and test please refer to the following link in github:

https://github.com/tusharkanta/ML_DL/blob/eva/S15/model_train_s15_6.py

For plotting and miscellaneous please refer to the following link:

https://github.com/tusharkanta/ML_DL/blob/eva/S15/utils.py

Training model and plot outputs are stored in google drive link:

Custom-model outputs

https://drive.google.com/drive/folders/1L9mOfceETJugJjY13khJJYuIViJs_McT?usp=sharing

https://drive.google.com/drive/folders/1VXUDcNkJvZIhRLDzFtjx1C0Btl6Y4G2Z?usp=sharing

U-Net ouput

https://drive.google.com/drive/folders/1ZZaJQESgw_2W5BaRgWHjZMcxu8FHswH3?usp=sharing

**Output Presentation**

**After 1st epoch of Run 1 (50k bg-fg, 100 fg , 50k mask, 50k depth), validation (test dataset) output. For this run taking bg_fg image plot was missed out.**

Ground Truth Mask:

![](https://github.com/tusharkanta/ML_DL/blob/eva/S15/assets/custom_model/50k/1stepoch/0_930_actual_mask.jpg)



Predicted Mask:

![](https://github.com/tusharkanta/ML_DL/blob/eva/S15/assets/custom_model/50k/1stepoch/0_930_predicted_mask.jpg)

skimage module's Structural Similarity index (SSIM) measure is a good way to calculate image similarities between 2 images and hence it is a good tool calculate accuracy of predicted image. SSM index calcuated using the tool: https://github.com/tusharkanta/ML_DL/blob/eva/S15/ssim.py

**SSIM index** for mask GT and prediction is **0.979**

![image-20200525183102575](https://github.com/tusharkanta/ML_DL/blob/eva/S15/assets/custom_model/50k/1stepoch/image-20200525183102575.png)

=====================================================

Ground Truth Depth:

![](https://github.com/tusharkanta/ML_DL/blob/eva/S15/assets/custom_model/50k/1stepoch/0_930_actual_depth.jpg)



Predicted Depth:

![](https://github.com/tusharkanta/ML_DL/blob/eva/S15/assets/custom_model/50k/1stepoch/0_930_predicted_depth.png)

**SSIM index** for depth GT and prediction is **0.611**

![image-20200525185332278](https://github.com/tusharkanta/ML_DL/blob/eva/S15/assets/image-20200525185332278.png)



**After last epoch of Run 2 (total100k bg-fg, 100 fg , 100k mask, 100k depth), validation (test dataset) output.** 

BG_FG Image:

![](https://github.com/tusharkanta/ML_DL/blob/eva/S15/assets/custom_model/100k/lastepoch/13_930_actual_bg_fg.jpg)

Ground Truth Mask:

![](https://github.com/tusharkanta/ML_DL/blob/eva/S15/assets/custom_model/100k/lastepoch/13_930_actual_mask.jpg)



Predicted Mask:

![](https://github.com/tusharkanta/ML_DL/blob/eva/S15/assets/custom_model/100k/lastepoch/13_930_predicted_mask.jpg)

**SSIM index** for mask GT and prediction is **0.98**

![image-20200525191331074](https://github.com/tusharkanta/ML_DL/blob/eva/S15/assets/custom_model/100k/lastepoch/image-20200525191331074.png)

Ground Truth Depth:

![](https://github.com/tusharkanta/ML_DL/blob/eva/S15/assets/custom_model/100k/lastepoch/13_930_actual_depth.jpg)

Predicted Depth:

![](https://github.com/tusharkanta/ML_DL/blob/eva/S15/assets/custom_model/100k/lastepoch/13_930_predicted_depth.png)

**SSIM index** for depth GT and prediction is **0.602**

![image-20200525191129185](https://github.com/tusharkanta/ML_DL/blob/eva/S15/assets/custom_model/100k/lastepoch/image-20200525191129185.png)



**After last epoch of last run (total 200k bg-fg, 100 fg , 200k mask, 200k depth), validation (test dataset) output.** 

Actual BG-FG image

![](https://github.com/tusharkanta/ML_DL/blob/eva/S15/assets/custom_model/200k/lastepoch/13_375_actual_bg_fg.jpg)



Ground Truth Mask

![](https://github.com/tusharkanta/ML_DL/blob/eva/S15/assets/custom_model/200k/lastepoch/13_375_actual_mask.jpg)

Predicted Mask

![](https://github.com/tusharkanta/ML_DL/blob/eva/S15/assets/custom_model/200k/lastepoch/13_375_predicted_mask.jpg)

**SSIM index** for mask GT and prediction is **0.953**

![image-20200525192603975](https://github.com/tusharkanta/ML_DL/blob/eva/S15/assets/custom_model/200k/lastepoch/image-20200525192603975.png)



Ground Truth Depth

![](https://github.com/tusharkanta/ML_DL/blob/eva/S15/assets/custom_model/200k/lastepoch/13_375_actual_depth.jpg)

Predicted Depth

![](https://github.com/tusharkanta/ML_DL/blob/eva/S15/assets/custom_model/200k/lastepoch/13_375_predicted_depth.png)

**SSIM index** for depth GT and prediction is **0.588**

![image-20200525192349382](https://github.com/tusharkanta/ML_DL/blob/eva/S15/assets/custom_model/200k/lastepoch/image-20200525192349382.png)



UNet Model Output: This was run for one run (6 epochs) with 30k BG-FG images, 100 BG images, 30k Mask and 30k Depth GT. This model was run with mask loss as the only criteria (depth loss was ignored) for the first 3 epochs and then depth loss as the only criteria (mask loss ignored) for the subsequent 3 epochs.

BG-FG image:

![](https://github.com/tusharkanta/ML_DL/blob/eva/S15/assets/classic_unet/2_448_actual_bg_fg.jpg)

Ground Truth Mask:

![](https://github.com/tusharkanta/ML_DL/blob/eva/S15/assets/classic_unet/2_560_actual_mask.jpg)

Predicted Mask:

![](https://github.com/tusharkanta/ML_DL/blob/eva/S15/assets/classic_unet/2_560_predicted_mask.jpg)

**SSIM index** for mask GT and prediction is **0.924**

![image-20200525194301348](https://github.com/tusharkanta/ML_DL/blob/eva/S15/assets/classic_unet/image-20200525194301348.png)

After 6 epochs 

BG-FG image

![](https://github.com/tusharkanta/ML_DL/blob/eva/S15/assets/classic_unet/5_560_actual_bg_fg.jpg)

Ground Truth Depth

![](https://github.com/tusharkanta/ML_DL/blob/eva/S15/assets/classic_unet/5_560_actual_depth.jpg)

Predicted Depth

![](https://github.com/tusharkanta/ML_DL/blob/eva/S15/assets/classic_unet/5_560_predicted_depth.jpg)

**SSIM index** for depth GT and prediction is **0.284**

![image-20200525194444522](https://github.com/tusharkanta/ML_DL/blob/eva/S15/assets/image-20200525194444522.png)

- **Submission**

  1. Share the link to the readme file for your Assignment 15A. Read the assignment again to make sure you do not miss any part which you need to explain. -2500
     
  - https://github.com/tusharkanta/ML_DL/blob/eva/S15/Readme.md
     
  2. Share the link to note book links
     
     - las run on custom depthmax model : https://github.com/tusharkanta/ML_DL/blob/eva/S15/S15_Final_Phase1_EVA_lastrun.ipynb
     - First run : https://github.com/tusharkanta/ML_DL/blob/eva/S15/S15_Final_Phase1_EVA.ipynb
     - 2nd run: https://github.com/tusharkanta/ML_DL/blob/eva/S15/S15_Final_Phase1_EVA_2.ipynb
     - 3rd run: https://github.com/tusharkanta/ML_DL/blob/eva/S15/S15_Final_Phase1_EVA_3.ipynb
     - 4th run: https://github.com/tusharkanta/ML_DL/blob/eva/S15/S15_Final_Phase1_EVA_4.ipynb
     
  3. U-Net run notebook

     ​     https://github.com/tusharkanta/ML_DL/blob/eva/S15/S15_Final_Phase1_EVA_unet.ipynb

  
