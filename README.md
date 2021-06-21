# Introduction

![gunshot](https://user-images.githubusercontent.com/60019124/122730222-928f3a00-d2ac-11eb-8be5-5056a007b914.png)

The Faster-Rcnn code provided above is used to tackle gunshot detection problem of Microsoft project 15. (see https://www.youtube.com/watch?v=0A4B2RIWb9o and https://r15hil.github.io/ICL-Project15-ELP/). The gunshot detection is solved by object detection method instead of any audio classification network method under certain condtions.

1. Limited number of forest gunshot dataset given with random labelled true location, which require generally 12s to involve single or multi gunshot in the extracted sound clip.
2. We do not have enough time for manually relabelling the audio dataset, as well as limited knowledge to identify gunshot from any other similar sounds in the forest.In some case, even gusnhot expert find diffculty to identify gunshot from those similar forest sound. e.g. tree branch.
3. Forest gunshot shoot in different angles, and may shoot from miles. Online gunshot dataset would not implement to this task directly.
4. Current director used in forest have only 0.5% precision. Detection need long manully post processing time. Hence, the given audio dataset would be limited and involve mistakes.

Hence, object detection provide faster labelling through mel-spectrum images, which are shown above. Mel spectrum also provide easy path to identify gunshot without specialty of gunshot audio sound. The Faster-Rcnn model provides high precision 95%-98% on gunshot detection. The detection bounding box can also be used to re-caculate both the start and end time of the gunshot. This reduce manully post processing time from 20 days to less than an hour. The more standard dataset also be generated for any audio classication alogorithm in the future developement.

# Set up

tensorflow 1.13-gpu (CUDA 10.0)\
keras 2.1.5\
h5py  2.10.0

The more detailed tutorial on setting up can be found on https://r15hil.github.io/ICL-Project15-ELP/gettingstarted.html.

#  Pre-trained mode

The lastest model is providel in .........
The lastest model is trained under multi-classes dataset including AK-47 gunshot, shotgun, monkey, thunder and hn, where hn represent hard negative. Hard negative are those high frequency mistake made by current detector, which are mostly peak without echo. Any sound have shape of the raindrop (just peak without echo) but can be appeared in different frequency domain with raindrop. The initial model is only train with gunshot and hn dataset due to limited given manual dataset. Then the network test under low threhold vaule, e.g.0.6 to obtain more false positive dataset or true gunshot among thoundsand hours forest sound file testing. 


![fp1](https://user-images.githubusercontent.com/60019124/122781713-ac4b7400-d2e2-11eb-8f8d-eba9c1367dc3.png)


![fp2](https://user-images.githubusercontent.com/60019124/122781894-d3a24100-d2e2-11eb-9d14-446aeb976629.png)

![gunshot1](https://user-images.githubusercontent.com/60019124/122781978-e583e400-d2e2-11eb-9d50-364dc6245e09.png)




 

