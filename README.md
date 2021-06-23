# Introduction

![gunshot](https://user-images.githubusercontent.com/60019124/122730222-928f3a00-d2ac-11eb-8be5-5056a007b914.png)
             Figure1. Gunshot and some example sounds that cause errors for the current detectors 

The Faster-Rcnn code provided above is used to tackle gunshot detection problem of Microsoft project 15. (see https://www.youtube.com/watch?v=0A4B2RIWb9o and https://r15hil.github.io/ICL-Project15-ELP/). The gunshot detection is solved by object detection method instead of any audio classification network method under certain condtions. 

1. Limited number of forest gunshot dataset given with random labelled true location, which require generally 12s to involve single or multi gunshot in the extracted sound clip. Random label means that the true location provided is not the extact location of the start of the gunshot. Gunshot may happen after the given true location in the range of one to twelve 12 second. The number of gunshot after true location also ranging for single gunshots over to 30 gunshots.
2. We do not have enough time for manually relabelling the audio dataset, as well as limited knowledge to identify gunshot from any other similar sounds in the forest.In some case, even gusnhot expert find diffculty to identify gunshot from those similar forest sound. e.g. tree branch.
3. Forest gunshot shoot in different angles, and may shoot from miles. Online gunshot dataset would not implement to this task directly.
4. Current director used in forest have only 0.5% precision.It will peak up sound like raindrop and thunder shown in Figure1. Detection need long manully post processing time. Hence, the given audio dataset would be limited and involve mistakes. 

The above conditions challenge audio classification method like MFCC with CNN. Hence, ojection detection method provide an easier approch under limited time. The object detection provide faster labelling through mel-spectrum images, which are shown above. Mel spectrum also provide easy path to identify gunshot without specialty of gunshot audio sound. The Faster-Rcnn model provides high precision 95%-98% on gunshot detection. The detection bounding box can also be used to re-caculate both the start and end time of the gunshot. This reduce manully post processing time from 20 days to less than an hour. The more standard dataset also be generated for any audio classication alogorithm in the future developement.

# Set up

tensorflow 1.13-gpu (CUDA 10.0)\
keras 2.1.5\
h5py  2.10.0
model: retrain.hdf5 can be download from https://drive.google.com/drive/folders/1AswdCXlv3cxjgTwge2dIbkBxJv8ztLMF?usp=sharing

The more detailed tutorial on setting up can be found on https://r15hil.github.io/ICL-Project15-ELP/gettingstarted.html.

#  Pre-trained model

The lastest model is trained under multi-classes dataset including AK-47 gunshot, shotgun, monkey, thunder and hn, where hn represent hard negative. Hard negative are those high frequency mistake made by current detector, which are mostly peak without echo. Any sound have shape of the raindrop (just peak without echo) but can be appeared in different frequency domain with raindrop. The initial model is only train with gunshot and hn dataset due to limited given manual dataset. Then the network test under low threhold vaule, e.g.0.6 to obtain more false positive dataset or true gunshot among thoundsand hours forest sound file testing. The model will then use those hard negatives to re-train. This technic also call hard mining. The following images Figure.2 show three hard negative dataset (monkey, thunder, hn) that used to train the lastest model.  

![fp2](https://user-images.githubusercontent.com/60019124/122781894-d3a24100-d2e2-11eb-9d14-446aeb976629.png)
![fp3](https://user-images.githubusercontent.com/60019124/122927325-5a1a5980-d39b-11eb-9824-42e5858053af.png)
Figure2. three type of hard negative enter into training 

The model also provide classification between different gunshot which are AK-47 and shotgun shown in Figure3.
![gunshot1](https://user-images.githubusercontent.com/60019124/122781978-e583e400-d2e2-11eb-9d50-364dc6245e09.png)
Figure3. Mel-spectrum image of AK-47 and shotgun


# result

The following testing result shown in Table1 is based on the experiment on over 400 hours forest sound clips given by our clients. The more detailed information on other models except Faster-Rcnn can be found in https://r15hil.github.io/ICL-Project15-ELP/.  

| Model name                   | current template detector | custom AI(Microsoft) | YOLOV4 | Faster-Rcnn |
|------------------------------|---------------------------|----------------------|--------|-------------|
| Gunshot dectection precision | 0.5%                      | 60%                  | 85%    | 95%         |

Table1. testing result from different models

The model also generate csv file shown in Figure4.It include the file name, label, offset, start time and end time. The offset is the left end axis of the mel-spectrum, where each mel-spectrum is coducted from 12s sound clips. The start time and end time can then caculated using the offset and the detection bounding coordinate. Therefore, this csv file would provide more standard dataset, where true location will more likely sit just before the gunshot happen. Each cutting sound clip are likely to include single gunshot. Therefore, this dataset would involve less noise and wrong labelling. It will benefit for future development on any audio classification method. 
![image](https://user-images.githubusercontent.com/60019124/122941913-635df300-d3a8-11eb-994f-3e9db29deb4c.png)

Figure4. example of csv file for gunshot labelling  

# current model weakness and Future developement

The number of hard negative dataset are still limited after 400 hours sound clips testing. Hard negative like bee and strong thunder sound(shown in Figure5) are the main predicted errors for the current model. In order to improve the model further, it should finding more hard negative using lower threhold of the network through thousand hours sound clip testing. Then, the obtained hard negative dataset can be use to re-train and improve the model. However, the current model can avoid those false postive by setting around 0.98 threhold valu, but it will reduce the recall rate by 5-10%. The missing gunshot mainly on long distance gunshot. This is also cause by very limted long distance gunshot provided in the training dataset.

![mistake](https://user-images.githubusercontent.com/60019124/122949470-44625f80-d3ae-11eb-9d19-d312a7183f0f.png)

Figure5. Mel-spectrum image of strong thunder and bee

# Reference

https://github.com/yhenon/keras-rcnn





 

