# microsoft-gunshot-faster-rcnn

![gunshot](https://user-images.githubusercontent.com/60019124/122730222-928f3a00-d2ac-11eb-8be5-5056a007b914.png)

The Faster-Rcnn code provided upabove is used to tackle problem of Microsoft project 15. (see https://www.youtube.com/watch?v=0A4B2RIWb9o). The gunshot detection is solved by 
object detection method instead of any audio classification network method under certain condtions.

1. Limited number of forest gunshot dataset given with random labelled true location, which require generally 12s to involve single or multi gunshot in the extracted sound clip.
2. We do not have enough time for manually relabelling the audio dataset, as well as limited knowledge to identify gunshot from any other similar sounds in the forest.In some case, even gusnhot expert find diffculty to identify gunshot from those similar forest sound. e.g. tree branch.
3. Forest gunshot shoot in different angles, and may shoot from miles. Online gunshot dataset would not implement to this task directly.
4. Current director used in forest have only 0.5% precision. Detection need long manully post processing time. Hence, the given audio dataset would involve mistakes.

Hence, object detection provide faster labelling through mel-spectrum images, which are shown above. Mel spectrum also provide easy path to identify gunshot without specialty of gunshot audio sound. 
 

