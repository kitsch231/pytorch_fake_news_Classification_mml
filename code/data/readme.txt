------------------------readme---------------------------------
1. This dataset contains 9528 tweets with corresponding images for rumor detection. 
2. Tweets are separated into training set and testing set. For each set, there are two files storing rumor and non-rumor tweets respectively.
3. 16 social context features are extracted for each tweet in 'social_feature.txt'
4. Images are sepreated into two folders according to their labels. Each image is named by the last part of it's url attached in corresponding tweet.
5. The data format in each txt file is as follow:
tweet id|user name|tweet url|user url|publish time| original?|retweet count|comment count|praise count|user id|user authentication type|user fans count|user follow count|user tweet count|publish platform
image1 url|image2 url|null
tweet content
6. Please cite the following paper when using this dataset:
[1] Zhiwei Jin, Juan Cao, Han Guo, Yongdong Zhang, Jiebo Luo: Multimodal Fusion with Recurrent Neural Networks for Rumor Detection on Microblogs. ACM Multimedia 2017: 795-816

Notice:
(1). each tweet has three lines,the first line contains 15 meta information separated by |,the second line lists the urls of images attached to the tweet,urls are also separated by | and a 'null' placeholder always exits,the third line is the text content(could be empty).
(2).user authentication type has three distinct value,0 for no authentication,1 for person authentication, 2 for organization authentication