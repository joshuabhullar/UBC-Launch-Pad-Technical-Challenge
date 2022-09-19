# UBC-Launch-Pad-Technical-Challenge

After reading all the prompts, I struggled with choosing which question to answer. Most of my expertise lies within Frontend Web, and I am interested in iOS development. Still, I decided to challenge myself with the Machine Learning prompt as it’s something fascinating that I am currently trying to learn. Although I think Machine Learning is very powerful in what it can do and is not at all just a buzzword, I can definitely see why the average person would say it is. The term Machine Learning has been thrown around for a while now, and it’s often used incorrectly and in a very exaggerated sense (especially in the media), to the point where many people believe Machine Learning is just a useless and unnecessary advancement in technology. In a very generic and technical sense, Machine Learning is using user data and algorithms to understand and mimic how humans learn and behave. However, contrary to the previously mentioned belief, there are also large groups of people who believe Machine Learning is something over-the-top such as a solution to world hunger and even the best invention since the invention of the wheel.
 
I believe that Machine Learning lies somewhere between these two beliefs, as a tool that won’t solve world hunger by itself, but can definitely aid in it. Machine learning is more than just a buzzword, but it’s also not the answer to all the world’s problems. Some examples of where Machine Learning has shown its power and has been used properly are the AlphaGo Machine Learning bot, which was the first ever machine to defeat a world-renowned Go champion using reinforcement learning (one of three basic machine learning models used to train a machine to make a sequence of complex decisions through trial and error by repetitively placing it in difficult situations), and also GPT-3 (Generative Pre-trained Transformer 3), which uses an autoregressive language model (a feed-forward model which predicts future values based on past values) in order to produce human-like text in the form of writing essays and even coding websites. Although these examples are amazing and impressive, I think one of the best examples of how Machine Learning is being used in practically everyone’s daily life is TikTok. 
 
I can admit, just like many other people my age, that I use TikTok way more than I should. TikTok’s recommended videos and their “For You” page have been known to be extremely accurate in showing videos that appeal to the interest of the user, which is due to TikTok’s incredible use of Machine Learning. Everything that a user does and could possibly do is recorded and fed into TikTok’s Machine Learning algorithm. From what videos you watch, how long you watch the video for, what you skip, what you like, what you comment on, and much more, TikTok’s algorithm (which is more like a model since it saves unique user data and code) saves and uses it all to personalize everyone’s individual For You page. A more specific example of how TikTok’s Machine Learning model works is collaborative filtering. Collaborative-filtering works in the way that, if, for example, Bob likes and comments on videos 1, 2, 3, and 4, and Billy likes and comments on videos 1 and 2, then Billy will also tend to like videos 3 and 4 and will be recommended the video on his For You page. The following is some code using Apache Spark (using PsSpark which is a Python API used for large-scale data processing) for collaborative filtering. (Use the raw view for easiest interpretation of the code and although I tried to teach myself Apache Spark to write this code, inspiration for the code heavily comes from https://medium.com/swlh/an-easy-guide-to-creating-a-tiktok-like-algorithm-3e9c954fb4e9.)

\# import libraries\
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

lines = spark.read.text("TEXT_FILE.txt").rdd
parts = lines.map(lambda row: row.value.split("::"))
liked_data = parts.map(lambda p: Row(COLUMN_1, COLUMN_2, like=like)

liked_df = spark.createDataFrame(liked_data)
(training, test) = liked_df.randomSplit([0.75, 0.25])

als = ALS(maxIter=4, regParam=0.05, users="COLUMN_1", 
items="COLUMN_2", ratingCol="like",
          coldStartStrategy="drop")
										
model = als.fit(training)
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="like",
                                predictionCol="prediction")
                       
rmse = evaluator.evaluate(predictions)
print("RMSE = " + str(rmse))
videoRecs = model.recommendForUsers(20)

What this code is essentially doing is training an ALS model (which stands for Alternating Least Square and is a matrix factorization algorithm used in collaborative filtering problems that factorizes a given matrix R into two factors X and Y such that R ≈ XYT using root-mean-square-error) to print out a video recommendation based on features one chooses to include in the dataset of a user’s interactions. 
 
Another factor TikTok’s algorithm considers is content-based filtering, which recommends certain video attributes by suggesting videos that share similar features (such as duration, sound, and text) with one another. So, for example, if a user views a video that is around 20 seconds long and has the hashtags “explore” and “travel”, and there exists another video that is also around 20 seconds long with the hashtags “explore” and “travel”, the second video will be recommended to the user. The following is some code for content-based filtering.
 
\# import libraries
 
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
 
data = pd.read_csv("DATA.csv")
 
\# computing the cosine similarity\
alg = cosine_similarity(data)
 
This code segment essentially uses cosine similarity (an algorithm that computes similarity as a normalized dot product of two variables or videos in this case) and videos from the past in order to recommend similar videos to the user.
 
The idea behind Machine Learning is that what’s easy for computers is hard for humans and what's easy for humans is hard for computers generally — a matter of recognition vs calculation. Machine learning is a method to achieve those things that used to be hard for computers, such as recognizing a dog in a picture. The way it’s done is by modelling after the human brain such that machine learning models have nodes and edges, similarly to how we humans have neurons and synapses. As for challenges faced in Machine Learning, one of the main challenges when adopting machine learning is inaccessible user data to use. However, TikTok does an amazing and detailed job of collecting user data, so well to the point where there are concerns regarding data security and privacy breaches when TikTok collects user data. Reading the user agreement when creating a TikTok account shows that TikTok’s privacy policy collects a large amount of data from the user, such as app/file names, audio and microphone input/output, and it even monitors keyboard strokes and patterns. The problem of data security for TikTok is not such an easy problem to solve, as a best-case solution would most likely try and balance the security of the user’s data while also trying to provide TikTok with the data it needs to perform its Machine Learning and make its For You page so addicting. Overall, Machine Learning is being used daily as an extremely powerful tool in examples I haven’t even mentioned yet. I think it’s safe for me to conclude that Machine Learning is not a buzzword, as its being ingrained into so many different aspects and parts of our lives that we don’t even recognize yet.
