# UBC-Launch-Pad-Technical-Challenge

After reading all the prompts, I was struggling with the choice of which question to answer. Most of my expertise lies within Frontend Web, and I have an interest in iOS development, but I decided to challenge myself with the Machine Learning prompt as it’s something fascinating that I am currently trying to learn. Although I think Machine Learning is very powerful in what it is capable of doing and is not at all just a buzzword, I can definitely see why the average person would say it is. The term Machine Learning has been thrown around for a while now, and it’s often used incorrectly and in a very exaggerated sense (especially in the media), to the point where many people believe Machine Learning is just a useless and unnecessary advancement in technology. In a very generic and technical sense, Machine Learning is the use of user data and algorithms in order to understand and mimic the way humans learn and behave. Contrary to the previously mentioned belief, there are also large groups of people who believe Machine Learning is something over-the-top such as a solution to world hunger and even the best invention since the invention of the wheel.
 
I personally believe that Machine Learning lies somewhere in between these two beliefs, as a tool that won’t solve world hunger by itself, but can definitely aid in it. Machine learning is more than just a buzzword, but of course it’s not the answer to all the world’s problems. Some examples of where Machine Learning has shown its power and has been used properly are the AlphaGo Machine Learning bot, which was the first ever machine to defeat a world-renowned Go champion using reinforcement learning (one of three basic machine learning models used to train a machine to make a sequence of complex decisions through trial and error by repetitively placing it in difficult situations), and also GPT-3 (Generative Pre-trained Transformer 3), which uses an autoregressive language model (a feed-forward model which predicts future values based on past values) in order to produce human-like text in the form of writing essays and even coding websites. Although these examples are amazing and impressive, I think one of the best examples of how Machine Learning is being used in practically everyone’s daily life is TikTok. 
 
I can admit, just like many other people my age, that I use TikTok way more than I should. TikTok’s recommended videos and their “For You” page have been known to be extremely accurate in showing videos that appeal to the interest of the user, which is due to TikTok’s incredible use of Machine Learning. Everything that a user does and could possibly do is recorded and fed into TikTok’s Machine Learning algorithm. From what videos you watch, how long you watch the video for, what you skip, what you like, what you comment on, and much more, TikTok’s algorithm (which is more like a model since it saves unique user data and code) saves and uses it all to personalize everyone’s individual For You page. A more specific example of how TikTok’s Machine Learning model works is collaborative-filtering. Collaborative-filtering works in the way that, if, for example, Bob likes and comments on videos 1, 2, 3, and 4, and Billy likes and comments on videos 1 and 2, then Billy will also tend to like videos 3 and 4 and will be recommended the video on his For You page. In terms of code, the following is some code using Apache Spark (PsSpark and Python APIs) that prints out a video recommendation based on features one chooses to include in the dataset of a user’s interactions.

# import libraries
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
lines = spark.read.text("TEXT_FILE.txt").rdd
parts = lines.map(lambda row: row.value.split("::"))
liked_data = parts.map(lambda p: Row(COLUMN_1, COLUMN_2, like=like)
liked_df = spark.createDataFrame(liked_data)
(training, test) = liked_df.randomSplit([0.75, 0.25])
als = ALS(maxIter=4, regParam=0.05, users="COLUMN_1", items="COLUMN_2", ratingCol="like",
          coldStartStrategy="drop")
model = als.fit(training)
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="like",
                                predictionCol="prediction")
                       
rmse = evaluator.evaluate(predictions)
print("RMSE = " + str(rmse))
videoRecs = model.recommendForUsers(20)
