
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt


# In[2]:


from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pyspark.sql.types import StringType, StructField, StructType
from pyspark.sql import Row
from pyspark.sql.functions import col
from pyspark.sql.functions import udf, split
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, VectorIndexer
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, LinearSVC, NaiveBayes, RandomForestClassifier, GBTClassifier
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GBTRegressor
from pyspark.ml.clustering import KMeans, GaussianMixture, LDA
from pyspark.ml.recommendation import ALS
from pyspark.ml.fpm import FPGrowth
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator, RegressionEvaluator, ClusteringEvaluator


# In[3]:


def CreateSparkContext(appname, parameters={"spark.eventLog.enabled": "false"}):
    '''
    appname: 应用程序命名。
    parameters: 传入多组键值对形式的参数，如{"spark.eventLog.enabled": "true", "spark.ui.showConsoleProgress":"false"}。
    具体参数可参考：https://spark.apache.org/docs/latest/configuration.html。
    '''
    global sc
    sparkConf = SparkConf()                       .setAppName(appname)                       .set("spark.ui.showConsoleProgress", "false")                       .setAll(parameters)
    sc = SparkContext(conf = sparkConf)
    print("master=" + sc.master)
    SetLogger(sc)
    SetPath(sc)
    return sc

def SetLogger(sc):
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)
    logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)
    logger.LogManager.getRootLogger().setLevel(logger.Level.ERROR)
    
def SetPath(sc):
    global path
    if sc.master[0:5] == "local" or sc.master[0:5] == "spark":
        path = "file:/Users/johnnie/pythonwork/workspace/PythonProject/data/"
    else:
        path = "hdfs://localhost:9000/user/hduser/test/data/"


# In[4]:


def CreateSparkSession(master, appname, configofk=None, configofv=None, conf=None):
    '''
    master: 指定master URL地址，可以是本地local，或者spark，或mesos等。
    appname: 应用程序命名。
    configofk: 指定要设置的参数，如："spark.eventLog.enabled"
    configofv: 指定要设置参数的值，可以是布尔值，数值等，具体参数设置可参考：https://spark.apache.org/docs/latest/configuration.html。
    conf: 传入SparkConf()形式的参数。
    '''
    global spark
    spark = SparkSession                 .master(master)                 .builder                 .appName(appname)                 .config(configofk, configofv, conf)                 .getOrCreate()
    return spark


# In[5]:


def LoadFile(file, delimiter=None, nonstandard=False, standard=False):
    '''
    non-standard: 非标准格式数据，如1|M|teacher|234242形式。
    standard: 标准格式数据，如.csv, .xlsx格式的数据表。
    file: 要加载的文件。
    delimiter: 非标准格式数据的分隔符，如, |等。
    '''
    if nonstandard:
        rawData = sc.textFile(path + file)
        lines = rawData.map(lambda x: x.split(delimiter))
        fieldnum = len(lines.first())
        fields = [StructField("f" + str(i), StringType(), True) for i in range(fieldnum)]
        schema = StructType(fields)
        df = spark.createDataFrame(lines, schema)
    if standard:
        df = spark.read.format(file.split(".")[1]).option("header", "true").option("delimiter", delimiter).load(path + file)
    return df


# In[6]:


def SplitData(df):
    '''
    df: 传入dataframe格式要切分的数据。默认比例为7:3。注意：输入的dataframe已经是去掉无用字段，只剩余特征字段和标签字段。
    '''
    train_df, test_df = df.randomSplit([0.7, 0.3])
    return train_df, test_df


# In[7]:


def FeaturesTransform(stringIndexer=False, inputColStringIndexer=None, outputColStringIndexer=None,
                                 oneHotEncoder=False, inputColOneHotEncoder=None, outputColOneHotEncoder=None,
                                 vectorAssembler=False, inputColsVectorAssembler=None, outputColsVectorAssembler=None,
                                 vectorIndexer=False, inputColsVectorIndexer=None, outputColsVectorIndexer=None, maxCategories=None):
    """
    stringIndexer: 将文字的分类特征字段转换为数值，类似于Scikit Learn中的LabelEncoder的功能。
    inputColStringIndexer: 输入要转换的字段，可以是单个字段，也可以是多个字段。
    outputColStringIndexer: 输出转换后的字段名，一般是一个字段。
    oneHotEncoder: 独热向量编码，将一个数值的分类特征字段转为多个字段。
    inputColOneHotEncoder: 输入要转换的字段，可以是单个字段，也可以是多个字段。
    outputColOneHotEncoder: 输出转换后的字段名，一般是一个字段。
    vectorAssembler: 将多个特征字段整合成一个特征的Vector，也就是特征数据向量化。
    inputColsVectorAssembler: 输入要转换的字段，可以是单个字段，也可以是多个字段。
    outputColsVectorAssembler: 输出转换后的字段名，一般是一个字段。
    vectorIndexer: 将不重复数值的数量小于等于maxCategories参数值所对应的字段视为分类字段，是否视为数值字段。这样做的目的，可以提高准确率。
    inputColsVectorIndexer: 输入要转换的字段，可以是单个字段，也可以是多个字段。
    outputColsVectorIndexer: 输出转换后的字段名，一般是一个字段。
    maxCategories: 数据中的最大类别的个数，找到最大类别的数目，填写在此处。
    """
    stagesList = []
    if stringIndexer:
        stagesList.append(StringIndexer(inputCol=inputColStringIndexer, outputCol=outputColStringIndexer))
    if oneHotEncoder:
        stagesList.append(OneHotEncoder(dropLast=False, inputCol=inputColOneHotEncoder, outputCol=outputColOneHotEncoder))
    if vectorAssembler:
        stagesList.append(VectorAssembler(inputCols=inputColsVectorAssembler, outputCol=outputColsVectorAssembler))
    if vectorIndexer:
        stagesList.append(VectorIndexer(inputCol=inputColsVectorIndexer, outputCol=outputColsVectorIndexer, maxCategories=maxCategories))
    return stagesList


# In[8]:


def SparkML(train_df, test_df=None, featuresCol='features', labelCol='label', binaryclass=False, multiclass=False, n_cluster=2, userCol='user', itemCol='item', ratingCol='rating', rank=10, userid=3, itemid=3, itemsCol='items', minSupport=0.3, minConfidence=0.8,
                   stringIndexer=False, inputColStringIndexer=None, outputColStringIndexer=None,
                   oneHotEncoder=False, inputColOneHotEncoder=None, outputColOneHotEncoder=None,
                   vectorAssembler=False, inputColsVectorAssembler=None, outputColsVectorAssembler=None,
                   vectorIndexer=False, inputColsVectorIndexer=None, outputColsVectorIndexer=None, maxCategories=None,
                   classification=False, logisticregression=False, decisiontreeclassifier=False, linearsvc=False, naivebayes=False, randomforestclassifier=False, gbtclassifier=False,
                   regression=False, linearregression=True, decisiontreeregressor=False, randomforestregressor=False, gbtregressor=False,
                   clustering=False, kmeans=False, gaussianmixture=False, lda=False,
                   recommendation=False, als=False,
                   association=False, fpgrowth=False):
    if classification:
        if logisticregression:
            stagesList = FeaturesTransform(stringIndexer=stringIndexer, inputColStringIndexer=inputColStringIndexer, outputColStringIndexer=outputColStringIndexer,
                                                         oneHotEncoder=oneHotEncoder, inputColOneHotEncoder=inputColOneHotEncoder, outputColOneHotEncoder=outputColOneHotEncoder,
                                                         vectorAssembler=vectorAssembler, inputColsVectorAssembler=inputColsVectorAssembler, outputColsVectorAssembler=outputColsVectorAssembler,
                                                         vectorIndexer=vectorIndexer, inputColsVectorIndexer=inputColsVectorIndexer, outputColsVectorIndexer=outputColsVectorIndexer, maxCategories=maxCategories)
            LRClassifier = LogisticRegression(featuresCol=featuresCol, labelCol=labelCol, predictionCol='Prediction', probabilityCol='Probability', rawPredictionCol='RawPrediction', 
                                                            standardization=True, maxIter=100, regParam=0.0, elasticNetParam=0.0, tol=1e-06, fitIntercept=True, threshold=0.5)
            paramGrid = ParamGridBuilder().addGrid(LRClassifier.maxIter, [10, 20, 50, 100, 200, 300, 500, 1000, 2000, 5000]).addGrid(LRClassifier.regParam, [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]).build()
            if binaryclass:
                evaluator = BinaryClassificationEvaluator(rawPredictionCol="RawPrediction", labelCol=labelCol, metricName="areaUnderROC")
            if multiclass:            
                evaluator = MulticlassClassificationEvaluator(labelCol=labelCol, predictionCol="Prediction", metricName="accuracy")
            LRCV = CrossValidator(estimator=LRClassifier, evaluator=evaluator, estimatorParamMaps=paramGrid, numFolds=10)
            stagesList.append(LRCV)
            LRC_Pipeline = Pipeline(stages=stagesList)
            LRC_PipelineModel = LRC_Pipeline.fit(train_df)
            LRC_Predicted = LRC_PipelineModel.transform(test_df)
            LRC_BestModel = LRC_PipelineModel.stages[-1].bestModel
            LRC_Probability = LRC_Predicted.select("Probability").toPandas()
            LRC_Prediction = LRC_Predicted.select("Prediction").toPandas()
            LRC_Score = evaluator.evaluate(LRC_Predicted)
            return LRC_BestModel, LRC_Predicted, LRC_Probability, LRC_Prediction, LRC_Score
        if decisiontreeclassifier:
            stagesList = FeaturesTransform(stringIndexer=stringIndexer, inputColStringIndexer=inputColStringIndexer, outputColStringIndexer=outputColStringIndexer,
                                                         oneHotEncoder=oneHotEncoder, inputColOneHotEncoder=inputColOneHotEncoder, outputColOneHotEncoder=outputColOneHotEncoder,
                                                         vectorAssembler=vectorAssembler, inputColsVectorAssembler=inputColsVectorAssembler, outputColsVectorAssembler=outputColsVectorAssembler,
                                                         vectorIndexer=vectorIndexer, inputColsVectorIndexer=inputColsVectorIndexer, outputColsVectorIndexer=outputColsVectorIndexer, maxCategories=maxCategories)
            DTClassifier = DecisionTreeClassifier(featuresCol=featuresCol, labelCol=labelCol, predictionCol='Prediction', probabilityCol='Probability', rawPredictionCol='RawPrediction', 
                                                                 maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0, impurity='gini', seed=None)
            paramGrid = ParamGridBuilder().addGrid(DTClassifier.impurity, ["gini", "entropy"]).addGrid(DTClassifier.maxDepth, [3, 5, 10, 15, 20, 25]).addGrid(DTClassifier.maxBins, [3, 5, 10, 50, 100, 200]).build()
            if binaryclass:
                evaluator = BinaryClassificationEvaluator(rawPredictionCol="RawPrediction", labelCol=labelCol, metricName="areaUnderROC")
            if multiclass:            
                evaluator = MulticlassClassificationEvaluator(labelCol=labelCol, predictionCol="Prediction", metricName="accuracy")
            DTCV = CrossValidator(estimator=DTClassifier, evaluator=evaluator, estimatorParamMaps=paramGrid, numFolds=10)
            stagesList.append(DTCV)
            DTC_Pipeline = Pipeline(stages=stagesList)
            DTC_PipelineModel = DTC_Pipeline.fit(train_df)
            DTC_Predicted = DTC_PipelineModel.transform(test_df)
            DTC_BestModel = DTC_PipelineModel.stages[-1].bestModel
            DTC_Probability = DTC_Predicted.select("Probability").toPandas()
            DTC_Prediction = DTC_Predicted.select("Prediction").toPandas()
            DTC_Score = evaluator.evaluate(DTC_Predicted)
            return DTC_BestModel, DTC_Predicted, DTC_Probability, DTC_Prediction, DTC_Score
        if linearsvc:
            stagesList = FeaturesTransform(stringIndexer=stringIndexer, inputColStringIndexer=inputColStringIndexer, outputColStringIndexer=outputColStringIndexer,
                                                         oneHotEncoder=oneHotEncoder, inputColOneHotEncoder=inputColOneHotEncoder, outputColOneHotEncoder=outputColOneHotEncoder,
                                                         vectorAssembler=vectorAssembler, inputColsVectorAssembler=inputColsVectorAssembler, outputColsVectorAssembler=outputColsVectorAssembler,
                                                         vectorIndexer=vectorIndexer, inputColsVectorIndexer=inputColsVectorIndexer, outputColsVectorIndexer=outputColsVectorIndexer, maxCategories=maxCategories)
            SVClassifier = LinearSVC(featuresCol=featuresCol, labelCol=labelCol, predictionCol='Prediction', rawPredictionCol='RawPrediction', 
                                                maxIter=100, regParam=0.0, tol=1e-06, fitIntercept=True, standardization=True, threshold=0.0)
            paramGrid = ParamGridBuilder().addGrid(SVClassifier.maxIter, [10, 20, 50, 100, 200, 300, 500, 1000, 2000, 5000]).addGrid(SVClassifier.regParam, [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]).build()
            if binaryclass:
                evaluator = BinaryClassificationEvaluator(rawPredictionCol="RawPrediction", labelCol=labelCol, metricName="areaUnderROC")
            if multiclass:            
                evaluator = MulticlassClassificationEvaluator(labelCol=labelCol, predictionCol="Prediction", metricName="accuracy")
            SVCV = CrossValidator(estimator=SVClassifier, evaluator=evaluator, estimatorParamMaps=paramGrid, numFolds=10)
            stagesList.append(SVCV)
            SVC_Pipeline = Pipeline(stages=stagesList)
            SVC_PipelineModel = SVC_Pipeline.fit(train_df)
            SVC_Predicted = SVC_PipelineModel.transform(test_df)
            SVC_BestModel = SVC_PipelineModel.stages[-1].bestModel
            SVC_Prediction = SVC_Predicted.select("Prediction").toPandas()
            SVC_Score = evaluator.evaluate(SVC_Predicted)
            return SVC_BestModel, SVC_Predicted, SVC_Prediction, SVC_Score
        if naivebayes:
            stagesList = FeaturesTransform(stringIndexer=stringIndexer, inputColStringIndexer=inputColStringIndexer, outputColStringIndexer=outputColStringIndexer,
                                                         oneHotEncoder=oneHotEncoder, inputColOneHotEncoder=inputColOneHotEncoder, outputColOneHotEncoder=outputColOneHotEncoder,
                                                         vectorAssembler=vectorAssembler, inputColsVectorAssembler=inputColsVectorAssembler, outputColsVectorAssembler=outputColsVectorAssembler,
                                                         vectorIndexer=vectorIndexer, inputColsVectorIndexer=inputColsVectorIndexer, outputColsVectorIndexer=outputColsVectorIndexer, maxCategories=maxCategories)
            NBClassifier = NaiveBayes(featuresCol=featuresCol, labelCol=labelCol, predictionCol='Prediction', probabilityCol='Probability', rawPredictionCol='RawPrediction', 
                                                 smoothing=1.0, modelType='multinomial', thresholds=None)
            paramGrid = ParamGridBuilder().addGrid(NBClassifier.smoothing, [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]).build()
            if binaryclass:
                evaluator = BinaryClassificationEvaluator(rawPredictionCol="RawPrediction", labelCol=labelCol, metricName="areaUnderROC")
            if multiclass:            
                evaluator = MulticlassClassificationEvaluator(labelCol=labelCol, predictionCol="Prediction", metricName="accuracy")
            NBCV = CrossValidator(estimator=NBClassifier, evaluator=evaluator, estimatorParamMaps=paramGrid, numFolds=10)
            stagesList.append(NBCV)
            NBC_Pipeline = Pipeline(stages=stagesList)
            NBC_PipelineModel = NBC_Pipeline.fit(train_df)
            NBC_Predicted = NBC_PipelineModel.transform(test_df)
            NBC_BestModel = NBC_PipelineModel.stages[-1].bestModel
            NBC_Probability = NBC_Predicted.select("Probability").toPandas()
            NBC_Prediction = NBC_Predicted.select("Prediction").toPandas()
            NBC_Score = evaluator.evaluate(NBC_Predicted)
            return NBC_BestModel, NBC_Predicted, NBC_Probability, NBC_Prediction, NBC_Score
        if randomforestclassifier:
            stagesList = FeaturesTransform(stringIndexer=stringIndexer, inputColStringIndexer=inputColStringIndexer, outputColStringIndexer=outputColStringIndexer,
                                                         oneHotEncoder=oneHotEncoder, inputColOneHotEncoder=inputColOneHotEncoder, outputColOneHotEncoder=outputColOneHotEncoder,
                                                         vectorAssembler=vectorAssembler, inputColsVectorAssembler=inputColsVectorAssembler, outputColsVectorAssembler=outputColsVectorAssembler,
                                                         vectorIndexer=vectorIndexer, inputColsVectorIndexer=inputColsVectorIndexer, outputColsVectorIndexer=outputColsVectorIndexer, maxCategories=maxCategories)
            RFClassifier = RandomForestClassifier(featuresCol=featuresCol, labelCol=labelCol, predictionCol='Prediction', probabilityCol='Probability', rawPredictionCol='RawPrediction', 
                                                                   maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0, impurity='gini', numTrees=20, featureSubsetStrategy='auto', seed=None, subsamplingRate=1.0)
            paramGrid = ParamGridBuilder().addGrid(RFClassifier.impurity, ["gini", "entropy"]).addGrid(RFClassifier.maxDepth, [3, 5, 10, 15, 20, 25]).addGrid(RFClassifier.maxBins, [3, 5, 10, 50, 100, 200]).addGrid(RFClassifier.numTrees, [5, 10, 20, 50, 100, 200]).addGrid(RFClassifier.subsamplingRate, [0.1, 0.2, 0.5, 0.8, 0.9, 1.0]).build()
            if binaryclass:
                evaluator = BinaryClassificationEvaluator(rawPredictionCol="RawPrediction", labelCol=labelCol, metricName="areaUnderROC")
            if multiclass:            
                evaluator = MulticlassClassificationEvaluator(labelCol=labelCol, predictionCol="Prediction", metricName="accuracy")
            RFCV = CrossValidator(estimator=RFClassifier, evaluator=evaluator, estimatorParamMaps=paramGrid, numFolds=10)
            stagesList.append(RFCV)
            RFC_Pipeline = Pipeline(stages=stagesList)
            RFC_PipelineModel = RFC_Pipeline.fit(train_df)
            RFC_Predicted = RFC_PipelineModel.transform(test_df)
            RFC_BestModel = RFC_PipelineModel.stages[-1].bestModel
            RFC_Probability = RFC_Predicted.select("Probability").toPandas()
            RFC_Prediction = RFC_Predicted.select("Prediction").toPandas()
            RFC_Score = evaluator.evaluate(RFC_Predicted)
            return RFC_BestModel, RFC_Predicted, RFC_Probability, RFC_Prediction, RFC_Score
        if gbtclassifier:
            stagesList = FeaturesTransform(stringIndexer=stringIndexer, inputColStringIndexer=inputColStringIndexer, outputColStringIndexer=outputColStringIndexer,
                                                         oneHotEncoder=oneHotEncoder, inputColOneHotEncoder=inputColOneHotEncoder, outputColOneHotEncoder=outputColOneHotEncoder,
                                                         vectorAssembler=vectorAssembler, inputColsVectorAssembler=inputColsVectorAssembler, outputColsVectorAssembler=outputColsVectorAssembler,
                                                         vectorIndexer=vectorIndexer, inputColsVectorIndexer=inputColsVectorIndexer, outputColsVectorIndexer=outputColsVectorIndexer, maxCategories=maxCategories)
            GBClassifier = GBTClassifier(featuresCol=featuresCol, labelCol=labelCol, predictionCol='Prediction',
                                                     maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0, lossType='logistic', maxIter=20, stepSize=0.1, seed=None, subsamplingRate=1.0)
            paramGrid = ParamGridBuilder().addGrid(GBClassifier.maxDepth, [3, 5, 10, 15, 20, 25]).addGrid(GBClassifier.maxBins, [3, 5, 10, 50, 100, 200]).addGrid(GBClassifier.maxIter, [5, 10, 20, 50, 100, 200]).addGrid(GBClassifier.stepSize, [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]).addGrid(GBClassifier.subsamplingRate, [0.1, 0.2, 0.5, 0.8, 0.9, 1.0]).build()           
            evaluator = MulticlassClassificationEvaluator(labelCol=labelCol, predictionCol="Prediction", metricName="accuracy")
            GBCV = CrossValidator(estimator=GBClassifier, evaluator=evaluator, estimatorParamMaps=paramGrid, numFolds=10)
            stagesList.append(GBCV)
            GBC_Pipeline = Pipeline(stages=stagesList)
            GBC_PipelineModel = GBC_Pipeline.fit(train_df)
            GBC_Predicted = GBC_PipelineModel.transform(test_df)
            GBC_BestModel = GBC_PipelineModel.stages[-1].bestModel
            GBC_Prediction = GBC_Predicted.select("Prediction").toPandas()
            GBC_Score = evaluator.evaluate(GBC_Predicted)
            return GBC_BestModel, GBC_Predicted, GBC_Prediction, GBC_Score
    if regression:
        if linearregression:
            stagesList = FeaturesTransform(stringIndexer=stringIndexer, inputColStringIndexer=inputColStringIndexer, outputColStringIndexer=outputColStringIndexer,
                                                         oneHotEncoder=oneHotEncoder, inputColOneHotEncoder=inputColOneHotEncoder, outputColOneHotEncoder=outputColOneHotEncoder,
                                                         vectorAssembler=vectorAssembler, inputColsVectorAssembler=inputColsVectorAssembler, outputColsVectorAssembler=outputColsVectorAssembler,
                                                         vectorIndexer=vectorIndexer, inputColsVectorIndexer=inputColsVectorIndexer, outputColsVectorIndexer=outputColsVectorIndexer, maxCategories=maxCategories)
            LRegressor = LinearRegression(featuresCol=featuresCol, labelCol=labelCol, predictionCol='Prediction',
                                                         standardization=True, fitIntercept=True, loss='squaredError', 
                                                         maxIter=100, regParam=0.0, elasticNetParam=0.0, tol=1e-06, epsilon=1.35)
            paramGrid = ParamGridBuilder().addGrid(LRegressor.maxIter, [10, 20, 50, 100, 200, 300, 500, 1000, 2000, 5000]).addGrid(LRegressor.regParam, [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]).build()
            evaluator = RegressionEvaluator(labelCol=labelCol, predictionCol="Prediction", metricName="rmse")
            LRCV = CrossValidator(estimator=LRegressor, evaluator=evaluator, estimatorParamMaps=paramGrid, numFolds=10)
            stagesList.append(LRCV)
            LR_Pipeline = Pipeline(stages=stagesList)
            LR_PipelineModel = LR_Pipeline.fit(train_df)
            LR_Predicted = LR_PipelineModel.transform(test_df)
            LR_BestModel = LR_PipelineModel.stages[-1].bestModel
            LR_Prediction = LR_Predicted.select("Prediction").toPandas()
            LR_Score = evaluator.evaluate(LR_Predicted)
            return LR_BestModel, LR_Predicted, LR_Prediction, LR_Score    
        if decisiontreeregressor:
            stagesList = FeaturesTransform(stringIndexer=stringIndexer, inputColStringIndexer=inputColStringIndexer, outputColStringIndexer=outputColStringIndexer,
                                                         oneHotEncoder=oneHotEncoder, inputColOneHotEncoder=inputColOneHotEncoder, outputColOneHotEncoder=outputColOneHotEncoder,
                                                         vectorAssembler=vectorAssembler, inputColsVectorAssembler=inputColsVectorAssembler, outputColsVectorAssembler=outputColsVectorAssembler,
                                                         vectorIndexer=vectorIndexer, inputColsVectorIndexer=inputColsVectorIndexer, outputColsVectorIndexer=outputColsVectorIndexer, maxCategories=maxCategories)
            DTRegressor = DecisionTreeRegressor(featuresCol=featuresCol, labelCol=labelCol, predictionCol='Prediction',
                                                                   maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0, impurity='variance', seed=None, varianceCol=None)
            paramGrid = ParamGridBuilder().addGrid(DTRegressor.maxDepth, [3, 5, 10, 15, 20, 25]).addGrid(DTRegressor.maxBins, [3, 5, 10, 50, 100, 200]).build()
            evaluator = RegressionEvaluator(labelCol=labelCol, predictionCol="Prediction", metricName="rmse")
            DTRCV = CrossValidator(estimator=DTRegressor, evaluator=evaluator, estimatorParamMaps=paramGrid, numFolds=10)
            stagesList.append(DTRCV)
            DTR_Pipeline = Pipeline(stages=stagesList)
            DTR_PipelineModel = DTR_Pipeline.fit(train_df)
            DTR_Predicted = DTR_PipelineModel.transform(test_df)
            DTR_BestModel = DTR_PipelineModel.stages[-1].bestModel
            DTR_Prediction = DTR_Predicted.select("Prediction").toPandas()
            DTR_Score = evaluator.evaluate(DTR_Predicted)
            return DTR_BestModel, DTR_Predicted, DTR_Prediction, DTR_Score  
        if randomforestregressor:
            stagesList = FeaturesTransform(stringIndexer=stringIndexer, inputColStringIndexer=inputColStringIndexer, outputColStringIndexer=outputColStringIndexer,
                                                         oneHotEncoder=oneHotEncoder, inputColOneHotEncoder=inputColOneHotEncoder, outputColOneHotEncoder=outputColOneHotEncoder,
                                                         vectorAssembler=vectorAssembler, inputColsVectorAssembler=inputColsVectorAssembler, outputColsVectorAssembler=outputColsVectorAssembler,
                                                         vectorIndexer=vectorIndexer, inputColsVectorIndexer=inputColsVectorIndexer, outputColsVectorIndexer=outputColsVectorIndexer, maxCategories=maxCategories)
            RFRegressor = RandomForestRegressor(featuresCol=featuresCol, labelCol=labelCol, predictionCol='Prediction',
                                                                     maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0, impurity='variance', subsamplingRate=1.0, seed=None, numTrees=20)
            paramGrid = ParamGridBuilder().addGrid(RFRegressor.maxDepth, [3, 5, 10, 15, 20, 25]).addGrid(RFRegressor.maxBins, [3, 5, 10, 50, 100, 200]).addGrid(RFRegressor.numTrees, [5, 10, 20, 50, 100, 200]).addGrid(RFRegressor.subsamplingRate, [0.1, 0.2, 0.5, 0.8, 0.9, 1.0]).build()
            evaluator = RegressionEvaluator(labelCol=labelCol, predictionCol="Prediction", metricName="rmse")            
            RFRCV = CrossValidator(estimator=RFRegressor, evaluator=evaluator, estimatorParamMaps=paramGrid, numFolds=10)
            stagesList.append(RFRCV)
            RFR_Pipeline = Pipeline(stages=stagesList)
            RFR_PipelineModel = RFR_Pipeline.fit(train_df)
            RFR_Predicted = RFR_PipelineModel.transform(test_df)
            RFR_BestModel = RFR_PipelineModel.stages[-1].bestModel
            RFR_Prediction = RFR_Predicted.select("Prediction").toPandas()
            RFR_Score = evaluator.evaluate(RFR_Predicted)
            return RFR_BestModel, RFR_Predicted, RFR_Prediction, RFR_Score
        if gbtregressor:
            stagesList = FeaturesTransform(stringIndexer=stringIndexer, inputColStringIndexer=inputColStringIndexer, outputColStringIndexer=outputColStringIndexer,
                                                         oneHotEncoder=oneHotEncoder, inputColOneHotEncoder=inputColOneHotEncoder, outputColOneHotEncoder=outputColOneHotEncoder,
                                                         vectorAssembler=vectorAssembler, inputColsVectorAssembler=inputColsVectorAssembler, outputColsVectorAssembler=outputColsVectorAssembler,
                                                         vectorIndexer=vectorIndexer, inputColsVectorIndexer=inputColsVectorIndexer, outputColsVectorIndexer=outputColsVectorIndexer, maxCategories=maxCategories)
            GBRegressor = GBTRegressor(featuresCol=featuresCol, labelCol=labelCol, predictionCol='Prediction',
                                                       maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0, subsamplingRate=1.0, lossType='squared', maxIter=20, stepSize=0.1, seed=None, impurity='variance')
            paramGrid = ParamGridBuilder().addGrid(GBRegressor.maxDepth, [3, 5, 10, 15, 20, 25]).addGrid(GBRegressor.maxBins, [3, 5, 10, 50, 100, 200]).addGrid(GBRegressor.maxIter, [5, 10, 20, 50, 100, 200]).addGrid(GBRegressor.stepSize, [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]).addGrid(GBRegressor.subsamplingRate, [0.1, 0.2, 0.5, 0.8, 0.9, 1.0]).build()   
            evaluator = RegressionEvaluator(labelCol=labelCol, predictionCol="Prediction", metricName="rmse")            
            GBRCV = CrossValidator(estimator=GBRegressor, evaluator=evaluator, estimatorParamMaps=paramGrid, numFolds=10)
            stagesList.append(GBRCV)
            GBR_Pipeline = Pipeline(stages=stagesList)
            GBR_PipelineModel = GBR_Pipeline.fit(train_df)
            GBR_Predicted = GBR_PipelineModel.transform(test_df)
            GBR_BestModel = GBR_PipelineModel.stages[-1].bestModel
            GBR_Prediction = GBR_Predicted.select("Prediction").toPandas()
            GBR_Score = evaluator.evaluate(GBR_Predicted)
            return GBR_BestModel, GBR_Predicted, GBR_Prediction, GBR_Score
    if clustering:
        if kmeans:
            stagesList = FeaturesTransform(stringIndexer=stringIndexer, inputColStringIndexer=inputColStringIndexer, outputColStringIndexer=outputColStringIndexer,
                                                         oneHotEncoder=oneHotEncoder, inputColOneHotEncoder=inputColOneHotEncoder, outputColOneHotEncoder=outputColOneHotEncoder,
                                                         vectorAssembler=vectorAssembler, inputColsVectorAssembler=inputColsVectorAssembler, outputColsVectorAssembler=outputColsVectorAssembler,
                                                         vectorIndexer=vectorIndexer, inputColsVectorIndexer=inputColsVectorIndexer, outputColsVectorIndexer=outputColsVectorIndexer, maxCategories=maxCategories)
            KCluster = KMeans(featuresCol=featuresCol, predictionCol='Prediction',
                                        k=n_cluster, initMode='k-means||', initSteps=2, tol=0.0001, maxIter=20, seed=None)
            paramGrid = ParamGridBuilder().addGrid(KCluster.initSteps, [1, 2, 5, 10, 20, 50, 100]).addGrid(KCluster.maxIter, [10, 20, 50, 100, 200, 500, 1000, 2000]).addGrid(KCluster.seed, [i for i in range(1001)]).build() 
            evaluator = ClusteringEvaluator(predictionCol='Prediction', featuresCol=featuresCol, metricName='silhouette')
            KMCV = CrossValidator(estimator=KCluster, evaluator=evaluator, estimatorParamMaps=paramGrid, numFolds=10)
            stagesList.append(KMCV)
            KMC_Pipeline = Pipeline(stages=stagesList)
            KMC_PipelineModel = KMC_Pipeline.fit(train_df)
            KMC_Predicted = KMC_PipelineModel.transform(train_df)
            KMC_BestModel = KMC_PipelineModel.stages[-1].bestModel
            KMC_Prediction = KMC_Predicted.select("Prediction").toPandas()
            KMC_Score = evaluator.evaluate(KMC_Predicted)
            return KMC_BestModel, KMC_Predicted, KMC_Prediction, KMC_Score
        if gaussianmixture:
            stagesList = FeaturesTransform(stringIndexer=stringIndexer, inputColStringIndexer=inputColStringIndexer, outputColStringIndexer=outputColStringIndexer,
                                                         oneHotEncoder=oneHotEncoder, inputColOneHotEncoder=inputColOneHotEncoder, outputColOneHotEncoder=outputColOneHotEncoder,
                                                         vectorAssembler=vectorAssembler, inputColsVectorAssembler=inputColsVectorAssembler, outputColsVectorAssembler=outputColsVectorAssembler,
                                                         vectorIndexer=vectorIndexer, inputColsVectorIndexer=inputColsVectorIndexer, outputColsVectorIndexer=outputColsVectorIndexer, maxCategories=maxCategories)
            GMCluster = GaussianMixture(featuresCol=featuresCol, predictionCol='Prediction', probabilityCol='Probability',
                                                       k=n_cluster, tol=0.01, maxIter=100, seed=None)
            paramGrid = ParamGridBuilder().addGrid(GMCluster.maxIter, [10, 20, 50, 100, 200, 500, 1000, 2000]).addGrid(GMCluster.seed, [i for i in range(1001)]).build() 
            evaluator = ClusteringEvaluator(predictionCol='Prediction', featuresCol=featuresCol, metricName='silhouette')
            GMCV = CrossValidator(estimator=GMCluster, evaluator=evaluator, estimatorParamMaps=paramGrid, numFolds=10)
            stagesList.append(GMCV)
            GMC_Pipeline = Pipeline(stages=stagesList)
            GMC_PipelineModel = GMC_Pipeline.fit(train_df)
            GMC_Predicted = GMC_PipelineModel.transform(train_df)
            GMC_BestModel = GMC_PipelineModel.stages[-1].bestModel
            GMC_Probability = GMC_Predicted.select("Probability").toPandas()
            GMC_Prediction = GMC_Predicted.select("Prediction").toPandas()
            GMC_Score = evaluator.evaluate(GMC_Predicted)
            return GMC_BestModel, GMC_Predicted, GMC_Probability, GMC_Prediction, GMC_Score
        if lda:
            stagesList = FeaturesTransform(stringIndexer=stringIndexer, inputColStringIndexer=inputColStringIndexer, outputColStringIndexer=outputColStringIndexer,
                                                         oneHotEncoder=oneHotEncoder, inputColOneHotEncoder=inputColOneHotEncoder, outputColOneHotEncoder=outputColOneHotEncoder,
                                                         vectorAssembler=vectorAssembler, inputColsVectorAssembler=inputColsVectorAssembler, outputColsVectorAssembler=outputColsVectorAssembler,
                                                         vectorIndexer=vectorIndexer, inputColsVectorIndexer=inputColsVectorIndexer, outputColsVectorIndexer=outputColsVectorIndexer, maxCategories=maxCategories)
            LDACluster = LDA(featuresCol=featuresCol,
                                      maxIter=20, seed=None, k=n_cluster, learningOffset=1024.0, learningDecay=0.51, subsamplingRate=0.05)
            paramGrid = ParamGridBuilder().addGrid(LDACluster.maxIter, [10, 20, 50, 100, 200, 500, 1000, 2000]).addGrid(LDACluster.seed, [i for i in range(1001)]).addGrid(LDACluster.subsamplingRate, [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]).build()  
            evaluator = ClusteringEvaluator(predictionCol='Prediction', featuresCol=featuresCol, metricName='silhouette')
            LDACV = CrossValidator(estimator=LDACluster, evaluator=evaluator, estimatorParamMaps=paramGrid, numFolds=10)
            stagesList.append(LDACV)
            LDA_Pipeline = Pipeline(stages=stagesList)
            LDA_PipelineModel = LDA_Pipeline.fit(train_df)
            LDA_Predicted = LDA_PipelineModel.transform(train_df)
            LDA_BestModel = LDA_PipelineModel.stages[-1].bestModel
            LDA_Topics = LDA_BestModel.describeTopics().toPandas()
            LDA_Score = evaluator.evaluate(LDA_Predicted)
            return LDA_BestModel, LDA_Topics, LDA_Score
    if recommendation:
        if als:
            ALSR = ALS(userCol=userCol, itemCol=itemCol, ratingCol=ratingCol,
                              rank=rank, maxIter=10, regParam=0.1, numUserBlocks=10, numItemBlocks=10, alpha=1.0, seed=1)
            ALSR_Model = ALSR.fit(train_df)
            ALSR_ForUsers = ALSR_Model.recommendForAllUsers(userid=userid)
            ALSR_ForItems = ALSR_Model.recommendForAllItems(itemid=itemid)          
            return ALSR_Model, ALSR_ForUsers, ALSR_ForItems
    if association:
        if fpgrowth:
            fpg = FPGrowth(minSupport=minSupport, minConfidence=minConfidence, itemsCol=itemsCol, predictionCol='Prediction')
            fpg_model = fpg.fit(train_df)
            fpg_freqItemsets = fpg_model.freqItemsets.toPandas()
            fpg_associationRules = fpg_model.associationRules.toPandas()     
            return fpg_model, fpg_freqItemsets, fpg_associationRules