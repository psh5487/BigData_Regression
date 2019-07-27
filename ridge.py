import numpy as np
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel,RidgeRegressionWithSGD, RidgeRegressionModel, LassoWithSGD, LassoModel

from pyspark import SparkContext

sc = SparkContext()

#load and parse the data
def parsePoint(line):
        values = [np.float(x) for x in line.replace(',', ' ').split(' ')]
	return LabeledPoint(values[6], values[0:6])

data = sc.textFile("/user/cloudera/hw1/train_nohead.csv")
wholedata = sc.textFile("/user/cloudera/hw1/wholedata.csv")

parsedData = data.map(parsePoint)
parsedWholeData = wholedata.map(parsePoint)

#Build the model
model = RidgeRegressionWithSGD.train(parsedData, iterations=100, step=0.1, regParam=0.01)

#Evaluate the model
valuesAndPreds = parsedWholeData.map(lambda p: (p.label, model.predict(p.features)))

RMSE = np.sqrt(
	valuesAndPreds \
	.map(lambda (v, p): (v - p)**2) \
	.reduce(lambda x, y: x + y) / valuesAndPreds.count()
	)
print("ridge regression output : \n")
print("RMSE = {0}\n".format(RMSE))

#save and load model
model.save(sc, "/user/cloudera/hw1/results/2015310884_ridge")
sameModel = RidgeRegressionModel.load(sc, "/user/cloudera/hw1/results/2015310884_ridge")


