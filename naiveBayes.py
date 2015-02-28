import math
import random
import csv

def loadData(file):
	lines = csv.reader(open(file, "rb"))
	data = list(lines)
	for i in range(len(data)):
		data[i] = [float(x) for x in data[i]]
	return data

def splitData(data, ratio):
	trainSize = int(len(data) * ratio)
	trainSet = []
	copy = list(data)

	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	
	return [trainSet, copy]

def separateByClass(data):
	separated = {}
	for i in range(len(data)):
		vector = data[i]
		if vector[-1] not in separated:
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	
	return separated

def mean(num):
	return sum(num) / float(len(num))

def stdDev(num):
	avg = mean(num)
	variance = sum([pow(x - avg, 2) for x in num])/float(len(num) - 1)
	return math.sqrt(variance)

def summarize(data):
	summaries = [(mean(attribute), stdDev(attribute)) for attribute in zip(*data)]
	#summaries = [(mean(attribute), stdDev(attribute)) for attribute in zip(*data)]
	del summaries[-1]
	return summaries

def summarizeByClass(data):
	separated = separateByClass(data)
	summaries = {}
	for classValue, instances in separated.iteritems():
		summaries[classValue] = summarize(instances)
	return summaries

#for an individual attribute
def calculateProbability(x, mean, stdDev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdDev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdDev)) * exponent

#for all the attributes belong to one instance
def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.iteritems():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdDev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdDev)
	return probabilities

#returns the largest probability of an instance belonging to a class
def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct / float(len(testSet))) * 100

filename = "pima-indians-diabetes.data.csv"
split = 0.67

dataSet = loadData(filename)
train, test = splitData(dataSet, split)
summaries = summarizeByClass(dataSet)
predictions = getPredictions(summaries, dataSet)

print "Accuracy %f" % getAccuracy(dataSet, predictions)
