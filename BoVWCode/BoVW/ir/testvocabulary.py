# import the necessary packages
from __future__ import print_function
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise
import numpy as np
import datetime
import h5py
import pandas as pd
import pickle
 

class SampleFeatures:
	def __init__(self, dbPath, verbose=True):
		# store the database path and the verbosity setting
		self.dbPath = dbPath
		self.verbose = verbose

	def getsample(self, samplePercent, path=None, randomState=None):
		# open the database and grab the total number of features
		db = h5py.File(self.dbPath)
		totalFeatures = db["features"].shape[0]
 
		# determine the number of features to sample, generate the indexes of the
		# sample, sorting them in ascending order to speedup access time from the
		# HDF5 database
		sampleSize = int(np.ceil(samplePercent * totalFeatures))
		print("Sampling: "+str(sampleSize)+" features out of "+str(totalFeatures))
		idxs = np.random.choice(np.arange(0, totalFeatures), (sampleSize), replace=False)
		idxs.sort()
		data = []

		# loop over the randomly sampled indexes and accumulate the features to
		# cluster
		for i in idxs:
			data.append(db["features"][i][2:])

		# cluster the data

		if path!=None:
			df = pd.DataFrame({'index':idxs})
			df.to_csv(path)
		db.close()
		return(data)


class TestVocabulary:
	def __init__(self, verbose=True):
		# store the database path and the verbosity setting
		self.verbose = verbose

	def fit(self, data, numClusters_vec, store="Yes", pathtocodebook = None, randomState=None):
		# open the database and grab the total number of features
		sse = []
		for k in numClusters_vec:
			print("Running k-means with "+str(k)+" clusters ")
			clt = MiniBatchKMeans(n_clusters=k, random_state=randomState,
				init='random', init_size=int(k*2), max_iter=50000, max_no_improvement=5000,
				n_init=int(max(5,k/100)), reassignment_ratio=0.01, tol=0.001, verbose=0)
			kmeans = clt.fit(data)
			centroids = kmeans.cluster_centers_
			#print([len(centroids), len(centroids[0])])
			pred_clusters = kmeans.predict(data)
			curr_sse = 0

			for i in list(range(len(data))):
				current_center = centroids[pred_clusters[i]]
				mypoint = np.asarray(data[i]).reshape(1,-1)
				current_center = np.asarray(current_center).reshape(1,-1)
				D = pairwise.euclidean_distances(mypoint, Y=current_center)[0]
				curr_sse += D**2
			sse.append(curr_sse)
			self._debug("cluster shape: {}".format(centroids.shape))
			if store=="Yes":
				# dump the clusters to file
				tempname = pathtocodebook+"/vocab_k"+str(k)+".cpickle"
				print("[INFO] storing cluster centers...")
				f = open(tempname, "wb")
				f.write(pickle.dumps(centroids))
				f.close()
		return(sse)
 
	def _debug(self, msg, msgType="[INFO]"):
		# check to see the message should be printed
		if self.verbose:
			print("{} {} - {}".format(msgType, msg, datetime.datetime.now()))