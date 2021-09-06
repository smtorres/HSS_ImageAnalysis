from __future__ import print_function
from imutils.feature import FeatureDetector_create, DescriptorExtractor_create
from imutils import paths
import argparse
import imutils
import cv2
import numpy as np
import h5py
import sys
import datetime
 

########################### CLASSES AND FUNCTIONS #################################

class DetectAndDescribe:
	def __init__(self, detector, descriptor):
		# store the keypoint detector and local invariant descriptor
		self.detector = detector
		self.descriptor = descriptor

	def describe(self, image, useKpList=True):
		# detect keypoints in the image and extract local invariant descriptors
		kps = self.detector.detect(image)
		(kps, descs) = self.descriptor.compute(image, kps)
 
		# if there are no keypoints or descriptors, return None
		if len(kps) == 0:
			return (None, None)
 
		# check to see if the keypoints should be converted to a NumPy array
		if useKpList:
			kps = np.int0([kp.pt for kp in kps])
 
		# return a tuple of the keypoints and descriptors
		return (kps, descs)

class BaseIndexer(object):
	def __init__(self, dbPath, estNumImages=500, maxBufferSize=50000, dbResizeFactor=2,
		verbose=True):
		# store the database path, estimated number of images in the dataset, max
		# buffer size, the resize factor of the database and the verbosity setting
		self.dbPath = dbPath
		self.estNumImages = estNumImages
		self.maxBufferSize = maxBufferSize
		self.dbResizeFactor = dbResizeFactor
		self.verbose = verbose
 
		# initialize the indexes dictionary
		self.idxs = {}

	def _writeBuffers(self):
		pass
 
	def _writeBuffer(self, dataset, datasetName, buf, idxName, sparse=False):
		# if the buffer is a list, then compute the ending index based on
		# the lists length
		if type(buf) is list:
			end = self.idxs[idxName] + len(buf)
 
		# otherwise, assume that the buffer is a NumPy/SciPy array, so
		# compute the ending index based on the array shape
		else:
			end = self.idxs[idxName] + buf.shape[0]
 
		# check to see if the dataset needs to be resized
		if end > dataset.shape[0]:
			self._debug("triggering `{}` db resize".format(datasetName))
			self._resizeDataset(dataset, datasetName, baseSize=end)
 
		# if this is a sparse matrix, then convert the sparse matrix to a
		# dense one so it can be written to file
		if sparse:
			buf = buf.toarray()
 
		# dump the buffer to file
		self._debug("writing `{}` buffer".format(datasetName))
		dataset[self.idxs[idxName]:end] = buf

	def _resizeDataset(self, dataset, dbName, baseSize=0, finished=0):
		# grab the original size of the dataset
		origSize = dataset.shape[0]
 
		# check to see if we are finished writing rows to the dataset, and if
		# so, make the new size the current index
		if finished > 0:
			newSize = finished
 
		# otherwise, we are enlarging the dataset so calculate the new size
		# of the dataset
		else:
			newSize = baseSize * self.dbResizeFactor
 
		# determine the shape of (to be) the resized dataset
		shape = list(dataset.shape)
		shape[0] = newSize
 
		# show the old versus new size of the dataset
		dataset.resize(tuple(shape))
		self._debug("old size of `{}`: {:,}; new size: {:,}".format(dbName, origSize,
			newSize))

	def _debug(self, msg, msgType="[INFO]"):
		# check to see the message should be printed
		if self.verbose:
			print("{} {} - {}".format(msgType, msg, datetime.datetime.now()))
 
	@staticmethod
	def featureStack(array, accum=None, stackMethod=np.vstack):
		# if the accumulated array is None, initialize it
		if accum is None:
			accum = array
 
		# otherwise, stack the arrays
		else:
			accum = stackMethod([accum, array])
 
		# return the accumulated array
		return accum


class FeatureIndexer(BaseIndexer):
	def __init__(self, dbPath, estNumImages=500, maxBufferSize=50000, dbResizeFactor=2,
		verbose=True):
		# call the parent constructor
		super(FeatureIndexer, self).__init__(dbPath, estNumImages=estNumImages,
			maxBufferSize=maxBufferSize, dbResizeFactor=dbResizeFactor,
			verbose=verbose)

		# open the HDF5 database for writing and initialize the datasets within
		# the group
		self.db = h5py.File(self.dbPath, mode="w")
		self.imageIDDB = None
		self.indexDB = None
		self.featuresDB = None
 
		# initialize the image IDs buffer, index buffer, and the keypoints +
		# features buffer
		self.imageIDBuffer = []
		self.indexBuffer = []
		self.featuresBuffer = None
 
		# initialize the total number of features in the buffer along with the
		# indexes dictionary
		self.totalFeatures = 0
		self.idxs = {"index": 0, "features": 0}

	def add(self, imageID, kps, features):
		# compute the starting and ending index for the features lookup
		start = self.idxs["features"] + self.totalFeatures
		end = start + len(features)
 
		# update the image IDs buffer, features buffer, and index buffer,
		# followed by incrementing the feature count
		self.imageIDBuffer.append(imageID)
		self.featuresBuffer = BaseIndexer.featureStack(np.hstack([kps, features]),
			self.featuresBuffer)
		self.indexBuffer.append((start, end))
		self.totalFeatures += len(features)
 
		# check to see if we have reached the maximum buffer size
		if self.totalFeatures >= self.maxBufferSize:
			# if the databases have not been created yet, create them
			if None in (self.imageIDDB, self.indexDB, self.featuresDB):
				self._debug("initial buffer full")
				self._createDatasets()
 
			# write the buffers to file
			self._writeBuffers()

	def _createDatasets(self):
		# compute the average number of features extracted from the initial buffer
		# and use this number to determine the approximate number of features for
		# the entire dataset
		avgFeatures = self.totalFeatures / float(len(self.imageIDBuffer))
		approxFeatures = int(avgFeatures * self.estNumImages)
 
		# grab the feature vector size
		fvectorSize = self.featuresBuffer.shape[1]
 
		# handle h5py datatype for Python 2.7
		if sys.version_info[0] < 3:
			dt = h5py.special_dtype(vlen=unicode)
 
		# otherwise use a datatype compatible with Python 3+
		else:
			dt = h5py.special_dtype(vlen=str)
 
		# initialize the datasets
		self._debug("creating datasets...")
		self.imageIDDB = self.db.create_dataset("image_ids", (self.estNumImages,),
			maxshape=(None,), dtype=dt)
		self.indexDB = self.db.create_dataset("index", (self.estNumImages, 2),
			maxshape=(None, 2), dtype="int")
		self.featuresDB = self.db.create_dataset("features",
			(approxFeatures, fvectorSize), maxshape=(None, fvectorSize),
			dtype="float")

	def _writeBuffers(self):
		# write the buffers to disk
		self._writeBuffer(self.imageIDDB, "image_ids", self.imageIDBuffer,
			"index")
		self._writeBuffer(self.indexDB, "index", self.indexBuffer, "index")
		self._writeBuffer(self.featuresDB, "features", self.featuresBuffer,
			"features")
 
		# increment the indexes
		self.idxs["index"] += len(self.imageIDBuffer)
		self.idxs["features"] += self.totalFeatures
 
		# reset the buffers and feature counts
		self.imageIDBuffer = []
		self.indexBuffer = []
		self.featuresBuffer = None
		self.totalFeatures = 0

	def finish(self):
		# if the databases have not been initialized, then the original
		# buffers were never filled up
		if None in (self.imageIDDB, self.indexDB, self.featuresDB):
			self._debug("minimum init buffer not reached", msgType="[WARN]")
			self._createDatasets()
 
		# write any unempty buffers to file
		self._debug("writing un-empty buffers...")
		self._writeBuffers()
 
		# compact datasets
		self._debug("compacting datasets...")
		self._resizeDataset(self.imageIDDB, "image_ids", finished=self.idxs["index"])
		self._resizeDataset(self.indexDB, "index", finished=self.idxs["index"])
		self._resizeDataset(self.featuresDB, "features", finished=self.idxs["features"])
 
		# close the database
		self.db.close()

################################################################################


### The actual code!
# This is very useful to run the code from the terminal and apply it to a particular set of images without changing the code EVERY TIME you change sources
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="Path to the directory that contains the images to be indexed")
ap.add_argument("-f", "--features-db", required=True,
	help="Path to where the features database will be stored")
ap.add_argument("-a", "--approx-images", type=int, default=500,
	help="Approximate # of images in the dataset")
ap.add_argument("-t", "--threshold", type=int, default=100,
	help="Hessian threshold", required=False)
ap.add_argument("-b", "--max-buffer-size", type=int, default=50000,
	help="Maximum buffer size for # of features to be stored in memory")
args = vars(ap.parse_args())
#args = {"dataset":"/media/smtorres/UElements/Getty/WomensMarch/imgtest", 
#"features-db":"output/testfeat2.hdf5",
#"approx-images":28,
#"max-buffer-size":100000}


# Detection of keypoints and extraction of features: Initialization
if imutils.is_cv2():
	detector = cv2.FeatureDetector_create("SURF")
# otherwise detect Fast Hessian keypoints in the image for OpenCV 3+
else:
	detector = cv2.xfeatures2d.SURF_create()
	if type(args['threshold'])==int:
		detector.setHessianThreshold(args['threshold'])

descriptor = DescriptorExtractor_create("RootSIFT")
dad = DetectAndDescribe(detector, descriptor)
 
# Feature indexer that helps us to have an idea of the flow of images processed (nice for debugging purposes)
fi = FeatureIndexer(args["features_db"], estNumImages=args["approx_images"],
	maxBufferSize=args["max_buffer_size"], verbose=True)

# Detection of keypoints and extraction of features: The actual loop
for (i, imagePath) in enumerate(sorted(paths.list_images(args["dataset"]))):
	# check to see if progress should be displayed
	if i > 0 and i % 10 == 0:
		fi._debug("processed {} images".format(i), msgType="[PROGRESS]")
 
	# extract the image filename (i.e. the unique image ID) from the image
	# path, then load the image itself
	filename = imagePath[imagePath.rfind("/") + 1:]
	image = cv2.imread(imagePath)
	try:
		image = imutils.resize(image, width=320)
	except:
		print(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
	# describe the image
	(kps, descs) = dad.describe(image)
 
	# if either the keypoints or descriptors are None, then ignore the image
	if kps is None or descs is None:
		continue
 
	# index the features
	fi.add(filename, kps, descs)
 
# finish the indexing process
fi.finish()