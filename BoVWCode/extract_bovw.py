from BoVW.ir.bagofvisualwords import BagOfVisualWords
from BoVW.indexer.bovwindexer import BOVWIndexer
import argparse
import pickle
import h5py
 
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--features-db", required=True,
	help="Path the features database")
ap.add_argument("-c", "--codebook", required=True,
	help="Path to the codebook")
ap.add_argument("-b", "--bovw-db", required=True,
	help="Path to where the bag-of-visual-words database will be stored")
ap.add_argument("-d", "--idf", required=True,
	help="Path to inverse document frequency counts will be stored")
ap.add_argument("-s", "--max-buffer-size", type=int, default=500,
	help="Maximum buffer size for # of features to be stored in memory")
args = vars(ap.parse_args())

vocab = pickle.loads(open(args["codebook"], "rb").read())
bovw = BagOfVisualWords(vocab)
 
# open the features database and initialize the bag-of-visual-words indexer
featuresDB = h5py.File(args["features_db"], mode="r")
bi = BOVWIndexer(bovw.codebook.shape[0], args["bovw_db"],
	estNumImages=featuresDB["image_ids"].shape[0],
	maxBufferSize=args["max_buffer_size"])

# loop over the image IDs and index
for (i, (imageID, offset)) in enumerate(zip(featuresDB["image_ids"], featuresDB["index"])):
	# check to see if progress should be displayed
	if i > 0 and i % 10 == 0:
		bi._debug("processed {} images".format(i), msgType="[PROGRESS]")
 
	# extract the feature vectors for the current image using the starting and
	# ending offsets (while ignoring the keypoints) and then quantize the
	# features to construct the bag-of-visual-words histogram
	features = featuresDB["features"][offset[0]:offset[1]][:, 2:]
	try:
		hist = bovw.describe(features)
	except:
		print("Faulty image: "+imageID)
		continue
 
	# add the bag-of-visual-words to the index
	bi.add(hist)
 
# close the features database and finish the indexing process
featuresDB.close()
bi.finish()
 
# dump the inverse document frequency counts to file
f = open(args["idf"], "wb")
f.write(pickle.dumps(bi.df(method="idf")))
f.close()