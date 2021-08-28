# HSS_ImageAnalysis

This is a brief overview of the files and scripts you will find here, as well as a list of potential resources regarding the tools that we reviewed in the workshop.

## Scripts

### CNN
During the workshop we trained two Convolutional Neural Networks (CNN) using [this script](https://colab.research.google.com/drive/1KFHwz8wjDdcFfsTmXfo-gwkKc-itN3MS). The code is in GoogleColab, and runs on a pre-configured virtual machine with a GPU. Thus, it does not require any software installation/compilation, and will run much faster that in machines without GPUs. 

The script covers the implementation of two CNNs. In the first one, we use MNIST data (pre-loaded in the Keras library) to identify handwritten numbers. In the second one, we import a pre-trained model (ResNet50) as our base model and retrain it to identify the level of "happiness" of multiple faces in our sample of interest. 

For the first example, you do not need to import or get data from external sources. The <tt>mnist.load_data()</tt> command will automatically download it for you. However, the second example requires you to request permission to download data, run a <tt>R</tt> script to get the same sample I used for training and testing, and then uploading them to your own Google Drive (that you also have to mount before running the code).

1. Visit the website [10k US Adult Faces Database](https://www.wilmabainbridge.com/facememorability2.html), from Dr. Wilma Bainbridge, and request the password to download that data.
2. Run the <tt>split_sample.R</tt> script to generate the training and testing datasets. Make sure to correctly specify the folder containing the data, and the output folder where the new datasets will be sotred.
3. Upload the training and testing datasets to your personal Google Drive.
4. Before running the Colab notebook, make sure to mount your Google Drive, and verify that the script is correctly importing the data.

