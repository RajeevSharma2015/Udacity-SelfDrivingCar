## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

This is second project of Term-1 curriculam, objective here is to build a program (neural network, data preprocessing and learning)to learn and recognize traffic signs. 


<img src="examples/output_18_0.png" width="680" alt="Combined Image" />


Overview
---
In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then try out your model on images of German traffic signs that you find on the web.

We have included an Ipython notebook that contains further instructions 
and starter code. Be sure to download the [Ipython notebook](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb). 

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting three files: 
* the Ipython notebook with the code
* the code exported as an html file
* a writeup report either as a markdown or pdf file 

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/481/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Project Writeup - Rajeev Sharma
Attached project report describes how project objectives achieved ![ Project Writeup ](https://github.com/RajeevSharma2015/Udacity-SelfDrivingCar/blob/master/P2-Term1/Traffic_Sign_Classifier.md)

## Project Outcome - Examples 
Images translated from RGB scale to grayscale for image processing, below given is a sampled depiction of same.
<img src="examples/grayscale.jpg" width="480" alt="Combined Image" />

Below shown examples are visualisation of training data (German Traffic Sign - RGB)
<img src="examples/output_18_0.png" width="580" alt="Combined Image" />

Unbalanced training samples category wise (labeled) histogram analysis. It clearly depicts data provided is unbalanced - also a graphical data summary provided.  
<img src="examples/output_19_0.png" width="580" alt="Combined Image" /> 

<img src="examples/output_20_0.png" width="580" alt="Combined Image" />


Data category with label visualisation for better understanding and analysis of training data.
<img src="examples/output_23_0.png" width="580" alt="Combined Image" />

<img src="examples/output_24_0.png" width="580" alt="Combined Image" />


Fake data generated to increase training samples and learning accuracy. A sample depicted below - how a few samples images distored to create fake data and balance learning examples.
<img src="examples/output_33_0.png" width="480" alt="Combined Image" />


Summary visualisation of balanced training data available for training a model (Neural network)
<img src="examples/output_34_1.png" width="580" alt="Combined Image" />

<img src="examples/output_39_0.png" width="580" alt="Combined Image" />


Images preprocessing - grayscaling 

<img src="examples/output_43_1.png" width="580" alt="Combined Image" />

<img src="examples/output_44_1.png" width="580" alt="Combined Image" />


Training and Testing accuracy - a visualisation of neural network learning progress
<img src="examples/output_79_1.png" width="680" alt="Combined Image" />


#### Recognized outcome of test images - a visualisation of result
<img src="examples/output_87_0.png" width="580" alt="Combined Image" />

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset and Repository

1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```

### Requirements for Submission
Follow the instructions in the `Traffic_Sign_Classifier.ipynb` notebook and write the project report using the writeup template as a guide, `writeup_template.md`. Submit the project code and writeup document.

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

