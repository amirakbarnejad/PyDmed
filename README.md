
[![button](button_quickstart2.png)](https://amirakbarnejad.github.io/Tutorial/tutorial_section1.html)

# PyDmed (Python Dataloader for Medical Imaging)

***Sample notebooks are available in the folder [sample notebooks](https://github.com/amirakbarnejad/PyDmed/tree/master/sample_notebooks/).*** 

The loading speed of hard drives is well below the processing speed of modern GPUs.
This is problematic for machine learning algorithms, specially for medical imaging datasets with large instances.

For example, consider the following case: we have a dataset containing 500
[whole-slide-images](https://en.wikipedia.org/wiki/Digital_pathology)
(WSIs) each of which are approximately 100000x100000. 
We want the dataloader to repeatedly do the following steps:
1. randomly select one of those huge images (i.e., WSIs).
2. crop and return a random 224x224 patch from the huge image.


***PyDmed solves this issue.*** 


# How It Works?
The following two classes are pretty much the whole API of PyDmed.
1. `BigChunk`: a relatively big chunk from a patient. It can be, e.g., a 5000x5000 patch from a huge whole-slide-image. 
2. `SmallChunk`: a small data chunk collected from a big chunk. It can be, e.g., a 224x224 patch cropped from a 5000x5000 big chunk. In the below figure, `SmallChunk`s are the blue small patches. 

The below figure illustrates the idea of PyDmed. 
As long as some `BigChunk`s are loaded into RAM, we can quickly collect some `SmallChunk`s and pass them to GPU(s).
As illustrated below, `BigChunk`s are loaded/replaced from disk time to time.  
![Alt Text](howitworks.gif)

[![button](button_quickstart2.png)](https://amirakbarnejad.github.io/Tutorial/tutorial_section1.html)

# Issues
We regularly check for possible issues and update pydmed. Please check out "Issues" if you faced any problems running pydmed.
If you couldn't find your issue there, please raise the issue so we can improve pydmed. 

# Installation
PyDmed is now available as a pyton local package. To use PyDmed one needs to have the folder called `PyDmed/` (by, e.g., cloning the repo).
Afterwards, the folder has to be added to `sys.path` as done in [sample notebook 1](https://github.com/amirakbarnejad/PyDmed/blob/8ef0f6b9282815498bc50bf31a827a8a7eeb48a8/sample_notebooks/sample_1_train_classifier.ipynb)
or in [the sample colab notebook](https://colab.research.google.com/drive/1WvntL-guv9JATJQWaS_Ww32DLBwGd9Ux?usp=sharing).

