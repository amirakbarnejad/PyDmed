
[![button](button_quickstart2.png)](https://amirakbarnejad.github.io/Tutorial/tutorial_section1.html)

# PyDmed (Python Dataloader for Medical Imaging)

*** Sample notebooks are available in the folder [sample notebooks](https://https://github.com/amirakbarnejad/PyDmed/tree/master/sample_notebooks). *** 

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
