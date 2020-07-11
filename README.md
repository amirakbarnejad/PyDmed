# PyDmed

The loading speed of hard drives is well below the processing speed of modern GPUs.
To handle this issue, pytorch provides multi-process dataloader where different processes simultaneously provide GPU(s) with data 
(for details, please refer to  [pytorch documentation](https://pytorch.org/docs/stable/data.html). This approach is quite effective for conventional datasets.
However, when it comes to medical imaging datasets with large instances, the above approach will not be effective anymore.

For instance, consider the following case: we have a dataset containing 500 whole-slide-images
[whole-slide-images](https://en.wikipedia.org/wiki/Digital_pathology)
(WSIs) each of which are approximately 100000x100000. 
In each iteration, we want the dataloader to do:
1. randomly select one of those huge images (i.e., WSIs).
2. crop and return a random 224x224 patch from the huge image.

In an mediocre machine, the second operation takes 30-40 mili seconds.
Therefoer, a naive dataloader can provide only 20-30 patches per second to GPU(s) which makes the processing speed of GPU(s) useless. 

***TODO:packagename can solve this issue.*** 

# How It Works?
The following two classes are pretty much the whole API of TODO:packagename.
1. `BigChunk`: a relatively big chunk from a patient. It can be, e.g., a 5000x5000 patch from a huge whole-slide-image. 
2. `SmallChunk`: a small data chunk collected from a big chunk. It can be, e.g., a 224x224 patch cropped from a 5000x5000 big chunk.

The below figure illustrates the idea of TODO:packagename. As long as some `BigChunk`s are loaded into RAM, we can quickly collect some `SmallChunk`s to pass to GPU.
As illustrated below, `BigChunk`s are loaded from disk time to time.  
![Alt Text](howitworks.gif)
