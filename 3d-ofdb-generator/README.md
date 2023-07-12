# Multi-view Fractal DataBase (MV-FractalDB) IROS 2021

## Abstruct

Multi-view image recognition is one of the solutions in order to avoid leaving weak viewpoints in robotics applications such as object manipulation, mobile robot services, and navigation robots. For example, a mobile robot in a home must judge an object category and the posture with a given image for household chores. The paper proposes a method for automatic multi-view dataset construction based on formula-driven supervised learning (FDSL). Although a data collection and human annotation of 3D objects are de nitely labor-intensive, we simultaneously and automatically generate 3D models,multi-view images, and their training labels in the proposed multi-view dataset. In order to create a large-scale multi-view dataset, we employ fractal geometry, which is considered the background information of many objects in the real world. It is expected that this background knowledge of the real world would allow convolutional neural networks (CNN) to acquire a better represen- tation in terms of any-view image recognition. We project in a circle from the rendered 3D fractal models to construct the Multi-view Fractal DataBase (MV- FractalDB), which is then used to make a pre-trained CNN model for improving the problem of multi-view image recognition. Since the dataset construction is automatic, the use of our MV-FractalDB does not require any 3D model de nition or additional manual annotations in the pre-training phase. According to the experimental results, the MV-FractalDB pre-trained model surpasses the accuracies with self- supervised methods (e.g., SimCLR and MoCo) and is close to supervised methods (e.g., ImageNet pre- trained model) in terms of performance rates on multi-view image datasets. Also, it was con rmed that MV-FractalDB pre-trained model has better convergence speed than the ImageNet pre-trained model on ModelNet40 dataset. Moreover, we demonstrate the potential for multi-view image recognition with FDSL.

![](mvfractal.gif)

## Installation
1. Create anaconda virtual environment.
```
$ conda create -n mvfdb python=3.x -y
$ conda activate mvfdb
```

2. Install requirement modules
```
$ conda install -c conda-forge openexr-python
$ pip install -r requirements.txt
```

## Running the code

We prepared execution file MVFractalDB_render.sh in the top directory. 
The execution file contains our recommended parameters. 
Please type the following commands on your environment. 
You can execute the fractal category search, the 3D fractal model generate, and the multi-view image render, MV-FractalDB Construction.

```bash MVFractalDB_render.sh```

The folder structure is constructed as follows.

```misc
./
  MVFractalDB/
    3DIFS_param/
        ExFractalDB-{category}/
            000000.csv
            000001.csv
            ...
    3Dmodel/
        MVFractalDB-{category}/
           000000/
           000000_0000.ply
           000000_0001.ply
           ...
         ...
    images/
        MVFractalDB-{category}/
           000000/
           000000_00000_000.png
           000000_00001_001.png
           ...
         ...
```
