# 3D-OFDB (One-instance FractalDataBase)

<p align="center"> <img src="../3D-OFDB1k.png" /> <p align="center">These images are the full set of images rendered from a single 3D-OFDB-1k viewpoint.</p>



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
