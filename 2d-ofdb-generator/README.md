# 2D-OFDB (One-instance FractalDataBase)
<p align="center"> <img src="../OFDB1k.png" /> <p align="center">These images show the full set of 2D-OFDB-1k</p>
<!-- ![2D-OFDB-1k](../OFDB1k.png "2D-OFDB-1k") -->

## Summary
In this repository, you can construct a dataset of 2D-OFDB. To create 2D-OFDB, perform a Fractal Category Search and render images using the parameters decided as categories.




## Requirements

* Python 3.x (worked at 3.7)

The code for generating 2D-OFDB is the same as the environment used for Pre-training.

## 2D-OFDB Construction

You can construct 2D-OFDB by running exe.sh and exe_parallel.sh. Choose which to use based on the environment you're using. The shell script file performs a Fractal Category Search and 2D-OFDB Construction.


```bash
bash exe.sh
```

For a faster generate fractal image, you shuold run the ```exe_parallel.sh``` as follows. You must adjust the thread parameter ```numof_thread=40``` in the script depending on your computational resource.

```bash
bash exe_parallel.sh
```


## Fractal Category Search

Run the code ```param_search/ifs_search.py``` to create fractal categories and their representative images. In our work, the basic parameters are ```--rate 0.2 --category 1000 --numof_point 100000```

```bash
python param_search/ifs_search.py --rate=0.2 --category=1000 --numof_point=100000  --save_dir='./data'
```

The structure of directories is constructed as follows.

```misc
./
  data/
    csv_rate20_category1000/
      00000.csv
      00001.csv
      ...
    rate20_category1000/
      00000.png
      00001.png
      ...
  param_search/
  ...
```

## 2D-OFDB Construction

Run the code ```fractal_renderer/make_2dofdb.py``` to construct 2D-OFDB.

```bash
python fractal_renderer/make_2dofdb.py
```

The code includes the following parameters.

```misc
--load_root: Category root with CSV file. You can find in "./data".
--save_root: Create the directory of FractalDB.)
--image_size_x: x-coordinate image size 
--image_size_y: y-coordinate image size
--pad_size_x: x-coordinate padding size
--pad_size_y: y-coordinate padding size
--iteration: #dot/#patch in a fractal image
--draw_type: Rendering type. You can select "{point, patch}_{gray, color}"
--weight_csv: Weight parameter. You can find "./fractal_renderer/weights")
```


The structure of rendered 2D-OFDB is constructed as follows.

```misc
./
  data/
    2d-ofdb-1000/
      00000/
        00000_00_count_0_flip0.png
        ...
      00001/
        00001_00_count_0_flip0.png
        ...
  ...
```

## Acknowledgements
The 2D-OFDB generation code is based on [FractalDB](https://github.com/hirokatsukataoka16/FractalDB-Pretrained-ResNet-PyTorch).

