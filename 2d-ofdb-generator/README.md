# 2D-OFDB (One-instance FractalDataBase)
<p align="center"> <img src="../OFDB1k.png" /> <p align="center">These images show the full set of 2D-OFDB-1k</p>
<!-- ![2D-OFDB-1k](../OFDB1k.png "2D-OFDB-1k") -->

## Summary
このレポジトリでは，あなたは，2D-OFDBのデータセット構築ができる．2D-OFDBを作成するには，Fractal Category Searchをして，カテゴリとして決定されたパラメータを使って画像のレンダリングを行います．
<!-- The 
The repository contains a Fractal Category Search, FractalDB Construction, Pre-training, and Fine-tuning in Python/PyTorch. -->



## Requirements

* Python 3.x (worked at 3.7)
* Pytorch 1.x (worked at 1.4)
* CUDA (worked at 10.1)
* CuDNN (worked at 7.6)
* Graphic board (worked at single/four NVIDIA V100)


## Execution file

We prepared execution files ```exe.sh``` and ```exe_parallel.sh``` in the top directory. The execution file contains our recommended parameters. Please type the following commands on your environment. You can execute the Fractal Category Search, FractalDB Construction, Pre-training, and Fine-tuning.

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

## FractalDB Construction

Run the code ```fractal_renderer/make_fractaldb.py``` to construct FractalDB.

```bash
python fractal_renderer/make_fractaldb.py
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
--weight_csv: Weight parameter. You can find "./fractal_renderer/weights"
--instance: #instance. 10 -> 1000 instances per category, 100 -> 10,000 instances per category')
```


The structure of rendered FractalDB is constructed as follows.

```misc
./
  data/
    FractalDB-1000/
      00000/
        00000_00_count_0_flip0.png
        00000_00_count_0_flip1.png
        00000_00_count_0_flip2.png
        00000_00_count_0_flip3.png
        ...
      00001/
        00001_00_count_0_flip0.png
        00001_00_count_0_flip1.png
        00001_00_count_0_flip2.png
        00001_00_count_0_flip3.png
        ...
  ...
```

