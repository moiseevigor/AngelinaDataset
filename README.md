# Angelina Braille Images Dataset

This dataset consists of labeled photos of Braille texts.

It includes 212 pages of two-sided printed Braille books and 28 handwritten studet works. Also 44 non-braille photos of various documents found in Internet are included as negative examples.
Each group of files is split into train and validation sets. Appropriate image lists are stored in ```train_*.txt``` and ```val_*.txt``` files in corresponding directories:

|   | train | val | total |
| ----- | :---: | :---: | ---: |
| books | 169 | 43 | 212 |
| handwritten | 22 | 6 | 28 |
| not braille | 44 |  | 44 |

<div align=center><img width="40%" src="pics/raw.jpg?raw=true"/>&nbsp;&nbsp;<img width="40%" src="pics/labeled.jpg?raw=true"/></div> 

Label files has [LabelMe](https://github.com/wkentaro/labelme) JSON format. For two-side pages only front side is labeled.

Each Braille symbol is labeled as 
* corresponding plain text letter or symbol (mainly Russian letter)
* '\~_number_' or '\~_number_\~' where _number_ is a digital representation of the Braille symbol (for example '\~3456' for digital sign)
* some special marks like '##' for digital sign

Tools for handling this dataset can be found at [Angelina Braille Reader](https://github.com/IlyaOvodov/AngelinaReader) repository.

Correspondance between Braille symbols and correspondance letters is defined at [letters.py](https://github.com/IlyaOvodov/AngelinaReader/blob/master/braille_utils/letters.py) file.
Some special symbols can be labeled in several ways. See [labeling_synonyms](https://github.com/IlyaOvodov/AngelinaReader/blob/328b0e0d30353d76ec06cde8560a876da5c46ab7/braille_utils/label_tools.py#L67) dict.
Various tools for handling labels can be found at [label_tools.py](https://github.com/IlyaOvodov/AngelinaReader/blob/master/braille_utils/label_tools.py) file.
Reading function for this dataset is implemented in [read_LabelMe_annotation](https://github.com/IlyaOvodov/AngelinaReader/blob/328b0e0d30353d76ec06cde8560a876da5c46ab7/data_utils/data.py#L325) function.
Implementation of PyTorch Dataset is [here](https://github.com/IlyaOvodov/AngelinaReader/blob/328b0e0d30353d76ec06cde8560a876da5c46ab7/data_utils/data.py#L171).         

------------------------------

X11 

```
xhost local:root
```

run interactively 

```
docker run --rm --gpus all --ipc=host -it --env="DISPLAY" -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" -v $PWD:/app braille python /app/train.py
```

Links
- infer on retinanet https://stackabuse.com/retinanet-object-detection-with-pytorch-and-torchvision/
- train retinanet https://colab.research.google.com/github/benihime91/pytorch_retinanet/blob/master/demo.ipynb
- train retinanet https://debuggercafe.com/object-detection-using-retinanet-with-pytorch-and-deep-learning/
- pretrained retinanet https://github.com/pytorch/vision/issues/3442
- fine tunning https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
- https://towardsdatascience.com/review-retinanet-focal-loss-object-detection-38fba6afabe4

validation los of retinanet
- frcnn https://stackoverflow.com/questions/60339336/validation-loss-for-pytorch-faster-rcnn/65347721#65347721
- https://discuss.pytorch.org/t/compute-validation-loss-for-faster-rcnn/62333/11