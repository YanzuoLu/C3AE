# [C3AE]( https://arxiv.org/abs/1904.05059 )

This is a unofficial keras implements of c3ae for age estimation. welcome to discuss ~ 

## structs
   - assets 
   - dataset (you`d better put dataset into this dir.)
   - detect (MTCNN and align)
   - download.sh (bash script of downloading dataset)
   - model (pretrain model will be here)
   - nets (all tainging code)
       - C3AE.py 
   - preproccessing (preprocess dataset)
   - tools (todo)

## Pretain mode(to do)
   >> to do

## required enviroments:
   numpy, tensorflow(1.8), pandas, feather, opencv, python=2.7
   
   >>> pip install -r requirements.txt

##  Preparation
*download*  imdb/wiki dataset and then *extract* those data to the "./dataset/" \
 [download wiki]( https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar) 
 [download imdb]( https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar)
 

## Preprocess:
    >>>  python preproccessing/dataset_proc.py -i ./dataset/wiki_crop --source wiki
    >>>  python preproccessing/dataset_proc.py -i ./dataset/imdb_crop --source imdb

## training: 
    >>> python C3AE.py -gpu -p c3ae_v16.h5 -s c3ae_v16.h5 --source wiki -w 10


## DETECT: 
   [mtcnn] (https://github.com/YYuanAnyVision/mxnet_mtcnn_face_detection):  detect\align\random erasing \
   ![trible box](https://raw.githubusercontent.com/StevenBanama/C3AE/master/assets/triple_boundbox.png)


### origin==paper
-------------------------

|source|dataset|MAE|
| -- | :--: | :--: |
| from papper | wiki | 6.57 |
| from papper | imdb| 6.44 |

### our == Exploring (to do)

|source|dataset|MAE|
| :--: | :--: | :--: |
| v84 | imdb-wiki| 7.1(without pretrain， -_-||) |


## Questions: 
   - only 10 bins in paper: why we got 12 category: we can split it as "[0, 10, ... 110 ]" by two points!\
   - Conv5 1 * 1 * 32, has 1056 params, which mean 32 * 32 + 32. It contains a conv(1 * 1 * 32) with bias 
   - feat: change [4 * 4 * 32] to [12] with 6156 params.As far as known, it may be compose of  conv(6144+12) ,pooling and softmax.
![params](https://raw.githubusercontent.com/StevenBanama/C3AE/master/assets/params.png)

# puzzlement:
   
