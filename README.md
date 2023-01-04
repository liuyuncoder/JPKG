# JPKG
The implementation of "Jointly Learning Propagating Features on the Knowledge Graph for Movie Recommendation" (JPKG).
>Liu, Y., Miyazaki, J., Chang, Q. (2022). Jointly Learning Propagating Features on the Knowledge Graph for Movie Recommendation. In: Strauss, C., Cuzzocrea, A., Kotsis, G., Tjoa, A.M., Khalil, I. (eds) Database and Expert Systems Applications. DEXA 2022. Lecture Notes in Computer Science, vol 13426. Springer, Cham. https://doi.org/10.1007/978-3-031-12423-5_1

The attention learning module and multi-hop propagation module of JPKG achieve attention-based multi-hop propagation feature learning by recursively calculating the different contributions of neighbors on the graph. The mutual learning module of JPKG combines the entity embeddings learned from the two aforementioned modules to help provide more accurate recommendations.

It is the optimization of "Knowledge Graph Enhanced Multi-Task Learning between Reviews and Ratings for Movie Recommendation" (KMRR).
> Liu Y, Miyazaki J, Chang Q. Knowledge graph enhanced multi-task learning between reviews and ratings for movie recommendation[C]//Proceedings of the 37th ACM/SIGAPP Symposium on Applied Computing. 2022: 1882-1889.

MKR is a multi-task learning method to learn cross features from ratings and reviews by fusing users and movies with their review knowledge entities in the same graph. The fusion graph structure enables the graph-link prediction task to learn the review entity features that are relevant to target users and movies, which further assists our recommendation task.
## Environments
- Python 3.9.7
- Tensorflow 2.6.0
- numpy 
- scipy
## Files in the folder
- data/
  - amazan/
    - amazon.cvs: the raw dataset of Amazon movies & TV.
    - 10-core/: the preprocessed dataset with 10-core from Amazon movies & TV dataset.
  - imdb/
    - imdb.cvs: the raw dataset of IMDb.
    - 10-core/: the preprocessed dataset with 10-core from IMDb dataset.
- src/
  - gat_layers_batch.py: the implementation of graph attention mechanism on link prediction task.
  - id_cross_layers.py: the implementation of mutual learning module.
  - id_cross_model_batch.py: the multi-task framework.
  - load_data.py: load data.
  - main.py: main.
  - preprocess.py: the preprocess of datasets to convert the original indices of review entities, users, and items into standard indices.
  - train_batch.py: the implementation of the training procudure.
## The description of datasets
- amazon.cvs
  - user_id: user ids.
  - item_id: item ids.
  - ratings: ratings from users to items.
  - review_identifiers: review entities extracted from reviews.
- imdb.cvs
  - user_id: user ids.
  - item_id: item ids.
  - ratings: ratings from users to items.
  - review_identifiers: review entities extracted from reviews.
  - identifier_inx: the item corresponding knowledge entities.
## Running the code
- cd src
- python read_ratings.py
- python preprocess.py
- python main.py

## Reference
if you find the code or data is useful, please cite our work:
```
@inproceedings{liu2022jointly,
  title={Jointly Learning Propagating Features on the Knowledge Graph for Movie Recommendation},
  author={Liu, Yun and Miyazaki, Jun and Chang, Qiong},
  booktitle={International Conference on Database and Expert Systems Applications},
  pages={3--16},
  year={2022},
  organization={Springer}
}
```
