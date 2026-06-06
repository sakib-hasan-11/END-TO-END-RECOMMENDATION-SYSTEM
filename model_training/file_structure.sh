#!/bin/bash

mkdir -p src/{configs,common,data,models,training,inference}

touch src/run.py

touch src/configs/base.py
touch src/configs/paths.py
touch src/configs/training.py

touch src/common/logger.py
touch src/common/s3_reader.py
touch src/common/utils.py

touch src/data/loaders.py
touch src/data/lookup_builder.py
touch src/data/tf_dataset.py

touch src/models/two_tower.py
touch src/models/ranking.py
touch src/models/layers.py

touch src/training/train_two_tower.py
touch src/training/train_ranking.py
touch src/training/mlflow_tracker.py

touch src/inference/embedding_generator.py
touch src/inference/candidate_retriever.py

find src -type d -exec touch {}/__init__.py \;