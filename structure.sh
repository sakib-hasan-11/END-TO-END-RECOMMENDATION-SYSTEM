#!/bin/bash

set -e

PROJECT_ROOT="model_features"

mkdir -p ${PROJECT_ROOT}

# Configs

mkdir -p ${PROJECT_ROOT}/src/configs

touch ${PROJECT_ROOT}/src/configs/__init__.py
touch ${PROJECT_ROOT}/src/configs/base.py
touch ${PROJECT_ROOT}/src/configs/test.py
touch ${PROJECT_ROOT}/src/configs/prod.py

# Jobs

mkdir -p ${PROJECT_ROOT}/src/jobs

touch ${PROJECT_ROOT}/src/jobs/__init__.py
touch ${PROJECT_ROOT}/src/jobs/split_dataset_job.py
touch ${PROJECT_ROOT}/src/jobs/build_user_features_job.py
touch ${PROJECT_ROOT}/src/jobs/build_item_features_job.py
touch ${PROJECT_ROOT}/src/jobs/build_ranking_features_job.py

# Feature Engineering

mkdir -p ${PROJECT_ROOT}/src/feature_engineering

touch ${PROJECT_ROOT}/src/feature_engineering/__init__.py
touch ${PROJECT_ROOT}/src/feature_engineering/splitter.py
touch ${PROJECT_ROOT}/src/feature_engineering/user_features.py
touch ${PROJECT_ROOT}/src/feature_engineering/item_features.py
touch ${PROJECT_ROOT}/src/feature_engineering/ranking_features.py

# Storage

mkdir -p ${PROJECT_ROOT}/src/storage

touch ${PROJECT_ROOT}/src/storage/__init__.py
touch ${PROJECT_ROOT}/src/storage/s3_reader.py
touch ${PROJECT_ROOT}/src/storage/s3_writer.py
touch ${PROJECT_ROOT}/src/storage/artifact_manager.py

# Common

mkdir -p ${PROJECT_ROOT}/src/common

touch ${PROJECT_ROOT}/src/common/__init__.py
touch ${PROJECT_ROOT}/src/common/constants.py
touch ${PROJECT_ROOT}/src/common/schemas.py
touch ${PROJECT_ROOT}/src/common/logger.py

# Tests

mkdir -p ${PROJECT_ROOT}/tests

touch ${PROJECT_ROOT}/tests/__init__.py

# Root Files

touch ${PROJECT_ROOT}/src/run.py
touch ${PROJECT_ROOT}/requirements.txt
touch ${PROJECT_ROOT}/README.md

echo "Project structure created successfully."
