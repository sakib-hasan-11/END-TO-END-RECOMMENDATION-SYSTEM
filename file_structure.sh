# mkdir src
# mkdir notebooks
# mkdir .github
# mkdir .github/workflows
# mkdir ETL
# touch .env


# mkdir ETL/pipeline
# touch ETL/pipeline/pipeline.py
# mkdir ETL/jobs
# touch ETL/jobs/__init__.py
# mkdir ETL/readers
# touch ETL/readers/__init__.py
# touch ETL/readers/s3_reader.py
# mkdir ETL/transformation
# touch ETL/transformation/__init__.py
# mkdir ETL/utils
# touch ETL/utils/__init__.py
# touch ETL/utils/spark_session.py
# mkdir ETL/writers
# touch ETL/writers/__init__.py
# touch ETL/writers/s3_writer.py
# echo "all files are created"

# aws s3 cp s3://recommendation-system-1149/raw-data/sample_data/articles_sample.parquet ./sample_data
# echo "articles data downloaded"
# aws s3 cp s3://recommendation-system-1149/raw-data/sample_data/customers_sample.parquet ./sample_data
# echo "customer data downloaded"
# aws s3 cp s3://recommendation-system-1149/raw-data/sample_data/transactions_sample.parquet ./sample_data
# echo "transactions data downloaded"
