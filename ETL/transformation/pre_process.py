from pyspark.sql import functions as F

from logs import logger


def preprocess(customers, articles, transactions, test: bool = False):
    logger.info("Starting preprocessing")

    try:
        if not transactions or transactions.count() == 0:
            raise ValueError("Transactions dataframe is empty")
        if not customers or customers.count() == 0:
            raise ValueError("Customers dataframe is empty")
        if not articles or articles.count() == 0:
            raise ValueError("Articles dataframe is empty")

        transactions = transactions.withColumn(
            "t_dat", F.to_timestamp("t_dat")
        ).withColumn("article_id", F.col("article_id").cast("string"))

        transactions = transactions.dropDuplicates(
            ["customer_id", "article_id", "t_dat"]
        )
        logger.info(f"Transactions: {transactions.count()} records")

        transactions = transactions.withColumn(
            "event_timestamp", F.to_timestamp("t_dat")
        ).withColumn("event_timestamp", F.col("event_timestamp").cast("timestamp"))

        transactions = transactions.filter(F.col("price") > 0)
        transactions = transactions.filter(F.col("customer_id").isNotNull()).filter(
            F.col("article_id").isNotNull()
        )

        for col_name, dtype in transactions.dtypes:
            if dtype == "string":
                transactions = transactions.withColumn(
                    col_name, F.trim(F.lower(F.col(col_name)))
                )

        numeric_cols = [
            c for c, t in customers.dtypes if t in ["int", "bigint", "double", "float"]
        ]
        for col_name in numeric_cols:
            median = customers.approxQuantile(col_name, [0.5], 0)[0]
            customers = customers.fillna({col_name: median})

        customers = customers.fillna(
            {"fashion_news_frequency": "UNKNOWN", "club_member_status": "UNKNOWN"}
        )

        for col_name, dtype in customers.dtypes:
            if dtype == "string":
                customers = customers.withColumn(
                    col_name, F.trim(F.lower(F.col(col_name)))
                )

        string_cols = [c for c, t in customers.dtypes if t == "string"]
        customers = customers.fillna("unknown", subset=string_cols)
        logger.info(f"Customers: {customers.count()} records")

        articles = articles.withColumn(
            "detail_desc", F.lower(F.trim(F.col("detail_desc")))
        )

        for col_name, dtype in articles.dtypes:
            if dtype == "string":
                articles = articles.withColumn(
                    col_name, F.trim(F.lower(F.col(col_name)))
                )

        string_cols = [c for c, t in articles.dtypes if t == "string"]
        articles = articles.fillna("unknown", subset=string_cols)

        numeric_cols = [
            c for c, t in articles.dtypes if t in ["int", "bigint", "double", "float"]
        ]
        for col_name in numeric_cols:
            median = articles.approxQuantile(col_name, [0.5], 0)[0]
            articles = articles.fillna({col_name: median})

        logger.info(f"Articles: {articles.count()} records")
        logger.info("Preprocessing completed")

        return customers, articles, transactions

    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise
