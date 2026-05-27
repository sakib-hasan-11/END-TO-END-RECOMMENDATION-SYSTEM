from logs import logger


def join_all(customer, article, transaction):
    logger.info("Joining tables")

    try:
        joined_df = transaction.join(customer, on="customer_id", how="left").join(
            article, on="article_id", how="left"
        )

        logger.info(
            f"Join completed: {joined_df.count()} records, {len(joined_df.columns)} columns"
        )
        return joined_df

    except Exception as e:
        logger.error(f"Error during join: {str(e)}")
        raise
