from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnan, count
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ETLPipeline:
    def __init__(self, app_name="ML ETL Pipeline"):
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .getOrCreate()
        logger.info("Spark session initialized")
        
    def extract(self, source_path, format="csv"):
        """Extract data from source"""
        logger.info(f"Extracting data from {source_path}")
        if format == "csv":
            return self.spark.read.option("header", "true").option("inferSchema", "true").csv(source_path)
        elif format == "parquet":
            return self.spark.read.parquet(source_path)
        elif format == "json":
            return self.spark.read.json(source_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
    def transform(self, df):
        """Transform data"""
        logger.info("Transforming data")
        
        # Data quality check - count nulls
        null_counts = {col_name: df.filter(col(col_name).isNull()).count() 
                      for col_name in df.columns}
        logger.info(f"Null counts: {null_counts}")
        
        # Data quality check - summary statistics
        df.describe().show()
        
        # Example transformation: Fill missing values
        for column in df.columns:
            if df.select(column).dtypes[0][1] == 'string':
                df = df.withColumn(column, when(col(column).isNull(), "unknown").otherwise(col(column)))
            else:
                df = df.withColumn(column, when(col(column).isNull(), 0).otherwise(col(column)))
                
        logger.info("Data transformation completed")
        return df
        
    def load(self, df, destination_path, format="parquet"):
        """Load data to destination"""
        logger.info(f"Loading data to {destination_path}")
        if format == "parquet":
            df.write.mode("overwrite").parquet(destination_path)
        elif format == "csv":
            df.write.mode("overwrite").option("header", "true").csv(destination_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        logger.info("Data loaded successfully")
            
    def run_pipeline(self, source_path, destination_path, source_format="csv", dest_format="parquet"):
        """Run the complete ETL pipeline"""
        df = self.extract(source_path, source_format)
        transformed_df = self.transform(df)
        self.load(transformed_df, destination_path, dest_format)
        logger.info("ETL pipeline completed successfully")
        return transformed_df

if __name__ == "__main__":
    pipeline = ETLPipeline()
    pipeline.run_pipeline(
        source_path="./data/raw/customer_data.csv",
        destination_path="./data/processed/customer_data_processed"
    )