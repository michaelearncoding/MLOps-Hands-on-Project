import pandas as pd
import numpy as np
import logging
from pyspark.sql import DataFrame as SparkDataFrame

class DataValidator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def check_missing_values(self, df, threshold=0.1):
        """
        Check if the percentage of missing values exceeds the threshold
        
        Args:
            df: Pandas DataFrame or PySpark DataFrame
            threshold: Maximum allowed missing percentage (default 10%)
            
        Returns:
            Dictionary with column names and pass/fail status
        """
        results = {}
        
        if isinstance(df, SparkDataFrame):
            # PySpark DataFrame
            total_rows = df.count()
            for col_name in df.columns:
                missing_count = df.filter(df[col_name].isNull()).count()
                missing_percentage = missing_count / total_rows
                results[col_name] = {
                    "missing_percentage": missing_percentage,
                    "passed": missing_percentage <= threshold
                }
        else:
            # Pandas DataFrame
            missing_percentage = df.isna().mean()
            for col_name, percentage in missing_percentage.items():
                results[col_name] = {
                    "missing_percentage": percentage,
                    "passed": percentage <= threshold
                }
                
        self.logger.info(f"Missing value check results: {results}")
        return results
        
    def check_data_types(self, df, expected_types):
        """
        Verify data types match expected types
        
        Args:
            df: Pandas DataFrame or PySpark DataFrame
            expected_types: Dictionary mapping column names to expected types
            
        Returns:
            Dictionary with column validation results
        """
        results = {}
        
        if isinstance(df, SparkDataFrame):
            # PySpark DataFrame
            actual_types = {f.name: f.dataType.simpleString() for f in df.schema.fields}
            for col_name, expected_type in expected_types.items():
                if col_name in actual_types:
                    actual_type = actual_types[col_name]
                    results[col_name] = {
                        "expected": expected_type,
                        "actual": actual_type,
                        "passed": expected_type in actual_type  # Simple substring check
                    }
                else:
                    results[col_name] = {
                        "error": f"Column {col_name} not found in the dataset"
                    }
        else:
            # Pandas DataFrame
            for col_name, expected_type in expected_types.items():
                if col_name in df.columns:
                    actual_type = df[col_name].dtype
                    # Simple check - convert both to strings and do a partial match
                    passed = expected_type.lower() in str(actual_type).lower()
                    results[col_name] = {
                        "expected": expected_type,
                        "actual": str(actual_type),
                        "passed": passed
                    }
                else:
                    results[col_name] = {
                        "error": f"Column {col_name} not found in the dataset"
                    }
                    
        self.logger.info(f"Data type check results: {results}")
        return results
        
    def check_value_range(self, df, column_ranges):
        """
        Check if values in columns fall within expected ranges
        
        Args:
            df: Pandas DataFrame or PySpark DataFrame
            column_ranges: Dictionary mapping column names to (min, max) tuples
            
        Returns:
            Dictionary with column validation results
        """
        results = {}
        
        if isinstance(df, SparkDataFrame):
            # PySpark DataFrame
            for col_name, (min_val, max_val) in column_ranges.items():
                if col_name in df.columns:
                    stats = df.select(col_name).summary("min", "max").collect()
                    actual_min = float(stats[0][1])
                    actual_max = float(stats[1][1])
                    
                    results[col_name] = {
                        "expected_range": (min_val, max_val),
                        "actual_range": (actual_min, actual_max),
                        "passed": actual_min >= min_val and actual_max <= max_val
                    }
                else:
                    results[col_name] = {
                        "error": f"Column {col_name} not found in the dataset"
                    }
        else:
            # Pandas DataFrame
            for col_name, (min_val, max_val) in column_ranges.items():
                if col_name in df.columns:
                    actual_min = df[col_name].min()
                    actual_max = df[col_name].max()
                    
                    results[col_name] = {
                        "expected_range": (min_val, max_val),
                        "actual_range": (actual_min, actual_max),
                        "passed": actual_min >= min_val and actual_max <= max_val
                    }
                else:
                    results[col_name] = {
                        "error": f"Column {col_name} not found in the dataset"
                    }
                    
        self.logger.info(f"Value range check results: {results}")
        return results

    def run_all_checks(self, df, expected_types=None, column_ranges=None, missing_threshold=0.1):
        """Run all data quality checks"""
        results = {
            "missing_values": self.check_missing_values(df, missing_threshold)
        }
        
        if expected_types:
            results["data_types"] = self.check_data_types(df, expected_types)
            
        if column_ranges:
            results["value_ranges"] = self.check_value_range(df, column_ranges)
            
        # Calculate overall pass/fail
        all_checks_passed = all(
            all(check["passed"] for check in category.values() if "passed" in check)
            for category in results.values()
        )
        
        results["overall_passed"] = all_checks_passed
        
        return results