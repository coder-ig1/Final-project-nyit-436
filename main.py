import pyspark.sql as ps
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql.functions import col, when
from pyspark.sql.types import DoubleType ,IntegerType
from sklearn.ensemble import IsolationForest

# Create a SparkSession
spark = ps.SparkSession.builder.appName("CybersecurityAnalysis").getOrCreate()

# Read the CSV files using Spark
training = spark.read.csv('data/labelled_training_data.csv', header=True, inferSchema=True)
testing = spark.read.csv('data/labelled_testing_data.csv', header=True, inferSchema=True)
validation = spark.read.csv('data/labelled_validation_data.csv', header=True, inferSchema=True)
training.printSchema()
training.show(5)
# Define the function to prepare the dataset
def prepare_dataset(df: ps.DataFrame) -> ps.DataFrame:
    """
    Prepare the dataset for training.
    """
    # Drop the columns that are not needed
    for i in ["eventId", "eventName", "hostname","args","processName","stackAddresses","threadId","evil"]:
        df = df.drop(i)
    
    # Convert the label column to a numeric value
    df = df.withColumn("sus", col("sus").cast(IntegerType()))
    
    # Convert the timestamp column to a numeric value
    df = df.withColumn("timestamp", col("timestamp").cast(DoubleType()))
    
    # Apply standardization to the timestamp column
    mean_timestamp = df.agg({"timestamp": "mean"}).collect()[0][0]
    std_timestamp = df.agg({"timestamp": "std"}).collect()[0][0]
    df = df.withColumn("timestamp", (col("timestamp") - mean_timestamp) / std_timestamp)
    # Define UDFs for mapping
    df = df.withColumn("processId", when(col("processId").isin([0, 1, 2]), 0).otherwise(1))
    
    # Map parent_process_id
    df = df.withColumn("parentProcessId", when(col("parentProcessId").isin([0, 1, 2]), 0).otherwise(1))
    
    # Map user_id
    df = df.withColumn("userId", when(col("userId") < 1000, 0).otherwise(1))
    
    # Map mount_namespace
    df = df.withColumn("mountNamespace", when(col("mountNamespace") == 4026531840, 0).otherwise(1))
    
    # Map return_value
    df = df.withColumn("returnValue", when(col("returnValue") == 0, 0)
                                     .when(col("returnValue") > 0, 1)
                                     .otherwise(2))
    
    return df

# Prepare the training dataset
training = prepare_dataset(training)
testing = prepare_dataset(testing)
validation = prepare_dataset(validation)
#print rows with nan values in testing data
#convert to pandas dataframe
pandas_testing = testing.toPandas()
pandas_validation = validation.toPandas()

#calculate the percentage of evil 
#show the number of evil colunms in the pd dataset

pandas_training = training.toPandas()

# Extract the features for Isolation Forest
features = pandas_training.drop("sus", axis=1)
sus_percentage = pandas_training["sus"].value_counts()[1] / len(pandas_training)
# Train the Isolation Forest model
isolation_forest = IsolationForest(random_state=42, contamination=sus_percentage, n_jobs=-1)
#replace nan values with 0
features = features.fillna(0)
isolation_forest.fit(features)
# Predict the labels for the training dataset
def test_isolation_forest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict the labels for the training dataset.
    """
    #if contains nan replace with 0
    df = df.fillna(0)
    # Extract the features for Isolation Forest
    features = df.drop("sus", axis=1)
    
    # Predict the labels
    predictions = isolation_forest.predict(features)
    
    # Create a new column for the predictions
    return predictions

    
    return df
def convert_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    convert predictions of -1 to 1 and 1 to 0"""
    # Convert the predictions to 0 and 1
    pd.DataFrame(df)
    df = (df == -1).astype(int)

    
    
    
    # Convert the predictions to a Pandas DataFrame
    
    
    return df
def compare_predictions(true_data: pd.DataFrame,predicted_data: pd.DataFrame):
    """
    compare the predictions to the actual values"""
   
    
    # Calculate the accuracy
    accuracy = (true_data["sus"] == predicted_data).sum() / len(true_data)
    print(f"Accuracy: {accuracy}")
    
    
    return accuracy
y_pred_testing = test_isolation_forest(pandas_testing)
y_pred_validation = test_isolation_forest(pandas_validation)
print(y_pred_testing)
print(y_pred_validation)

#compare the predictions to the actual values
print(compare_predictions(pandas_testing,convert_predictions(y_pred_testing)))
#print the 1 values

# Stop the SparkSession


""" 
# Show the first 5 rows of the data
training.show(5)
testing.show(5)
validation.show(5)
#print datatype of each row
training.printSchema()
# Calculate the correlation matrix
# Convert to a Pandas DataFrame for plotting
#convert sus to a numeric value from string

training = training.withColumn("sus", training["sus"].cast("int"))
corr_matrix = training.toPandas().corr(numeric_only=True)

# Plot the correlation matrix heatmap
# use between -1 and 1 for the colorbar
plt.imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(np.arange(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(np.arange(len(corr_matrix.columns)), corr_matrix.columns)
plt.show()
 #drop event name because it is the same as event id
training = training.drop("event_name")
#drop hostname because it is just the name of the computer
training = training.drop("hostname")
#drop process id because it is highly corrilated to event thread id
training = training.drop("process_id")
 """

# Stop the SparkSession
spark.stop()
