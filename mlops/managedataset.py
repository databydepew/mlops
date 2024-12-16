# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from google.cloud import aiplatform
from google.cloud.aiplatform_v1beta1 import DatasetServiceClient, GcsSource, ImportDataConfig
from google.cloud import bigquery
# Add this import at the top of your file
import tensorflow_data_validation as tfdv

class Aiplatforminit:
    def __init__(self, project_id: str, location: str, staging_bucket: str):
        self.project_id = project_id
        self.location = location
        self.staging_bucket = staging_bucket

       
    def initizatlize(self):
        aiplatform.init(project=self.project_id, location=self.location)


class ManagedDataset(Aiplatforminit):
    def __init__(self, bq_table: str, bq_dataset: str, project_id: str, location: str, staging_bucket: str):
        self.bq_table = bq_table
        self.bq_dataset = bq_dataset
        self.project_id = project_id
        self.fqdn_bq = f"bq://{project_id}.{bq_dataset}.{bq_table}"
        self.dataserviceclient = DatasetServiceClient(client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"})
        self.bqclient = bigquery.Client(project=project_id)
        self.staging_bucket = staging_bucket

        super().__init__(project_id, location, staging_bucket)


# Create a Vertex AI managed dataset from the BigQuery table
    def create_versioned_dataset(self) -> str:
        DATASET_NAME = f"managed_{self.bq_table.split('.')[-1]}"
        
        # Dataset metadata
        dataset = {
            "display_name": DATASET_NAME,
            # The metadata_schema_uri is a reference to the schema that describes the structure of the dataset.
            # This URI points to a YAML file stored in Google Cloud Storage (GCS) that defines the expected format and fields
            # of the dataset for the AI Platform service. It ensures that the dataset complies with the required schema
            # for successful processing and management by the AI Platform.
            "metadata_schema_uri": "gs://google-cloud-aiplatform/schema/dataset/schema-v0.0.1.yaml",
            "metadata": {},
        }


        operation = self.dataset_client.create_dataset(
            parent=f"projects/{self.project_id}/locations/{self.location}",
            dataset=dataset,
        )
        dataset_info = operation.result()
        dataset_name = dataset_info.name

        # Import BigQuery data into the dataset
        bigquery_source = {
            "bigquery_source": {"input_uri": f"bq://{self.fqdn_bq}"}
        }
        import_configs = [
            ImportDataConfig(data_item_labels={}, bigquery_source=bigquery_source)
        ]
        self.dataset_client.import_data(name=dataset_name, import_configs=import_configs)
        
        print(f"Vertex AI Dataset created with ID: {dataset_name}")
        return dataset_name

        # [ NOTES ]
        #in the metadat field here are some useful fields

        # current_date = datetime.datetime.now().isoformat()

        # dataset = {
        #     "display_name": DATASET_NAME,
        #     "metadata_schema_uri": "gs://google-cloud-aiplatform/schema/dataset/schema-v0.0.1.yaml",
        #     "metadata": {
        #         "dataSource": {
        #             "type": "BigQuery",
        #             "tableName": self.fqdn_bq
        #         },
        #         "datasetInfo": {
        #             "purpose": "Loan refinance prediction",
        #             "features": list(self.schema.keys()),
        #             "targetVariable": "refinance"
        #         },
        #         "versionControl": {
        #             "version": "1.0",
        #             "creationDate": current_date,
        #             "lastModifiedDate": current_date
        #         },
        #         "ownership": {
        #             "team": "Data Science Team",
        #             "contact": "data-science@example.com"
        #         },
        #         "compliance": {
        #             "dataSensitivity": "Medium",
        #             "applicableStandards": ["GDPR", "CCPA"]
        #         }
        #     },
        # }

    def vdata_stats(self):
        dataset = self.dataserviceclient.get_dataset(name=self.dataset_name)
        print(f"Dataset ID: {dataset.name}")

        # Query data from BigQuery
        query = f"SELECT * FROM `{self.fqdn_bq}`"
        df = self.bqclient.query(query).to_dataframe()

        # Generate statistics using TFDV
        stats = tfdv.generate_statistics_from_dataframe(df)

        # Visualize the statistics
        tfdv.visualize_statistics(stats)

        # You can also get a summary of the statistics as a proto
        stats_proto = stats.datasets[0]

        # Print some basic statistics
        print(f"Number of examples: {stats_proto.num_examples}")
        print(f"Number of features: {len(stats_proto.features)}")

        # Print statistics for each feature
        for feature in stats_proto.features:
            print(f"\nFeature: {feature.name}")
            print(f"  Type: {feature.type}")
            if feature.HasField('num_stats'):
                print(f"  Mean: {feature.num_stats.mean}")
                print(f"  Std dev: {feature.num_stats.std_dev}")
                print(f"  Min: {feature.num_stats.min}")
                print(f"  Max: {feature.num_stats.max}")
            elif feature.HasField('string_stats'):
                print(f"  Unique: {feature.string_stats.unique}")

        return stats
    
    def get_data_as_df(self):
        query = f"SELECT * FROM `{self.fqdn_bq}`"
        return self.bqclient.query(query).to_dataframe()

    def validate_data(self):
        schema = {
            "interest_rate": "float",
            "loan_amount": "int",
            "loan_balance": "int",
            "loan_to_value_ratio": "float",
            "credit_score": "int",
            "debt_to_income_ratio": "float",
            "income": "int",
            "loan_term": "int",
            "loan_age": "int",
            "home_value": "int",
            "current_rate": "float",
            "rate_spread": "float",
            "refinance": "int"
        }
        df = self.get_data_as_df()

        # Ensure data types match the schema
        for column, dtype in schema.items():
            if dtype == "int":
                df[column] = df[column].astype(int)
            elif dtype == "float":
                df[column] = df[column].astype(float)


    def xy_split(self,test_size=0.2, target_column=''):
        # Split the data into features and target
        df = self.get_data_as_df()
        if target_column:
            X = df.drop(columns=['refinance'])
            y = df['refinance']

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        else:
            print("No target column specified. Cannot perform X-Y split.")
            #perform normal datapreprocess split
            #TODO: implement data preprocessing and feature engineering

        return X_train, X_test, y_train, y_test
