from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.query import dict_factory
import pandas as pd

CASSANDRA_KEYSPACE = "credit"
CASSANDRA_TABLE = "credit_data"

class configuration():

    
    def __init__(self):
        self.cluster = Cluster
        self.PlainTextAuthProvider = PlainTextAuthProvider
        self.dict_factory = dict_factory
  

    def get_configuration(self) -> pd.DataFrame:

        try:
            
            
            print("Connecting to Cassandra Database: credit")
            cloud_config =  {
            ##'secure_connect_bundle': '<</PATH/TO/>>secure-connect-health-insurance-premium-prediction.zip'
                        'secure_connect_bundle': r'C:\\data science\\Internship projects\\credit card defaulters\\Credit_card_default_prediction_with_mlflow\secure_bundle\secure-connect-credit.zip'
            }   

            auth_provider = PlainTextAuthProvider('PhGlatcyaNvkYeTNazxKUpeo', 'c_u3tM4DUkkOE945mcdKSjouj4HvvfkfMl0r.Z7fEN17Seqgy0iREqXl,oe-D3LlNe,Bfub50q.eT0AUrvk41a2FfoDuXT8bjtwPgZGOM.EKd15Npso_1QNPdpcrWY0L')
            ## client_id and secret
            cluster = Cluster(cloud=cloud_config, auth_provider= auth_provider)
            session = cluster.connect()

            row = session.execute("select release_version from system.local").one()
            if row:
                print(row[0],"Successfully connected to cassandra database: 'credit'")
            else:
                print("An error occurred.")
            session.row_factory = dict_factory

               
            # data= pd.DataFrame()
            ##sql_test="SELECT * FROM insurance.insurance LIMIT 300"
            ##se = session.execute(sql_test)
            ##print(se)
            cql_query = "SELECT * FROM {}.{};".format(CASSANDRA_KEYSPACE, CASSANDRA_TABLE)
            # Fetch data from Cassandra
            cassandra_data = session.execute(cql_query)

            # Convert data to a Pandas DataFrame
            df = pd.DataFrame(list(cassandra_data))
            df.to_csv("dataset/dataset.csv",mode="w", index=False,header=True)
            session.shutdown()
            return df

            
            
            
            
            
        except Exception as e:
            print(e)  

    # def run(self):    ## calling the thread
    #     try:
    #         self.get_configuration()
    #     except Exception as e:
    #         raise e          