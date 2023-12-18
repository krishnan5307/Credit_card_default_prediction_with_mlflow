from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.query import dict_factory
import pandas as pd



class configuration():

    
    def __init__(self):
        self.cluster = Cluster
        self.PlainTextAuthProvider = PlainTextAuthProvider
        self.dict_factory = dict_factory
  

    def get_configuration(self) -> pd.DataFrame:

        try:
            
            
            print("Connecting to Cassandra Database: credit_card_data")
            cloud_config =  {
                ##'secure_connect_bundle': '<</PATH/TO/>>secure-connect-health-insurance-premium-prediction.zip'
                'secure_connect_bundle': 'secure_bundle\secure-connect-credit-card-data.zip'    
         
            }   
    
            auth_provider = self.PlainTextAuthProvider('qeTfkUKhQWNfGsGBwRTbzAfQ', 'dv2Xf,+7zZuRstzPm42eDrOOZMMw_MZ+MsmtIUrY_fN+Q4H7PW.iqhjitnlM+,xo5kbohu1XZb514MMPXFWYkT-0WiUBhQY-qRjwNqJueb6,hMC5hk5mbUWd.nMaUkgQ')
            cluster = self.cluster(cloud=cloud_config, auth_provider= auth_provider)
            session = cluster.connect()
            session.row_factory = dict_factory

               
            data= pd.DataFrame()
            ##sql_test="SELECT * FROM insurance.insurance LIMIT 300"
            ##se = session.execute(sql_test)
            ##print(se)

            sql_query = "SELECT * FROM credit_card.credit_card"
            for row in session.execute(sql_query):
                data = data.append(pd.DataFrame(row, index=[0]))
            ##    data = pd.concat(pd.DataFrame(row, index=[0]))
            data = data.reset_index(drop=True).fillna(pd.np.nan)    
            data.to_csv("dataset/dataset.csv",mode="w", index=False,header=True)
            session.shutdown()
            return data

            
            
            
            
            
        except Exception as e:
            print(e)  

    # def run(self):    ## calling the thread
    #     try:
    #         self.get_configuration()
    #     except Exception as e:
    #         raise e          