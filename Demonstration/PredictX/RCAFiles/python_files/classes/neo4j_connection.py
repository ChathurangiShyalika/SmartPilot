from neo4j import GraphDatabase

# class to establish connection to neo4j
class Neo4jConnection:
    def __init__(self, uri, user, pwd):
        self.__uri = uri
        self.__user = user
        self.__pwd = pwd
        self.__driver = None
        try:
            self.__driver = GraphDatabase.driver(self.__uri, auth=(self.__user, self.__pwd))
        except Exception as e:
            print("Failed to create the driver:", e)
            
    def close(self):
        if self.__driver is not None:
            self.__driver.close()
            
    def query(self, query, parameters=None, db=None):
        assert self.__driver is not None, "Driver not initialized!"
        session = None
        response = None
        try: 
            session = self.__driver.session(database=db) if db is not None else self.__driver.session() 
            #response = (session.run(query, parameters))
            response = list(session.run(query, parameters))
        except Exception as e:
            print("Query failed:", e)
        finally: 
            if session is not None:
                session.close()
        
        #return pd.DataFrame([r.values() for r in response], columns=response.keys())
        return response
        
    def multi_query(self, multi_line_query, parameters=None, db=None):
        for li in multi_line_query.splitlines():
                print(li)
                result=self.query(li, parameters=None, db=None)
                print(result)