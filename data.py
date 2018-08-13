import psycopg2
from psycopg2.extras import DictCursor
import os 


class Data:

    def __init__(self):
        self.connection_string = os.environ.get('PSQL_CONNECTION_STRING', False)
        self.conn = psycopg2.connect(self.connection_string)
        self.cursor = self.conn.cursor(cursor_factory=DictCursor)

    def get_live_products(self):
        query = """
        select product_id,
               Image as fname             
        from product_productimage pi
        inner join product_product p
        on p.id = pi.product_id
        where sort_order = 0
        and image not like '%na_image%'
        """
        self.cursor.execute(query)
        result = list()
        for row in self.cursor:
            result.append({'id':row[0], 'fname':row[1]})
        return result

    def clear_similarity_table(self):
        cursor = self.conn.cursor()
        query = """
        truncate product_productsimilarity;
        """
        cursor.execute(query)
        self.conn.commit()

    def save_similarities(self, distances):
        cursor = self.conn.cursor()
        insert_query = 'insert into \
                        product_productsimilarity \
                        (product_id, similar_products) values %s'
        psycopg2.extras.execute_values (
            cursor, 
            insert_query, 
            distances, 
            template=None, 
            page_size=100
        )
        self.conn.commit()

    def close_connections(self):
        self.conn.close()