from app import mysql, session
from flask import session
import os ,datetime



class Table:
    def __init__(self, table_name, *args):
        self.table = table_name
        self.columns = "(%s)" %",".join(args)
        self.columnsList = args
    
    def logout(self):
        session.pop("email",None)
        session.pop("logged_in",None)
        session.pop("id",None)
        session.pop("name",None)
        
    def getall(self):
        cur = mysql.connection.cursor()
        result = cur.execute("SELECT * FROM %s" %self.table)
        data = cur.fetchall(); 
        return data
    
    #check if user already exists
    
    def getbooks():
        cur = mysql.connection.cursor()
        result = cur.execute("SELECT * FROM book")
        data = cur.fetchall(); 
        return data
     #get all the values from the table
    
    def getone(self, search, value):
        cur = mysql.connection.cursor()
        result = cur.execute("SELECT user_id,name,email,password FROM %s WHERE %s = \"%s\"" %(self.table, search, value))
        if result > 0: 
            data = {}; 
            data = cur.fetchone()
            cur.close(); 
            return data
        
    def getone1(self, search, value):
        cur = mysql.connection.cursor()
        result = cur.execute("SELECT id,name,email,password FROM %s WHERE %s = \"%s\"" %(self.table, search, value))
        if result > 0: 
            data = {}; 
            data = cur.fetchone()
            cur.close(); 
            return data
        
    def insert(self, *args):
        data = ""
        for arg in args: #convert data into string mysql format
            data += "\"%s\"," %(arg)
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO %s%s VALUES(%s)" %(self.table, self.columns, data[:len(data)-1]))
        mysql.connection.commit()
        cur.close()
        
    def delete(self, book_id):
        cur = mysql.connection.cursor()
        cur.execute("DELETE FROM %s WHERE %s = \"%s\"" %(self.table, "book_id", book_id))
        mysql.connection.commit()
        cur.close()
        
    def delete1(self, book1_id):
        cur = mysql.connection.cursor()
        cur.execute("DELETE FROM %s WHERE %s = \"%s\"" %(self.table, "book1_id", book1_id))
        mysql.connection.commit()
        cur.close()
        
    def delete_title(self, title):
        cur = mysql.connection.cursor()
        cur.execute("DELETE FROM %s WHERE %s = \"%s\"" %(self.table, "title", title))
        mysql.connection.commit()
        cur.close()
    
    def delete_book(self, user_id,book_id):
        cur = mysql.connection.cursor()
        cur.execute("DELETE FROM %s WHERE %s = \"%s\" AND %s = \"%s\"" %(self.table, "book_id", book_id, "user_id", user_id))
        mysql.connection.commit()
        cur.close()
        
    def delete_book1(self, user_id,book1_id):
        cur = mysql.connection.cursor()
        cur.execute("DELETE FROM %s WHERE %s = \"%s\" AND %s = \"%s\"" %(self.table, "book1_id", book1_id, "user_id", user_id))
        mysql.connection.commit()
        cur.close()
        
    def update(self, book_id):
        cur = mysql.connection.cursor()
        cur.execute("UPDATE %s SET likes_count = likes_count+1 WHERE %s = \"%s\"" %(self.table, "book_id", book_id))
        mysql.connection.commit()
        cur.close()
        
    def update14(self, book_id):
        cur = mysql.connection.cursor()
        cur.execute("UPDATE %s SET likes_count = likes_count+1 WHERE %s = \"%s\"" %(self.table, "bk_id", book_id))
        mysql.connection.commit()
        cur.close()
        
    def update27(self, book1_id):
        cur = mysql.connection.cursor()
        cur.execute("UPDATE %s SET likes_count = likes_count+1 WHERE %s = \"%s\"" %(self.table, "book1_id", book1_id))
        mysql.connection.commit()
        cur.close()
        
    def update_title(self, id, title):
        cur = mysql.connection.cursor()
        cur.execute("UPDATE %s SET title = \"%s\" WHERE %s = \"%s\"" %(self.table,title, "id", id))
        mysql.connection.commit()
        cur.close()
        
    def updatelike(self, review_id):
        cur = mysql.connection.cursor()
        cur.execute("UPDATE %s SET likes = likes+1 WHERE %s = \"%s\"" %(self.table, "review_id", review_id))
        mysql.connection.commit()
        cur.close()
        
    #execute mysql code from python
    def sql_raw(execution):
        cur = mysql.connection.cursor()
        cur.execute(execution)
        mysql.connection.commit()
        cur.close()

        