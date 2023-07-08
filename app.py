from flask import Flask, render_template, request, redirect, flash, session,url_for,jsonify,json
from flask_session import Session
from flask_mysqldb import MySQL
from datetime import date
import yaml
from functools import wraps
import base64
import json
import datetime
import jwt
import razorpay
import requests
from bs4 import BeautifulSoup as bs
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import random



popular_df = pickle.load(open('popular.pkl','rb'))
pt = pickle.load(open('pt.pkl','rb'))
books = pickle.load(open('books.pkl','rb'))
similarity_scores = pickle.load(open('similarity_scores.pkl','rb'))


app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.config["SECRET KEY"] = "tisha123"
Session(app)



# Configure db
db = yaml.full_load(open('db.yaml'))
app.config['MYSQL_HOST'] = db['mysql_host']
app.config['MYSQL_USER'] = db['mysql_user']
app.config['MYSQL_PASSWORD'] = db['mysql_password']
app.config['MYSQL_DB'] = db['mysql_db']

mysql = MySQL(app)
from models import Table

def random_book(mood):
    # Step 1: Load dataset
    books_df = pd.read_csv('books_with_moods1.csv')

    # Step 2: Preprocess text data
    books_df['description'] = books_df['description'].fillna('')
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(books_df['description'])

    # Step 3: Split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, books_df['mood'], test_size=0.2, random_state=42)

    # Step 4: Train mood classification model on training set
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # Step 5: Evaluate model on test set
    y_pred = clf.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(f'Precision: {precision_score(y_test, y_pred, average="weighted", zero_division=1)}')
    print(f'Recall: {recall_score(y_test, y_pred, average="weighted", zero_division=1)}')
    print(f'F1 score: {f1_score(y_test, y_pred, average="weighted", zero_division=1)}')
    # Step 6: Get user input
    user_mood = mood
    # Step 7: Recommend books based on mood
    user_tfidf = tfidf.transform([user_mood])
    user_pred = clf.predict(user_tfidf)[0]
    user_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    books_df['similarity'] = user_similarities
    recommended_books = books_df.loc[books_df['mood'] == user_pred].sort_values(by=['similarity'], ascending=False).head(20)
    print(f'Here are the top 10 books recommended for {user_mood} mood:')
    print(recommended_books[['title']])
    if len(recommended_books) == 0:
        return('Sorry, there are no books recommended')
    else:
        # Choose a random book from the recommended books
        random_book = recommended_books.sample(n=1)

        # Print the title and author of the random book
        print(f'Here is a random book recommended for {user_mood} mood:')
        return(random_book["title"].iloc[0])
        

def scrape_and_run(genre):
    print(genre)
    page = requests.get("https://www.goodreads.com/shelf/show/" + genre)
    soup = bs(page.content, 'html.parser')
    titles = soup.find_all('a', class_='bookTitle')
    authors = soup.find_all('a', class_='authorName')
    ratings = soup.find_all('span', class_='greyText smallText')
    cur = mysql.connection.cursor()
    cur.execute("TRUNCATE TABLE %s"%(genre))
    for title, author,ratings in zip(titles, authors,ratings): 
        ratings = ratings.get_text()
        ratings = ratings.strip()
        ratings = ratings.split(" ")
        link="https://www.goodreads.com" + title['href']
        users = Table(genre, "title", "author", "ratings","link")
        users.insert(title.get_text(),author.get_text(),float(ratings[2]),link)
        
       

def check_for_token(func):
    @wraps(func)
    def wrap(*args, **kwargs):
        token = request.args.get('token')
        if not token:
            return jsonify({'message':'Missing token'}), 403
        try:
            data= jwt.decode(token, app.config["SECRET KEY"])
        except:
            return jsonify({'message':'Invalid token'}), 403
        return func(*args, **kwargs)
    return wrap


def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session and session['logged_in']:
            return f(*args, **kwargs)
        else:
            flash("You need to login first","danger")
            return redirect('/login')
    return wrap

def log_in_user(email):
        users = Table("users", "name", "phone", "email", "password")
        user = users.getone("email", email)
        session['logged_in'] = True
        session['id'] = user[0]
        session['name'] = user[1]
        session['email'] = user[2]
        token = jwt.encode({'user': email, 'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=60)}, app.config["SECRET KEY"])
        print(token)
        
def log_in_user1(email):
        users = Table("admin", "name", "phone", "email", "password")
        user = users.getone1("email", email)
        session['logged_in'] = True
        session['id'] = user[0]
        session['name'] = user[1]
        session['email'] = user[2]
        
def searchb(book):
    cur = mysql.connection.cursor()
    resultValue = cur.execute("SELECT * FROM %s WHERE %s = \"%s\"" %("book", "title", book))
    if resultValue > 0:
        userDetails1 = cur.fetchall()
        return userDetails1
        
def recommend(title):   
    index = np.where(pt.index == title)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:5]
    data = [] 
    for i in similar_items:
        item = []
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
        data.append(item)
    print(data)
    return(data)


@app.route('/',methods=['GET', 'POST'])
def index():
    data1=[]
    if 'id' in session:
        cur = mysql.connection.cursor()
        id=session['id']
        cur = mysql.connection.cursor()
        resultValue = cur.execute("SELECT * FROM %s WHERE %s = \"%s\"" %("wishlist", "user_id", id))
        if resultValue > 0:
            userDetails = cur.fetchall()
            for user in userDetails:
                book_id=user[2]
                cur = mysql.connection.cursor()
                resultValue1 = cur.execute("SELECT title FROM %s WHERE %s = \"%s\"" %("book", "book_id", book_id))
                if resultValue1 > 0:
                    userDetails1 = cur.fetchall()
                    title=userDetails1[0][0] 
                    data=recommend(title)
                data1.extend(data)

    cur = mysql.connection.cursor()
    resultValue1 = cur.execute("SELECT name,feedback,date_of_feedback FROM feedback")
    if resultValue1 > 0:
        userDetails1 = cur.fetchall()
    if request.method == 'POST':
        if request.form["section"] == "search":
            userDetails2 = request.form
            book=userDetails2["searchbook"]
            return redirect(url_for('search', name=book)) 
        else:
            userDetails = request.form
            book_id = userDetails['section']
            return redirect(url_for('viewmore', bookid=book_id))
    return render_template('index.html',context1=userDetails1, book_name = list(popular_df['Book-Title'].values),
                           author=list(popular_df['Book-Author'].values),
                           image=list(popular_df['Image-URL-M'].values),
                           votes=list(popular_df['num_ratings'].values),
                           rating=list(popular_df['avg_rating'].values),data=data1,x=5)
    

    
@app.route('/register', methods=['GET', 'POST'])
def register():
    users = Table("users", "name", "phone", "email", "password")
    users.logout()
    if request.method == 'POST':
        userDetails = request.form
        name = userDetails['name']
        phone = userDetails['phone']
        email = userDetails['email']
        password = userDetails['password']
        users.insert(name,phone,email,password)
        log_in_user(email)
        flash("Registration successful", "success")
        return redirect('/login')
    
    return render_template('register.html')

@app.route("/login", methods = ['GET', 'POST'])
def login():
    users = Table("users", "name", "phone", "email", "password")
    users.logout()
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if email=="tishanegandhi27@gmail.com":
            users1=Table("admin", "name", "phone", "email", "password")
            user=users1.getone1("email",email)
            log_in_user1(email)
            return redirect('/admin')
        else:
            user = users.getone("email", email)
            if user is None:
                flash("Invalid Email",'danger')
                return render_template('login.html')
            else:
                og_pass=user[3]
                if password!=og_pass:
                    flash("Password invalid", 'danger')
                    return render_template('login.html')
                else:
                    log_in_user(email)
                    return redirect('/')           

    return render_template('login.html')

@app.route("/wishlist",methods=['GET', 'POST'])
@login_required
def wishlist():
    ud=[]
    id=session['id']
    cur = mysql.connection.cursor()
    resultValue = cur.execute("SELECT * FROM %s WHERE %s = \"%s\"" %("wishlist", "user_id", id))
    if resultValue > 0:
        userDetails = cur.fetchall()
        for user in userDetails:
            book_id=user[2]
            cur = mysql.connection.cursor()
            resultValue1 = cur.execute("SELECT title,author,rating,cover,book_id FROM %s WHERE %s = \"%s\"" %("book", "book_id", book_id))
            if resultValue1 > 0:
                userDetails1 = cur.fetchall()
                ud.extend(userDetails1)
    cur = mysql.connection.cursor()
    resultValue22 = cur.execute("SELECT * FROM %s WHERE %s = \"%s\"" %("wishlist1", "user_id", id))
    if resultValue22 > 0:
        userDetails22 = cur.fetchall()
        for user1 in userDetails22:
            bk_id=user1[2]
            cur = mysql.connection.cursor()
            resultValue14 = cur.execute("SELECT title,author,rating,cover,bk_id FROM %s WHERE %s = \"%s\"" %("books_with_moods", "bk_id", bk_id))
            if resultValue14 > 0:
                userDetails15 = cur.fetchall()
                ud.extend(userDetails15)
    countt=len(ud)-1
    if request.method == 'POST':
        userDetails2 = request.form
        book=userDetails2["searchbook"]
        return redirect(url_for('search', name=book))        
    return render_template('wishlist.html',context=ud,context3=countt)

@app.route("/contact",methods=['GET', 'POST'])
@login_required
def contact():
    if request.method == 'POST':
        if request.form["section_name"] == "Send":
            userDetails = request.form
            name = userDetails['name']
            subject = userDetails['drop']
            email = userDetails['email']
            message = userDetails['message']
            if subject=="Query":
                users = Table("queries", "name", "email", "query", "date_of_query")
                users.insert(name,email,message,date.today())
                flash("Query submitted","success")
                return redirect('/contact') 
            else:
                users = Table("feedback", "name", "email", "feedback", "date_of_feedback")
                users.insert(name,email,message,date.today())
                flash("Feedback submitted","success")
                return redirect('/contact') 
        if request.form["section_name"] == "book_search":
            userDetails2 = request.form
            book=userDetails2["searchbook"]
            return redirect(url_for('search', name=book))
            
    return render_template('contactus.html')

@app.route("/mv",methods=['GET', 'POST'])
def mv():
    reviewdetails=None
    if request.method == 'POST':
        userDetails = request.form
        book_id = userDetails['section']
    else:
        book_id = request.args.get('context')
    cur = mysql.connection.cursor()
    resultValue = cur.execute("SELECT * FROM %s WHERE %s = \"%s\"" %("books_with_moods", "title", book_id))
    if resultValue > 0:
        bookdetails = cur.fetchall()
    cur = mysql.connection.cursor()
    resultValue = cur.execute("SELECT * FROM %s WHERE %s = \"%s\"" %("book1_reviews", "title", book_id))
    if resultValue > 0:
        reviewdetails = cur.fetchall()
    client=razorpay.Client(auth=("rzp_test_DhfZoCvuvrYovO","VWY4hrseIMrgbqNdKqGRbaIS"))
    payment=client.order.create({'amount':355*100, 'currency':'INR', 'payment_capture':1})
    return render_template('viewmoremv.html',context=bookdetails,context2=reviewdetails,payment=payment)

@app.route("/search",methods=['GET', 'POST'])
@login_required
def search():
    book1 = request.args.get('name')
    booksearch=searchb(book1)
    if booksearch == None:
        return redirect(url_for('googleapi'))
    data=recommend(book1)
    if request.method == 'POST':
        if request.form["section"] == "search":
            userDetails2 = request.form
            book=userDetails2["searchbook"]
            return redirect(url_for('search', name=book))
        else:
            userDetails = request.form
            book_id = userDetails['section']
            return redirect(url_for('viewmore', bookid=book_id))
    return render_template("search.html",book=booksearch,book1=book1,data=data,i=4,book_name = list(popular_df['Book-Title'].values),
                           author=list(popular_df['Book-Author'].values),
                           image=list(popular_df['Image-URL-M'].values),
                           votes=list(popular_df['num_ratings'].values),
                           rating=list(popular_df['avg_rating'].values))

@app.route("/searchbook",methods=['GET','POST'])
@login_required
def googleapi():
    return render_template('index1.html')

@app.route("/book",methods=['GET','POST'])
@login_required
def read():
    return render_template('book.html')

@app.route("/about",methods=['GET', 'POST'])
def about():
    if request.method == 'POST':
        userDetails2 = request.form
        book=userDetails2["searchbook"]
        return redirect(url_for('search', name=book))
    return render_template('aboutus.html')

@app.route("/admin",methods=['GET', 'POST'])
@login_required
def admin():
    cur = mysql.connection.cursor()
    resultValue = cur.execute("SELECT book_id,title,author FROM book")
    if resultValue > 0:
        userDetails1 = cur.fetchall()
    cur = mysql.connection.cursor()
    if request.method == 'POST':
        if request.form["section"] == "Add now":
            userDetails = request.form
            category = userDetails['category']
            title = userDetails['title']
            author = userDetails['author']
            rating = userDetails['rating']
            likes = userDetails['likes']
            price = userDetails['price']
            year = userDetails['year']
            genre = userDetails['genre']
            summary = userDetails['summary']
            if category == "Trending":
                users = Table("book", "title", "author", "rating", "likes_count","price","year","genre","summary")
                users.insert(title,author,rating,likes,price,year,genre,summary)
                return redirect('/admin') 
            else:
                users = Table("book1", "title", "author", "rating", "likes_count","price","year","genre","summary")
                users.insert(title,author,rating,likes,price,year,genre,summary)
                return redirect('/admin') 
        else:
            userDetails = request.form
            book_id = userDetails['section']
            if int(book_id) < 40:
                users = Table("book", "title", "author", "rating", "likes_count","price","year","genre","summary")
                users.delete(book_id)
                return redirect('/admin') 
            else:
                users = Table("book1", "title", "author", "rating", "likes_count","price","year","genre","summary")
                users.delete1(book_id)
                return redirect('/admin') 
    return render_template('adminhome.html',context=userDetails1)

@app.route('/viewmore' ,methods=['GET', 'POST'])
@login_required
def viewmore():
    reviewdetails=None
    review=None
    client=razorpay.Client(auth=("rzp_test_DhfZoCvuvrYovO","VWY4hrseIMrgbqNdKqGRbaIS"))
    payment=client.order.create({'amount':355*100, 'currency':'INR', 'payment_capture':1})
    book_id = request.args.get('bookid')
    cur = mysql.connection.cursor()
    resultValue = cur.execute("SELECT title,author,summary,cover,rating,book_id,likes_count FROM %s WHERE %s = \"%s\"" %("book", "title", book_id))
    if resultValue > 0:
        bookdetails = cur.fetchall()
        cur = mysql.connection.cursor()
        resultValue = cur.execute("SELECT name,review,likes,review_id FROM %s WHERE %s = \"%s\"" %("book_reviews", "book_id", bookdetails[0][5]))
        if resultValue > 0:
            reviewdetails = cur.fetchall()
    if request.method == 'POST':
        if request.form["section"] == "search":
            userDetails2 = request.form
            book=userDetails2["searchbook"]
            return redirect(url_for('search', name=book)) 
        else: 
            userDetails4 = request.form
            review=userDetails4["review"]
            book_id=userDetails4["section"]
            users = Table("book_reviews", "book_id", "review", "likes", "name")
            users.insert(book_id,review,0,session['name'])
            cur = mysql.connection.cursor()
            res = cur.execute("SELECT title FROM %s WHERE %s = \"%s\"" %("book", "book_id", book_id))
            if res > 0:
                bookk = cur.fetchall()
                title=bookk[0]
                return redirect(url_for('viewmore', bookid=title))
    return render_template('viewmore.html',context=bookdetails,payment=payment, context2=reviewdetails)  

@app.route('/viewmore1', methods=["GET",'POST'])
@login_required
def viewmore1():
    if request.method == 'POST':
        like=request.form
        likes=like.get("like")
        users = Table("book", "title", "author", "rating", "likes_count","price","year","genre","summary")
        users.update(likes)
        cur = mysql.connection.cursor()
        res = cur.execute("SELECT title FROM %s WHERE %s = \"%s\"" %("book", "book_id", likes))
        if res > 0:
            bookk = cur.fetchall()
            title=bookk[0]
            return redirect(url_for('viewmore', bookid=title))
        
@app.route('/viewmoremv1', methods=["GET",'POST'])
@login_required
def viewmoremv1():
    if request.method == 'POST':
        like=request.form
        likes=like.get("like")
        users = Table("books_with_moods", "title", "author", "rating", "likes_count","desc","cover","mood")
        users.update14(likes)
        cur = mysql.connection.cursor()
        res = cur.execute("SELECT title FROM %s WHERE %s = \"%s\"" %("books_with_moods", "bk_id", likes))
        if res > 0:
            bookk = cur.fetchall()
            title=bookk[0]
            return redirect(url_for('mv', context=title))
        
@app.route('/viewmore2', methods=["GET",'POST'])
@login_required
def viewmore2():
    if request.method == 'POST':
        wishlist=request.form
        wish=wishlist.get("wishlist")
        users = Table("wishlist", "user_id", "book_id")
        users.insert(session['id'],wish)
        cur = mysql.connection.cursor()
        res = cur.execute("SELECT title FROM %s WHERE %s = \"%s\"" %("book", "book_id", wish))
        if res > 0:
            bookk = cur.fetchall()
            title=bookk[0]
            return redirect(url_for('viewmore', bookid=title))
        
@app.route('/viewmoremv2', methods=["GET",'POST'])
@login_required
def viewmoremv2():
    if request.method == 'POST':
        wishlist=request.form
        wish=wishlist.get("wishlist")
        users = Table("wishlist1", "user_id", "book_id")
        users.insert(session['id'],wish)
        cur = mysql.connection.cursor()
        res = cur.execute("SELECT title FROM %s WHERE %s = \"%s\"" %("books_with_moods", "bk_id", wish))
        if res > 0:
            bookk = cur.fetchall()
            title=bookk[0]
            return redirect(url_for('mv', context=title))
        
@app.route('/viewmoremv3', methods=["GET",'POST'])
@login_required
def viewmoremv3():
    if request.method == 'POST':
        userDetails4 = request.form
        review=userDetails4["review"]
        book_id=userDetails4["section"]
        users = Table("book1_reviews", "title", "review", "likes", "name")
        users.insert(book_id,review,0,session['name'])
        return redirect(url_for('mv', context=book_id))
        
@app.route('/charge', methods=['POST'])
@login_required
def charge():
    if request.method == 'POST':
        client=razorpay.Client(auth=("rzp_test_DhfZoCvuvrYovO","VWY4hrseIMrgbqNdKqGRbaIS"))
        payment=client.order.create({'amount':200*100, 'currency':'INR', 'payment_capture':1})
        return redirect(url_for('viewmore', payment=payment))
@app.route('/likes', methods=["GET",'POST'])
@login_required
def likes():
    if request.method == 'POST':
        like=request.form
        likes=like.get("like")
        cur = mysql.connection.cursor()
        resultValue = cur.execute("SELECT book_id FROM %s WHERE %s = \"%s\"" %("book", "title", likes))
        if resultValue > 0:
            book_id = cur.fetchall()
            users = Table("book", "title", "author", "rating", "likes_count","price","year","genre","summary")
            users.update(book_id[0])
            return redirect('/')
        
@app.route('/wish', methods=["GET",'POST'])
@login_required
def wish():
    if request.method == 'POST':
        wishlist=request.form
        wish=wishlist.get("wishlist")
        users = Table("wishlist", "user_id", "book_id")
        cur = mysql.connection.cursor()
        resultValue = cur.execute("SELECT book_id FROM %s WHERE %s = \"%s\"" %("book", "title", wish))
        if resultValue > 0:
            book_id = cur.fetchall()
            users.insert(session['id'],book_id[0])
            return redirect('/')
        
@app.route('/like1', methods=["GET",'POST'])
@login_required
def like1():
    if request.method == 'POST':
        review_like=request.form
        likee=review_like.get("like")
        users = Table("book_reviews", "review_id", "book_id", "review", "likes", "name")
        users.updatelike(likee)
        cur = mysql.connection.cursor()
        resultValue = cur.execute("SELECT book_id FROM %s WHERE %s = \"%s\"" %("book_reviews", "review_id", likee))
        if resultValue > 0:
            book_id = cur.fetchall()
            return redirect(url_for('viewmore', bookid=book_id[0]))
        
@app.route('/remove',methods=["GET",'POST'])
@login_required
def remove():
    if request.method == 'POST':
        remove_book=request.form
        book_id=remove_book.get("section")
        users = Table("wishlist", "wishlist_id", "user_id", "book_id")
        users.delete_book(session['id'],book_id)
        return redirect('/wishlist')
    
@app.route('/genre',methods=["GET","POST"])
def genre():
    genre=""
    userDetails=[]
    if request.method == 'POST':
        scrape=request.form
        genre=scrape.get("section")
        cur = mysql.connection.cursor()
        resultValue = cur.execute("SELECT * FROM %s"%(genre))
        if resultValue > 0:
            userDetails = cur.fetchall()
    return render_template('genre.html',genre=genre,context1=userDetails)

@app.route('/mood',methods=["GET","POST"])
def mood():
    mood=""
    userDetails148=[]
    if request.method == 'POST':
        if request.form["mood"] == "happy":
            mood='happy'
        if request.form["mood"] == "romantic":
            mood='romantic'
        if request.form["mood"] == "uplifting":
            mood='uplifting'
        if request.form["mood"] == "scary":
            mood='scary'
        if request.form["mood"] == "neutral":
            mood='neutral'
        if request.form["mood"] == "suspense":
            mood='suspense'
    cur = mysql.connection.cursor()
    resultValue1 = cur.execute("SELECT title FROM %s WHERE %s = \"%s\"" %("books_with_moods", "mood", mood))
    if resultValue1 > 0:
        userDetails148 = cur.fetchall()
    random_book = random.choice(list(userDetails148))
    rb=random_book[0]
    cur = mysql.connection.cursor()
    resultValue1 = cur.execute("SELECT * FROM %s WHERE %s = \"%s\"" %("books_with_moods", "title",rb ))
    if resultValue1 > 0:
        userDetails1485 = cur.fetchall()
    cur = mysql.connection.cursor()
    resultValue1 = cur.execute("SELECT * FROM %s WHERE %s = \"%s\" and %s != \"%s\"" %("books_with_moods", "mood", mood, "title",rb))
    if resultValue1 > 0:
        userDetails1483 = cur.fetchall()
    return render_template('mood.html',mood=mood,book=userDetails1485,data=userDetails1483)
    
        
@app.route('/dump',methods=['GET'])
def showjson():
    with open("dump.json") as file:
        data = json.load(file)
        return jsonify(data)

@app.route('/dump', methods=['POST'])
def addOne():
    title = {'title' : request.json['title'],'author' : request.json['author'],'rating' : request.json['rating'], 'id' : request.json['id']}
    title1=request.json['title']
    author1=request.json['author']
    rating1=request.json['rating']
    id1=request.json['id']
    with open("dump.json") as file:
        data = json.load(file)
        data.append(title)
        with open("dump.json", mode='w') as f:
            f.write(json.dumps(data))
    users = Table("book", "title","author","rating","id")
    users.insert(title1,author1,rating1,id1)
    return jsonify({'book' : "Book added"})
        

@app.route('/dump/<int:id>', methods=['PUT'])
def editOne(id):
    with open("dump.json") as file:
        data = json.load(file)
    langs = [book for book in data if book['id'] == id]
    langs[0]['title'] = request.json['title']
    title=request.json['title']
    with open("dump.json", mode='w') as f:
        f.write(json.dumps(data))
    users = Table("book", "title", "author", "rating", "likes_count","price","year","genre","summary")
    users.update_title(id,title)
    return jsonify({'book' : "Book updated"})

@app.route('/dump/<string:title>', methods=['DELETE'])
def removeOne(title):
    with open("dump.json") as file:
        data = json.load(file)
    lang = [book for book in data if book['title'] == title]
    data.remove(lang[0])
    with open("dump.json", mode='w') as f:
        f.write(json.dumps(data))
    users = Table("book", "title", "author", "rating", "likes_count","price","year","genre","summary")
    users.delete_title(title)
    return jsonify({'book' : "Book deleted"})

if __name__ == '__main__':
    app.run(debug=True,port=8080,use_reloader=False)
