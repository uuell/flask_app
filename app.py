import sqlite3
from flask import Flask, render_template, request, redirect, url_for, session, g, flash, jsonify
import base64
import easyocr
import deepl

app = Flask(__name__)
app.secret_key = 'your_secret_key'
DATABASE = 'site.db'  # Define the database file name
auth_key = "3998ee77-3ce9-4b7c-bff0-65db7106880b:fx"  # Deepl API key

def get_db():
    """Connects to the specific database."""
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row  # Return rows as dictionary-like objects
    return db

@app.teardown_appcontext
def close_connection(exception):
    """Closes the database again at the end of the request."""
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

# This function initializes the database schema.
def init_db():
    """Initializes the database schema."""
    with app.app_context():
        db = get_db()
        with app.open_resource('schema.sql', mode='r') as f:
            db.cursor().executescript(f.read())
        db.commit()

# Flask CLI command to initialize the database
@app.cli.command('initdb')
def initdb_command():
    """Creates the database tables."""
    init_db()
    print('Initialized the database.')

def query_db(query, args=(), one=False):
    """Queries the database and returns a list of dictionaries."""
    cur = get_db().execute(query, args)
    rv = cur.fetchall()
    cur.close()
    return (rv[0] if rv else None) if one else rv

def execute_db(query, args=()):
    """Executes a write query (INSERT, UPDATE, DELETE)."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute(query, args)
    conn.commit()
    cur.close()

def is_logged_in():
    return 'user_id' in session

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['GET', 'POST'])
def translate():
    if request.method == 'POST':
        reader = easyocr.Reader(['ja', 'en'], gpu=False)
        text_array = []
        recognized_text = ""
        translated_text = ""
        target_language = request.form.get('target_language', 'en').upper() # DeepL expects uppercase language codes
        image_filename = None

        if 'image_data' in request.form:
            image_data = request.form['image_data'].split(',')[1]
            image_bytes = base64.b64decode(image_data)
            results = reader.readtext(image_bytes)
            for bbox, text, prob in results:
                print(f"bbox: {bbox}, text: {text}, prob: {prob}")
                recognized_text += text + " " # Accumulate recognized text
                translated_text = deepl.Translator(auth_key).translate_text(text, target_lang=target_language).text + " "
            recognized_text = recognized_text.strip()
            translated_text = translated_text.strip()
        elif 'image_file' in request.files:
            image_file = request.files['image_file']
            image_bytes = image_file.read() # Read the file content as bytes
            image_filename = image_file.filename
            results = reader.readtext(image_bytes)
            for bbox, text, prob in results:
                print(f"bbox: {bbox}, text: {text}, prob: {prob}")
                text_array.append(text)
            translated_text_array = deepl.Translator(auth_key).translate_text(text_array, target_lang=target_language)
            for text in translated_text_array:
                recognized_text += text_array[translated_text_array.index(text)] + " "
                translated_text += text.text + " "
            recognized_text = recognized_text.strip()
            translated_text = translated_text.strip()

        if is_logged_in():
            user_id = session['user_id']
            execute_db("""
                INSERT INTO translation_history (user_id, image_filename, recognized_text, translated_text, target_language)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, image_filename, recognized_text, translated_text, target_language))
            get_db().commit() # Ensure the data is written to the database

        return jsonify({'recognized_text': recognized_text, 'translated_text': translated_text})
    return render_template('translate.html')

@app.route('/account')
def account():
    if is_logged_in():
        # Example: Fetch user data from the database
        user = query_db('SELECT username FROM users WHERE id = ?', (session['user_id'],), one=True)
        history = query_db('SELECT recognized_text, translated_text, target_language, timestamp FROM translation_history WHERE user_id = ? ORDER BY timestamp DESC', (session['user_id'],))
        return render_template('account.html', user=user, history=history)
    else:
        return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        error = None
        user = query_db('SELECT id, password FROM users WHERE username = ?', (username,), one=True)

        if user is None:
            error = 'Invalid username'
        elif user['password'] != password:  # In a real app, compare hashed passwords!
            error = 'Invalid password'

        if error is None:
            session['user_id'] = user['id']
            return redirect(url_for('account'))
        flash(error, 'error')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        error = None

        if not username:
            error = 'Username is required'
        elif not password:
            error = 'Password is required'
        elif password != confirm_password:
            error = 'Passwords do not match'
        elif query_db('SELECT id FROM users WHERE username = ?', (username,), one=True) is not None:
            error = 'Username already exists'

        if error is None:
            execute_db('INSERT INTO users (username, password) VALUES (?, ?)', (username, password)) # In real app, hash password!
            flash('Signup successful! Please log in.', 'success')
            return redirect(url_for('login'))
        flash(error, 'error')
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)