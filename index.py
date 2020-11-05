from flask import Flask, redirect, url_for

app=Flask(__name__)



# CREATING A HOME PAGE ------  NORMAL ROUTING
@app.route('/')
def home():
    return 'hello <h1>Hi in h1 tag</h1>'


# PARAMETER ROUTING
@app.route('/<name>')
def user(name):
    return f'Hello {name}'


# ROUTE REDIRECTING
@app.route('/admin')
def admin():
    return redirect(url_for('home'))  # YOU HAVE TO GIVE NAME OF THE FUNCTION.

if __name__ == '__main__':
    app.run()


