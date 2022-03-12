from flask import Flask, flash, redirect, render_template, \
    request, url_for
app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'


@app.route('/')
def index():
    return render_template('./test.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)