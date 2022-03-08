# coding:UTF-8
from app import views
__author__ = 'fxg'

from flask import Flask

app = Flask(__name__)


app.config.from_object('config')
