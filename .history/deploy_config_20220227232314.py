#coding=UTF-8

bind = '192.168.223.133:8000' #绑定的端口
workers = 1 #worker数量
backlog = 2048
debug = True
proc_name = 'gunicorn.pid'
pidfile = '/var/log/gunicorn/debug.log'
loglevel = 'debug'
