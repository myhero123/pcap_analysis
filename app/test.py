from app import app
from flask import render_template, request, flash, redirect, url_for, send_from_directory
from .forms import Upload, ProtoFilter
from .utils.upload_tools import allowed_file, get_filetype, random_name
from .utils.pcap_decode import PcapDecode
from .utils.pcap_filter import get_all_pcap, proto_filter, showdata_from_id
from .utils.proto_analyzer import common_proto_statistic, pcap_len_statistic, http_statistic, dns_statistic, most_proto_statistic
from .utils.flow_analyzer import time_flow, data_flow, get_host_ip, data_in_out_ip, proto_flow, most_flow_statistic
from .utils.ipmap_tools import getmyip, get_ipmap, get_geo
from .utils.data_extract import web_data, telnet_ftp_data, mail_data, sen_data, client_info
from .utils.except_info import exception_warning
from .utils.file_extract import web_file, ftp_file, mail_file, all_files
from scapy.all import rdpcap
import os
import hashlib
app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'


@app.route('/')
def index():
    return render_template('./test.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
