#!/usr/bin/python3
# Uses requests library to get dataset from github
import requests

csv_url='https://raw.githubusercontent.com/mikaylaedwards/flask_model/master/data/marketing_new.csv'
req = requests.get(csv_url)
url_content = req.content

with open('data/marketing.csv', 'wb') as csv_file:
    csv_file.write(url_content)

