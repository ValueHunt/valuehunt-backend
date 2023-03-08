import time

from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask import request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv, find_dotenv
from PIL import Image
import io
import pymongo
import os
import requests
import urllib
# ************************************Web Scraping ***************************
from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

# ***************************************Tensorflow and Models *************************
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import cv2

import functools
from concurrent.futures import ThreadPoolExecutor

cate_model = load_model(
    'C:/Users/dell/Desktop/ValueHunt/valuehunt-backend/category.h5')
style_model = load_model(
    'C:/Users/dell/Desktop/ValueHunt/valuehunt-backend/SameStyle.h5')


def getCategory(img_byte):

    try:

        classes = ['Blazer',
                'Body',
                'Dress',
                'Hat',
                'Hoodie',
                'Longsleeve',
                'Outwear',
                'Pants',
                'Polo',
                'Shirt',
                'Shoes',
                'Shorts',
                'Skirt',
                'T-Shirt',
                'Top',
                'Undershirt']

        img = Image.open(io.BytesIO(img_byte))
        # Resize the image (to the same size our model was trained on)
        img = img.resize((299, 299))
        img=np.array(img)
        # Rescale the image (get all values between 0 and 1)
        img = img/255.
        # Make a prediction
        pred = cate_model.predict(np.expand_dims(img, axis=0))

        # Get the predicted class
        if len(pred[0]) > 1:
            # if more than one output, take the max
            pred_class = classes[pred.argmax()]
        else:
            pred_class = classes[int(tf.round(pred)[0][0])]

        return pred_class
    except :
        return  f'Image is Corrupted'


def getStyle(image):

    try:

        classes = ['animal', 'cartoon', 'chevron', 'floral', 'geometry', 'houndstooth', 'ikat', 'letter_numb', 'plain', 'polka dot', 'scales', 'skull', 'squares',
                'stars',
                'stripes',
                'tribal']

        image_pil = Image.open(io.BytesIO(image))

        image_np = np.array(image_pil)

        # Convert RGB image to BGR
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        # image_bgr = cv2.imread(file_path, cv2.IMREAD_COLOR)  Previous code

        # Convert to HSV for creating a mask
        image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        # Convert to grayscale that will actually be used for training, instead of color image
        image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # Create a mask that detects the red rectangular tags present in each image
        mask = cv2.inRange(image_hsv, (0, 255, 255), (0, 255, 255))

        # Get the coordinates of the red rectangle in the image,
        # But take entire image if mask fails to detect the red rectangle
        if len(np.where(mask != 0)[0]) != 0:
            y1 = min(np.where(mask != 0)[0])
            y2 = max(np.where(mask != 0)[0])
        else:
            y1 = 0
            y2 = len(mask)

        if len(np.where(mask != 0)[1]) != 0:
            x1 = min(np.where(mask != 0)[1])
            x2 = max(np.where(mask != 0)[1])
        else:
            x1 = 0
            x2 = len(mask[0])

        # Crop the grayscle image along those coordinates
        image_cropped = image_gray[y1:y2, x1:x2]

        # Resize the image to 100x100 pixels size
        image_100x100 = cv2.resize(image_cropped, (100, 100))

        # Save image as in form of array of 10000x1
        image_arr = image_100x100.flatten()

        image_arr = image_arr/255
        image_arr = image_arr.reshape(1, 100, 100, 1)
        pred = np.round(style_model.predict(image_arr))

        if len(pred[0]) > 1:  # check for multi-class
            # if more than one output, take the max
            pred_class = classes[pred.argmax()]
        else:
            # if only one output, round
            pred_class = classes[int(tf.round(pred)[0][0])]

        return pred_class
    except:
        return 'Image is Corrupted'


# ********************************Backend **************************




app = Flask(__name__)
CORS(app)
actual_token = os.environ.get("token")

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["ValueHunt"]


API = os.environ.get("ScrapyAPI")

def get_url(url):
    payload = {'api_key': API, 'url': url}
    proxy_url = 'http://api.scraperapi.com/?' + urllib.parse.urlencode(payload)
    return proxy_url


def compare(a,b):
    
    if float(a['Price'])>float(b['Price']):
        return 1
    return -1
    
def preprocessPrice(li):

    for a in li:
        # print('before ',a['Price'])
        comma=a['Price'].find(',')
        if(comma!=-1):
            a['Price']=a['Price'][:comma]+a['Price'][comma+1:]

        adecimal= a['Price'].find('.')

        if(adecimal==len(a['Price'])-1):
            a['Price']=a['Price'][0:adecimal]

        # print('after ',a['Price'])
    return li

def getAmazonData(color,style,category):
    
    amazon_output_data=[]

    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36"

    options = webdriver.ChromeOptions()
    options.headless = True
    options.add_argument(f'user-agent={user_agent}')
    options.add_argument("--window-size=1920,1080")
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--allow-running-insecure-content')
    options.add_argument("--disable-extensions")
    options.add_argument("--proxy-server='direct://'")
    options.add_argument("--proxy-bypass-list=*")
    options.add_argument("--start-maximized")
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--no-sandbox')
    # chrome_options.add_argument('window-size=1920x1080')
    driver = webdriver.Chrome(service=Service('chromedriver.exe'),options=options)
    
    query=f'{color} {style} {category}'

    driver.get('https://www.amazon.in/s?' + urllib.parse.urlencode({'k': query}))
    
    # print ('https://www.amazon.in/s?' + urllib.parse.urlencode({'k': query}))
    content = driver.page_source
    # print('/*******************************************')
    # print(content)
    soup = BeautifulSoup(content,'lxml')
    try:
        for prod in soup.findAll('div', attrs={'class':'a-section a-spacing-base a-text-center'}):
            # print('*********************3333333333333333333333333****************')
            ProdLink=prod.find('a', attrs={'class':'a-link-normal s-no-outline'})['href']
            ProdLink= 'https://www.amazon.in/'+ProdLink
            ImageSrc=prod.find('img')['src'].split(',')[0]
            Label=prod.find('span',attrs={'class':'a-size-base-plus a-color-base a-text-normal'}).get_text()
            Price = prod.find('span',attrs={'a-price-whole'})

            if(Price):
                Price=Price.get_text()

            if(Price):
                check_cate=Label.lower().split(' ')
                category=category.lower()
                if(category in check_cate):
                    amazon_output_data.append({'ImageSrc':ImageSrc,'Label':Label,'Price':Price,"ProdLink":ProdLink})

                    # print('************************AMzonCategory*************************************')
                    # print(amazon_output_data)
                    # print('*************************************************************')

        # print('************************AMzonCategory*************************************')
        # print(amazon_output_data)
        # print('*************************************************************')

        amazon_output_data=preprocessPrice(amazon_output_data)

        amazon_output_data=sorted(amazon_output_data,key=functools.cmp_to_key(compare))

        amazon_output_data=amazon_output_data[:4]

  
        if(len(amazon_output_data)==0):
            return 'No Data Found'
        return amazon_output_data
    except Exception as e :
        return f'SOmething went Wrong {e}'




def getMyntraData(color, style, category):

    query=f'{color} {category} {style}'

    myntra_output_data = []
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36"

    options = webdriver.ChromeOptions()
    options.headless = True
    options.add_argument(f'user-agent={user_agent}')
    options.add_argument("--window-size=1920,1080")
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--allow-running-insecure-content')
    options.add_argument("--disable-extensions")
    options.add_argument("--proxy-server='direct://'")
    options.add_argument("--proxy-bypass-list=*")
    options.add_argument("--start-maximized")
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--no-sandbox')
    # chrome_options.add_argument('window-size=1920x1080')
    driver = webdriver.Chrome(service=Service(
        'chromedriver.exe'), options=options)

    driver.get("https://www.myntra.com/"+query)

    content = driver.page_source
    soup = BeautifulSoup(content, 'lxml')

    try:
        for element in soup.findAll('li', attrs={'class': 'product-base'}):

            prod = element.find('a')
            Label = prod.find('h4', attrs={'class': "product-product"}).text

            ProdLink = 'https://www.myntra.com/'+prod['href']
            ImageSrc = prod.find('source')

            Price = prod.find('span', attrs={'class': 'product-discountedPrice'})

            if(ImageSrc and Price):
                ImageSrc = ImageSrc.find('img')['src']
                Price = int(Price.get_text().split()[1])

                myntra_output_data.append(
                    {'ImageSrc': ImageSrc, 'Label': Label, 'Price': str(Price), "ProdLink": ProdLink})
            else:
                ImageSrc = None
                Price = None

        myntra_output_data=preprocessPrice(myntra_output_data)

        myntra_output_data=sorted(myntra_output_data,key=functools.cmp_to_key(compare))

        myntra_output_data=myntra_output_data[:4]
        # print('*****************************INside Myntra*****************************')
        # print(myntra_output_data)
        if(len(myntra_output_data)==0):
            return 'No Data Found'

        return myntra_output_data
    except:
        return 'No Data Found'

# for x in getMyntraData('Gucci', 'red', 'floral', 'shirt', 38, 'cotton'):
#     print('*************************************************************')
#     print(x)
#     print('*************************************************************')

def getFlipkartData(brand, color, style, category, size, clothType):
    pass


def getAjioData(brand, color, style, category, size, clothType):
    pass


def getColor(clothImg):
    pass


# *****************************Validator****************
def validate(token):
    if(token == actual_token):
        return True
    return False


# *********************************Insert Contact***************************
def insertContact(name, email, msg):
    try:

        col = db["contact"]
        data = {'name': name, 'email': email, 'msg': msg}

        if(col.find_one(data)):

            return jsonify('Contact Already Present')

        user_id = col.insert_one(data).inserted_id

        return jsonify('Contact Details has been Saved')

    except:
        return jsonify("It's our problem not yours")

# ********************************Insert Vh*****************************


def insertVh(clothImg, brand):

    try:
        im = Image.open(clothImg)
        rgb_im = im.convert('RGB')
        image_bytes = io.BytesIO()
        rgb_im.save(image_bytes, format='JPEG')
        image = {
            'data': image_bytes.getvalue()
        }
        col = db["VH"]
        data = {'clothImg': image, 
                'brand': brand}

        if(col.find_one(data)):

            return jsonify('Image Already Present')

        user_id = col.insert_one(data).inserted_id

        return jsonify('Image has been Saved')

    except:
        return jsonify("It's our problem not yours")


# ***********************************Contact Route************************
@app.route("/contact", methods=['POST'])
def contact():
    headers = request.headers
    bearer = headers.get('Authorization')
    token = bearer.split()[1]
    # print(config)
    if(validate(token)):
        name = request.json['name']
        email = request.json['email']
        msg = request.json['msg']

        res = insertContact(name, email, msg)
        return res

    return jsonify('You are not authenticated')


# ***********************************VH Route************************
@app.route("/vh", methods=['POST'])
def vh():
    headers = request.headers
    bearer = headers.get('Authorization')
    token = bearer.split()[1]
    if(validate(token)):

        clothImg = request.files['clothImg']
        # file_path = os.path.join('/tmp', clothImg.filename) 
        # clothImg.save(file_path)

        image_bytes = clothImg.read()
        # size = request.form.get('size')
        brand = request.form.get('brand')
        # clothType = request.form.get('clothType')

        insertVh(clothImg, brand)

        # color = getColor(clothImg)
        color='red'
        
        style = getStyle(image_bytes)
        print('********************Style*************************')
        print(style)
        print('*********************************************')
        category = getCategory(image_bytes)
        print('******************Category***************************')
        print(category)
        print('*********************************************')

        if(category =='Image is Corrupted' or style == 'Image is Corrupted'):
            return jsonify('Image is Corrupted')

        if(category =='Body'):
            category='Kurti'
        if(category =='Outwear'):
            category='Jacket'
        with ThreadPoolExecutor(max_workers=1) as executor:

    # Pass the parameters to the functions using the `args` parameter
            amazon_future = executor.submit(getAmazonData, color, style, category)
            myntra_future = executor.submit(getMyntraData, color, style, category)

            # Wait for the results
            amazon = amazon_future.result()
            myntra = myntra_future.result()

        # amazon=getAmazonData( color, style, category)
        # amazon = getAmazonData(brand, color, style, category, size, clothType)
        print('******************Amazon***************************')
        print(amazon)
        print('*********************************************')
        # myntra = getMyntraData(brand, color, style, category, size, clothType)
        print('*********************Myntra************************')
        print(myntra)
        print('*********************************************')
        # flipkart = getFlipkartData(brand, color, style, category, size, clothType)
        # ajio = getAjioData(brand, color, style, category, size, clothType)

        res = jsonify({'amazon':amazon,'myntra' : myntra}) #add this here (ajio,flipkart)
        # print('*********************REs************************')
        # # print(res.json())
        # print('*********************************************')
        return res

    return jsonify('You are not authenticated')
