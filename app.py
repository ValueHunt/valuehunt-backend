from io import BytesIO
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
import io
import pymongo
import os
import requests
import urllib
# ************************************Web Scraping ***************************
from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.service import Service as ChromeService
from time import sleep
# ***************************************Tensorflow and Models *************************
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import cv2
from tensorflow import keras

import functools
from concurrent.futures import ThreadPoolExecutor

cate_model = load_model('./category.h5')
style_model = load_model('./style.h5')
color_model = load_model('./Color.h5')


def getCategory(img_byte):

    try:

        classes = ['Cap', 'Hoodie', 'Jeans',
                   'Shirt', 'Shoes', 'T-Shirt', 'Vest']

        img = Image.open(io.BytesIO(img_byte))
        # Resize the image (to the same size our model was trained on)
        img = img.resize((299, 299))
        img = np.array(img)
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
    except:
        return f'Image is Corrupted'


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


# ***************************** Color Model ************************


def getColor(clothImg_bytes):
    try:
        # Load image from bytes
        img = Image.open(BytesIO(clothImg_bytes))
        img = img.resize((90, 90))  # Resize the image to (90, 90) if necessary
        test_image = keras.preprocessing.image.img_to_array(img)

        test_image = np.expand_dims(test_image, axis=0)
        test_image /= 255.0

        # class_names = {v: k for k, v in train_ds.class_indices.items()}
        # print(class_names)
        class_names = {0: 'black', 1: 'blue', 2: 'brown', 3: 'green', 4: 'grey', 5: 'orange',
                       6: 'pink', 7: 'purple', 8: 'red', 9: 'silver', 10: 'white', 11: 'yellow'}
        predicted_labels = color_model.predict(test_image)
        predicted_label = class_names[np.argmax(predicted_labels)]

        # print("Predicted label:", predicted_label)
        return predicted_label
    except:
        return 'Image is Corrupted'


# ********************************Backend **************************


app = Flask(__name__)
CORS(app)
actual_token = os.environ.get("token")

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["ValueHunt"]


API = os.environ.get("ScrapyAPI")


def compare(a, b):

    if float(a['Price']) > float(b['Price']):
        return 1
    return -1


def preprocessPrice(li):

    for a in li:
        # print('before ',a['Price'])
        comma = a['Price'].find(',')
        if (comma != -1):
            a['Price'] = a['Price'][:comma]+a['Price'][comma+1:]

        adecimal = a['Price'].find('.')

        if (adecimal == len(a['Price'])-1):
            a['Price'] = a['Price'][0:adecimal]

        # print('after ',a['Price'])
    return li


def checkTshirt(check_cat):

    li=['t' , 't-' , 't-shirt' ,'t-shirt-' , 'tshirt' ,'tshirts','polo','layer']

    for i in li:
        if i in check_cat:
            return True
        
    return False



def getAmazonData(brand, color, style, category):

    amazon_output_data = []
    abrand = brand
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
    options.add_argument('ignore-certificate-errors')
    # chrome_options.add_argument('window-size=1920x1080')
    driver = webdriver.Chrome(service=Service(
        'chromedriver.exe'), options=options)
    
    if(category=='Shirt'):
        category='Shirts'

    query = f'{color} {style} {category}'

    if (brand != 'No Brand'):
        query = f'{color} {category}'

    url = 'https://www.amazon.in/s?' + urllib.parse.urlencode({'k': query})

    if (brand != 'No Brand'):
        brand = brand.replace(' ', '+')

        url += '&rh=n%3A1571271031%2Cp_89%3A'+brand

    url += '&s=price-asc-rank'
    print('*****************************url******************')
    print(url)
    driver.get(url)
    sleep(3)

    content = driver.page_source

    soup = BeautifulSoup(content, 'lxml')
    try:
        for prod in soup.findAll('div', attrs={'class': 'a-section a-spacing-base a-text-center'}):
            # print('*********************3333333333333333333333333****************')
            ProdLink = prod.find(
                'a', attrs={'class': 'a-link-normal s-no-outline'})['href']
            ProdLink = 'https://www.amazon.in/'+ProdLink
            ImageSrc = prod.find('img')['src'].split(',')[0]
            Label = prod.find('span', attrs={
                              'class': 'a-size-base-plus a-color-base a-text-normal'})
            Price = prod.find('span', attrs={'a-price-whole'})

            BrandCheck = prod.find(
                'span', attrs={'class': 'a-size-base-plus a-color-base'})

            
            if (Label):
                Label = Label.get_text()

            if (BrandCheck):
                BrandCheck = BrandCheck.get_text()

            # print('***************Brand Check *.........',BrandCheck)
            if (Price):
                Price = Price.get_text()

            if (Price):
                check_cate = Label.lower().split(' ')
                category = category.lower()

                
                if (category == 'shirts'):
                    category='shirt'
                
                if (category in check_cate):

                    if (category == 'shirt' and checkTshirt(check_cate)):
                        continue
                    # if(category == 'shirt' and ('t-' or 't') in check_cate):
                    #     print('+++++++++++++++++++---------------------------',category,temp,check_cate,ProdLink)
                    # print('***************check category******')
                    # print(category,temp,check_cate,ProdLink)
                   
                    
                    if (abrand != 'No Brand' and abrand == BrandCheck):
                        amazon_output_data.append(
                            {'ImageSrc': ImageSrc, 'Label': Label, 'Price': Price, "ProdLink": ProdLink})
                    elif (abrand == 'No Brand'):
                        amazon_output_data.append(
                            {'ImageSrc': ImageSrc, 'Label': Label, 'Price': Price, "ProdLink": ProdLink})

        amazon_output_data = preprocessPrice(amazon_output_data)

        amazon_output_data = sorted(
            amazon_output_data, key=functools.cmp_to_key(compare))

        amazon_output_data = amazon_output_data[:4]

        if (len(amazon_output_data) == 0):
            return 'No Data Found'
        return amazon_output_data
    except Exception as e:
        return f'Something went Wrong'


def getMyntraData(brand, color, style, category):

    if(style=='plain'):
        style='casual-plain'

    query = f'{color} {style} {category}'

    if (brand != 'No Brand'):
        query = f'{color} {category}'

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
    options.add_argument('ignore-certificate-errors')
    driver = webdriver.Chrome(service=Service(
        'chromedriver.exe'), options=options)

    query = query.replace(' ', '-')
    url = "https://www.myntra.com/"+query

    if (brand == 'Adidas'):
        brand = 'ADIDAS'

    if (brand != 'No Brand'):
        brand = brand.replace(' ', '%20')

        url += '?f=Brand%3A'+brand + '&sort=price_asc'
    else:
        url += '?sort=price_asc'

    print('*****************************url******************')
    print(url)

    driver.get(url)
    sleep(3)

    content = driver.page_source
    soup = BeautifulSoup(content, 'lxml')

    try:
        for element in soup.findAll('li', attrs={'class': 'product-base'}):

            prod = element.find('a')
            Label = prod.find('h4', attrs={'class': "product-product"}).text

            ProdLink = 'https://www.myntra.com/'+prod['href']
            ImageSrc = prod.find('source')

            Price = prod.find(
                'div', attrs={'class': 'product-price'})

            disP = Price.find(
                'span', attrs={'class': 'product-discountedPrice'})
            if (disP):
                Price = disP
                # print('.....................',Price)

            if (ImageSrc and Price):
                # print('......................................')
                # print(ProdLink,Price)
                ImageSrc = ImageSrc.find('img')['src']
                Price = Price.get_text().split()[1]

                myntra_output_data.append(
                    {'ImageSrc': ImageSrc, 'Label': Label, 'Price': Price, "ProdLink": ProdLink})
            else:
                ImageSrc = None
                Price = None

        myntra_output_data = preprocessPrice(myntra_output_data)

        myntra_output_data = sorted(
            myntra_output_data, key=functools.cmp_to_key(compare))

        myntra_output_data = myntra_output_data[:4]

        if (len(myntra_output_data) == 0):
            return 'No Data Found'

        return myntra_output_data
    except:
        return 'No Data Found'


# ********************** Flipkart ****************************


def getFlipkartData(brand, color, style, category):
    # if category == 'Hat':
    #     category = 'cap'

    color = color.capitalize()

    flipkart_output_data = []
    brandurl = f'https://www.flipkart.com/search?q={category}&otracker=search&otracker1=search&marketplace=FLIPKART&as-show=on&sort=price_asc&style={style}&p%5B%5D=facets.color%255B%255D%3D{color}&p%5B%5D=facets.brand%255B%255D%3D'
    # if category == 'Hat':
    #     category = 'cap'

    # brandurl = f'https://www.flipkart.com/search?q={category}&otracker=search&otracker1=search&marketplace=FLIPKART&as-show=on&sort=price_asc&color={color}&p[]=facets.brand%255B%255D%3D'
    url = f'https://www.flipkart.com/search?q={category}&otracker=search&otracker1=search&marketplace=FLIPKART&as-show=on&sort=price_asc&style={style}&p%5B%5D=facets.color%255B%255D%3D{color}'
    prodLink = 'https://www.flipkart.com'
    if brand == 'No Brand' or brand == '':
        response = requests.get(url)
        queryURL = url
    else:
        if ' ' in brand:
            brand = brand.replace(' ', '%2B')
        brandurl += brand
        response = requests.get(brandurl)
        queryURL = brandurl
    print('*****************************  🔎  url     ******************')
    print(queryURL)
    soup = BeautifulSoup(response.content, 'lxml')
    try:
        products = soup.find_all('div', {'class': '_13oc-S'})

        for pr in products:
            prName = pr.find_all('a', {'class': 'IRpwTa'})
            prPrice = pr.find_all('div', {'class': '_30jeq3'})
            prLink = pr.find_all('a', {'class': '_2UzuFa'})
            prImageLink = pr.find_all('img', {'class': '_2r_T1I'})
            for itemImageSrc, itemName, itemPrice, itemLink in zip(prImageLink, prName, prPrice, prLink):
                if itemLink.find('span', {'class': '_192laR'}) is None and (category.lower() in itemName.text.lower().split(' ') or (category == 'Hoodie' and ('full' and 'sleeve') in itemName.text.lower().split(' ')) or (category == 'T-Shirt' and (('T' and 'Shirt') or ('t' and 'shirt') or ('t-shirt') or ('T-Shirt') or ('T' and 'shirt') or ('t' and 'Shirt')) in itemName.text.lower().split(' '))):
                    flipkart_output_data.append({'ImageSrc': (itemImageSrc.get(
                        'src')), 'Label': itemName.text, 'Price': itemPrice.text[1:], 'ProdLink': prodLink+(itemLink.get('href'))})

        flipkart_output_data = preprocessPrice(flipkart_output_data)

        flipkart_output_data = sorted(
            flipkart_output_data, key=functools.cmp_to_key(compare))

        flipkart_output_data = flipkart_output_data[:4]

        if (len(flipkart_output_data) == 0):
            return 'No Data Found'
        return flipkart_output_data
    except Exception as e:
        return f'Something went Wrong {e}'


def getAjioData(brand, color, style, category):
    options = webdriver.ChromeOptions()
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36"
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--incognito')
    options.add_argument('--headless')
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


    color = color.capitalize()
    if category == 'Jeans':
        url = f'https://www.ajio.com/s/jeans-3571-88891?query=%3Arelevance%3Averticalcolorfamily%3A{color}&curated=true&curatedid=jeans-3571-88891&gridColumns=5&segmentIds='
    else:
        url = f'https://www.ajio.com/search/?text={style} {color} {category}&gridColumns=5'
    driver = webdriver.Chrome(
        options=options, service=ChromeService('chromedriver.exe'))
    try:
        if brand == 'No Brand' or brand == '':
            queryURL = url
            driver.get(url)
        else:
            brandurl = ''
            if category == 'Jeans':
                brand = brand.upper()
                if ' ' in brand:
                    brand = brand.replace(' ', '%20')
                brandurl += f'https://www.ajio.com/s/jeans-3571-88891?query=%3Arelevance%3Averticalcolorfamily%3A{color}%3Abrand%3A{brand}&curated=true&curatedid=jeans-3571-88891&gridColumns=5&segmentIds='
                queryURL = brandurl
            else:
                if ' ' in brand:
                    brand = brand.replace(' ', '%20')
                brandurl += f'https://www.ajio.com/search/?query=%3Arelevance%3Abrand%3A{brand}&text={color} {category}&gridColumns=5'
                queryURL = brandurl
                driver.get(brandurl)
            driver.get(brandurl)
        print('*****************************  🔎 url   ******************')
        print(queryURL)
        while True:

            last_height = driver.execute_script(
                "return document.body.scrollHeight")
            driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);")
            sleep(1)
            new_height = driver.execute_script(
                "return document.body.scrollHeight")
            soup = BeautifulSoup(driver.page_source, 'lxml')

            products = soup.find_all(
                'div', {'class': 'item rilrtl-products-list__item item'})
            if new_height == last_height or len(products) >= 20:
                break
        baseurl = 'https://www.ajio.com'
        ajio_output_data = []
        for pr in products:
            if pr.find('img').get('src') != None:
                # if (category == 'Shirt' and ('t-shirts' not in (pr.find('div', {'class': 'nameCls'}).text) and 'T-Shirts' not in (pr.find('div', {'class': 'nameCls'}).text) and 't-shirt' not in (pr.find('div', {'class': 'nameCls'}).text) and 'T-Shirt' not in (pr.find('div', {'class': 'nameCls'}).text) and 'T-shirt' not in (pr.find('div', {'class': 'nameCls'}).text) and 'Top' not in (pr.find('div', {'class': 'nameCls'}).text) and 'top' not in (pr.find('div', {'class': 'nameCls'}).text) and 'TOP' not in (pr.find('div', {'class': 'nameCls'}).text))):
                name = pr.find('div', {'class': 'nameCls'}).text
                if (category.lower() in name.lower().split(' ')):
                    ajio_output_data.append({'ImageSrc': pr.find('img').get('src'), 'Label': pr.find('div', {'class': 'nameCls'}).text, 'Price': pr.find(
                        'span', {'class': 'price'}).text[1:], 'ProdLink': (baseurl+pr.find('a', {'class': 'rilrtl-products-list__link'}).get('href'))})
                # else:
                #     ajio_output_data.append({'ImageSrc': pr.find('img').get('src'), 'Label': pr.find('div', {'class': 'nameCls'}).text, 'Price': pr.find(
                #         'span', {'class': 'price'}).text[1:], 'ProdLink': (baseurl+pr.find('a', {'class': 'rilrtl-products-list__link'}).get('href'))})

        ajio_output_data = preprocessPrice(ajio_output_data)

        ajio_output_data = sorted(
            ajio_output_data, key=functools.cmp_to_key(compare))

        ajio_output_data = ajio_output_data[:4]

        if (len(ajio_output_data) == 0):
            return 'No Data Found'
        return ajio_output_data
    except Exception as e:
        return f'Something went Wrong {e}'


# ***************************** Validator ****************
def validate(token):
    if (token == actual_token):
        return True
    return False


# ********************************* Insert Contact ***************************
def insertContact(name, email, msg):
    try:

        col = db["contact"]
        data = {'name': name, 'email': email, 'msg': msg}

        if (col.find_one(data)):

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

        if (col.find_one(data)):

            return jsonify('Image Already Present')

        user_id = col.insert_one(data).inserted_id

        return jsonify('Image has been Saved')

    except:
        return jsonify("It's our problem not yours")


# *********************************** Contact Route ************************
@app.route("/contact", methods=['POST'])
def contact():
    headers = request.headers
    bearer = headers.get('Authorization')
    token = bearer.split()[1]
    # print(config)
    if (validate(token)):
        name = request.json['name']
        email = request.json['email']
        msg = request.json['msg']

        res = insertContact(name, email, msg)
        return res

    return jsonify('You are not authenticated')


# *********************************** VH Route ************************
@app.route("/vh", methods=['POST'])
def vh():
    headers = request.headers
    bearer = headers.get('Authorization')
    token = bearer.split()[1]
    if (validate(token)):

        clothImg = request.files['clothImg']

        image_bytes = clothImg.read()

        brand = request.form.get('brand')

        insertVh(clothImg, brand)

        with ThreadPoolExecutor(max_workers=3) as executor:

            category_future = executor.submit(getCategory, image_bytes)

            style_future = executor.submit(getStyle, image_bytes)

            color_future = executor.submit(getColor,  image_bytes)

            category = category_future.result()
            style = style_future.result()
            color = color_future.result()

        print('********************  Style    *************************')
        print(style)
        print('*********************************************')

        print('******************      Category   ***************************')
        print(category)
        print('*********************************************')

        print(
            '******************   Color ***************************')
        print(color)
        print('*********************************************')

        if (category == 'Image is Corrupted' or style == 'Image is Corrupted' or color == 'Your Image is Corupted'):
            return jsonify('Image is Corrupted')

        with ThreadPoolExecutor(max_workers=4) as executor:

            # Pass the parameters to the functions using the `args` parameter
            amazon_future = executor.submit(
                getAmazonData, brand, color, style, category)
            myntra_future = executor.submit(
                getMyntraData, brand, color, style, category)
            ajio_future = executor.submit(
                getAjioData, brand, color, style, category)
            flipkart_future = executor.submit(
                getFlipkartData, brand, color, style, category)

            # Wait for the results
            amazon = amazon_future.result()
            myntra = myntra_future.result()
            ajio = ajio_future.result()
            flipkart = flipkart_future.result()

        print('******************   Amazon     ***************************')
        print(amazon)
        print('*********************************************')

        print('*********************    Myntra     ************************')
        print(myntra)
        print('*********************************************')
        print('*********************    Flipkart    ************************')
        print(flipkart)
        print('*********************************************')
        print('*********************    Ajio    ************************')
        print(ajio)
        print('*********************************************')

        res = jsonify({'amazon': amazon, 'myntra': myntra,
                      'flipkart': flipkart, 'ajio': ajio})
  
        return res

    return jsonify('You are not authenticated')