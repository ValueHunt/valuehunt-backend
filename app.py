from flask import Flask
from flask import request,jsonify
from flask_cors import CORS
from dotenv import load_dotenv, find_dotenv
from PIL import Image
import io
import pymongo
import os

# config = dotenv_values(".env")
app = Flask(__name__)
CORS(app)

actual_token = os.environ.get("token")

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["ValueHunt"]



def getAmazonData(brand,color,style,size,clothType):
    pass

def getFlipkartData(brand,color,style,size,clothType):
    pass

def getMyntraData(brand,color,style,size,clothType):
    pass

def getAjioData(brand,color,style,size,clothType):
    pass

def getColorOfCloth(clothImg):
    pass

def getStyleOfCloth(clothImg):
    pass





# *****************************Validator****************
def validate(token):
    if(token==actual_token):
        return True
    return False

# *********************************Insert Contact***************************
def insertContact(name,email,msg):
    try:

        col = db["contact"]
        data={'name':name,'email':email,'msg':msg}

        if(col.find_one(data)):

            return jsonify('Contact Already Present')
        

        user_id = col.insert_one(data).inserted_id

        return jsonify('Contact Details has been Saved')

    except:
        return jsonify("It's our problem not yours")

# ********************************Insert Vh*****************************
def insertVh(clothImg,size,brand,typeOfCloth):
    try:
        im = Image.open(clothImg)
        rgb_im = im.convert('RGB')
        image_bytes = io.BytesIO()
        rgb_im.save(image_bytes, format='JPEG')
        image = {
            'data': image_bytes.getvalue()
            
            
            }
        col = db["VH"]
        data={'clothImg':image,'size':size,'brand':brand,'typeOfCloth':typeOfCloth}

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
        name=request.json['name']
        email=request.json['email']
        msg=request.json['msg']
       
        res=insertContact(name, email, msg)
        return res
    
    
    return jsonify('You are not authenticated')
            


# ***********************************VH Route************************
@app.route("/vh", methods=['POST'])
def vh():
    headers = request.headers
    bearer = headers.get('Authorization')    
    token = bearer.split()[1]
    if( validate(token)):
        
        clothImg=request.files['clothImg']
        size=request.form.get('size')
        brand=request.form.get('brand')
        clothType=request.form.get('clothType')
       
        insertVh(clothImg,size,brand,clothType)

        color=getColorOfCloth(clothImg)
        style=getStyleOfCloth(clothImg)

        amazon=getAmazonData(brand,color,style,size,clothType)
        flipkart=getFlipkartData(brand, color, style, size, clothType)
        ajio=getAjioData(brand, color, style, size, clothType)
        myntra=getMyntraData(brand, color, style, size, clothType)

        res = jsonify({amazon,flipkart,ajio,myntra})
        return res
    
    
    return jsonify('You are not authenticated')
            

