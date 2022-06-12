import numpy as np
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)

dic = {0 : 'Cassava Bacterial Blight', 1 : 'Cassava Brown Streak Disease', 2 :'Cassava Green Mite', 3 : 'Cassava Mosaic Disease', 4 : 'Healthy'}

model = load_model('cassava_model.h5')

model.make_predict_function()

def predict_label(img_path):
    i = image.load_img(img_path, target_size=(256,256))
    i = image.img_to_array(i)
    i = i.reshape(1, 256,256,3)
    p = np.argmax(model.predict(i))
    return dic[p]

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)