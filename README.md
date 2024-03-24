# AI for Personalized Customer Service Chatbots

AI chatbot is a machine learning based intelligent chatbot designed to provide human-like conversation. Chatbot is created with intention to shorten the gap between business and customer.

# Technologies Used:
- Python3
- Django Framework
- Tensorflow
- NLTK
- Numpy
- HTML, CSS
- JavaScript

# How to run ChatBot on your computer 

- Install the required packages.
"pip install -r requirements.txt"

- First create `secret key` for project

- Requirements for creating new key:
	- Be a minimum of 50 characters in length
	- Contain a minimum of 5 unique characters
	- Not be prefixed with "django-insecure-"

- Now open project directory where `settings.py` is located.
- Create new `.env` file and add the newly generated `secret key`
- .env file should look like this:

"SECRET_KEY = 'dlm*zt#1-3g!xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'"

- Save the .env file
- Do the migrations
"python manage.py makemigrations" 
"python manage.py migrate"

- And run the project.
"python manage.py runserver"

# Modifying .json file
-You can change intents, tags and responses by modifying .json file 

***
- ChatBot image are taken from [flaticon](https://www.flaticon.com/free-icons/bot)
- User image are taken from [freeiconspng](https://www.freeiconspng.com/img/7563)
***
