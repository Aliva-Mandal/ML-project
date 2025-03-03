# ML-project

Installation and Execution Guide 
1️ Install Required Dependencies Run the following command to install all necessary Python libraries:

pip install pandas numpy torch tensorflow scikit-learn 

2️ Prepare the Dataset Place your dataset (spotify.csv) inside a dataset/ directory. Ensure the CSV contains columns: track_name, artist_name, and lyrics. 
3️ Train the Models Execute the script to train both TensorFlow (Keras) and PyTorch models: 

python your_script.py 
This will preprocess data, train models, and save them in the models/ directory. 

The tokenizer (tokenizer.pkl), Keras model (keras_model.h5), and PyTorch model (pytorch_model.pth) will be saved. 
4️ Predict a Song from Lyrics Run the script again, and enter a lyric snippet when prompted:

python your_script.py Example: Enter a song lyric snippet: Hello, is it me you're looking for? TensorFlow Identified Song ID: 12 PyTorch Identified Song ID: 12
