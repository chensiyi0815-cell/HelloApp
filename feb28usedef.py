from transformers import pipeline
from PIL import Image
import streamlit as st

def ageClassifier(imgFilename,modelname):
  # Load the age classification pipeline
  # The code below should be placed in the main part of the program
  age_classifier = pipeline("image-classification", model = modelname)
  
  image_name = imgFilename
  image_name = Image.open(image_name).convert("RGB")
  
  # Classify age
  age_predictions = age_classifier(image_name)
  return age_prediction

def main():
  # Streamlit UI
  st.header("Title: Age Classification using ViT")
  age_predection = ageClassifier("middleagedMan.jpg",modelname = nateraw/vit-age-classifier)

  st.write(age_predictions)
  st.write("Predicted Age Range:")
  st.write(f"Age range: {age_predictions[0]['label']}")

  st.write('Done')

if __name__ == "__main__":
    main()
