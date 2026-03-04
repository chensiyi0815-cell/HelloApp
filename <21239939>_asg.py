# Assignment: ISOM5240 Individual Assignment - Storytelling Application
# Objective: Process an image to generate a 50-100 word story and convert to audio. [cite: 5, 6, 7]
# Target Audience: 3-10 year-old kids. 

# Dependency Installation (Run this in a cell if using .ipynb) [cite: 31]
# !pip install "transformers < 5.0.0" streamlit torch sentencepiece librosa

import streamlit as st
from transformers import pipeline
import torch
import os

# ---------------------------------------------------------
# 1. Model Loading with Caching
# ---------------------------------------------------------
# Caching models ensures they are loaded only once to save memory and time. 
@st.cache_resource
def load_ai_models():
    # Stage 1: Image Captioning (Requirement 21) [cite: 21, 22]
    # Recommended model: Salesforce/blip-image-captioning-base
    caption_pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    
    # Stage 2: Story Generation (Requirement 24) [cite: 24]
    # We use 'TinyStories-1M' which uses simple vocabulary suitable for 3-10 year olds. 
    story_pipe = pipeline("text-generation", model="roneneldan/TinyStories-1M")
    
    # Stage 3: Text-to-Speech Conversion (Requirement 25) [cite: 25, 27]
    tts_pipe = pipeline("text-to-audio", model="Matthijs/mms-tts-eng")
    
    return caption_pipe, story_pipe, tts_pipe

# ---------------------------------------------------------
# 2. Main Application UI and Logic
# ---------------------------------------------------------
def main():
    # Set page configuration for a kid-friendly interface 
    st.set_page_config(page_title="Kids' Magic Storybook", page_icon="🦄")
    st.header("✨ Turn Your Image into a Magical Story!")
    st.write("Welcome, little explorer! Upload a picture to start your adventure.")

    # Initialize the pre-trained models [cite: 11]
    img_to_text, text_to_story, story_to_audio = load_ai_models()

    # Requirement: Image Input - Upload by filename in working directory [cite: 13, 14]
    uploaded_file = st.file_uploader("Select an Image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save the file locally to the current working directory 
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display the uploaded image 
        st.image(uploaded_file, caption="Your Uploaded Image", use_container_width=True)

        # Processing Stages
        with st.status("The magic wand is working..."):
            
            # --- Stage 1: Image Processing & Captioning --- [cite: 20, 21]
            st.write("Step 1: Analyzing the picture...")
            caption_result = img_to_text(uploaded_file.name)[0]["generated_text"]
            st.info(f"I see: {caption_result}")

            # --- Stage 2: Story Generation (50-100 words) --- [cite: 15, 16]
            st.write("Step 2: Writing a story just for you...")
            # Prompt Engineering to ensure kid-friendly content
            prompt = f"Once upon a time, there was {caption_result}. A simple story for a child: "
            
            # Constraints: min_new_tokens=50 and max_new_tokens=100 for word count 
            story_output = text_to_story(
                prompt, 
                max_new_tokens=100, 
                min_new_tokens=50, 
                do_sample=True,
                temperature=0.8
            )
            full_story = story_output[0]['generated_text'].replace(prompt, "").strip()
            
            st.subheader("📖 Your Magical Story:")
            st.write(full_story)

            # --- Stage 3: Text-to-Speech Conversion --- [cite: 17, 18, 27]
            st.write("Step 3: Preparing the narrator's voice...")
            audio_data = story_to_audio(full_story)
            
            # Interactive Audio Playback [cite: 18, 38]
            st.audio(audio_data["audio"], sample_rate=audio_data["sampling_rate"])
            st.success("Your story is ready! Click play to listen. 🔊")

if __name__ == "__main__":
    main()
