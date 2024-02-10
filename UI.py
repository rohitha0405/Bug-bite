import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the trained model
model = tf.keras.models.load_model('./trained_model.h5')

# Define the path to your CSV file
train_csv_path = "./train/_annotations.csv"
train_image_dir = "./train/"

# Read the CSV file into train_df
train_df = pd.read_csv(train_csv_path)
train_df = train_df[train_df['class'] != 'bug-bite']   #its not a proper class so its not necessary to be there in the dataset

# Verify unique classes after removing 'bug-bite'
print(train_df['class'].unique())
# Define batch size
batch_size = 64

# Initialize ImageDataGenerator for training data
train_datagen = ImageDataGenerator(rescale=1./255)

# Create train_generator
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    directory=train_image_dir,
    x_col="filename",
    y_col="class",
    target_size=(224, 224),  # Adjusted target size
    batch_size=batch_size,
    class_mode="categorical"
)

# Function to preprocess the uploaded image
def load_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # normalize the image
    return img_array

# Function to predict bug type
def predict_bug_type(image_array):
    prediction = model.predict(image_array)
    class_labels = train_generator.class_indices
    predicted_class = np.argmax(prediction)
    predicted_label = list(class_labels.keys())[predicted_class]
    confidence_score = np.max(prediction)
    return predicted_label, confidence_score, prediction

# Function to display prediction and confidence score
def display_prediction(prediction, confidence_score):
    st.write("### Predicted Class:", prediction)
    st.write("### Confidence Score:", confidence_score)

# Function to visualize model predictions
def visualize_predictions(prediction):
    labels = list(train_generator.class_indices.keys())
    fig, ax = plt.subplots()
    ax.bar(labels, prediction[0])
    ax.set_xlabel('Bug Types')
    ax.set_ylabel('Probability')
    ax.set_title('Bug Bite Prediction Probabilities')
    st.pyplot(fig)

# Function to get user feedback
def get_user_feedback():
    st.write("Was the prediction correct?")
    feedback = st.radio("Feedback", ("Yes", "No"))
    return feedback

# Function to get recommendations for bug classes
def get_recommendations(bug_class):
    recommendations = {
        'Bed Bug': {
            'Precautions': "1. Wash bedding and clothes in hot water to kill any bed bugs.\n2. Vacuum frequently, including mattresses, box springs, and bed frames.\n3. Seal cracks and crevices in walls, floors, and furniture.\n4. Use mattress encasements to trap bed bugs and prevent them from biting.\n5. Declutter your home to eliminate hiding spots for bed bugs.",
            'Treatment': "1. Use insecticides specifically labeled for bed bugs, following the manufacturer's instructions carefully.\n2. Use steam or heat treatments to kill bed bugs and their eggs in infested areas.\n3. Hire a pest control professional with experience in bed bug extermination.\n4. Launder infested items in hot water and dry them on the highest heat setting.\n5. Dispose of infested items properly to prevent spreading bed bugs to other areas."
        },
        'orumcek': {
            'Precautions': "1. Keep your home clean and clutter-free to reduce hiding spots for spiders.\n2. Use spider repellents or natural remedies like peppermint oil to deter spiders from entering your home.\n3. Seal cracks and gaps in walls, floors, and windows to prevent spiders from entering.\n4. Remove outdoor debris like leaves and woodpiles, which can attract spiders.\n5. Install screens on windows and doors to keep spiders out while allowing fresh air to circulate.",
            'Treatment': "1. Remove spider webs and egg sacs using a vacuum cleaner or broom.\n2. Use insecticides labeled for spider control, focusing on areas where spiders are commonly found.\n3. Hire a pest control professional for severe spider infestations or if you're unsure how to safely handle the situation.\n4. Use spider traps or sticky traps to capture spiders and monitor their activity.\n5. Seal entry points around your home to prevent spiders from returning."
        },
        'insan-pire': {
            'Precautions': "1. Wash bedding and clothes in hot water to kill any fleas and their eggs.\n2. Vacuum frequently, focusing on carpets, rugs, and upholstered furniture.\n3. Use mattress covers and pillow protectors to trap fleas and prevent them from biting.\n4. Avoid sharing personal items like towels and bedding, which can spread fleas.\n5. Treat pets for fleas regularly with veterinarian-approved products.",
            'Treatment': "1. Apply anti-itch creams or lotions to relieve itching and discomfort from flea bites.\n2. Use antihistamines to reduce allergic reactions and inflammation caused by flea bites.\n3. Use insecticides labeled for flea control on carpets, furniture, and pet bedding.\n4. Wash affected areas with soap and water to clean flea bites and prevent infection.\n5. Seek medical attention for severe reactions or if flea bites become infected."
        },
        'Tick': {
            'Precautions': "1. Avoid tick-infested areas like tall grass, wooded areas, and leaf piles, especially during peak tick season.\n2. Wear protective clothing, including long sleeves, pants, and closed-toe shoes, when hiking or spending time outdoors.\n3. Use insect repellents containing DEET or picaridin on exposed skin and clothing to repel ticks.\n4. Perform tick checks after outdoor activities, paying special attention to areas like the scalp, armpits, and groin.\n5. Shower soon after coming indoors to wash away any ticks that may be crawling on your skin.",
            'Treatment': "1. Use fine-tipped tweezers to grasp the tick as close to the skin's surface as possible.\n2. Gently pull upward with steady pressure, being careful not to twist or crush the tick's body.\n3. Clean the bite area with soap and water to reduce the risk of infection.\n4. Apply antiseptic to the bite to prevent bacterial growth and promote healing.\n5. Watch for signs of tick-borne illness, such as fever, rash, or flu-like symptoms, and seek medical attention if necessary."
        },
        'sinek': {
            'Precautions': "1. Keep food stored securely in airtight containers to prevent attracting flies.\n2. Clean spills and crumbs promptly, especially in kitchens and dining areas.\n3. Use insect screens on windows and doors to prevent flies from entering your home.\n4. Dispose of garbage regularly and keep outdoor trash cans tightly sealed.\n5. Clean kitchen appliances and counters regularly to remove food residue and potential breeding sites for flies.",
            'Treatment': "1. Use fly swatters or fly traps to capture and eliminate flies indoors.\n2. Clean up breeding sites like decaying organic matter and standing water around your home.\n3. Use insecticides labeled for fly control in areas where flies are a persistent problem.\n4. Repair damaged window screens and door seals to prevent flies from entering your home.\n5. Hire a pest control professional for severe fly infestations or if home remedies are ineffective."
        },
        'kene': {
            'Precautions': "1. Avoid wooded and grassy areas where ticks are commonly found, especially during peak tick season.\n2. Wear protective clothing, including long sleeves, pants, and closed-toe shoes, when hiking or spending time outdoors.\n3. Use insect repellents containing DEET or picaridin on exposed skin and clothing to repel ticks.\n4. Perform tick checks after outdoor activities, paying special attention to areas like the scalp, armpits, and groin.\n5. Shower soon after coming indoors to wash away any ticks that may be crawling on your skin.",
            'Treatment': "1. Use fine-tipped tweezers to grasp the tick as close to the skin's surface as possible.\n2. Gently pull upward with steady pressure, being careful not to twist or crush the tick's body.\n3. Clean the bite area with soap and water to reduce the risk of infection.\n4. Apply antiseptic to the bite to prevent bacterial growth and promote healing.\n5. Watch for signs of tick-borne illness, such as fever, rash, or flu-like symptoms, and seek medical attention if necessary."
        },
        'ari': {
            'Precautions': "1. Keep food stored securely in airtight containers to prevent attracting bees.\n2. Clean up food and drink spills promptly, especially when dining outdoors.\n3. Seal cracks and crevices in walls, floors, and windows to prevent bees from entering your home.\n4. Use insect screens on windows and doors to keep bees out while allowing fresh air to circulate.\n5. Avoid wearing strong fragrances and bright-colored clothing, which can attract bees.",
            'Treatment': "1. Use bee sting kits containing antihistamines and corticosteroids for allergic reactions to bee stings.\n2. Remove the bee sting promptly by scraping it out with a blunt object, such as a credit card.\n3. Clean the sting area with soap and water to reduce the risk of infection.\n4. Apply cold compresses or ice packs to reduce swelling and alleviate pain.\n5. Seek medical attention for severe allergic reactions or if multiple bee stings occur."
        },
        'pire': {
            'Precautions': "1. Use insect repellents containing DEET, picaridin, or oil of lemon eucalyptus to repel mosquitoes.\n2. Wear long sleeves, pants, and socks to cover exposed skin when outdoors, especially during peak mosquito times.\n3. Use bed nets treated with insecticides to protect sleeping areas from mosquitoes, especially in areas with high malaria transmission.\n4. Avoid outdoor activities during dawn and dusk, when mosquitoes are most active.\n5. Remove standing water around your home to eliminate mosquito breeding sites.",
            'Treatment': "1. Apply anti-itch creams or lotions containing hydrocortisone or calamine to relieve itching and discomfort from mosquito bites.\n2. Use antihistamines to reduce allergic reactions and inflammation caused by mosquito bites.\n3. Use cold compresses or ice packs to reduce swelling and alleviate pain from mosquito bites.\n4. Apply insect repellents containing ammonia or baking soda to neutralize mosquito saliva and reduce itching.\n5. Use insecticides or mosquito nets to protect yourself from mosquito bites when traveling to areas with high mosquito activity."
        }
        # Add recommendations for other bug classes here
    }
    return recommendations.get(bug_class, {})



def hero_section():
    st.title("Bug Bite Detection App")
    st.write("""
        ## Welcome to the Bug Bite Detection App!
        Discover the type of bug bite you've encountered.
        """)

def features_section():
    st.header("Key Features")
    st.write("""
        - Detect the type of bug bite from an uploaded image.
        - Learn about different bug bite types.
        - Get in touch with experts for further assistance.
        """)

def testimonials_section():
    st.header("Testimonials")
    st.write("""
        "This app is amazing! It helped me identify a mosquito bite quickly."
        - John Doe
        
        "I was skeptical at first, but this app accurately identified the bug bite type."
        - Jane Smith
        """)

def call_to_action():
    st.write("""
        Ready to identify your bug bite? Upload an image now!
        """)

def large_content():
    st.header("Learn About Bug Bites")
    st.write("""
        Bug bites can cause discomfort and sometimes even serious health issues. It's important to be able to identify different types of bug bites to know how to treat them properly.
        
        Here are some common types of bug bites:
        - **Bed Bug**: Typically found in beds and furniture. Causes itchy red welts.
        - **Spider Bite**: Can vary in severity depending on the spider species. May cause redness, swelling, and pain.
        - **Human Flea Bite**: Usually found on the lower body. Causes itchy red bumps.
        - **Tick Bite**: Can transmit diseases like Lyme disease and Rocky Mountain spotted fever.
        - **Mosquito Bite**: Causes red, itchy bumps and can transmit diseases like malaria and dengue fever.
        - **Fly Bite**: May cause pain, swelling, and redness.
        - **Bee Sting**: Can cause pain, swelling, and allergic reactions in some individuals.
        
        Understanding the characteristics and recommended treatments for each type of bug bite can help you take appropriate action if you or someone you know gets bitten.
        """)

def bug_bite_detection():
    st.title('Bug Bite Detection')
    st.write("""
        ### Upload an image of a bug bite to detect its type.
        """)

    # Upload a single image file
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        img_array = load_image(uploaded_file)

        # Get the model's prediction
        prediction, confidence_score, raw_prediction = predict_bug_type(img_array)

        # Display prediction and confidence score
        display_prediction(prediction, confidence_score)

        # Visualize predictions
        visualize_predictions(raw_prediction)

        # Get recommendations
        recommendations = get_recommendations(prediction)

        if recommendations:
            st.header("Recommendations")
            st.write("### Precautions:")
            st.write(recommendations.get('Precautions', 'No precautions found.'))
            st.write("### Treatment:")
            st.write(recommendations.get('Treatment', 'No treatment found.'))

        # Get user feedback
        feedback = get_user_feedback()
        # You can handle feedback here

def contact_us():
    st.title("Contact Us")
    st.write("""
        ### Get in touch with our experts for further assistance.
        - Email: rohitha2605@gmail.com
        - Phone: +91 9154947666
        - Address: IARE, Dundigal...
        """)

def about_project():
    st.title("About Project")
    st.write("""
        ## Bug Bite Detection App
        This project aims to help people identify different types of bug bites
        through image recognition technology. By leveraging machine learning,
        we can quickly determine the type of bug bite based on uploaded images.
        
        ### Our Mission
        Our mission is to provide a user-friendly platform that empowers individuals
        to identify bug bites accurately and take appropriate actions.
        
        ### Team Members
        - Donga Poojitha (TEam-Lead)
        - Tanneru Rohitha (BAckend Developer)
        - Mohiuddin(UI/UX Framework)
        
        ### Technologies Used
        - Python
        - TensorFlow
        - Streamlit
        """)

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Bug Bite Detection", "Contact Us", "About Project"])

    if page == "Home":
        hero_section()
        st.write("---")
        features_section()
        st.write("---")
        testimonials_section()
        st.write("---")
        call_to_action()
        st.write("---")
        large_content()

    elif page == "Bug Bite Detection":
        bug_bite_detection()

    elif page == "Contact Us":
        contact_us()

    elif page == "About Project":
        about_project()

if __name__ == "__main__":
    main()
