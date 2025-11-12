import streamlit as st
import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Function to calculate the distance between two points
def calculate_distance(p1, p2):
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

# Function to recognize hand gestures
def get_gesture(hand_landmarks):
    # Extract key finger landmarks
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    # List of fingertip positions (excluding the thumb)
    fingers = [index_tip, middle_tip, ring_tip, pinky_tip]

    # Gesture recognition logic
    # Thumbs Up 👍
    if thumb_tip.y < thumb_ip.y and all(f.y > middle_mcp.y for f in fingers):
        return "Thumbs Up"

    # Thumbs Down 👎
    elif thumb_tip.y > thumb_ip.y and all(f.y > middle_mcp.y for f in fingers):
        return "Thumbs Down"

    # Open Hand ✋ (All fingers extended)
    elif all(f.y < middle_mcp.y for f in fingers) and thumb_tip.y < thumb_ip.y:
        return "Open Hand"

    # Fist ✊ (All fingers curled)
    elif all(f.y > middle_mcp.y for f in fingers) and calculate_distance(thumb_tip, index_tip) < 0.05:
        return "Fist"

    # Index Finger Up ☝️
    elif index_tip.y < index_mcp.y and all(f.y > middle_mcp.y for f in fingers[1:]):
        return "Index Finger Up"

    # Peace Sign ✌️ (Index and middle fingers extended)
    elif index_tip.y < index_mcp.y and middle_tip.y < middle_mcp.y and all(f.y > middle_mcp.y for f in fingers[2:]):
        return "Peace Sign"

    # Rock Sign 🤘 (Index and pinky fingers extended)
    elif index_tip.y < index_mcp.y and pinky_tip.y < middle_mcp.y and all(f.y > middle_mcp.y for f in fingers[1:3]):
        return "Rock Sign"

    # OK Sign 👌 (Thumb and index finger touching)
    elif calculate_distance(thumb_tip, index_tip) < 0.05 and all(tip.y > middle_mcp.y for tip in [middle_tip, ring_tip, pinky_tip]):
        return "OK Sign"

    return "No Recognized Gesture"

# Streamlit UI
st.set_page_config(page_title="Hand Gesture Recognition", layout="wide")
st.title("✋ Hand Gesture Recognition")

# Create a placeholder for the video feed
video_placeholder = st.empty()

# The camera input will prompt the user for access
st.write("Allow camera access to begin gesture recognition.")
image_file_buffer = st.camera_input(" ")

if image_file_buffer:
    # Read the image from the buffer
    bytes_data = image_file_buffer.getvalue()
    frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)
    
    # Convert BGR to RGB (MediaPipe expects RGB input)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    # Prepare layout with two columns, one for the video and one for the gesture label
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Live Video Feed")
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get and display gesture
                gesture = get_gesture(hand_landmarks)
                cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Convert BGR back to RGB for Streamlit display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Display the processed frame
        st.image(frame, channels="RGB")
        
    with col2:
        st.subheader("Recognized Gesture")
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                gesture = get_gesture(hand_landmarks)
                st.write(f"**Gesture:** {gesture}")
                if gesture == "Thumbs Up":
                    st.write("👍")
                elif gesture == "Thumbs Down":
                    st.write("👎")
                elif gesture == "Open Hand":
                    st.write("✋")
                elif gesture == "Fist":
                    st.write("✊")
                elif gesture == "Index Finger Up":
                    st.write("☝️")
                elif gesture == "Peace Sign":
                    st.write("✌️")
                elif gesture == "Rock Sign":
                    st.write("🤘")
                elif gesture == "OK Sign":
                    st.write("👌")
        else:
            st.write("No hand detected.")

else:
    # This block displays the initial "blank square"
    st.markdown("<div style='border: 2px solid #ccc; height: 400px; width: 100%;'></div>", unsafe_allow_html=True)
