# Video to GIF Transcription and Tagging Tool

This tool allows users to transcribe audio from video files, generate GIFs, upload them to GIPHY, and create relevant tags and captions using Anthropic's Claude model. The tool uses various libraries such as `cv2`, `moviepy`, `pydub`, and `streamlit` to perform these tasks.

This project aims to create a versatile platform that converts videos into GIFs and stickers, with features that allow for adding captions and watermarks, and the option to animate these elements. The project also integrates multiple sources for video input, including direct uploads, YouTube links, and Google Photos, while providing additional functionalities such as uploading GIFs to Giphy, downloading them in a ZIP file, and editing captions. A key aspect of the project involves using Google OAuth for login and Stripe for payment processing. Currently, the project is built on the Streamlit framework.

### Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Environment Variables](#environment_variables)
4. [Features](#Features)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/FlicLabs/videotogif-public.git
   cd videotogif
   ```

2. **Create and activate a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Set up environment variables:**

   - Create a `.env` file in the root directory of the project and add the following environment variables:

   ```env
   ANTHROPIC_API_KEY=your_anthropic_api_key
   OPENAI_API_KEY=your_OpenAI_API_Key
   ```

   - Set Google Photos API credentials in a `secret\credentials.json` file

4. **Install the required Python packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Start the Application**:

   ```bash
   streamlit run main.py
   ```

2. **Select Input Source**:

   - **Upload Video File**: Choose a video file from your local machine.
   - **YouTube Link**: Provide a YouTube video URL.
   - **YouTube Channel**: Enter a channel link and specify the number of videos to download.
   - **Google Photos**: Authenticate with Google and select an album.

3. **Customize Your GIFs**:

   - Set text options such as font, size, color, shadow, outline, etc.
   - Add a watermark if desired.
   - Optionally upload GIFs to GIPHY.

4. **Generate and Download**:
   - Create GIFs based on your selections.
   - Download the GIFs as a ZIP file.

### Environment_Variables

Set API credentials in a `secret\credentials.json` file.
The following environment variables need to be set in a `.env` file:

- `ANTHROPIC_API_KEY`: API key for Anthropic's Claude model.
- `ALBUM_ID`: ID of Google Photos Album.
- `OPENAI_API_KEY`: ID of Google Photos Album.

### Features

- **Video Downloading**: Download videos from YouTube (individual links or entire channels) and Google Photos albums.
- **Video Processing**: Compress, resize, and trim videos. Detect and skip corrupt or soundless videos.
- **GIF Generation**: Create GIFs or stickers with or without text overlays. Add captions, timestamps, and watermarks.
- **Text Customization**: Customize text font, size, color, shadow, outline, and positioning.
- **GIPHY Upload**: Option to upload generated GIFs to GIPHY.
- **User Authentication**: Authenticate with Google Photos API to access user albums.

### ASSIGNMENT TASK

- Understand the entire code flow.
- Create an amazing full-stack application with killer UI/UX, login/signup, database, and subscription integration.
- Redesign the backend code as a FastAPI for smooth integration with the frontend.
- Automate the Tenor API, including the automatic upload of GIFs to a specific account on the Tenor website.
- Fix the Edit Caption functionality in the dashboard, which is currently not working.
- Develop a database for the full-stack application, including login/signup, subscription purchasing, and credit storage.
- Integrate subscriptions using Stripe, ensuring a proper connection between the frontend and backend to secure all confidential data.

### PAID OPEN AI KEY

- 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
- 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

If in case they stop working then use any free version and focus on other aspects by ignoring this step.

Thankyou!! Looking forward to have you in the team as an founding member for this application. 
