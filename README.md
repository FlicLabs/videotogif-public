# Video to GIF Transcription and Tagging Tool

This tool allows users to transcribe audio from video files, generate GIFs, upload them to GIPHY, and create relevant tags and captions using Anthropic's Claude model. The tool uses various libraries such as `cv2`, `moviepy`, `pydub`, and `streamlit` to perform these tasks.

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
