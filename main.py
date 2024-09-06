import io
import os
import cv2
import json
import shutil
import base64
import yt_dlp
import openai
import zipfile
import logging
import tempfile
import requests
import anthropic
import numpy as np
import mediapipe as mp
import streamlit as st
from time import sleep
from openai import OpenAI
from textwrap import dedent
from natsort import natsorted
from dotenv import load_dotenv
from selenium import webdriver
from st_paywall import add_auth
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from PIL import ImageDraw, ImageFont, ImageSequence, Image

load_dotenv()

# st.title("Welcome to Video To Gif Generator!")
# st.markdown("""
# ### Subscribe for Access!
# - **🚀 Innovative Tools:** Get access to state-of-the-art tools that will enhance your Gif Production.
# - **🔒 Secure:** Your data's safety is our top priority.
# - **📈 Scalable:** Our solutions grow with your needs, ensuring long-term value.
# - **🛠️ Easy Integration:** Seamlessly integrates with your existing Gif Libraries.
# ### How to Access!
# - 🎟️ Click the **Login with Google** to subscribe and unlock amazing features!

# """)
# add_auth(required=True)

Json_Formatting_iteration_limit = 5
LLM_Model = "claude-3-5-sonnet-20240620"
SCOPES = ['https://www.googleapis.com/auth/photoslibrary.readonly']
has_caption_result = False

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    level=logging.INFO
)

logging.info("Streamlit app started")


def resize_video_to_360p(input_file):
    clip = VideoFileClip(input_file)

    target_height = 360

    aspect_ratio = clip.size[0] / clip.size[1]
    new_width = int(target_height * aspect_ratio)

    resized_clip = clip.resize(height=target_height)

    temp_output_file = input_file.replace(".mp4", "_temp_360p.mp4")

    resized_clip.write_videofile(temp_output_file, codec='libx264')

    os.remove(input_file)
    os.rename(temp_output_file, input_file)


def process_videos_in_folder_to_360(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".mp4"):
            input_file = os.path.join(folder_path, filename)
            resize_video_to_360p(input_file)
            print(f"Processed {filename}")


def edit_gif(gif_path, new_text, font_path, font_size, shadow_offset, shadow_color, outline_color, font_color1, font_color2, output_folder="gifs", backup_folder="Back up Gifs without texts"):
    backup_path = os.path.join(backup_folder, os.path.basename(gif_path))

    edited_gif_path = os.path.join(output_folder, os.path.basename(gif_path))
    if os.path.exists(edited_gif_path):
        os.remove(edited_gif_path)
        print(f"{edited_gif_path} has been deleted.")
    else:
        print(f"{edited_gif_path} does not exist.")

    add_text_to_gif(backup_path, edited_gif_path, new_text, font_path, font_size,
                    shadow_offset, shadow_color, outline_color, font_color1, font_color2)

    st.success(f"Edited GIF saved at {
               edited_gif_path} and original backed up at {backup_path}")


def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def check_caption(base64_image):
    api_key = os.getenv('OPENAI_API_KEY')
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Does this image contain any LARGE SIZED CAPTIONS?\n- STRICTLY answer in format:\n  If yes respond 'True' else 'False'"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }
    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    result = response.json()
    message_content = result['choices'][0]['message']['content']
    return eval(message_content)


def download_video_from_YT_link(url, output_path='.'):
    try:
        ydl_opts = {
            'format': 'best',
            'outtmpl': f'{output_path}/%(title)s.%(ext)s',
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print("Download completed!")
    except Exception as e:
        print(f"An error occurred: {e}")


def download_YT_channel_ALL_videos(channel_url, output_path='.'):
    try:
        ydl_opts = {
            'format': 'best',
            'outtmpl': f'{output_path}/%(title)s.%(ext)s',
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([channel_url])
        print("Download completed!")
    except Exception as e:
        print(f"An error occurred: {e}")


def download_YT_channel_Specific_Number_videos(channel_url, output_path='.', max_videos=2):
    try:
        ydl_opts = {
            'format': 'best',
            'outtmpl': f'{output_path}/%(title)s.%(ext)s',
            'max_downloads': max_videos,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([channel_url])
        print(f"Download of up to {max_videos} videos completed!")
    except Exception as e:
        print(f"An error occurred: {e}")


def get_gif_files(folder):
    return [f for f in os.listdir(folder) if f.endswith('.gif')]


def count_files(directory):
    file_count = 0
    for root, dirs, files in os.walk(directory):
        file_count += len(files)
    return file_count


def zip_specific_folder(folder_name, zip_file_name="gif_files.zip"):
    with zipfile.ZipFile(zip_file_name, 'w') as zipf:
        for root, _, files in os.walk(folder_name):
            for file in files:
                zipf.write(os.path.join(root, file))
    return zip_file_name


def is_video_corrupt(video_path):
    print("entered")
    try:
        clip = VideoFileClip(video_path)
        clip.reader.close()
        return False
    except Exception as e:
        return True


def trim_silence_start(video_path, output_path, silence_threshold=-41.39, chunk_size=10):
    video = VideoFileClip(video_path)
    audio_path = tempfile.mktemp(suffix='.wav')
    video.audio.write_audiofile(audio_path)

    audio = AudioSegment.from_file(audio_path)
    nonsilent_ranges = detect_nonsilent(
        audio, min_silence_len=chunk_size, silence_thresh=silence_threshold)

    if nonsilent_ranges:
        start_trim = nonsilent_ranges[0][0]
        trimmed_video = video.subclip(start_trim / 1000, video.duration)
        trimmed_video.write_videofile(output_path, codec='libx264')

        print(f"Trimmed video saved to {output_path}")
    else:
        print("No non-silent part detected.")


def transcribe_audio_with_timestamps(audio_file_path: str):
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    audio_file_temp = "output_files/audio_temp.wav"
    extract_audio_from_video(audio_file_path, audio_file_temp)

    sleep(1)
    audio_file = open(audio_file_temp, "rb")
    transcript = client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-1",
        response_format="verbose_json",
        timestamp_granularities=["word"]
    )
    return transcript


def split_video_into_segments(video_path: str, segment_length: int = 30, overlap: int = 5) -> list:
    video = VideoFileClip(video_path)
    duration = int(video.duration)
    segments = []

    for start_time in range(0, duration, segment_length - overlap):
        end_time = min(start_time + segment_length, duration)
        segment_path = f"output_files/segment_{start_time}_{end_time}.mp4"
        video.subclip(start_time, end_time).write_videofile(
            segment_path, codec="libx264")
        segments.append(segment_path)

    video.close()
    return segments


def compress_gif_if_large(input_path: str, output_path: str, max_size_mb: int = 8, max_width: int = 500) -> None:
    file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
    print(f"Original GIF size: {file_size_mb:.2f} MB")

    if file_size_mb <= max_size_mb:
        print("GIF size is within the limit. No compression needed.")
        return


def upload_gif_to_giphy(file_path: str, tags: str, GIPHY_API_KEY, title="", source_post_url="https://jackjay.io", source_image_url="", max_retries=3) -> None:
    url = "https://upload.giphy.com/v1/gifs"
    giphy_api_key = GIPHY_API_KEY

    if giphy_api_key is None:
        raise ValueError(
            "API key not found. Please set the GIPHY_API_KEY environment variable.")

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    compressed_file_path = "compressed_" + file_path
    compress_gif_if_large(file_path, compressed_file_path)
    file_to_upload = compressed_file_path if os.path.exists(
        compressed_file_path) else file_path

    print(" --Gis are uploadeding")
    print(file_to_upload)
    for attempt in range(max_retries):
        try:
            with open(file_to_upload, 'rb') as file:
                payload = {
                    'api_key': giphy_api_key,
                    'tags': tags,
                    'title': title,
                    'source_post_url': source_post_url,
                    'source_image_url': source_image_url
                }
                files = {'file': file}

                print(f"Attempt {attempt + 1} to upload GIF...")
                response = requests.post(url, data=payload, files=files)
                print(f"Response Status Code: {response.status_code}")
                print(f"Response Content: {response.content}")

                response.raise_for_status()
                data = response.json()
                gif_url = "https://giphy.com/gifs/" + str(data['data']['id'])
                print(
                    f"GIF uploaded successfully! [Click here to visit the GIF]({gif_url})")
                return
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries:
                sleep(30 ** attempt)
            else:
                print("Max retries reached. GIF upload failed.")


def get_api_client():
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if api_key is None:
        raise ValueError(
            "API key not found. Please set the ANTHROPIC_API_KEY environment variable.")
    return anthropic.Client(api_key=api_key)


def generate_tags_for_gif_critic(Input_Text: str, iteration_limit: int = Json_Formatting_iteration_limit) -> str:
    api_key = os.getenv('ANTHROPIC_API_KEY')

    if api_key is None:
        raise ValueError(
            "API key not found. Please set the ANTHROPIC_API_KEY environment variable.")

    client = get_api_client()

    try:
        response = client.messages.create(
            model=LLM_Model,
            max_tokens=4000,
            temperature=0,
            messages=[{
                "role": "user",
                "content": dedent(f"""
                Your task is to carefully read the `INPUT TEXT` and output it in the specified JSON format.
                **IMPORTANT NOTE:**
                    - Output tags as specified in JSON format by removing any unnecessary formatting. This will be used as direct JSON format. Respond with an array of captions in JSON. ENSURE THE TAGS ARE EXACTLY AS IN THE INPUT TEXT. NO SINGLE WORD OR LETTER SHOULD BE DIFFERENT OR NEW.
                    - OUTPUT ONLY in specified JSON format. ABSOLUTELY NOTHING ELSE. NOT EVEN ANY SINGLE NOTE OR SINGLE CHARACTER OTHER THAN JSON FILE.
                **INPUT TEXT:**
                {Input_Text}

                **OUTPUT FORMAT:**
                    {{

                        "tags": "tag_1, tag_2, ... tag_20"
                    }}
                """)
            }]
        )
    except anthropic.errors.AnthropicError as e:
        print(f"An error occurred: {e}")
        return None

    response_content = response.content[0].text
    response_content = response_content.strip().replace(
        "```json", "").replace("```", "")

    try:
        temp = json.loads(response_content)
        if isinstance(temp, dict) and "tags" in temp and isinstance(temp["tags"], str):
            print("-- response_content")
            print(response_content)
            return response_content
        else:
            if iteration_limit > 0:
                return generate_tags_for_gif_critic(response_content, iteration_limit - 1)
            else:
                return None
    except json.JSONDecodeError:
        if iteration_limit > 0:
            return generate_tags_for_gif_critic(response_content, iteration_limit - 1)
        else:
            return None


def generate_timestamps_gif_captions_critic(input_text: str, iteration_limit: int = Json_Formatting_iteration_limit) -> str:
    client = get_api_client()

    try:
        response = client.messages.create(
            model=LLM_Model,
            max_tokens=4000,
            temperature=0,
            messages=[{
                "role": "user",
                "content": f"""Your task is to carefully read the `INPUT TEXT` and output it in the specified JSON format.
                **IMPORTANT NOTE:**
                    - Output captions sentences as specified in JSON format by removing any unnecessary formatting. This will be used as direct JSON format. Respond with an array of captions in JSON. ENSURE THE SELECTED SENTENCES and values ARE EXACTLY AS IN THE INPUT TEXT. NO SINGLE WORD OR LETTER SHOULD BE DIFFERENT OR NEW.
                    - OUTPUT ONLY in specified JSON format. ABSOLUTELY NOTHING ELSE. NOT EVEN ANY SINGLE NOTE OR SINGLE CHARACTER OTHER THAN JSON FILE.

                **INPUT TEXT:**
                {input_text}

                **OUTPUT FORMAT:**
                {{

                    "captions_with_timestamps": [
                        {{
                            "caption": "Sentence 1",
                            "start": "start_time",
                            "end": "end_time"
                        }},
                        {{
                            "caption": "Sentence 2",
                            "start": "start_time",
                            "end": "end_time"
                        }},
                        ...
                        {{
                            "caption": "Sentence n",
                            "start": "start_time",
                            "end": "end_time"
                        }}
                    ]
                }}
                """
            }]
        )
    except anthropic.errors.AnthropicError as e:
        print(f"An error occurred: {e}")
        return None

    response_content = response['content'].strip().replace(
        "```json", "").replace("```", "")

    try:
        temp = json.loads(response_content)
        if isinstance(temp, dict) and "captions_with_timestamps" in temp and isinstance(temp["captions_with_timestamps"], list):
            print("-- response_content")
            print(response_content)
            return response_content
        else:
            if iteration_limit > 0:
                return generate_timestamps_gif_captions_critic(response_content, iteration_limit - 1)
            else:
                return None
    except json.JSONDecodeError:
        if iteration_limit > 0:
            return generate_timestamps_gif_captions_critic(response_content, iteration_limit - 1)
        else:
            return None


def generate_timestamps_gif_captions(text_transcript: dict, gif_lines: str) -> dict:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        raise ValueError(
            "API key not found. Please set the openai_api_key environment variable.")

    client = OpenAI(api_key=openai_api_key)
    responses = []

    try:
        prompt = (
            f"""**Task:**
                Identify timestamps for sentences in `INPUT 1` within `INPUT 2` and output them in the desired `OUTPUT FORMAT`.

            **Understanding the Inputs:**
                - `INPUT 1`: Contains a single sentence, referred to as `sentence_1`.
                - `INPUT 2`: Contains a set of randomly ordered sentences with timestamps. Each timestamp might have multiple sentences, formatted like `"sentence_1 sentence_2"` along with start and end times.

            **Challenge:**
                Determine the precise moment `sentence_1` appears in `INPUT 2`. This may involve considering factors like:
                    - Whether `sentence_1` appears in the first half of the timestamp.
                    - The number of words in the combined sentences compared to `sentence_1` alone.

            **Output:**
                The desired output is the timestamp at which `sentence_1` appears in `INPUT 2`.

            **IMPORTANT NOTE:**
                - ENSURE THE SENTENCES IN THE OUTPUT ARE EXACTLY AS IN THE INPUT TEXT. NO SINGLE sentence OR LETTER SHOULD BE DIFFERENT OR NEW.
                - OUTPUT ONLY in specified JSON format. ABSOLUTELY NOTHING ELSE. NOT EVEN ANY SINGLE NOTE OR SINGLE CHARACTER OTHER THAN EXPECTED JSON FORMAT as specified in `OUTPUT FORMAT`.
                - For a given sentence in (INPUT_1), determine the time it most likely appears in (INPUT_2).

            **INPUT 1:**
            {gif_lines}

            **INPUT 2:**
            {text_transcript}

            **OUTPUT FORMAT:**
            {{

                "captions_with_timestamps": [
                    {{
                        "caption": "Sentence 1",
                        "start": "start_time",
                        "end": "end_time"
                    }},
                    {{
                        "caption": "Sentence 2",
                        "start": "start_time",
                        "end": "end_time"
                    }},
                    ...
                    {{
                        "caption": "Sentence n",
                        "start": "start_time",
                        "end": "end_time"
                    }}
                ]
            }}"""
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            responses.append(chunk.choices[0].delta.content)

    response_text = "".join(responses)
    print("response_text")
    print(response_text)
    if not response_text:
        raise ValueError("Response text is empty")

    start_index = response_text.find('{')
    end_index = response_text.rfind('}') + 1
    if start_index == -1 or end_index == -1:
        raise ValueError("JSON part not found in the response text")

    json_part = response_text[start_index:end_index]
    response_content = json.loads(json_part)

    try:
        temp = json.loads(json_part)
        if isinstance(temp, dict) and "captions_with_timestamps" in temp and isinstance(temp["captions_with_timestamps"], list):
            print(temp["captions_with_timestamps"])
            return response_content
        else:
            return generate_timestamps_gif_captions_critic(response_content)
    except json.JSONDecodeError:
        return generate_timestamps_gif_captions_critic(response_content)


def generate_tags_for_gif(gif_text: str) -> dict:
    word_count = len(gif_text.split()) + 1
    max_tags = 20 - word_count

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        raise ValueError(
            "API key not found. Please set the openai_api_key environment variable.")

    client = OpenAI(api_key=openai_api_key)
    responses = []

    try:
        prompt = (
            f"""Take a deep breath and think step by step:
                    You are making tags for a search engine of gifs and memes, people often use these to express emotion. Here are examples of the most used terms:
                        Happy birthday
                        Good night
                        Cat
                        Congratulations
                        Crying
                        Yes
                        Funny
                    The gif is of a person saying “{gif_text}” give {max_tags} single word syllables or multi word expressions related to the original text that work as more top gif search terms each separated by a comma in order of best suggestions.

                    IMPORTANT NOTE:
                        - The count of tags should be less than or equal to {max_tags}
                        - STRICTLY answer in JSON format:
                            {{
                                "tags": "tag_1, tag_2, ... tag_{max_tags}"
                            }}"""
        )

        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                responses.append(chunk.choices[0].delta.content)

        response_text = "".join(responses)
        if not response_text:
            raise ValueError("Response text is empty")

        start_index = response_text.find('{')
        end_index = response_text.rfind('}') + 1
        if start_index == -1 or end_index == -1:
            raise ValueError("JSON part not found in the response text")

        json_part = response_text[start_index:end_index]
        response_content = json.loads(json_part)

        existing_tags = response_content.get('tags', '').split(', ')
        existing_tags = existing_tags[:max_tags]

        new_tags = gif_text.split() + existing_tags
        new_tags = new_tags[:20]
        new_tags = list(set(new_tags))

        response_content['tags'] = ', '.join(new_tags)

        return response_content

    except json.JSONDecodeError:
        print("Failed to decode JSON response")
        return None
    except KeyError as e:
        print(f"Key error: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def create_GIF_caption_identifier_task_critic(Input_Text: str, iteration_limit: int = Json_Formatting_iteration_limit) -> str:
    api_key = os.getenv('ANTHROPIC_API_KEY')

    if api_key is None:
        raise ValueError(
            "API key not found. Please set the ANTHROPIC_API_KEY environment variable.")

    client = get_api_client()

    try:
        response = client.messages.create(
            model=LLM_Model,
            max_tokens=4000,
            temperature=0,
            messages=[{
                "role": "user",
                "content": f"""Your task is to carefully read the `INPUT TEXT` and output it in the specified JSON format.
                **IMPORTANT NOTE:**
                    - Output captions sentences as specified in JSON format by removing any unnecessary formatting. This will be used as direct JSON format. Respond with an array of captions in JSON. ENSURE THE SELECTED SENTENCES ARE EXACTLY AS IN THE INPUT TEXT. NO SINGLE WORD OR LETTER SHOULD BE DIFFERENT OR NEW.
                    - OUTPUT ONLY in specified JSON format. ABSOLUTELY NOTHING ELSE. NOT EVEN ANY SINGLE NOTE OR SINGLE CHARACTER OTHER THAN JSON FILE.

                **INPUT TEXT:**
                {Input_Text}

                **OUTPUT FORMAT:**
                    {{

                        "captions": [
                            "Sentence 1",
                            "Sentence 2",
                            ...
                            "Sentence n"
                        ]
                    }}
                """
            }]
        )
    except anthropic.errors.AnthropicError as e:
        print(f"An error occurred: {e}")
        return None

    response_content = response.content[0].text
    response_content = response_content.strip().replace(
        "```json", "").replace("```", "")

    try:
        temp = json.loads(response_content)
        if isinstance(temp, dict) and "captions" in temp and isinstance(temp["captions"], list):
            return response_content
        else:
            if iteration_limit > 0:
                return create_GIF_caption_identifier_task_critic(response_content, iteration_limit - 1)
            else:
                return None
    except json.JSONDecodeError:
        if iteration_limit > 0:
            return create_GIF_caption_identifier_task_critic(response_content, iteration_limit - 1)
        else:
            return None


def create_gif_caption_identifier_task(Text_Transcript: str) -> str:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        raise ValueError(
            "API key not found. Please set the openai_api_key environment variable.")

    client = OpenAI(api_key=openai_api_key)
    responses = []

    try:
        prompt = (
            f"""As a GIF caption expert, you're tasked with identifying sentences in the `INPUT TEXT` that can be turned into perfect text overlays for GIFs and outputting them in the expected `OUTPUT FORMAT`.
                **IMPORTANT NOTE:**
                    - ENSURE THE SELECTED SENTENCES ARE EXACTLY AS IN THE INPUT TEXT. NO SINGLE WORD OR LETTER SHOULD BE DIFFERENT OR NEW.
                    - OUTPUT ONLY in specified JSON format. ABSOLUTELY NOTHING ELSE. NOT EVEN ANY SINGLE NOTE OR SINGLE CHARACTER OTHER THAN EXPECTED JSON FORMAT as specified in `OUTPUT FORMAT`.
                    - list all sentences that are suitable for GIF captions, ensuring they are concise, relevant, and engaging.
                    - Don't accept any language except English.
                    - Keep GIF captions very short. Strictly adhere to this guideline. Each sentence should have fewer than 6 words

                **INPUT TEXT:**
                {Text_Transcript}

                **OUTPUT FORMAT:**
                    {{
                        "captions": [
                            "Sentence 1",
                            "Sentence 2",
                            ...
                            "Sentence n"
                        ]
                    }}
            """
        )

        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                responses.append(chunk.choices[0].delta.content)

        response_text = "".join(responses)
        print("-- response_text")
        print(response_text)
        if not response_text:
            raise ValueError("Response text is empty")

        start_index = response_text.find('{')
        end_index = response_text.rfind('}') + 1
        if start_index == -1 or end_index == -1:
            raise ValueError("JSON part not found in the response text")

        response_content = response_text[start_index:end_index]

        try:
            temp = json.loads(response_content)
            if isinstance(temp, dict) and "captions" in temp and isinstance(temp["captions"], list):
                print("-- response_content")
                print(response_content)
                return response_content
            else:
                return create_GIF_caption_identifier_task_critic(response_content)
        except json.JSONDecodeError:
            return create_GIF_caption_identifier_task_critic(response_content)

    except json.JSONDecodeError:
        print("Failed to decode JSON response")
        return None
    except KeyError as e:
        print(f"Key error: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def extract_audio_from_video(video_path, audio_path):
    clip = VideoFileClip(video_path)
    if clip.audio is not None:
        clip.audio.write_audiofile(audio_path)
        clip.close()
        return True
    else:
        return False


def check_silence(audio_path, silence_threshold=-39.2, chunk_size=10):
    audio = AudioSegment.from_file(audio_path)
    silent = True

    for start in range(0, len(audio), chunk_size):
        chunk = audio[start:start + chunk_size]
        if chunk.dBFS > silence_threshold:
            silent = False
            break

    return silent


def has_sound(video_path):
    audio_path = "output_files\\temp_audio_1.wav"
    if extract_audio_from_video(video_path, audio_path):
        is_silent = check_silence(audio_path)
        return not is_silent
    else:
        return False


def save_video_clip(video_path: str, start_time: float, end_time: float, output_path: str) -> None:
    video = VideoFileClip(video_path)
    clip = video.subclip(start_time, end_time)
    clip.write_videofile(output_path, codec="libx264")
    clip.close()
    video.close()


def wrap_text(text, font, max_width):
    lines = []
    words = text.split()

    for word in words:
        if font.getbbox(word)[2] > max_width:
            print(f"Skipping text wrapping: font size is too large for the max width ({
                  max_width}).")
            return []

    while words:
        line = ''
        while words:
            test_line = line + words[0] + ' '
            if font.getbbox(test_line)[2] <= max_width:
                line = test_line
                words.pop(0)
            else:
                break
        lines.append(line.strip())

    return lines


def add_text_to_gif(input_gif_path, output_gif_path, text, font_path, font_size=50, shadow_offset=(4, 4), shadow_color=(0, 0, 0, 128), outline_color=(0, 0, 0, 128), font_color1=(255, 255, 255, 255), font_color2=(204, 204, 255, 255), line_spacing=10, boldness=2):
    font = ImageFont.truetype(font_path, font_size)
    original_gif = Image.open(input_gif_path)
    frames = []

    max_width = original_gif.size[0] - 20
    lines = wrap_text(text, font, max_width)

    for frame_index, frame in enumerate(ImageSequence.Iterator(original_gif)):
        frame = frame.convert("RGBA")
        draw = ImageDraw.Draw(frame)

        frame_width = frame.size[0]
        frame_height = frame.size[1]

        total_text_height = sum([draw.textbbox((0, 0), line, font=font)[
                                3] for line in lines]) + (len(lines) - 1) * line_spacing

        y = frame_height - total_text_height - 10
        current_font_color = font_color1 if frame_index % 10 < 5 else font_color2

        for line in lines:
            text_bbox = draw.textbbox((0, 0), line, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_x = (frame_width - text_width) // 2

            draw.text(
                (text_x + shadow_offset[0], y + shadow_offset[1]), line, font=font, fill=shadow_color)

            for x_offset in [-boldness, 0, boldness]:
                for y_offset in [-boldness, 0, boldness]:
                    if x_offset != 0 or y_offset != 0:
                        draw.text((text_x + x_offset, y + y_offset),
                                  line, font=font, fill=outline_color)

            draw.text((text_x, y), line, font=font, fill=current_font_color)
            y += text_height + line_spacing

        frames.append(frame)

    duration = original_gif.info.get('duration', 100)

    frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        disposal=2
    )

    print(f"GIF with text saved at {output_gif_path}")


def add_watermark_to_gif_by_text(input_gif_path, output_gif_path, text, font_path,  position='top-right', font_size=50, shadow_offset=(4, 4), shadow_color=(0, 0, 0, 100), outline_color=(255, 255, 255, 100), font_color1=(0, 0, 0, 100), font_color2=(0, 0, 0, 100), line_spacing=10, boldness=4):
    font = ImageFont.truetype(font_path, font_size)
    original_gif = Image.open(input_gif_path)
    frames = []

    max_width = original_gif.size[0] - 20
    lines = wrap_text(text, font, max_width)

    for frame_index, frame in enumerate(ImageSequence.Iterator(original_gif)):
        frame = frame.convert("RGBA")
        draw = ImageDraw.Draw(frame)

        total_text_height = sum(draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox(
            (0, 0), line, font=font)[1] for line in lines) + (len(lines) - 1) * line_spacing

        frame_width, frame_height = frame.size
        current_font_color = font_color1 if frame_index % 10 < 5 else font_color2

        if position == 'top-left':
            y = 10
            x = 10
        elif position == 'top-right':
            y = 10
            x = frame_width - \
                max(draw.textbbox((0, 0), line, font=font)
                    [2] for line in lines) - 10
        elif position == 'bottom-left':
            y = frame_height - total_text_height - 10
            x = 10
        elif position == 'bottom-right':
            y = frame_height - total_text_height - 10
            x = frame_width - \
                max(draw.textbbox((0, 0), line, font=font)
                    [2] for line in lines) - 10
        else:
            raise ValueError(
                "Invalid position. Choose from 'top-left', 'top-right', 'bottom-left', 'bottom-right'.")

        for line in lines:
            text_bbox = draw.textbbox((0, 0), line, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            draw.text(
                (x + shadow_offset[0], y + shadow_offset[1]), line, font=font, fill=shadow_color)

            for x_offset in [-boldness, 0, boldness]:
                for y_offset in [-boldness, 0, boldness]:
                    if x_offset != 0 or y_offset != 0:
                        draw.text((x + x_offset, y + y_offset), line,
                                  font=font, fill=outline_color)

            draw.text((x, y), line, font=font, fill=current_font_color)
            y += text_height + line_spacing

        frames.append(frame)

    duration = original_gif.info.get('duration', 100)

    frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        disposal=2
    )

    print(f"GIF with watermark saved at {output_gif_path}")


def add_watermarks_to_gifs_in_folder_by_text(input_folder, output_folder, text, font_path, position):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.gif'):
            input_gif_path = os.path.join(input_folder, filename)
            output_gif_path = os.path.join(output_folder, filename)
            add_watermark_to_gif_by_text(
                input_gif_path, output_gif_path, text, font_path,  position=position)


def add_watermark_to_gif_by_image(input_gif_path, output_gif_path, watermark_image_path, corner='top-left', transparency=300, max_watermark_size=(200, 200), shadow_offset=(5, 5), shadow_color=(54, 69, 79, 128)):
    original_gif = Image.open(input_gif_path)
    watermark = Image.open(watermark_image_path).convert("RGBA")
    watermark.thumbnail(max_watermark_size, Image.LANCZOS)

    alpha = watermark.split()[3]
    alpha = alpha.point(lambda p: p * transparency / 255.0)
    watermark.putalpha(alpha)

    shadow = Image.new(
        'RGBA', (watermark.size[0] + shadow_offset[0], watermark.size[1] + shadow_offset[1]), shadow_color)
    shadow.paste(watermark, shadow_offset, watermark)

    frames = []
    for frame_index, frame in enumerate(ImageSequence.Iterator(original_gif)):
        frame = frame.convert("RGBA")

        frame_width, frame_height = frame.size
        watermark_width, watermark_height = watermark.size

        if corner == 'top-left':
            x, y = 10, 10
        elif corner == 'top-right':
            x, y = frame_width - watermark_width - 10, 10
        elif corner == 'bottom-left':
            x, y = 10, frame_height - watermark_height - 10
        elif corner == 'bottom-right':
            x, y = frame_width - watermark_width - 10, frame_height - watermark_height - 10
        else:
            raise ValueError(
                "Invalid corner value. Use 'top-left', 'top-right', 'bottom-left', or 'bottom-right'.")

        transparent_layer = Image.new("RGBA", frame.size, (255, 255, 255, 0))
        transparent_layer.paste(shadow, (x, y), shadow)
        transparent_layer.paste(
            watermark, (x + shadow_offset[0], y + shadow_offset[1]), watermark)

        combined = Image.alpha_composite(frame, transparent_layer)

        frames.append(combined)

    duration = original_gif.info.get('duration', 100)

    frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        disposal=2
    )

    print(f"GIF with watermark saved at {output_gif_path}")


def add_watermarks_to_gifs_in_folder_by_image(input_folder, output_folder, watermark_image_path, position):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.gif'):
            input_gif_path = os.path.join(input_folder, filename)
            output_gif_path = os.path.join(output_folder, filename)
            add_watermark_to_gif_by_image(
                input_gif_path, output_gif_path, watermark_image_path, position)


def add_text_to_image(image, text, font_path, font_size=50, shadow_offset=(4, 4), shadow_color=(0, 0, 0, 128), outline_color=(0, 0, 0, 128), font_color=(255, 255, 255, 255), line_spacing=10, boldness=2):
    font = ImageFont.truetype(font_path, font_size)
    draw = ImageDraw.Draw(image)

    max_width = image.size[0] - 20
    lines = wrap_text(text, font, max_width)

    frame_width = image.size[0]
    frame_height = image.size[1]

    total_text_height = sum([draw.textbbox((0, 0), line, font=font)[
                            3] for line in lines]) + (len(lines) - 1) * line_spacing

    y = frame_height - total_text_height - 10

    for line in lines:
        text_bbox = draw.textbbox((0, 0), line, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = (frame_width - text_width) // 2

        draw.text(
            (text_x + shadow_offset[0], y + shadow_offset[1]), line, font=font, fill=shadow_color)

        for x_offset in [-boldness, 0, boldness]:
            for y_offset in [-boldness, 0, boldness]:
                if x_offset != 0 or y_offset != 0:
                    draw.text((text_x + x_offset, y + y_offset),
                              line, font=font, fill=outline_color)

        draw.text((text_x, y), line, font=font, fill=font_color)
        y += text_height + line_spacing

    return image


def extract_and_annotate_first_frame(video_path, output_image_path, text, font_path, font_size=50, shadow_offset=(4, 4), shadow_color=(0, 0, 0, 128), outline_color=(0, 0, 0, 128), font_color=(255, 255, 255, 255), line_spacing=10, boldness=2):
    destination_path = video_path
    cap = cv2.VideoCapture(destination_path)
    success, frame = cap.read()
    cap.release()

    if not success:
        print(f"Failed to extract the first frame from {destination_path}")
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    annotated_image = add_text_to_image(pil_image, text, font_path, font_size,
                                        shadow_offset, shadow_color, outline_color, font_color, line_spacing, boldness)

    annotated_image.save(output_image_path, format='PNG')

    print(f"Image with text saved at {output_image_path}")


def generate_gif_without_text(mp4_file: str, output_gif_path, speed_factor=4) -> None:
    try:
        gif_file = 'output_sticker_without_text.gif'
        clip = VideoFileClip(mp4_file)
        clip = clip.speedx(factor=speed_factor)

        fps = clip.fps
        if fps is None:
            raise ValueError(
                "FPS value could not be determined from the video file.")

        clip.write_gif(gif_file, fps=fps)
        clip.close()
        print(f"GIF created: {gif_file}")
    except Exception as e:
        print(f"Error converting {mp4_file} to GIF: {e}")


def generate_Sticker_without_text(video_path, output_gif_path):
    segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(
        model_selection=1)
    cap = cv2.VideoCapture(video_path)

    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        height, width, channels = frame.shape
        RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = segmentation.process(RGB)
        mask = results.segmentation_mask
        mask = (mask * 255).astype(np.uint8)

        alpha_channel = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)[1]

        rgba_frame = cv2.cvtColor(RGB, cv2.COLOR_RGB2RGBA)
        rgba_frame[:, :, 3] = alpha_channel

        pil_frame = Image.fromarray(rgba_frame)

        frames.append(pil_frame)

    cap.release()
    cv2.destroyAllWindows()

    frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=35,
        loop=0,
        disposal=2,
        transparency=0
    )

    print(f"-- GIF saved at {output_gif_path}")


def generate_gifs(video_path, font_size, shadow_offset_x, shadow_offset_y, shadow_color, outline_color, font_color1, font_color2, line_spacing, boldness, option_sticker_or_Gif, selected_font, AI_Vision_state):
    try:
        print("Transcribing video...")
        text_transcript = transcribe_audio_with_timestamps(video_path)

        if not text_transcript.text or "Speech Recognition could not understand audio" in text_transcript.text:
            print("No potential gifs in the video.")
            return

        print("Video transcribed successfully")

        print("Identifying GIF-worthy captions...")
        gif_sentences = create_gif_caption_identifier_task(
            text_transcript.text)
        if not gif_sentences:
            print("No potential gifs in the video.")
            return

        print("Captions identified successfully")

        temp = json.loads(gif_sentences)
        gif_sentences_list = temp["captions"]
        if not gif_sentences_list:
            print("No potential gifs in the video.")
            return

        audio_path = "output_files/audio.wav"
        extract_audio_from_video(video_path, audio_path)

        clips_paths = {}
        print("Creating GIFs with captions by setting timestamps and trimming clips...")
        transcription_with_timestamps = text_transcript.words
        timestamps_of_gif_captions = generate_timestamps_gif_captions(
            transcription_with_timestamps, gif_sentences_list)

        for caption_data in timestamps_of_gif_captions["captions_with_timestamps"]:
            start_time = float(caption_data['start'])
            end_time = float(caption_data['end'])
            caption = caption_data['caption']
            if start_time is not None:
                output_path = f"output_files/{caption}.mp4"
                temp_max_length_video = VideoFileClip(video_path).duration
                if (start_time >= 0.5) and (end_time + 0.5 <= temp_max_length_video):
                    save_video_clip(video_path, start_time -
                                    0.5, end_time + 0.5, output_path)
                elif (end_time + 0.5 <= temp_max_length_video):
                    save_video_clip(video_path, start_time,
                                    end_time + 0.5, output_path)
                elif (start_time >= 0.5):
                    save_video_clip(video_path, start_time -
                                    0.5, end_time, output_path)
                else:
                    save_video_clip(video_path, start_time,
                                    end_time, output_path)
                clips_paths[caption] = output_path
        print("Video clips created successfully")

        clips_paths_trimmed = {}
        for caption, video_path in clips_paths.items():
            print("-- video_path")
            print(video_path)
            output_path = f"trimmed/{caption.replace(' ', '_')}.mp4"
            output_path = output_path.replace("output_files/", "", 1)
            trim_silence_start(video_path, output_path)
            clips_paths_trimmed[caption] = output_path

        gif_files = []
        print("Generating Gifs...")
        for caption, video_path in clips_paths_trimmed.items():
            output_path = f"gifs/{caption.replace('_', ' ')}.gif"
            gif_files.append(output_path)
            if option_sticker_or_Gif == 'Sticker':
                generate_Sticker_without_text(
                    video_path, "output_sticker_without_text.gif")
            else:
                generate_gif_without_text(
                    video_path, "output_sticker_without_text.gif")

            if not os.path.exists("Back up Gifs without texts"):
                os.makedirs("Back up Gifs without texts")
            target_file_path = os.path.join(
                "Back up Gifs without texts", f"{caption}.gif")
            shutil.copy("output_sticker_without_text.gif", target_file_path)
            print(f"'{"output_sticker_without_text.gif"}' has been copied to '{
                  "Back up Gifs without texts"}' as '{f"{caption}.gif"}'")

            shadow_offset = (shadow_offset_x, shadow_offset_y)

            has_caption_result = False
            if AI_Vision_state:
                with Image.open(r"output_sticker_without_text.gif") as img:
                    frames = [frame.copy()
                              for frame in ImageSequence.Iterator(img)]
                    selected_frames = [frames[0],
                                       frames[len(frames)//2], frames[-1]]

                    for i, frame in enumerate(selected_frames):
                        base64_image = encode_image(frame)
                        has_caption = check_caption(base64_image)
                        print(f"Frame {['first', 'middle', 'last']
                              [i]}: Has Caption? {has_caption}")
                        print(type(has_caption))
                        if (has_caption):
                            has_caption_result = True
                            break

            if not has_caption_result:
                add_text_to_gif(r"output_sticker_without_text.gif", rf"gifs\{
                                caption}.gif", caption, selected_font, font_size, shadow_offset, shadow_color, outline_color, font_color1, font_color2, line_spacing, boldness)
            else:
                shutil.copyfile(
                    r"output_sticker_without_text.gif", rf"gifs\{caption}.gif")

        print("All steps completed!")

        return gif_files

    except Exception as e:
        print(f"An error occurred: {e}")


def upload_gifs_to_giphy(gif_files, GIPHY_API_KEY):
    for gif_file in gif_files:
        temp_gif_file = gif_file
        temp_gif_file = (temp_gif_file.replace("""watermarked_gifs\\""", ""))
        temp_gif_file = (temp_gif_file.replace("""gifs\\""", ""))
        temp_gif_file = (temp_gif_file.replace("_", " ")).split(".")[0]
        print("-- Generating Tags for: ")
        print(temp_gif_file)

        tags = generate_tags_for_gif(temp_gif_file)
        tags = tags["tags"]
        print("--tags")
        print(tags)

        if tags is None:
            print(f"Failed to generate tags for {
                  (gif_file.replace("_", " ")).split(".")[0]}")
        else:
            try:
                upload_gif_to_giphy(gif_file, tags, GIPHY_API_KEY)
            except ValueError as e:
                print(f"Error uploading to GIPHY: {e}")


def convert_to_mp4(input_path, output_path):
    clip = VideoFileClip(input_path)
    clip.write_videofile(output_path, codec='libx264')


def process_video(video_path, font_size, shadow_offset_x, shadow_offset_y, shadow_color, outline_color, font_color1, font_color2, line_spacing, boldness, option_sticker_or_Gif, selected_font, AI_Vision_state):
    try:
        filename = video_path
        base, ext = os.path.splitext(filename)

        if ext.lower() != ".mp4":
            print("-- CONVERTED INTO MP4 file")
            convert_to_mp4(filename, f"{download_path}/input_video.mp4")
            filename = f"{download_path}/input_video.mp4"

        segments = split_video_into_segments(filename)
        for segment in segments:
            try:
                gif_files = generate_gifs(segment, font_size, shadow_offset_x, shadow_offset_y, shadow_color, outline_color,
                                          font_color1, font_color2, line_spacing, boldness, option_sticker_or_Gif, selected_font, AI_Vision_state)
                print("-- gif_files")
                print(gif_files)

                delete_non_mp4_files("output_files")
            except Exception as e:
                print(f"Error generating GIFs for segment {segment}: {e}")
                continue

    except Exception as e:
        print(f"An error occurred with video {video_path}: {e}")
    return 0


def delete_files_in_folder(folder_path):
    try:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            if os.path.isfile(file_path):
                os.remove(file_path)
            else:
                print(f"Skipped (not a file): {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def delete_non_mp4_files(directory):
    for filename in os.listdir(directory):
        if not filename.lower().endswith('.mp4'):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")


def authenticate():
    """Authenticate the user with Google Photos API."""
    creds = None
    if os.path.exists(r'secret/token.json'):
        creds = Credentials.from_authorized_user_file(
            r'secret/token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                r'secret/credentials.json', SCOPES)
            creds = flow.run_local_server(port=8080)
        with open(r'secret/token.json', 'w') as token:
            token.write(creds.to_json())
    return creds


def list_albums(creds):
    """List albums in the Google Photos library and return a dictionary."""
    url = 'https://photoslibrary.googleapis.com/v1/albums'
    headers = {
        'Authorization': f'Bearer {creds.token}'
    }
    response = requests.get(url, headers=headers)
    albums = response.json().get('albums', [])
    album_dict = {album['title']: album['id'] for album in albums}
    return album_dict


def list_shared_albums(creds):
    """List shared albums in the Google Photos library and return a dictionary."""
    url = 'https://photoslibrary.googleapis.com/v1/sharedAlbums'
    headers = {
        'Authorization': f'Bearer {creds.token}'
    }
    response = requests.get(url, headers=headers)
    shared_albums = response.json().get('sharedAlbums', [])
    shared_album_dict = {album.get('title', 'No Title'): album.get(
        'id', 'No ID') for album in shared_albums}
    return shared_album_dict


def get_shared_album_videos(creds, album_id):
    """Retrieve videos from a shared album."""
    video_urls = []
    media_items_url = 'https://photoslibrary.googleapis.com/v1/mediaItems:search'
    body = {
        "albumId": album_id,
        "pageSize": 100
    }
    headers = {
        'Authorization': f'Bearer {creds.token}'
    }
    response = requests.post(media_items_url, headers=headers, json=body)

    print("Media Items Response JSON:", response.json())

    if response.status_code != 200:
        print("Error fetching media items:", response.json())
        return []

    items = response.json().get('mediaItems', [])
    for item in items:
        if 'video' in item['mimeType']:
            video_urls.append(item['baseUrl'] + "=dv")

    return video_urls


def download_all_videos(video_urls, download_path):
    for idx, url in enumerate(video_urls):
        try:
            response = requests.get(url)
            video_path = os.path.join(download_path, f'video_{idx + 1}.mp4')
            with open(video_path, 'wb') as video_file:
                video_file.write(response.content)
            print(f'Downloaded: {video_path}')
        except Exception as e:
            print(f"An error occurred with video {url}: {e}")


st.set_page_config(
    page_title="Video To Gif Generator",
    page_icon="➡️",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'About': "This is web application that allows users to generate GIFs from videos with text overlays corresponding to spoken words in the video."
    }
)


def main():
    st.markdown(
        """
        <style>
        .stImage {
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            border-radius: 10px;
            margin: 10px;
        }
        .stImage:hover {
            transform: scale(1.05);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "<div style='text-align: left;'><h1 style='display: inline-block; margin: 0; padding: 0;'>Video To Gif Generator ➡️</h1></div>",
        unsafe_allow_html=True
    )
    upload_option = st.selectbox("  Choose an option:", ("Upload Video File",
                                 "Upload by YouTube Link", "Select from YouTube Channel", "Select from Google Photos"))
    AI_Vision_state = st.toggle("Use AI vision", value=True)

    logging.info(f"Upload option: {upload_option}")

    video_file = None
    is_video_file = False

    if upload_option == "Upload Video File":
        video_file = st.file_uploader(
            "Upload a video file", type=["mp4", "mov", "avi"])
        if video_file:
            st.video(video_file)
            save_path = os.path.join(download_path, video_file.name)
            with open(save_path, "wb") as f:
                f.write(video_file.getbuffer())
            st.success(f"File Uploaded: {video_file.name}")
            is_video_file = True

    elif upload_option == "Upload by YouTube Link":
        is_video_file = True
        YT_Video_Link = st.text_input("Enter YouTube Video Link")

    elif upload_option == "Select from YouTube Channel":
        is_video_file = True
        YT_channel_url = st.text_input("Enter YouTube Channel Link")
        option_YT_videos_count = st.radio(
            'Do you want to download all Videos?', ('Yes', 'Limited Count'))
        if option_YT_videos_count == "Limited Count":
            YT_Video_count = st.number_input(
                'Enter an integer:', min_value=1, max_value=200, step=1)

    elif upload_option == "Select from Google Photos":
        st.write("Authenticating with your Google account to access albums...")
        is_video_file = True
        creds = authenticate()
        album_dict = list_albums(creds)
        shared_album_dict = list_shared_albums(creds)
        all_albums = {**album_dict, **shared_album_dict}
        album_choice = st.selectbox("Select an Album", list(all_albums.keys()))

    if is_video_file or video_file:
        col1, col2, col3 = st.columns(3)

        with col1:
            option_sticker_or_Gif = st.radio(
                'Choose an Option:', ('Gif', 'Sticker'))

        with col2:
            option_watermark = st.radio(
                'Want to add Watermark?', ('No', 'Yes'))

        with col3:
            GIPHY_upload = st.radio(
                'Want to Upload to Gifs Library website?', ('No', 'Yes'))

        water_mark = None
        if option_watermark == 'Yes':
            col1, col2, col3 = st.columns(3)
            with col1:
                watermark_type = st.radio(
                    "Select watermark type", ('Text', 'Image'))
            with col2:
                if watermark_type == 'Text':
                    water_mark = st.text_input("Enter watermark text")
                elif watermark_type == 'Image':
                    watermark_image = st.file_uploader(
                        "Upload a watermark image", type=["png"])
                    if watermark_image is not None:
                        image = Image.open(watermark_image)
                        st.image(
                            image, caption='Uploaded Watermark Image', use_column_width=True)
                        watermark_image_path = os.path.join(
                            "intermediate", "watermark_image.png")
                        image.save(watermark_image_path)
                        st.success(f"Image saved successfully at {
                                   watermark_image_path}")
            with col3:
                position = st.radio("Select watermark position", [
                                    'top-right', 'top-left', 'bottom-right', 'bottom-left'])

        GIPHY_API_KEY = None
        if GIPHY_upload == 'Yes':
            col1, col2 = st.columns(2)
            with col1:
                Tenor_or_Giphy = st.radio('', ('Giphy', 'Tenor'))
            with col2:
                if Tenor_or_Giphy == "Tenor":
                    TENOR_state = st.toggle("Use Existing Profile", value=True)
                    if not TENOR_state:
                        TENOR_EMAIL = st.text_input("E mail:", "")
                        TENOR_PASSWORD = st.text_input(
                            "Password:", type="password")
                        if st.button("Sign In to Tenor"):
                            # tenor login functionality
                            pass
                else:
                    GIPHY_API_KEY = st.text_input(
                        "Enter your GIPHY API Key", type="password")

        output_gifs_folder_path = "gifs"

        st.sidebar.header("Text Customization")
        font_size = st.sidebar.slider(
            "Font Size", min_value=1, max_value=300, value=95)

        boldness = st.sidebar.slider(
            "Boldness", min_value=1, max_value=10, value=2)

        col1, col2 = st.sidebar.columns(2)
        with col1:
            font_color1 = st.color_picker("Font Color 1", "#FFFFFF")
            outline_color = st.color_picker("Outline Color", "#000000")
        with col2:
            font_color2 = st.color_picker("Font Color 2", "#995DFF")
            shadow_color = st.color_picker("Shadow Color", "#000000")

        shadow_offset_x, shadow_offset_y = st.sidebar.slider(
            "Shadow Offset", min_value=-20, max_value=20, value=(4, 6))
        line_spacing = st.sidebar.slider(
            "Line Spacing", min_value=0, max_value=50, value=10)

        font_options = {
            'Oswald': r'fonts\Oswald.ttf',
            'Chase Dreams': r'fonts\Chase Dreams.ttf',
            'Arial': r'fonts\Arial.ttf',
            'Nelhinco': r'fonts\Nelhinco.ttf',
            'RacingPunch': r'fonts\RacingPunch.ttf',
            'Scripto': r'fonts\Scripto.ttf',
            'Cartoon Blocks': r'fonts\Cartoon Blocks.ttf',
            'BloodAttack': r'fonts\BloodAttack.ttf',
            'Ghost Blaze': r'fonts\Ghost Blaze.ttf',
            '28 Days Later': r'fonts\28 Days Later.ttf'
        }
        font_option = st.sidebar.selectbox(
            'Choose an option:',
            list(font_options.keys())
        )

        selected_font = font_options[font_option]
        preview_caption = st.sidebar.text_input(
            "Preview Caption", "Sample Caption")

        if video_file:
            if st.sidebar.button("Preview Caption"):
                logging.info("Preview Caption button clicked")

                output_image_path = "output.png"
                shadow_color_rgba = tuple(
                    int(shadow_color[i:i+2], 16) for i in (1, 3, 5)) + (128,)
                outline_color_rgba = tuple(
                    int(outline_color[i:i+2], 16) for i in (1, 3, 5)) + (128,)
                font_color1_rgba = tuple(
                    int(font_color1[i:i+2], 16) for i in (1, 3, 5)) + (128,)

                files = os.listdir("uploads")
                if files:
                    first_file = os.path.join("uploads", files[0])
                    print(first_file)
                else:
                    print("No files found in the folder")

                video_path = first_file
                output_image_path = "output.png"

                extract_and_annotate_first_frame(
                    video_path, output_image_path, preview_caption, selected_font, font_size,
                    shadow_offset=(shadow_offset_x, shadow_offset_y),
                    shadow_color=shadow_color_rgba, outline_color=outline_color_rgba,
                    font_color=font_color1_rgba, line_spacing=line_spacing, boldness=boldness
                )

                st.sidebar.image(
                    output_image_path, caption='Preview of text Customization', use_column_width=True)

        if st.button("Generate"):
            logging.info("Generate button clicked")
            if upload_option == "Select from Google Photos":
                selected_album_id = all_albums[album_choice]
                video_urls = get_shared_album_videos(creds, selected_album_id)
                with st.spinner('Downloading videos from Google Photos...'):
                    download_all_videos(video_urls, download_path)

            if upload_option == "Upload by YouTube Link":
                with st.spinner('Downloading videos from YouTube...'):
                    download_video_from_YT_link(YT_Video_Link, download_path)

            if upload_option == "Select from YouTube Channel":
                with st.spinner('Downloading videos from YouTube...'):
                    if option_YT_videos_count == "Limited Count":
                        download_YT_channel_Specific_Number_videos(
                            YT_channel_url, download_path, max_videos=YT_Video_count)
                    else:
                        download_YT_channel_ALL_videos(
                            YT_channel_url, download_path)

            with st.spinner('Generating Gifs...'):
                try:
                    if not os.path.exists("output_files"):
                        os.makedirs("output_files")
                    delete_files_in_folder("output_files")

                    if not os.path.exists("trimmed"):
                        os.makedirs("trimmed")
                    delete_files_in_folder("trimmed")

                    if not os.path.exists(output_gifs_folder_path):
                        os.makedirs(output_gifs_folder_path)
                    delete_files_in_folder(output_gifs_folder_path)

                    if not os.path.exists("watermarked_gifs"):
                        os.makedirs("watermarked_gifs")
                    delete_files_in_folder("watermarked_gifs")

                    video_files = os.listdir(download_path)
                    sorted_video_files = sorted(video_files)

                    process_videos_in_folder_to_360(download_path)

                    for video_file in sorted_video_files:
                        video_path = os.path.join(download_path, video_file)
                        if is_video_corrupt(video_path) or not has_sound(video_path):
                            continue
                        result = process_video(video_path, font_size, shadow_offset_x, shadow_offset_y, shadow_color, outline_color,
                                               font_color1, font_color2, line_spacing, boldness, option_sticker_or_Gif, selected_font, AI_Vision_state)
                        delete_files_in_folder("output_files")
                        delete_files_in_folder("trimmed")

                    if option_watermark == 'Yes':
                        output_gifs_folder_path = "watermarked_gifs"
                        if watermark_type == "Text":
                            add_watermarks_to_gifs_in_folder_by_text('gifs', output_gifs_folder_path, f" {
                                                                     water_mark} ", selected_font, position)
                        elif watermark_type == "Image":
                            add_watermarks_to_gifs_in_folder_by_image(
                                'gifs', output_gifs_folder_path, watermark_image_path, position)

                    result = count_files(output_gifs_folder_path)
                    st.write(f"Generated {result} gifs")

                except Exception as e:
                    st.error(f"An error occurred: {e}")

                delete_files_in_folder(download_path)

        if (GIPHY_upload == 'Yes') and bool([f for f in os.listdir('gifs') if os.path.isfile(os.path.join('gifs', f))]):
            if st.button("Upload Gifs"):
                logging.info("Upload Gifs button clicked")

                with st.spinner('Uploading Gifs...'):
                    gif_files = []
                    if option_watermark == 'Yes':
                        output_gifs_folder_path = "watermarked_gifs"

                    for root, dirs, files in os.walk(output_gifs_folder_path):
                        for file in files:
                            gif_files.append(os.path.join(root, file))
                    if Tenor_or_Giphy == "Tenor":
                        if gif_files:
                            for gif_file in gif_files:
                                temp_gif_file = gif_file
                                temp_gif_file = (temp_gif_file.replace(
                                    """watermarked_gifs\\""", ""))
                                temp_gif_file = (
                                    temp_gif_file.replace("""gifs\\""", ""))
                                temp_gif_file = (temp_gif_file.replace(
                                    "_", " ")).split(".")[0]
                                print("-- Generating Tags for: ")
                                print(temp_gif_file)

                                tags = generate_tags_for_gif(temp_gif_file)
                                tags = tags["tags"]
                                print("--tags")
                                print(tags)
                                tags_list = [tag.strip()
                                             for tag in tags.split(",")]
                                print("--tags_list")
                                print(tags_list)

                                if tags is None:
                                    print(f"Failed to generate tags for {
                                          (gif_file.replace("_", " ")).split(".")[0]}")
                                else:
                                    try:
                                        if TENOR_state and (Tenor_or_Giphy == "Tenor"):
                                            # tenore gif upload
                                            pass
                                    except ValueError as e:
                                        print(f"Error uploading to GIPHY: {e}")
                    else:
                        if gif_files:
                            try:
                                upload_gifs_to_giphy(gif_files, GIPHY_API_KEY)
                                st.success("GIFs uploaded successfully!")
                            except Exception as e:
                                st.error(f"Error uploading GIFs: {e}")

        if option_watermark == 'Yes':
            output_gifs_folder_path = "watermarked_gifs"

        zip_file_path = zip_specific_folder(output_gifs_folder_path)
        with open(zip_file_path, 'rb') as f:
            data = f.read()

        if bool([f for f in os.listdir('gifs') if os.path.isfile(os.path.join('gifs', f))]):
            st.download_button(
                label="Download All GIFs",
                data=data,
                file_name="gif_files.zip",
                mime="application/zip"
            )

        if bool([f for f in os.listdir('gifs') if os.path.isfile(os.path.join('gifs', f))]):
            gif_folder = "watermarked_gifs" if option_watermark == 'Yes' else 'gifs'
            if 'delete_key' not in st.session_state:
                st.session_state.delete_key = None
            if st.button("Display GIFs"):
                st.session_state.display_gifs = True

            if st.session_state.delete_key:
                gif_path = os.path.join(
                    gif_folder, st.session_state.delete_key)
                if os.path.exists(gif_path):
                    try:
                        os.remove(gif_path)
                        st.success(
                            f"{st.session_state.delete_key} has been deleted.")
                    except Exception as e:
                        st.error(f"Error deleting {
                                 st.session_state.delete_key}: {e}")
                else:
                    st.warning(
                        f"File {st.session_state.delete_key} does not exist.")
                st.session_state.delete_key = None

            if st.session_state.get('display_gifs', False):
                gif_files = get_gif_files(gif_folder)
                if gif_files:
                    num_columns = 3
                    columns = st.columns(num_columns)
                    for idx, gif_file in enumerate(gif_files):
                        gif_path = os.path.join(gif_folder, gif_file)
                        col = columns[idx % num_columns]
                        with col:
                            st.image(gif_path, use_column_width=True)
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button(f"Delete {gif_file}", key=f"delete_{gif_file}"):
                                    st.session_state.delete_key = gif_file
                            with col2:
                                if st.button(f"Edit {gif_file}", key=f"edit_{gif_file}"):
                                    st.session_state.edit_key = gif_file

                            if st.session_state.get('edit_key') == gif_file:
                                st.write(f"Editing {gif_file}")
                                new_text = st.text_input(
                                    "Enter new text for the GIF:", key=f"new_text_{gif_file}")

                                col1, col2, col3, col4, col5, col6 = st.columns(
                                    6)
                                with col1:
                                    font_size = st.slider(
                                        "Font Size", min_value=1, max_value=300, key=f"font_size_1_{gif_file}")
                                with col2:
                                    boldness = st.slider(
                                        "Boldness", min_value=1, max_value=10, key=f"boldness_slider_1_{gif_file}")
                                with col3:
                                    font_color1 = st.color_picker(
                                        "Font Color 1", "#FFFFFF", key=f"font_color1_{gif_file}")
                                with col4:
                                    font_color2 = st.color_picker(
                                        "Font Color 2", "#995DFF", key=f"font_color2_{gif_file}")
                                with col5:
                                    outline_color = st.color_picker(
                                        "Outline Color", "#000000", key=f"outline_color1_{gif_file}")
                                with col6:
                                    shadow_color = st.color_picker(
                                        "Shadow Color", "#000000", key=f"shadow_color1_{gif_file}")

                                apply_col, cancel_col = st.columns(2)
                                with apply_col:
                                    if st.button("Apply Edit", key=f"Apply_Edit_{gif_file}"):
                                        edit_gif(os.path.join(gif_folder, gif_file), new_text, selected_font, font_size, (
                                            shadow_offset_x, shadow_offset_y), shadow_color, outline_color, font_color1, font_color2)
                                        st.session_state.edit_key = None
                                with cancel_col:
                                    if st.button("Cancel", key=f"Cancel_Edit_{gif_file}"):
                                        st.session_state.edit_key = None
                else:
                    st.write("No GIFs found.")

    if st.button("RESET ALL"):
        logging.info("RESET ALL button clicked")

        if os.path.exists(r"secret/token.json"):
            os.remove(r"secret/token.json")

        if os.path.exists(r"output.png"):
            os.remove(r"output.png")

        if os.path.exists(r"gif_files.zip"):
            os.remove(r"gif_files.zip")

        if os.path.exists(r"output_sticker_without_text.gif"):
            os.remove(r"output_sticker_without_text.gif")

        if not os.path.exists(r"Back up Gifs without texts"):
            os.makedirs(r"Back up Gifs without texts")
        delete_files_in_folder(r"Back up Gifs without texts")

        if not os.path.exists(r"uploads"):
            os.makedirs(r"uploads")
        delete_files_in_folder(r"uploads")

        if not os.path.exists("output_files"):
            os.makedirs("output_files")
        delete_files_in_folder("output_files")

        if not os.path.exists("trimmed"):
            os.makedirs("trimmed")
        delete_files_in_folder("trimmed")

        if not os.path.exists("watermarked_gifs"):
            os.makedirs("watermarked_gifs")
        delete_files_in_folder("watermarked_gifs")

        if not os.path.exists("gifs"):
            os.makedirs("gifs")
        delete_files_in_folder("gifs")

        if not os.path.exists("intermediate"):
            os.makedirs("intermediate")
        delete_files_in_folder("intermediate")


if __name__ == '__main__':
    GIF_COUNT = 0
    GIPHY_API_KEY = None
    option_sticker_or_Gif = None
    download_path = "uploads"
    output_gifs_folder_path = "gifs"

    if not os.path.exists(download_path):
        os.makedirs(download_path)
    delete_files_in_folder(download_path)

    if not os.path.exists("output_files"):
        os.makedirs("output_files")

    if not os.path.exists("trimmed"):
        os.makedirs("trimmed")

    if not os.path.exists(output_gifs_folder_path):
        os.makedirs(output_gifs_folder_path)

    if not os.path.exists("watermarked_gifs"):
        os.makedirs("watermarked_gifs")

    if not os.path.exists("intermediate"):
        os.makedirs("intermediate")

    if not os.path.exists(r"Back up Gifs without texts"):
        os.makedirs(r"Back up Gifs without texts")
    main()
