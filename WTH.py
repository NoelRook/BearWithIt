#Imports for AWS API
import boto3, json
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent
import io
from contextlib import closing 
from botocore.exceptions import BotoCoreError, ClientError
from pydub import AudioSegment
import pyaudio 
import uuid
import requests
import time
import asyncio
import sounddevice

#Imports for video
import cv2
from deepface import DeepFace
import time
from playsound import playsound
import os

# Create an &BR; client in the &region-us-east-1; Region.
bedrock = boto3.client('bedrock-runtime', region_name='ap-south-1')
polly = boto3.client("polly", region_name="ap-south-1")
transcribe_client = boto3.client('transcribe', region_name='ap-southeast-1')
# bedrock_runtime.list_foundation_models()

# Load face and eye cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
ALERT_SOUND_PATH = "Teddy.wav"  # Replace with the path to your sound file
# Start capturing video
eyes_closed_start_time = None  # To track when eyes are first detected as closed
warning_displayed = False      # To ensure warning is displayed only once per event

def prompt_model(input):
  kwargs = {
    "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
    "contentType": "application/json",
    "accept": "application/json",
    "body": json.dumps({
      "anthropic_version": "bedrock-2023-05-31",
      "max_tokens": 1000,
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "Answer in 1 sentence" + input
            }
          ]
        }
      ]
    })
  }    
  response = bedrock.invoke_model(**kwargs)
  body = json.loads(response['body'].read())
  return (body['content'][0]['text'])

def play_audio_from_bytes(audio_bytes) :
# Convert the byte string to an audio segment
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
    if audio.channels == 2:
        audio = audio.set_channels(1)
    
    audio_raw_data = audio.raw_data
    
    p = pyaudio.PyAudio()
    
    # Open a PyAudio stream
    stream = p.open(format=p.get_format_from_width(audio.sample_width),
                    channels=audio.channels, 
                    rate=audio.frame_rate, 
                    output=True)
    
    # Play audio
    with closing(stream):
        stream.write(audio_raw_data)
    p.terminate()
    
def transcribe_file(job_name, job_uri, transcribe_client):
    transcribe_client.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={"MediaFileUri": job_uri},
        MediaFormat="wav",
        LanguageCode="en-US",
        Settings={
            'VocabularyName':'WTHVocabulary'
        }
    )

    max_tries = 60
    while max_tries > 0:
        max_tries -= 1
        job = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
        job_status = job["TranscriptionJob"]["TranscriptionJobStatus"]
        if job_status in ["COMPLETED", "FAILED"]:
            print(f"Job {job_name} is {job_status}.")
            if job_status == "COMPLETED":
                '''
                print(
                    f"Download the transcript from\n"
                    f"\t{job['TranscriptionJob']['Transcript']['TranscriptFileUri']}."
                )
                '''

                url = job['TranscriptionJob']['Transcript']['TranscriptFileUri']
                

                # Fetch the transcript file
                response = requests.get(url)

                # Raise an error if the request failed
                response.raise_for_status()

                # Parse JSON if the response is JSON
                transcript = response.json()
                #response = urlopen(job['TranscriptionJob']['Transcript']['TranscriptFileUri'])
                #data = json.loads(response.read())
                text = transcript['results']['transcripts'][0]['transcript']
                print(text)
                return text
            break
        else:
            print(f"Waiting for {job_name}. Current status is {job_status}.")
        time.sleep(10)

def delete_job(job_name, transcribe_client):
    """
    Deletes a transcription job. This also deletes the transcript associated with
    the job.

    :param job_name: The name of the job to delete.
    :param transcribe_client: The Boto3 Transcribe client.
    """
    try:
        transcribe_client.delete_transcription_job(TranscriptionJobName=job_name)
        logger.info("Deleted job %s.", job_name)
    except ClientError:
        logger.exception("Couldn't delete job %s.", job_name)
        raise

def study(text):
  print(text)
  if text == "study time" :
    return True
  elif text == "stop studying.":
    return False 
  else:
    return False

# Returns True if "hello bear" is first 2 words of final text
def activated(text):
  text = text.replace(",", "")
  text = text.replace(".", "")
  words = text.lower().split()
  print(words)
  return words[:2] in [["hello", "bear"]] and len(words) > 2

# Strips "hey bear" or "hi bear"
def get_action(text):

  text = text.lower()
  text = text.replace(",", "")
  text = text.replace(".", "")

  # Convert text to lowercase and check for greetings
  if text.lower().startswith("hello bear"):
    return text[11:].strip()  # Remove "hey bear, "
  # elif text.lower().startswith("hi bear"):
  #   return text[8:].strip()  # Remove "hi bear, "
  else:
    return None

def video():
   cap = cv2.VideoCapture(0)
   eyes_closed_start_time = None

   while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert grayscale frame to RGB format
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face_roi = rgb_frame[y:y + h, x:x + w]
        gray_face_roi = gray_frame[y:y + h, x:x + w]

        # Detect eyes in the face ROI
        eyes = eye_cascade.detectMultiScale(gray_face_roi, scaleFactor=1.1, minNeighbors=10, minSize=(15, 15))

        if len(eyes) >= 2:
            eye_status = "Eyes Open"
            eyes_closed_start_time = None  # Reset the timer
            warning_displayed = False     # Reset warning flag
        else:
            eye_status = "Eyes Closed"
            if eyes_closed_start_time is None:
                eyes_closed_start_time = time.time()  # Start timing
            else:
                elapsed_time = time.time() - eyes_closed_start_time
                if elapsed_time > 5: #and not warning_displayed:
                    if os.path.exists('Teddy.wav'):
                        print("File exists")
                        playsound(ALERT_SOUND_PATH)
                    else:
                        print("File does not exist")
                    print("This student is asleep") 
                    
                    #warning_displayed = True  # Display the warning only once

        # Perform emotion analysis on the face ROI
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

        # Determine the dominant emotion
        emotion = result[0]['dominant_emotion']

        # Draw rectangle around face and label with predicted emotion and eye status
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, f"{eye_status}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Real-time Emotion and Eye Status Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
      cap.release()
      cv2.destroyAllWindows()
      break


# Hi hehe
class MyEventHandler(TranscriptResultStreamHandler):
    # video = False
    # if video is True:
    #   video()
    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        # This handler can be implemented to handle transcriptions as needed.
        # Here's an example to get started.
        results = transcript_event.transcript.results


        for result in results:
            # print(result.__dict__)
            
            for alt in result.alternatives:
                if result.is_partial == False:
                  command = alt.transcript
                  print("Full sentence: " + command)
                  # "Hey Bear" key phrase heard
                  if activated(command):
                    # Get action from command 
                    prompt = get_action(command)                   
                    print("Action: " + prompt)
 
                    
                    if study(prompt) == True:
                        print('execute timer and image recognition')
                        model_response = "Okay let's study for 1 Hour"
                        response = polly.synthesize_speech(VoiceId= "Brian",
                                                          OutputFormat="mp3",
                                                          Text=model_response)
                        video()

                        # break
                    elif study(prompt) == False:
                      print('execute AI')
                      model_response = prompt_model(prompt)
                      response = polly.synthesize_speech(VoiceId= "Brian",
                                                        OutputFormat="mp3",
                                                        Text=model_response)
                    elif prompt == None:
                      print('Did not understand action')
                      model_response = "Sorry, I didn't understand what you said."
                      response = polly.synthesize_speech(VoiceId= "Brian",
                                                          OutputFormat="mp3",
                                                          Text=model_response)
                      
                              
                    audio_bytes = response['AudioStream'].read()
                    play_audio_from_bytes(audio_bytes)

async def mic_stream():
    # This function wraps the raw input stream from the microphone forwarding
    # the blocks to an asyncio.Queue.
    loop = asyncio.get_event_loop()
    input_queue = asyncio.Queue()

    def callback(indata, frame_count, time_info, status):
        loop.call_soon_threadsafe(input_queue.put_nowait, (bytes(indata), status))

    # Be sure to use the correct parameters for the audio stream that matches
    # the audio formats described for the source language you'll be using:
    # https://docs.aws.amazon.com/transcribe/latest/dg/streaming.html
    stream = sounddevice.RawInputStream(
        channels=1,
        samplerate=16000,
        callback=callback,
        blocksize=1024 * 2,
        dtype="int16",
    )
    # Initiate the audio stream and asynchronously yield the audio chunks
    # as they become available.q
    with stream:
        while True:
            indata, status = await input_queue.get()
            yield indata, status

async def write_chunks(stream):
    # This connects the raw audio chunks generator coming from the microphone
    # and passes them along to the transcription stream.
    async for chunk, status in mic_stream():
        await stream.input_stream.send_audio_event(audio_chunk=chunk)
    await stream.input_stream.end_stream()

async def basic_transcribe():
    # Setup up our client with our chosen AWS region
    client = TranscribeStreamingClient(region="us-east-1")

    # Start transcription to generate our async stream
    stream = await client.start_stream_transcription(
        language_code="en-US",
        media_sample_rate_hz=16000,
        media_encoding="pcm"
    )

    # Instantiate our handler and start processing events
    handler = MyEventHandler(stream.output_stream)
    await asyncio.gather(write_chunks(stream), handler.handle_events()) 

def main():
  try:
    loop = asyncio.get_event_loop()
    loop.run_until_complete(basic_transcribe())
    loop.close()
  except: 
    loop = asyncio.get_event_loop()
    loop.run_until_complete(basic_transcribe())
    loop.close()

  
if __name__ == "__main__":
    main()