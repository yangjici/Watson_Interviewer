from watson_developer_cloud import TextToSpeechV1, SpeechToTextV1, NaturalLanguageUnderstandingV1 as NLU
import watson_developer_cloud.natural_language_understanding.features.v1 as features
from recorder import Recorder
import pyaudio
import wave
import os
import json
import numpy as np
import pandas as pd
from streamcharts import StreamCharts
import datetime
import time 
import webbrowser

class Interviewer(object):

    def __init__(self, ):
        self.MIN_ANSWER_LEN = 5
        self.MIN_CONFIDENCE = 0.60
        self.SMALL_TALK = ['I see.', 'Got it.', 'Ok', 'Interesting']
        self.POSITIVE_REMARK = ["Good.", "Excellent!", "Sounds great!", "That's awesome!", "Wonderful!"]
        self.NEGATIVE_REMARK = ["I'm sad to hear that.", "That doesn't sound very good.", "I'm sad to hear that.", "ah", "Someone forgot to have their coffee today"]
        self.questions = ['Tell me about yourself',
            'Tell me about a recent project that you worked on',
            'What are your greatest weaknesses?',
            'What did you dislike the most about your last job?',
            'If you were an animal, which one would you want to be?',
            'What are your hobbies?',
            'What is your greatest professional achievement?',
            'Why do you want to work here?',
            'What are your strengths?',
            'Where do you see yourself in five years?',
            'What type of work environment do you prefer?',
            "What's a time you disagreed with a decision that was made at work?",
            'Why was there a gap in your employment?',
            'Can you explain why you changed career paths?',
            'How do you deal with pressure or stressful situations?',
            'What would your first 30, 60, or 90 days look like in this role?',
            'What are your salary requirements?',
            'How many tennis balls can you fit into a limousine?',
            'Are you planning on having children?',
            'How many ping pong balls fit on a 737?',
            'Describe a difficult work situation / project and how you overcame it',
            'How are you different from the competition?',
            'Do you take work home with you?',
            'How do you view yourself? Whom do you compare yourself to?',
            'What motivates you',
            'What did you like most about your last job?',
            'What did you dislike most about your last job?',
            'Why should I take a risk on you?']
        self.text_to_speech = TextToSpeechV1(
            x_watson_learning_opt_out=True)  # Optional flag
        self.speech_to_text = SpeechToTextV1(
            x_watson_learning_opt_out=False)
        self.nlu = NLU(
            version='2017-02-27')

        self.TEMPFILE = './temp/output.wav'
        self.answers, self.sentiments = [], []


    def transcribe_audio(self, stt, path_to_audio_file):
        with open(self.TEMPFILE, 'rb') as audio_file:
            return stt.recognize(audio_file,content_type='audio/wav')

    def say(self, text, output_filename='./temp/output.wav'):
        if os.path.isfile(output_filename):
            os.remove(output_filename)
        with open(output_filename,'wb') as audio_file:
            audio_file.write(self.text_to_speech.synthesize(text, accept='audio/wav',
                                          voice="en-US_AllisonVoice"))
        chunk = 1024  
        f = wave.open(output_filename,"rb")  
        p = pyaudio.PyAudio()  
        stream = p.open(format = p.get_format_from_width(f.getsampwidth()),  
                        channels = f.getnchannels(),  
                        rate = f.getframerate(),  
                        output = True)  
        data = f.readframes(chunk)
        while data:  
            stream.write(data)  
            data = f.readframes(chunk)  
        stream.stop_stream()  
        stream.close()  
        p.terminate()

    def listen(self):
        if os.path.isfile(self.TEMPFILE):
            os.remove(self.TEMPFILE)       

        recorder = Recorder(self.TEMPFILE)
        print('Recording')
        recorder.record_to_file()
        print('Stopped')

        confidence, transcript = 0, ''

        result = self.transcribe_audio(self.speech_to_text, self.TEMPFILE)

        if len(result['results']) > 0:
            confidence = result['results'][0]['alternatives'][0]['confidence']
            transcript = result['results'][0]['alternatives'][0]['transcript'].strip()
        return transcript, confidence

    def analyze_sentiment(self, answer):
        result = self.nlu.analyze(text=answer,features=[features.Keywords(),features.Sentiment()])
        if result['keywords']:
            keywords = result['keywords'][0]
            keyword = keywords['text']
        else:
            keyword = None
        sentiment = result['sentiment']['document']['score']
        return sentiment, keyword

    def discuss_sentiment(self, sentiment, keyword):
        if sentiment < -0.45:
            self.say(self.NEGATIVE_REMARK[np.random.choice(range(len(self.NEGATIVE_REMARK)))])
        elif sentiment > 0.25:
            self.say(self.POSITIVE_REMARK[np.random.choice(range(len(self.POSITIVE_REMARK)))])
            if np.random.rand() > 0.25 and keyword:
                self.say("I'd like to hear more about " + keyword + " next time")
        else:
            self.say(self.SMALL_TALK[np.random.choice(range(len(self.SMALL_TALK)))])

    def run_interview(self):
        self.answers, self.sentiments = [], []


        chart = StreamCharts(window_size=20)
        webbrowser.open(chart.chart_url, new=2)

        time.sleep(10)

        i = 0
        self.say('Let us begin the interview.')

        while i < len(self.questions):
            self.say(self.questions[i])
            answer, confidence = self.listen()
            if (len(answer) >= self.MIN_ANSWER_LEN) and (confidence >= self.MIN_CONFIDENCE):
                sentiment, keyword = self.analyze_sentiment(answer)
                self.answers.append(answer)
                self.sentiments.append(sentiment)
                i += 1
                self.discuss_sentiment(sentiment, keyword)
                if not keyword:
                    keyword = ""
                x_label = keyword #" ".join(answer.split()[:5])
                y_value = sentiment
                chart.update(x_label=x_label, y_value=y_value)

            elif confidence < self.MIN_CONFIDENCE:
                self.say("Let's try that again, I couldn't understand you.")
            elif len(answer) < self.MIN_ANSWER_LEN:
                self.say("Let's try that again, please provide a longer answer.")
        self.say("Thank you for your time.  We will get back to you shortly about our decision. Thank you, and thank you Galvanize and IBM Cognitive Builder Faire.")
        # Close the stream when done plotting
        chart.close()