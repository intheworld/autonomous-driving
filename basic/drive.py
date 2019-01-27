import argparse
import base64
from datetime import datetime
import os
import shutil
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
from keras.models import load_model
import utils

sio = socketio.Server()
model = None
prev_image_array = None

MAX_SPEED = 25
MIN_SPEED = 10
speed_limit = MAX_SPEED


class Drive:

    def __init__(self):
        self.args = None

    @sio.on('telemetry')
    def telemetry(self, sid, data):
        if data:
            # The current steering angle of the car
            steering_angle = float(data["steering_angle"])
            # The current throttle of the car
            throttle = float(data["throttle"])
            # The current speed of the car
            speed = float(data["speed"])
            # The current image from the center camera of the car
            image = Image.open(BytesIO(base64.b64decode(data["image"])))
            # save frame
            if self.args.image_folder != '':
                timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
                image_filename = os.path.join(self.args.image_folder, timestamp)
                image.save('{}.jpg'.format(image_filename))

            try:
                image = np.asarray(image)  # from PIL image to numpy array
                image = utils.preprocess(image)  # apply the preprocessing
                image = np.array([image])  # the model expects 4D array

                # predict the steering angle for the image
                steering_angle = float(model.predict(image, batch_size=1))
                # lower the throttle as the speed increases
                # if the speed is above the current speed limit, we are on a downhill.
                # make sure we slow down first and then go back to the original max speed.
                global speed_limit
                if speed > speed_limit:
                    speed_limit = MIN_SPEED  # slow down
                else:
                    speed_limit = MAX_SPEED
                throttle = 1.0 - steering_angle ** 2 - (speed / speed_limit) ** 2

                print('{} {} {}'.format(steering_angle, throttle, speed))
                self.send_control(steering_angle, throttle)
            except Exception as e:
                print(e)

        else:
            # NOTE: DON'T EDIT THIS.
            sio.emit('manual', data={}, skip_sid=True)


    @sio.on('connect')
    def connect(self, sid, environ):
        print("connect ", sid)
        self.send_control(0, 0)

    def send_control(self, steering_angle, throttle):
        sio.emit(
            "steer",
            data={
                'steering_angle': steering_angle.__str__(),
                'throttle': throttle.__str__()
            },
            skip_sid=True)

    def main(self):
        parser = argparse.ArgumentParser(description='Remote Driving')
        parser.add_argument(
            'model',
            type=str,
            help='Path to model h5 file. Model should be on the same path.'
        )
        parser.add_argument(
            'image_folder',
            type=str,
            nargs='?',
            default='',
            help='Path to image folder. This is where the images from the run will be saved.'
        )
        args = parser.parse_args()
        self.args = args

        model = load_model(args.model)

        if args.image_folder != '':
            print("Creating image folder at {}".format(args.image_folder))
            if not os.path.exists(args.image_folder):
                os.makedirs(args.image_folder)
            else:
                shutil.rmtree(args.image_folder)
                os.makedirs(args.image_folder)
            print("RECORDING THIS RUN ...")
        else:
            print("NOT RECORDING THIS RUN ...")

        # wrap Flask application with engineio's middleware
        app = Flask(__name__)
        app = socketio.Middleware(sio, app)

        # deploy as an eventlet WSGI server
        eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

