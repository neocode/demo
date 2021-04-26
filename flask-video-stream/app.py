from flask import Flask, render_template, Response, request
from flask_bootstrap import Bootstrap
from pathlib import Path
from camera import Camera
import argparse
# import logging, logging.config

# logger = logging.getLogger(__name__)

camera = Camera()
camera.run()

app = Flask(__name__)
Bootstrap(app)


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering or Chrome Frame,
    and also to cache the rendered page for 10 minutes
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers["Cache-Control"] = "public, max-age=0"
    return r


def gen(camera, mask):
    # logger.debug("Starting stream")
    while True:
        frame = camera.get_frame(mask)
        yield (b'--frame\r\n'b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')


@app.route("/")
@app.route("/mask_1")
def mask_1():
    # logger.debug("Requested stream page")
    return render_template("mask_1.html")


@app.route("/mask_2")
def mask_2():
    # logger.debug("Requested stream page")
    return render_template("mask_2.html")


@app.route("/mask_3")
def mask_3():
    # logger.debug("Requested stream page")
    return render_template("mask_3.html")


@app.route("/mask_4")
def mask_4():
    # logger.debug("Requested stream page")
    return render_template("mask_4.html")


@app.route("/video_feed")
def video_feed():
    mask = request.args.get('mask_path')
    return Response(gen(camera, mask), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--port',type=int,default=5000, help="Running port")
    parser.add_argument("-H","--host",type=str,default='0.0.0.0', help="Address to broadcast")
    args = parser.parse_args()
    # logger.debug("Starting server")
    app.run(host=args.host, port=args.port)
