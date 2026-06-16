from autobahn.twisted.component import Component, run
from twisted.internet.defer import inlineCallbacks
from autobahn.twisted.util import sleep

from pathlib import Path

from face_recogniztion_manager import FaceNotFoundError, FaceRecognitionManager, MultipleFacesFoundError
from face_store_manager import FaceStoreManager


# def sight(frame):
#     # This function is called every time the robot moves
#     print(frame)

# DEFAULT_IMAGE = Path("camera_image.jpg")
# def do_face_recognition(image_path: str = str(DEFAULT_IMAGE), threshold: float = 0.5) -> str:
#     """
#     Recognize the face in `image_path` against local face_db storage.
#     Returns the predicted name or "Unknown".
#     """
#     name, distance = recognize(image_path=image_path, threshold=threshold)

#     if distance is None:
#         print("No known faces in database.")
#     else:
#         print(f"Recognition result: {name} (distance={distance:.4f}, threshold={threshold})")


@inlineCallbacks
def main(session, details):
    face_store_manager = FaceStoreManager()

    yield session.call("rie.vision.face.find")
    frames = yield session.call("rom.sensor.sight.read")

    frame = frames[0]
    # image_data = frame["data"]["body.head.front"]


    # Save image from camera stream
    img_bytes = frames[0]["data"]["body.head.eyes"]  # this is already JPEG bytes
    with open("camera_image.jpg", "wb") as f:
        f.write(img_bytes)

    # detect if there is a face in the image
    try:
        recognition_manager = FaceRecognitionManager("camera_image.jpg")
        facial_features = recognition_manager.get_facial_features()
    # if match, print hello name

    # if no mathc, print face not recognized


    except FaceNotFoundError:
        print("No face found. Please stand in front of the camera and try again.")
        return
    except MultipleFacesFoundError:
        print("Multiple faces found. Please make sure only one person is visible.")
        return

    # Check if the facial features have anything in commong with the database
    name = face_store_manager.check_match_in_DB(facial_features, threshold=0.22)
    
    # If no match, say hello and ask for name
    # No match found
    if name is None:

        # Ask for the user's name
        yield session.call(
            "rie.dialogue.say",
            text="Hello, I do not think we have met before. What is your name?"
        )

        print("Waiting for user name...")

        # Listen for spoken input
        response = yield session.call("rie.dialogue.stt.read")

        print(response)
        name = response[0]["data"]["body"]["text"]

        print(f"Recognized name: {name}")

        # Confirm the name
        yield session.call(
            "rie.dialogue.say",
            text=f"Nice to meet you, {name}"
        )

        # Store in database
        face_store_manager.add_person(name, facial_features)

        print(f"Stored new person in database: {name}")

    
    # If match, print the name of the match
    else:
        # print("")
        yield session.call("rie.dialogue.say", text=f"Hello, {name}")
        yield sleep(2)
        print(f"Hello, {name}")
    # encode the face in the captured image

    
    # print(faceStoreManager.recognize("camera_image.jpg"))
    # do_face_recognition()

    # yield session.call("rom.sensor.sight.stream")
    session.leave()  # Close the connection with the robot

wamp = Component(
	transports=[{
		"url": "ws://wamp.robotsindeklas.nl",
		"serializers": ["msgpack"],
		"max_retries": 0
	}],
	realm="rie.REDACTED_REALM",
)

wamp.on_join(main)

if __name__ == "__main__":
	run([wamp])


# from pathlib import Path

# from face_store import recognize


# DEFAULT_IMAGE = Path("camera_image.jpg")


# def do_image_recognition(image_path: str = str(DEFAULT_IMAGE), threshold: float = 0.5) -> str:
#     """
#     Recognize the face in `image_path` against local face_db storage.
#     Returns the predicted name or "Unknown".
#     """
#     name, distance = recognize(image_path=image_path, threshold=threshold)

#     if distance is None:
#         print("No known faces in database.")
#     else:
#         print(f"Recognition result: {name} (distance={distance:.4f}, threshold={threshold})")

#     return name


# if __name__ == "__main__":
#     do_image_recognition()
