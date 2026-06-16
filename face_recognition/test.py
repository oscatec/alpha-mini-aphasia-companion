from face_store_manager import FaceStoreManager
from face_recogniztion_manager import (
    FacialFeatures,
    FaceNotFoundError,
    FaceRecognitionManager,
    MultipleFacesFoundError,
)


def average_facial_features(features: list[FacialFeatures]) -> FacialFeatures:
    if not features:
        raise ValueError("Cannot average an empty feature list.")

    return FacialFeatures(
        pixels=[
            sum(feature.pixels[idx] for feature in features) / len(features)
            for idx in range(len(features[0].pixels))
        ],
        histogram=[
            sum(feature.histogram[idx] for feature in features) / len(features)
            for idx in range(len(features[0].histogram))
        ],
        edge_pixels=[
            sum(feature.edge_pixels[idx] for feature in features) / len(features)
            for idx in range(len(features[0].edge_pixels))
        ],
        gradient_pixels=[
            sum(feature.gradient_pixels[idx] for feature in features) / len(features)
            for idx in range(len(features[0].gradient_pixels))
        ],
    )


def main() -> None:
    face_store_manager  = FaceStoreManager()
    # path = "test_images/martijn/"
    image_path = "test_images/martijn/martijn 2.jpg"
    
    # get the facial features
    try:
        recognition_manager = FaceRecognitionManager(image_path)
        facial_features = recognition_manager.get_facial_features()

        # recognition_manager = FaceRecognitionManager(path + "martijn.jpg")
        # facial_features_martijn = recognition_manager.get_facial_features()

        # recognition_manager = FaceRecognitionManager(path + "martijn 2.jpg")
        # facial_features_martijn_2 = recognition_manager.get_facial_features()

        # recognition_manager = FaceRecognitionManager(path + "martijn 3.jpg")
        # facial_features_martijn_3 = recognition_manager.get_facial_features()

        # avg_facial_features = average_facial_features([
        #     facial_features_martijn,
        #     facial_features_martijn_2,
        #     facial_features_martijn_3,
        # ])
    except FaceNotFoundError:
        print("No face found. Please stand in front of the camera and try again.")
        return
    except MultipleFacesFoundError:
        print("Multiple faces found. Please make sure only one person is visible.")
        return
    # except FileNotFoundError as error:
    #     print(error)
    #     return
    # except ValueError as error:
    #     print(error)
    #     return
    # except RuntimeError as error:
    #     print(error)
    #     return

    # print(facial_features)

    # Check if the facial features have anything in commong with the database

    name = face_store_manager.check_match_in_DB(facial_features, threshold=0.2)
    # name = face_store_manager.check_match_in_DB(facial_features_martijn_2, threshold=0.2)
    # name = face_store_manager.check_match_in_DB(facial_features_martijn_3, threshold=0.2)
    
    # If no match, say hello and ask for name
    if name == None:
        print("I have not seen you before, what is your name?")
        name = input()
        # asks for name
        face_store_manager.add_person(name, facial_features)
    
    # If match, print the name of the match
    else:
        print(f"Hello, {name}")
    # encode the face in the captured image


if __name__ == "__main__":
    main()
