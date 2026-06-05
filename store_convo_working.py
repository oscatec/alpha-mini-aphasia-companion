"""
HRI Coding Assignment - PwA Introductory Conversation Robot
Uses Gemini for LLM responses and the Alpha Mini robot's STT/TTS modules.
"""

from autobahn.twisted.component import Component, run
from twisted.internet.defer import inlineCallbacks
from autobahn.twisted.util import sleep
from google import genai
import json
import os


#TODO: if the robot forgets about the systemp prompt after the introduction and keeps asking about movies and pizza, then we can do this:
# make another prompt with guidelines for the conversation and topics to talk about. But this is only given to the robot when introduction is False
# and dialogue is True

# Gemini client and persistent chat session (maintains conversation history)
API_KEY = "REDACTED_API_KEY" # Insert your Gemini API key here. 
chatbot = genai.Client(api_key=API_KEY)

SYSTEM_PROMPT = (
    "Your name is Mini. You are a warm, friendly social robot. "
    "Your role is to have a short introductory conversation with a patient who has aphasia." \
    "Keep every response to short, simple sentences. "
    "After the introduction, ask the user:\n"
    "- how was their day" \
    "- how their family is doing" 
)

EXIT_WORDS = ("quit", "exit", "goodbye", "bye", "stop", ":q")

# Create one chat session. This preserves the full conversation history
chat = chatbot.chats.create(
    model="gemini-2.5-flash",
    config=genai.types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
)


# Shared state for ASR callback
is_speaking = False
finish_dialogue = False
query = ""

conversation = []

def asr(frames):
    """Callback invoked by the robot's STT stream on each recognised utterance."""
    global finish_dialogue, query, is_speaking

    if is_speaking:
        return

    if frames["data"]["body"]["final"]:
        recognized_text = str(frames["data"]["body"]["text"]).strip()

        if not recognized_text:
            return
        
        query = recognized_text
        print(f"[ASR] '{query}'")
        finish_dialogue = True
        

# Main robot session
@inlineCallbacks
def main(session, details):
    global finish_dialogue, query, is_speaking

    # Configure language and posture
    yield session.call("rie.dialogue.config.language", lang="en")
    yield session.call("rom.optional.behavior.play", name="BlocklyStand")


    # Let the LLM generate the opening greeting
    opening = chat.send_message("Begin the conversation by greeting and asking for the name of the user")
    print(f"[Robot] {opening.text}")

    is_speaking = True
    yield session.call("rie.dialogue.say", text=opening.text)
    yield sleep(0.5)
    is_speaking = False

    # Subscribe to ASR events once; the stream is opened/closed each turn
    yield session.subscribe(asr, "rie.dialogue.stt.stream")

    introduction = True
    while introduction:
        finish_dialogue = False
        query = ""
        yield session.call("rie.dialogue.stt.stream")

        # Wait until ASR delivers a final result
        while not finish_dialogue:
            yield sleep(0.1)

        yield session.call("rie.dialogue.stt.close")  

        print(f"[User] '{query}'")

        print("the robot should now extract the name of the user")
        response = chat.send_message(f"Look at the following message and extract the name of the person. Reply with only the name of the person, {query}. Only the name, no full stop")
        target_name = response.text 
        print("the extracted name is", target_name)
        user_info = lookup_info(target_name)

        # send the information to Gemini
        if user_info:
            follow_up = chat.send_message(f"Remember this information about the user {user_info}. Just remember it, do NOT repeat the information as it is. Say that you remember one piece of the information. Only one. Then ask a follow up about it. For example, if they like walking, ask them where they like to walk.")
            yield session.call("rie.dialogue.say", text=follow_up.text)
            print(f"[robot] {follow_up.text}")
            # verification = chat.send_message(f"Say, {target_name}, you are very cool")
            # yield session.call("rie.dialogue.say", text=verification.text)
        
        else:
            follow_up = chat.send_message(f"Tell the user you have not met them yet. Ask them their age.")
            yield session.call("rie.dialogue.say", text=follow_up.text)
            print(f"[robot] {follow_up.text}")

        introduction = False
        dialogue = True
        print("introduction is over")
    
    while dialogue:
        # Reset state and start listening
        finish_dialogue = False
        query = ""
        yield session.call("rie.dialogue.stt.stream")

        # Wait until ASR delivers a final result
        while not finish_dialogue:
            yield sleep(0.3)

        # Stop listening BEFORE the robot speaks so it cannot hear itself
        yield session.call("rie.dialogue.stt.close")
        yield sleep(0.3)

        print(f"[User] '{query}'")
        conversation.append(query)

        if query.lower() in EXIT_WORDS:
            introduction = False
            dialogue = False
            farewell = chat.send_message(
                "The patient wants to end the conversation. Say a warm one-sentence goodbye."
            )
            print(f"[Robot] {farewell.text}")
            yield session.call("rie.dialogue.say", text=farewell.text)

        elif query:
            response = chat.send_message(query)
            print(f"[Robot] {response.text}")
            is_speaking = True
            yield session.call("rie.dialogue.say", text=response.text)
            print(f"[robot] {response.text}")
            yield sleep(0.3)
            is_speaking = False
            

        else:
            # Empty transcription-ask the patient to repeat
            yield session.call(
                "rie.dialogue.say",
                text="Sorry, I didn't catch that. Could you say it again?",
            )

        # Brief pause so the robot's audio fully stops before re-opening STT
        yield sleep(1)
    print("the conversation if finished")

    store_info(conversation)

    yield session.call("rom.optional.behavior.play", name="BlocklyCrouch")
    session.leave()

def store_info(conversation):
    filename = 'sample.json'
    request = chat.send_message(f"Your name is Mini. You are a warm, friendly social robot. "
    "Your role is to have a short introductory conversation with a patient who has aphasia. Take the " 
    "following conversation, {conversation}, and reply with a summary of the user's information, such as their age, hobbies, etc,... " 
    "The information should be timeless: in a sense that it should not be related to temporary things like mood etc,.." \
    "Also, the name of the user is always present in the conversation. Your reply should be " 
    "[name of the user]: the summarized information you selected")
    text = request.text
    print("the information is", text)
    
    split_text = text.split(": ")

    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with open(filename, 'r') as f:
            data = json.load(f)
    else:
        data = []

    # Check if the user already exists in the list
    user_found = False
    for entry in data:
        if entry["name"].lower() == split_text[0].lower():
            entry["information"] += f", {split_text[1]}"
            user_found = True
            break 

    # If the user doesn't exist, create a brand new entry
    if not user_found:
        new_entry = {
            "name": split_text[0],
            "information": split_text[1]
        }
        data.append(new_entry)

    # save the file
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def lookup_info(user_name):
    filename = 'sample.json'
    # make sure file is there
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        return f"Error: '{filename}' does not exist or is empty."

    # open the file
    with open(filename, 'r') as f:
        data = json.load(f)

    # search for the information by the name
    for person_dict in data:
        # Check if the name is there 
        if person_dict["name"].lower() == user_name.lower():
            # return the information
            return person_dict["information"]

    # 4. If the loop finishes without finding the name
    else:
        print(f"User '{user_name}' was not found in the file.")
        return

    

# WAMP connection 
wamp = Component(
    transports=[{
        "url": "ws://wamp.robotsindeklas.nl",
        "serializers": ["msgpack"], 
        "max_retries": 0,
    }],
    realm="rie.REDACTED_REALM",
)

wamp.on_join(main)

if __name__ == "__main__":
    run([wamp])
