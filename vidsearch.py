from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import chromadb
import json
import requests 


# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
     "Qwen/Qwen2.5-VL-3B-Instruct",
     torch_dtype="auto",
     # attn_implementation="flash_attention_2",  good optimization, but pain in the ass to get it to work on windows T_T 
     device_map="auto",

 )


# The default range for the number of visual tokens per image in the model is 4-16384
# Qwen2.5vl splits each frame into 28x28 pixel patches, each patch = 1 visual token

# Pixel formula and settings for Qwen 2.5 vl 
# da formulas:   resolution^2 / (28*28 or 784) == multiplier, SO THEN   min pixels: multiplier * 28 * 28 
#ref:  400x400 = 196 * 28 * 28, 512 x 512 = 324 * 28 * 28
# gameboy sp res is 240 x 160. So (240 * 160)/784 = 48.9, so min pixels: 49 * 28 * 28

max_pixels=1280 * 28 * 28
min_pixels = 196 * 28 * 28

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", min_pixels = min_pixels, max_pixels = max_pixels )

#setting up the ChromaDB 
chroma_cl = chromadb.Client()
collect = chroma_cl.get_or_create_collection(name="video_analysis")

def store_data(video, Jdata):
    for i, segment in enumerate(Jdata):
        ider = f"{video}_seg_{i}" # creates unique id for video title: MLPs4x10.mp4_seg_0
        da_info = f"""
        Time: {segment['start_time']} to {segment['end_time']}
        Description: {segment['description']} 
        Actions: {segment['actions']}
        Body Language: {segment['body_language']}
        Emotions: {segment['emotions']}
        Visuals: {segment['visuals']}
        Key Details: {segment['key_details']}

    """
        collect.add(
            documents = [da_info],
            ids = [ider],
            metadatas = [{
                "video": video,
                "start": segment['start_time'],
                "end": segment['end_time']
            }]
        )
        print(f"stored seg {i}")


#SET UP DEFAULT PROMPT :P
default_prompt = """Analyze this video frame by frame and provided detailed JSON output
    {
    For the ENTIRE video, return a JSON array with each of these objects. For timestamps use format: (MM:SS) :
    "start_time": "MM:SS",
    "end_time": "MM:SS",
    "description": "TWo people are sitting at a table, one is looking at his phone while eating a hot dog",
    "actions": "person_1: speaking and sitting", "person_2: on his phone while eating a hot dog",
    "body_language": "person_1: confident posture, direct eye contact", "person_2: distracted, leaning forward",
    "emotions": "person_1: passionate", "person_2: excited",
    "visuals": "",
    "key_details": []

    }


    DA RULES:
    Be specific about actions. DONT SAY "talking", but be descriptive, "gesturing with his hands, looking down while speaking".
    Describe ALL body language objectively (posture, movements, etc)
    Include all visuals including captions, text, images, graphcis, clothing, devices (created section for visuals)
    Use short segments like 2-5 seconds
    DO NOT stretch timestaps, make continuous
    DO NOT over fragment the video
    ONLY DESCRIBE WHAT YOU SEE. DONT MAKE UP SCENES
    OUTPUT ONLY VALID JSON, NO OTHER TEXT. 
 """


def analyze_vid(vid, prompt= None, fps=2.5):
    if prompt is None:
        prompt = default_prompt
    
    messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": vid,
                "fps": fps,
            },
            {"type": "text", "text": prompt},
        ],
    }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # takes messages dict, strips the vids and imgs as args, including return vid kwargs which is vid settings
    vision_info = process_vision_info(messages)
    images = vision_info[0] # stores imgs in 1st index
    videos = vision_info[1] # stores vids in 2nd index

    #mostly boilerplate from HuggingFace,  text takes formated txt from apply_chat_template(), images,video,fps all take from vision_info, padding = pads sequence for same vid length, return_tensors outputs as pytorch tensors, **video_kwargs unpacks extra video settings from process_vision_info   
    inputs = processor(text = [text], images = images, videos = videos, padding = True, return_tensors = "pt")
    # takes inputs and moves to GPU memory 
    inputs = inputs.to("cuda")
    #unpacks inputs runs through qwen model. max token is equal to 2^11, google recommends it so :P
    generate_ids = model.generate(**inputs, max_new_tokens = 2048 )

    #trims off user input tokens, leaves only generated tokens. 
    # slices off length from input aka len(input_ids) and keeps what's after
    generate_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generate_ids)
    ]

    # converts token ids back into redeable text using batch_decode. 
    # skip_special = true to remove any special tokens generated by model
    # clean_up_tokenization_spaces = false keeps spacing as generated by ai

    output_text = processor.batch_decode(
    generate_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)


    raw = output_text[0] #gets string from text, also the word raw is funny
    pretty_output = raw.replace("```json", "").replace("```", "").strip()
    try:
      parsed = json.loads(pretty_output)
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed: {e}")
        print("Raw output was:")
        print(pretty_output)
        return None
    print(parsed)
    return parsed


        
def ask_llama(q):

    # goes through DB
    search = searchQUERY(q)
    # gets docs
    docs = search['documents'][0]
    # for loop adds new line break, formating basically 
    context = ""
    for x in docs:
        context += x + "\n"


    prompt = f"""Based ONLY on the following video data provided, answer all questions. If the answer is not in the data provided, tell the user that data can't be found in the video
                Video data: {context}
                Question: {q} """
    reply = requests.post('http://localhost:11434/api/generate',
        json={
            'model': 'ai_model', # replace with prefered model like qwen3:8b for ex.
            'prompt': prompt,
            'stream': False
        }
    )
    return reply.json()['response']

def searchQUERY(query):
    results = collect.query(
        query_texts = [query],
        n_results = 2
    )
    return results 


if __name__ == "__main__":
    result = analyze_vid(r"vid file path here")
    store_data("vid title", result) # replace vid title

    answer = ask_llama("what was the man doing while talking?")
    print(answer)

