## Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework

### AIM:
To design and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation.

### PROBLEM STATEMENT:
Current methods for evaluating the performance of complex, fine-tuned transformer models, such as BART, for Named Entity Recognition (NER) often require technical expertise and command-line interfaces, which significantly limits accessibility for domain experts (e.g., historians, legal analysts) who need to interactively test and provide feedback on the model's output in real-time. This lack of an accessible, interactive interface slows down the iterative development and quality assurance process

### DESIGN STEPS:

#### STEP 1: Import Libraries and Load Environment Variables 

Import the necessary Python libraries: os, json, requests, gradio, and dotenv.

Load the .env file to access the Hugging Face API key and model endpoints securely.

#### STEP 2: Define Helper Function for API Calls

Create a get_completion() function that sends HTTP POST requests to the Hugging Face Inference API.

Include Authorization headers for secure access using the API token.

#### STEP 3: Define the Named Entity Recognition (NER) Function

Use the get_completion() function to send input text to the NER model endpoint.

Process the JSON response and extract named entities.

#### STEP 4: Token Merging (Optional Enhancement)

Implement a merge_tokens() helper function to merge subword tokens (e.g., “Cal” + “##ifornia” → “California”) for cleaner entity visualization.

#### STEP 5: Build Gradio Interface

Create a Gradio interface using gr.Interface() with:

Input: Textbox for entering text.

Output: HighlightedText for displaying entities.

Example texts for quick testing.

Launch the application using demo.launch(share=True) to generate a public link for access.

### PROGRAM:

```py
import os
import json
import requests
import gradio as gr
from dotenv import load_dotenv, find_dotenv

# Load .env variables
_ = load_dotenv(find_dotenv())
hf_api_key = "hf_api_key"
API_URL = "https://router.huggingface.co/hf-inference/models/dslim/bert-base-NER"

# HF API CALL
def get_completion(inputs, parameters=None, ENDPOINT_URL=API_URL):
    headers = {
        "Authorization": f"Bearer {hf_api_key}",
        "Content-Type": "application/json"
    }

    data = {"inputs": inputs}
    if parameters:
        data.update({"parameters": parameters})

    response = requests.post(ENDPOINT_URL, headers=headers, data=json.dumps(data))
    text = response.content.decode("utf-8").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for part in text.split("\n"):
            try:
                return json.loads(part)
            except:
                continue
        raise ValueError(f"Invalid JSON returned from model: {text}")


# TOKEN MERGING + FORMATTING FOR GRADIO
def format_entities_for_gradio(entities):
    """
    Converts HF NER output into the format required by Gradio HighlightedText.
    Gradio expects: {"text": "...", "entities": [{"entity": "...", "start": x, "end": y}]}
    """

    formatted = []
    for e in entities:
        entity_type = e.get("entity", e.get("entity_group"))
        if entity_type is None:
            continue

        start = e.get("start")
        end = e.get("end")
        if start is None or end is None:
            continue

        formatted.append({
            "entity": entity_type,
            "start": start,
            "end": end
        })

    return formatted

# MAIN NER FUNCTION
def ner(input_text):
    output = get_completion(input_text)

    if not isinstance(output, list):
        raise ValueError(f"Unexpected model output: {output}")

    entities = format_entities_for_gradio(output)

    return {"text": input_text, "entities": entities}


# GRADIO UI
gr.close_all()

demo = gr.Interface(
    fn=ner,
    inputs=gr.Textbox(label="Enter text", lines=3),
    outputs=gr.HighlightedText(label="NER Result"),
    title="NER Application – Fine-tuned BART Model",
    description=(
        "Prototype application for Named Entity Recognition using a fine-tuned BART model, "
        "deployed with Gradio for evaluation and user interaction."
    ),
    examples=[
        ["My name is Rithik and I live in Chennai."],
        ["Google was founded by Larry Page and Sergey Brin."],
        ["Prathik works at DeepLearningAI in Bangalore."]
    ]
)

demo.launch(share=True, server_port=int(os.environ.get("PORT3", 7860)))
```
### OUTPUT:

![alt image](https://github.com/user-attachments/assets/bee2eca2-ae67-421b-9bde-b5cd1b4f4901)

### RESULT:
The Named Entity Recognition (NER) prototype was successfully developed using the fine-tuned BERT model (dslim/bert-base-NER) and deployed through the Gradio interface. The system efficiently identifies and highlights entities such as names, locations, and organizations from user-provided text input.
