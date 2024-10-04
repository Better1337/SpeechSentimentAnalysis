import whisper

def transcribe_audio(file_path, model_name="base", device="cuda", language="en"):
    model = whisper.load_model(model_name, device=device) 

    result = model.transcribe(file_path, language=language)

    return result["text"]

