from flask import Flask, request
from flask_cors import CORS
import json
import os
from transformers import pipeline
import zipfile
from git import Repo
import threading
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)
CORS(app)

# Load the Universal Sentence Encoder
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


def semantic_similarity(sentence1, sentence2, embed):

  # # Example sentences
  # sentence1 = "Hello, how are you?"
  # sentence2 = "I'm doing fine, thank you."

  # Encode the sentences
  embedding1 = embed([sentence1])[0]
  embedding2 = embed([sentence2])[0]

  # Reshape the embeddings
  embedding1 = np.reshape(embedding1, (1, -1))
  embedding2 = np.reshape(embedding2, (1, -1))

  # Calculate cosine similarity
  similarity_score = cosine_similarity(embedding1, embedding2)

  print("Similarity Score:", similarity_score[0][0])

  threshold = 0.35

  if(similarity_score[0][0] > threshold):
    return True
  else:
    return False

# Create upload folder if it doesn't exist
UPLOAD_FOLDER = 'upload'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Create gitFiles folder if it doesn't exist
GIT_FILES_FOLDER = 'upload'
if not os.path.exists(GIT_FILES_FOLDER):
    os.makedirs(GIT_FILES_FOLDER)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return json.loads('{"status":"No file part"}')

    file = request.files['file']
    if file.filename == '':
        return json.loads('{"status":"No selected file"}')
    if file:
        file.save(os.path.join(UPLOAD_FOLDER, file.filename))
        # unzip_file()
        thread1 = threading.Thread(target=unzip_file)
        thread1.start()
        return json.loads('{"status":"Sucessfully uploaded"}')

@app.route('/download_repo', methods=['POST'])
def download_repo():
    github_link = request.json.get('github_link')
    if not github_link:
        return 'No GitHub link provided'
    try:
        repo = Repo.clone_from(github_link, GIT_FILES_FOLDER)
        print("Names corresponding to the specified extensions:", get_ext_names("gitFiles/"))
        return json.loads('{"status":"Sucessfully Cloned"}')
    except Exception as e:
        return f'Error downloading repository: {str(e)}'

@app.route('/query', methods=['POST'])
def query():
    prompt = request.json.get('query')
    print("Prompt :",prompt)
    if not prompt:
        return json.loads('{"status":"No query found"}')
    else:
        #Abstraction of the working of current code
        s1 = "The development of artificial intelligence has revolutionized various industries, leading to significant advancements in automation, data analysis, and decision-making processes, ultimately shaping the way we live and work in the modern world."
        #Requirement specified by the user for the new task
        # s2 = "Climate change poses a grave threat to our planet's ecosystems, biodiversity, and human societies, necessitating urgent action to mitigate its impacts through sustainable practices, renewable energy adoption, and international cooperation for effective climate policies and adaptation strategies."
        print("Similar :",semantic_similarity(s1,prompt,embed))
        # Path we will get from s1...
        path = "upload"
        createFile(path,prompt)
        return json.loads('{"status":"Sucessfully Query recieved"}')


def unzip_file():
    zip_folder_path="upload"
    destination_directory="upload"
    # Get the list of files in the zip folder
    files = os.listdir(zip_folder_path)
    
    # Look for the zip file in the folder
    zip_file = None
    for file in files:
        if file.endswith('.zip'):
            zip_file = file
            break
    if zip_file:
        zip_file_path = os.path.join(zip_folder_path, zip_file)
        try:
            # Create the destination directory if it doesn't exist
            if not os.path.exists(destination_directory):
                os.makedirs(destination_directory)

            # Extract the contents of the zip file
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(destination_directory)
                

        except Exception as e:
            print(f"An error occurred while extracting the zip file: {e}")

        finally:
            os.remove(zip_file_path)
            print("Names corresponding to the specified extensions:", get_ext_names("upload/"))

            
    else:
        print("No zip file found in the specified folder.")
    

def get_ext_names(current_folder):
    valid_lang = ["Python", "JavaScript", "Java", "C++", "C#", "Ruby", "Swift", "Go", "Rust", "TypeScript", "Kotlin", "PHP", "HTML", "CSS", "SQL", "R", "MATLAB", "Perl", "Shell", "Objective-C", "Assembly", "Lua", "Dart", "Scala", "Haskell", "Groovy", "Julia", "Cobol", "Fortran", "Lisp", "Scheme", "Tcl", "Ada", "Prolog", "Elixir", "Erlang", "Clojure", "F#", "Verilog", "VHDL", "ActionScript", "CoffeeScript", "Delphi", "Elm", "Eiffel", "Forth", "FoxPro", "LabVIEW", "Logo", "ML", "Pascal", "PL/SQL", "PostScript", "PowerShell", "PureScript", "Smalltalk", "ABAP", "Apex", "J", "Objective-J", "Max/MSP", "Oz", "Crystal", "AWK", "Bourne shell", "C shell", "Cilk", "ClojureScript", "Cython", "D", "E", "Emacs Lisp", "Fish", "Gosu", "IDL", "Io", "JScript", "Lex", "Nim", "OCaml", "Parrot", "Racket", "REXX", "S", "SPARK", "Chapel", "Dylan", "Idris", "K", "Limbo", "M4", "Magma", "Maple", "Mathematica", "Mercury", "MQL4", "MUMPS", "MQL5", "NetLogo", "NXT-G", "Oberon", "Objective-C++", "Objective-J", "Pop-11", "Processing", "Pure Data", "Q", "REALbasic", "REBOL", "Revolution", "S-Lang", "SAS", "Scratch", "ShaderLab", "Silex", "Squirrel", "Squeak", "Stata", "Suneido", "SuperCollider", "TeX", "TI-BASIC", "Turing", "UnrealScript", "Vala/Genie", "VBA", "VBScript", "VimL", "Visual Basic .NET", "WebAssembly", "Winbatch", "X10", "xBase", "XC", "Xojo", "XQuery", "XSB", "XSLT", "Yorick", "Z notation", "Zap", "Zeno", "Zsh", "C*", "C--", "Elm", "HTML5", "Haxe", "JScript", "LiveScript", "PureBasic", "Kotlin", "Ceylon", "TypeScript", "PPL (Perl)", "PPL (Python)", "Ceylon", "ECMAScript", "JScript", "Dart", "JSX", "Squirrel", "Hack", "Objective-J", "Ring", "Q#", "T-SQL", "MAXScript", "Apex", "Ada", "ASP", "MATLAB", "Scheme", "Racket", "AutoIt", "BlitzMax", "CFML (ColdFusion)", "Coq", "CSP", "E", "Erlang", "F#", "Forth", "Groovy", "IDL", "JScript.NET", "LabVIEW", "LiveCode", "MQL4", "MQL5", "MS-DOS batch", "MUMPS", "LabVIEW", "Octave", "Pike", "PowerShell", "Q", "Ring", "Scala", "Stata", "Tcl", "UnityScript", "VBScript", "Xojo", "Z shell", "Delphi", "Pascal", "PL/SQL", "ABAP", "RPG (OS/400)", "RPG (report program generator)", "Scratch", "Shell", "Swift", "T-SQL", "Windows PowerShell", "XQuery", "Clojure", "Cobol", "REXX", "Elixir", "Io", "Prolog", "JRuby", "Jython", "Ring", "BeanShell", "OpenCL", "VHDL", "Verilog", "WebAssembly", "IDL", "Cython", "Pike", "GAP", "APL", "FORTH", "Logtalk", "ECL", "JADE", "Red", "Boo", "Ada", "QML", "OpenEdge ABL", "ROOP", "BCPL", "Rust", "PCASTL", "Flavors", "Transact-SQL", "ColdFusion", "Groovy", "Groovy (JVM)", "Fantom", "Smalltalk", "OpenCL", "GPU-accelerated JavaScript", "GameMaker Language", "PureScript", "Clojure", "Elm", "Kotlin", "Flow", "Shen", "Idris", "Squirrel", "ABAP", "WebAssembly", "Chapel", "Opa", "Max", "Oz", "Rust", "Mirah", "Frink", "Opa", "CoffeeScript", "Elm", "TypeScript", "Dart", "Nim", "Crystal", "Haxe", "Futhark", "Wren"]

    # Get Extension list
    def get_file_extensions(folder_path):
        extensions = set()
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                filename, extension = os.path.splitext(file)
                extensions.add(extension.lower())
        return extensions

    extensions_to_search = get_file_extensions(current_folder)

    # File path
    file_path = "json/languages.json"

    names = []

    # Read JSON data from file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Iterate over each language
    not_Prog_Lang = []
    for language, info in data.items():
        extensions = info.get("extensions", [])
        for ext in extensions_to_search:
            if ext in extensions:
                if language in valid_lang:
                    names.append(language)
                else:
                    not_Prog_Lang.append(language)
                break
    return names
    
def comparision(paragraph):
    np= pipeline("text-generation", model="distilgpt2")
    
    generated_text = np(paragraph, max_length=100, num_return_sequences=1)
    
    generated_text = generated_text[0]['generated_text'].strip()
    
    lines2 = generated_text.split("\n")
    file_extension2 = None
    content2 = ""
    create_file2 = False

    # Iterate through each line of the generated text
    for line in lines2:
        line_lower2 = line.lower()
        if "create a" in line_lower2 or "build a" in line_lower2 or "make a" in line_lower2:
            create_file2 = True
            # Extract the file extension (if specified)
            words2 = line_lower2.split()

            for word in words2:
                if "." in word:
                    file_extension2 = word.split(".")[-1].strip()
                    break
        else:
            content2 += line + "\n"

    # Remove trailing newline characters from content
    content2 = content2.strip()

    return file_extension2, create_file2, content2

def extract_info_from_paragraph(paragraph):
    # Load the text generation pipeline
    nlp= pipeline("text-generation", model="distilgpt2")

    # Generate text based on the input paragraph
    generated_text = nlp(paragraph, max_length=100, num_return_sequences=1)

    # Extract relevant information from the generated text
    generated_text = generated_text[0]['generated_text'].strip()
    lines = generated_text.split("\n")

    file_extension = None
    file_name = None
    content = ""
    create_file = False

    # Iterate through each line of the generated text
    for line in lines:
        line_lower = line.lower()
        if "create a" in line_lower or "build a" in line_lower or "make a" in line_lower:
            create_file = True
            # Extract the file extension (if specified)
            words = line_lower.split()
            for word in words: 
                if "." in word:
                    file_name = word.split(".")[0].strip()
                    break
                
            for word in words:
                if "." in word:
                    file_extension = word.split(".")[-1].strip()
                    break
        else:
            content += line + "\n"

    content = content.strip()

    return file_extension, create_file, content, file_name

def create_file_in_folder(folder_name, file_extension, content, create_file, file_name):
    if create_file:
        try:
            # Determine the full path based on the provided folder name
            file_path = os.path.join(os.getcwd(), folder_name)

            # Create the folder if it doesn't exist
            os.makedirs(file_path, exist_ok=True)

            with open(os.path.join(file_path, f"{file_name}.{file_extension}"), "w") as file:
                file.write(content)

            print(f"File '{file_path}/{file_name}.{file_extension}' created successfully with the provided content.")
        except Exception as e:
            print(f"An error occurred: {e}")

def createFile(folder_name, paragraph):
    # paragraph = """Create a file named bav.html file
    # This is the content of the file.
    # It can contain multiple lines.
    # """
    
#     store_path = "C:/Users/nayan/Downloads/hackathon"
    # folder_name = "frontEnd"
    print("Hangigyayayaaya")
    file_extension, create_file, content, file_name = extract_info_from_paragraph(paragraph)
    file_extension2, create_file2, content2 = comparision(paragraph)
    c=list(content)
    c2=list(content2)
    x=len(c)
    y=len(c2)
    print("Hangigyayayaaya")

    strg=""
    if(x>y):
      for i in range(0,x):
        if(c[i]==c2[i]):
          strg+=c[i]
        else:
          break
    else:
      for i in range(0,y):
        if(c[i]==c2[i]):
          strg+=c2[i]
        else:
          break

#     create_file_with_extension(file_extension, strg, create_file, store_path)
    create_file_in_folder(folder_name, file_extension, strg, create_file, file_name)
    

createFile("upload","""Create a file named bav.html file
    # This is the content of the file.
    # It can contain multiple lines.""")
if __name__ == '__main__':
    app.run(debug=True)