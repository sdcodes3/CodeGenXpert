
# CodeGenXpert

This project aims to develop an AI-powered code generation platform tailored to company-specific codebases. The platform automates the creation of boilerplate code, scaffolding, and repetitive tasks, facilitating rapid development while ensuring high coding standards. The platform leverages machine learning (ML) and artificial intelligence (AI) components to analyze the company-specific codebase, identify patterns, and generate new code aligned with the project's architecture, functionalities, and coding standards.

![alt text](https://i.ibb.co/w6HrwrL/form.png)



## Documentation

[Documentation](https://linktodocumentation)

In our project, we have developed a comprehensive solution for automating various coding tasks using cutting-edge technologies and frameworks. At the core of our system, we have employed Transformers, a state-of-the-art deep learning architecture, and pre-trained models to tackle a multitude of coding-related tasks. Leveraging the power of these models, our system is capable of performing tasks such as code completion, code summarization, and even code generation.

To provide a user-friendly interface for interacting with our system, we have designed a sleek and intuitive front-end using HTML and CSS. This front-end allows users to input a GitHub repository URL, enabling seamless integration with their existing codebase. Upon receiving the repository URL, our system proceeds to download the repository, providing the necessary data for analysis and processing.

To further streamline the user experience and facilitate seamless communication between the front-end and the back-end, we have integrated Flask, a lightweight and efficient web framework for Python. Flask acts as the bridge between the HTML/CSS front-end and our back-end processing pipeline. When a user inputs a prompt or request via the front-end interface, Flask orchestrates the necessary actions to invoke our model API.

Behind the scenes, our model API utilizes the pre-trained Transformers model to generate structured code snippets based on the provided prompt and the analyzed GitHub repository data. These code snippets are automatically structured to align with the architecture, functionalities, and coding standards observed within the repository. The generated code is then returned to the user via the Flask framework, completing the end-to-end workflow seamlessly.

Overall, our system offers a sophisticated yet user-friendly solution for automating coding tasks, empowering developers to accelerate development cycles, maintain high coding standards, and streamline their workflows with ease.
## Features

- Generate Code
- Live previews
- Structured Code


## Demo

```bash
  from transformers import AutoTokenizer, AutoModelWithLMHead

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeBERTa-base", use_auth_token=True)
model = AutoModelWithLMHead.from_pretrained("microsoft/CodeBERTa-base", use_auth_token=True)

# Define the code
code = """
def hello_world():
    print("Hello, world!")
"""

# Encode the code and generate a description
inputs = tokenizer.encode(code, return_tensors='pt')
outputs = model.generate(inputs, max_length=50, num_return_sequences=5, temperature=0.6)

# Decode the outputs to get the descriptions
for output in outputs:
    description = tokenizer.decode(output, skip_special_tokens=True)
    print(description)
```


## Installation

Requirement CodeGenXpert 

```bash
  pip install transformers
  pip install accelarate
  pip install datasets
  pip install torch
  pip install langchain
  pip install huggingface_hub
```


    
## Usage/Examples

```py
from transformers import AutoTokenizer, AutoModelWithLMHead

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeBERT-base", use_auth_token=True)
model = AutoModelWithLMHead.from_pretrained("microsoft/CodeBERT-base", use_auth_token=True)

# Define the code
code = """
def hello_world():
    print("Hello, world!")
"""

# Encode the code and generate a description
inputs = tokenizer.encode(code, return_tensors='pt')
outputs = model.generate(inputs, max_length=50, num_return_sequences=5, temperature=0.6)

# Decode the outputs to get the descriptions
for output in outputs:
    description = tokenizer.decode(output, skip_special_tokens=True)
    print(description)
```


# Fine Tuning on CodeBERT

```py
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments

# Load pre-trained model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
model = RobertaForSequenceClassification.from_pretrained('microsoft/codebert-base')

# Prepare training data
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
train_labels = train_labels
train_dataset = tf.data.Dataset.from_tensor_slices((
    {"input_ids": train_encodings['input_ids'], "attention_mask": train_encodings['attention_mask']},
    train_labels
))

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
)

# Create trainer
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
)

# Train the model
trainer.train()

```
## Roadmap

- Our proposed model workflow encompasses two main stages: directory structure analysis and code generation based on user prompts.

- Firstly, the model undergoes a comprehensive analysis of the entire directory structure, systematically cataloging all file paths along with their respective extensions. This process entails capturing the hierarchical organization of files within the directory, ensuring a thorough understanding of the project's architecture.

- Upon initialization, the model assimilates the directory pattern as its input and proceeds to comprehend its structural nuances. Through this phase, the model gains insights into the organization of files, their relationships, and the distribution of various file types across the directory structure.

![alt text](https://i.ibb.co/Yyy5TB8/Flow-Chart.png)

- Subsequently, when a user issues a query or prompt, the model springs into action by embedding both the prompt text and the textual contents of all files within the project directory. This embedding process encapsulates the semantic representations of the prompt and the file contents, enabling efficient vectorization for subsequent analysis.

- Once the embedding is complete, the model proceeds to compare the similarity between the embedded representations. Utilizing advanced similarity metrics, the model evaluates the degree of resemblance between the prompt and the file contents. If the similarity falls below a predefined threshold, typically set at 0.5 for robustness, the model infers a divergence between the prompt and the existing codebase.

- The Universal Sentence Encoder is employed to convert prompt texts and file contents into fixed-size vectors for semantic representation.

- Embeddings generated by the Universal Sentence Encoder facilitate similarity comparison between user prompts and directory file texts.

- The model utilizes Universal Sentence Encoder embeddings to assess semantic similarity, enabling detection of discrepancies between prompts and existing codebase.

- Universal Sentence Encoder embeddings serve as the foundation for determining whether generated code snippets align with the architecture and functionalities of the project directory.

- By leveraging Universal Sentence Encoder embeddings, the model ensures efficient and effective code generation while maintaining adherence to project-specific coding standards.

![alt text](https://i.ibb.co/55t5t1S/Visual-Storytelling.png)

- In response to such disparities, the model initiates code generation, leveraging the textual contents of the files as a reference. Employing techniques such as neural language modeling or code generation algorithms, the model autonomously generates code snippets mirroring the existing structures within the files. These generated code snippets are then stored in their respective file paths, seamlessly integrating with the project's directory structure.

- In essence, our approach entails a meticulous analysis of the directory structure followed by intelligent code generation, ensuring alignment with existing patterns and coding standards. By employing advanced natural language processing and machine learning techniques, our model facilitates efficient code augmentation while maintaining the integrity of the project's architecture and functionalities.


## Contributing

Contributions are always welcome!

See `contributing.md` for ways to get started.

Please adhere to this project's `code of conduct`.

