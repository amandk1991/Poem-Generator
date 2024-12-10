# Poem Generator 

This project generates short, contextually relevant poems based on input prompts using a fine-tuned GPT-2 model hosted on Hugging Face. The model is trained on a poetry dataset by Merve, which consists of poems from various themes and styles.

## Features
- **Poem Generation**: Generate short poems based on any input prompt (e.g., "A beautiful sunset").
- **Fine-Tuned GPT-2 Model**: The model is fine-tuned on the **Poetry dataset by Merve**, which includes poems from diverse genres and topics.
- **Hugging Face Integration**: The model is available for easy access via the Hugging Face API.

## Dataset
The model is trained on the **Poetry by Merve** dataset, which consists of poems of various lengths, genres, and themes. This dataset helps the model understand the structure and creativity of poetry, enabling it to generate poems that are contextually relevant to the input prompt.

Dataset details:
- **Name**: Poetry by Merve
- **Content**: Poems with different themes such as nature, love, and introspection.

## Files
- `Poem-Generator.ipynb`: Jupyter Notebook containing the code for generating poems.
- Model hosted on Hugging Face: [mehwish67/poem_Generator](https://huggingface.co/mehwish67/poem_Generator).

## How to Run
To quickly set up and run the project, follow these steps:

1. Clone the repository or access it through GitHub.
    ```bash
    git clone https://github.com/amandk1991/Poem-Generator.git
    ```
2. Open the `Poem-Generator.ipynb` file in **Google Colab** or **Jupyter Notebook**.

3. In the first cell of the notebook, install the required dependencies:
    ```python
    !pip install transformers
    ```

4. Next, load the fine-tuned GPT-2 model and tokenizer from Hugging Face. This will allow you to generate poems based on an input prompt.
    ```python
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    # Load the pre-trained model and tokenizer from Hugging Face
    model_name = "mehwish67/poem_Generator"  # Hugging Face model name
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Function to generate poems
    def generate_poem(prompt, max_length=50):
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
        poem = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return poem

    # Example usage
    print(generate_poem("A beautiful sunset"))
    ```

5. You can now generate poems by running the `generate_poem()` function with any prompt of your choice. For example:
    ```python
    print(generate_poem("A rainy day"))
    ```

6. Optionally, you can open the notebook directly in **Google Colab**:
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amandk1991/Poem-Generator/blob/main/Poem-Generator.ipynb)



## Acknowledgments
- **GPT-2 Model**: Pre-trained model from Hugging Face.
- **Poetry Dataset by Merve**: The dataset used to fine-tune the model.

Feel free to explore and modify the project as needed!
