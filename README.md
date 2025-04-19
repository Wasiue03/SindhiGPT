# Sindhi GPT Language Model

A transformer-based language model for Sindhi text generation, built with PyTorch. This model can generate coherent Sindhi text, autocomplete sentences, and assist in Sindhi NLP tasks.

## Features

- Trained on Sindhi text corpus
- Handles Right-to-Left (RTL) script properly
- Character-level tokenization for complete Sindhi character support
- Adjustable context window and generation parameters

## Prerequisites

- Python 3.7+
- PyTorch
- Pandas

```bash
pip install torch pandas
```
1. Data Preparation
Place your Sindhi text data in one of these formats:

Plain text file (sindhi_text.txt)

CSV file with a 'text' column (sindhi_articles.csv)

Example text file:

سنڌي ٻولي سنڌ جي ثقافت جو اهم حصو آهي.
سنڌ جي تاريخ وڏي پراڻي آهي.
2. Training the Model
python
from sindhi_gpt import train_model

# Path to your Sindhi text data
data_path = "sindhi_text.txt"  

# Train the model
model = train_model(
    data_path=data_path,
    batch_size=32,
    block_size=128,
    max_iters=8000
)
3. Generating Text
python
from sindhi_gpt import generate_sindhi

# Generate from prompt
prompt = "سنڌي ٻولي"
generated_text = generate_sindhi(
    model, 
    prompt, 
    max_new_tokens=100,
    temperature=0.8
)

print(generated_text)
Example Output:

سنڌي ٻولي سنڌ جي ثقافت جو اهم حصو آهي. هي ٻولي سنڌ جي تاريخ سان گڏوگڏ ان جي روايتن ۽ ثقافت کي به ظاهر ڪري ٿي.
 - Parameter	Default	Description
 - batch_size	32	Number of sequences processed in parallel
 - block_size	128	Context window size
 - max_iters	8000	Training iterations
 - learning_rate	2e-4	Optimizer learning rate
 - temperature	0.8	Generation creativity (0.1-1.0)

# Contributing
Contributions are welcome! Please open an issue or pull request for:

Additional Sindhi text datasets
Model architecture improvements
Bug fixes
