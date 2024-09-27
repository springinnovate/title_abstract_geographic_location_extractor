# Start from the base image
FROM therealspring/convei_abstract_classifier:latest

# Set environment variables (if needed)
ENV PYTHONUNBUFFERED=1

# Install openpyxl and transformers
RUN pip install openpyxl transformers

# Pre-download the NER model and cache it
RUN python -c "from transformers import AutoModelForTokenClassification, AutoTokenizer; AutoModelForTokenClassification.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english'); AutoTokenizer.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english')"

RUN pip install geograpy3
RUN pip install lxml_html_clean

RUN python -c "import nltk; nltk.downloader.download('maxent_ne_chunker'); nltk.downloader.download('words'); nltk.downloader.download('treebank'); nltk.downloader.download('maxent_treebank_pos_tagger'); nltk.downloader.download('punkt'); nltk.downloader.download('averaged_perceptron_tagger'); nltk.downloader.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng'); nltk.download('maxent_ne_chunker_tab')"

# Optionally set the working directory
WORKDIR /workspace
