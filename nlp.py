import re
import spacy
from spacy import displacy
import streamlit as st
import fitz
import string
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown
import os
import pathlib
import textwrap
import PIL.Image
from sklearn.feature_extraction.text import CountVectorizer
import gensim
import gensim.corpora as corpora
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter

# Custom stopwords list
custom_stopwords = {"http", "https", "www"}

# Function to remove emojis using regex
def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"  # dingbats
        u"\U000024C2-\U0001F251"  # enclosed characters
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

def main():
    st.title("GalvatronAI NLP")

    # File uploader for PDF files
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        # Extract text from the PDF file using PyMuPDF (fitz)
        filename = uploaded_file.name  # Get the filename
        pdf_content = uploaded_file.read()  # Read the file content as bytes

        with fitz.open(stream=pdf_content, filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()

        # Load SpaCy model
        nlp = spacy.load("en_core_web_sm")  # Replace with your desired model
        doc = nlp(text)

        # Render the named entity visualization as HTML
        html = displacy.render(doc, style="ent", jupyter=False)

        # Custom CSS to make the result scrollable within a box
        st.markdown("<h2 style='text-align: center; color: black;'>Named Entity Recognition</h2>", unsafe_allow_html=True)
        st.markdown("""
            <style>
            .scroll-box {
                max-height: 400px; /* Set the max height of the box */
                overflow-y: scroll; /* Enable vertical scrolling */
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: #f9f9f9;
            }
            </style>
            """, unsafe_allow_html=True)

        # Place the rendered HTML inside the scrollable box for NER
        st.markdown(f'<div class="scroll-box">{html}</div>', unsafe_allow_html=True)

        # After NER: Remove stopwords, custom stopwords, numbers, symbols, and emojis from the text
        filtered_tokens = [
            token.text.lower() for token in doc  # Convert token to lowercase
            if not token.is_stop  # SpaCy's default stopwords
            and token.text.lower() not in custom_stopwords  # Custom stopwords like "http", "https"
            and not token.is_punct  # Remove punctuation
            and not token.is_digit  # Remove numbers
            and token.text not in string.punctuation  # Remove special punctuation symbols
            and not re.match(r'[^\w\s]', token.text)  # Remove symbols like #$%^&
            and token.is_alpha  # Keep only alphabetic tokens
        ]

        # Recreate a string from the filtered tokens and remove emojis
        filtered_text = " ".join(filtered_tokens)
        filtered_text = remove_emojis(filtered_text)

        # Generate WordCloud
        st.markdown("<h2 style='text-align: center; color: black;'>WordCloud Visualization</h2>", unsafe_allow_html=True)
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(filtered_text)

        # Display WordCloud using Matplotlib
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")  # Disable axes
        st.pyplot(fig)
        fig.savefig("wordcloud.png")

        def to_markdown(text):
            text = text.replace('•', '  *')
            return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

        genai.configure(api_key="AIzaSyCFI6cTqFdS-mpZBfi7kxwygewtnuF7PfA")

        img = PIL.Image.open("wordcloud.png")
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(["You are a professional Data Analyst, write the complete conclusion and actionable insight based on the image", img], stream=True)
        response.resolve()
        st.write(response.text)



        def get_top_ngrams(corpus, n=2, top_n=10):
            vec = CountVectorizer(ngram_range=(n, n), stop_words='english').fit(corpus)
            bag_of_words = vec.transform(corpus)
            sum_words = bag_of_words.sum(axis=0) 
            words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
            words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
            return words_freq[:top_n]


        # Generate top 10 bigrams
        corpus = [filtered_text]
        top_bigrams = get_top_ngrams(corpus, n=2, top_n=10)

        # Display top 10 bigrams as a bar chart
        st.markdown("<h2 style='text-align: center; color: black;'>Top 10 Bigrams</h2>", unsafe_allow_html=True)
        bigrams, freq = zip(*top_bigrams)
        fig, ax = plt.subplots()
        ax.barh(bigrams, freq, color='skyblue')
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Bigrams")
        ax.set_title("Top 10 Bigrams")
        ax.invert_yaxis()  # Invert y-axis to show the highest bar at the top
        st.pyplot(fig)
        fig.savefig("bigram.png")


        def to_markdown(text):
            text = text.replace('•', '  *')
            return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

        genai.configure(api_key="AIzaSyCFI6cTqFdS-mpZBfi7kxwygewtnuF7PfA")

        img = PIL.Image.open("bigram.png")
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(["You are a professional Data Analyst, write the complete conclusion and actionable insight based on the image", img], stream=True)
        response.resolve()
        st.write(response.text)



        # Topic Modeling with LDA
        st.markdown("<h2 style='text-align: center; color: black;'>Topic Modeling Visualization</h2>", unsafe_allow_html=True)

        # Create a dictionary and corpus for LDA
        dictionary = corpora.Dictionary([filtered_tokens])
        corpus = [dictionary.doc2bow(text) for text in [filtered_tokens]]

        # Run LDA topic modeling
        num_topics = 3  # Set the number of topics
        lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

        # Display topics
        topics = lda_model.print_topics(num_words=5)
        #for idx, topic in topics:
            #st.write(f"**Topic {idx}:** {topic}")

        # Prepare data for visualization
        topic_words = []
        for idx, topic in lda_model.show_topics(formatted=False):
            topic_words.append([word[0] for word in topic])

        # Create bar plots for each topic
        fig, axes = plt.subplots(nrows=num_topics, figsize=(15, 6 * num_topics))
        for i, ax in enumerate(axes.flatten()):
            words, weights = zip(*lda_model.show_topic(i, topn=5))
            ax.barh(words, weights, color='skyblue')
            ax.set_title(f'Topic {i + 1}')
            ax.invert_yaxis()  # Invert y-axis to show the highest bar at the top

        st.pyplot(fig)
        fig.savefig("lda_topics.png")


        def to_markdown(text):
            text = text.replace('•', '  *')
            return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

        genai.configure(api_key="AIzaSyCFI6cTqFdS-mpZBfi7kxwygewtnuF7PfA")

        img = PIL.Image.open("lda_topics.png")
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(["You are a professional Data Analyst, write the complete conclusion and actionable insight based on the image", img], stream=True)
        response.resolve()
        st.write(response.text)


        # Enforce splitting into smaller chunks using CharacterTextSplitter
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key="AIzaSyCFI6cTqFdS-mpZBfi7kxwygewtnuF7PfA")
        # Now, handle zero-shot capability by querying the LLM directly without any retrieved documents
        point_template = """
        Summarize the key points from the document in bullet form.

        Document: "{text}"

        Summary (use bullet points):
        """
        point_prompt = PromptTemplate(input_variables=["text"], template=point_template)
        point_llm_chain = LLMChain(llm=llm, prompt=point_prompt)

        # Make sure to pass `text` to the run function
        point_response = point_llm_chain.run(text=text)

        # Display the zero-shot response
        st.markdown("### Key Points based on the Documents")
        st.write(point_response)




if __name__ == "__main__":
    main()
