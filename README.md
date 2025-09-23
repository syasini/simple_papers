# Simple Papers

![Simple Papers Logo](media/logo_github.png)

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Bedrock](https://img.shields.io/badge/AWS%20Bedrock-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)](https://aws.amazon.com/bedrock/)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com/)
[![ElevenLabs](https://img.shields.io/badge/ElevenLabs-000000?style=for-the-badge&logo=elevenlabs&logoColor=white)](https://elevenlabs.io/)
[![agentic-doc](https://img.shields.io/badge/agentic--doc-4285F4?style=for-the-badge&logo=python&logoColor=white)](https://pypi.org/project/agentic-doc/)

A better way to read scientific papers. 

Simple Papers transforms complex academic papers into digestible, simplified content with AI-powered explanations, interactive annotations, and audio narration.

Check out the app [here](https://simplepapers.streamlit.app/)!

## ✨ Features

- **📄 Smart Paper Simplification**: Makes complex academic papers easy to understand
- **🎯 Interactive Annotations**: Click on highlighted sections for simplified explanations
- **🔊 Audio Narration**: Listen to paper sections with multiple voice options
- **🔍 Keyword Extraction**: Automatically extract and link technical terms to Wikipedia
- **📱 Two Usage Modes**: Use online with pre-parsed papers or locally with your own PDFs

## 📚 Available Papers

- [Attention Is All You Need](https://simplepapers.streamlit.app/?paper=1706.03762v7)
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://simplepapers.streamlit.app/?paper=2210.03629)
- [XGBoost: A Scalable Tree Boosting System](https://simplepapers.streamlit.app/?paper=1603.02754v3)
- [Latent Dirichlet Allocation](https://simplepapers.streamlit.app/?paper=jair03-lda)
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://simplepapers.streamlit.app/?paper=2201.11903v6)
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://simplepapers.streamlit.app/?paper=2112.10752v2)
- [A Unified Approach to Interpreting Model Predictions](https://simplepapers.streamlit.app/?paper=1705.07874v2)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://simplepapers.streamlit.app/?paper=1810.04805v2)
- [A Practical Guide for Evaluating LLMs and LLM-Reliant Systems](https://simplepapers.streamlit.app/?paper=2506.13023v2)
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://simplepapers.streamlit.app/?paper=1908.10084v1)
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://simplepapers.streamlit.app/?paper=2005.11401v4)

## 🛠️ Technologies Used

- [AWS Bedrock](https://aws.amazon.com/bedrock/) - AI foundation models 
- [LangChain](https://langchain.com/) - Framework for building AI applications
- [agentic-doc](https://pypi.org/project/agentic-doc/) - Document parsing and extraction
- [ElevenLabs](https://elevenlabs.io/) - AI voice synthesis for audio narration

## ⚙️ How It Works


Simple Papers can be used in two ways:

**🌐 Online Mode (Streamlit Community Cloud)**
- Access pre-parsed and simplified academic papers
- No setup required - just click and read
- Perfect for exploring research without the complexity

**💻 Local/Dev Mode**
- Parse and simplify your own PDF papers
- Full control over processing and customization
- Ideal for researchers working with specific documents
- **Note**: Currently a proof-of-concept that requires multiple API keys for the end-to-end workflow. This complexity will be streamlined if the app gains sufficient user engagement and traction.

## 🚀 Installation

```bash
git clone https://github.com/syasini/simple_papers.git
cd simple_papers
pip install -e .
```

## 🏃 Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501` and start simplifying papers!

## 📋 Requirements

- Python 3.12+
- Streamlit
- API keys for local processing (see `secrets.toml.example` for setup details)
- AWS credentials (for advanced features)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.