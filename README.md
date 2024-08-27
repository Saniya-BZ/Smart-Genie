# Smart-Genie


SmartGenie is an innovative application designed to streamline exam preparation and enhance students' study efficiency through three distinct functionalities.
## PDF  Query:
This feature enables users to upload PDF documents containing study materials. Once uploaded, the PDFs are processed and split into manageable text chunks. Users can then interact with the content via a conversational interface powered by Groq’s ChatGroq API. This setup allows students to ask questions related to the uploaded material and receive contextually accurate answers, making it easier to grasp complex topics and find specific information.

## Summarize Video:
This functionality caters to users who want to quickly understand the content of YouTube videos or websites. By entering a URL and providing a Groq API key, users can fetch and summarize content from the specified source. The app extracts the essential information and presents it as a concise summary, saving users from having to watch lengthy videos or read extensive web pages.

## Question Paper Generation:
Designed to assist with exam preparation, this feature allows users to generate customized question papers based on their study materials. Users upload PDFs, and the app processes these documents to create questions focused on user-defined topics. This tailored approach helps students practice specific areas of study and test their knowledge effectively.
The app utilizes advanced language models and document processing techniques from langchain and Groq to deliver these functionalities. The interface is styled with a navy blue sidebar for easy navigation and a clean, user-friendly design to enhance the overall user experience. By integrating document retrieval, content summarization, and question generation, SmartGenie offers a comprehensive tool for efficient study and preparation.

## Software Requirements Specification
### Hardware
Computer or mobile device with internet access
Sufficient storage for uploading PDFs
Compatible with modern web browsers

### Software
Streamlit framework for the web interface
Groq API for natural language processing
LangChain for document and video content processing
Python programming languageLibraries: HuggingFace Embeddings, Chroma, PyPDFLoader, RecursiveCharacterTextSplitter, etc.


