import os
import sys
from dotenv import load_dotenv

load_dotenv()
# Add the project root to Python path for direct execution
project_root = os.getenv("PROJECT_ROOT")
sys.path.append(project_root)

from typing import Dict, List, Optional
from perplexia_ai.core.chat_interface import ChatInterface
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

"""Part 1 - Query Understanding implementation.

This implementation focuses on:
- Classify different types of questions
- Format responses based on query type
- Present information professionally
"""

class QueryUnderstandingChat(ChatInterface):
    """Week 1 Part 1 implementation focusing on query understanding."""
    
    def __init__(self):
        if not os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") == "":
            raise ValueError("GOOGLE_API_KEY is not set")
        
        self.llm = None
        self.query_classifier_prompt = None
        self.response_prompts = {}
    
    def initialize(self) -> None:
        """Initialize components for query understanding.
        
        Students should:
        - Initialize the chat model
        - Set up query classification prompts
        - Set up response formatting prompts
        """
        # TODO: Students implement initialization

        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        vector = embeddings.embed_query("hello, world!")
        print(f"Vector: {vector[:10]}")

        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

        # Prompt
        # prompt = PromptTemplate.from_template(
        #     "You are an assistant for question-answering tasks. Use the following pieces of retrieved "
        #     "context to answer the question. IMPORTANT: Always cite your sources by mentioning "
        #     "the document name, page number, and section number when providing information. Format your "
        #     "response in a structured json. \n"
        #     "json schema: {output_json_schema}\n"
        #     "If you don't know the answer, just say that you don't know. Don't make up any information. "
        #     "Use three sentences maximum and keep the answer concise.\n"
        #     "Question: {question} \n"
        #     "Context: {context} \n"
        #     "Don't mention the source in the answer. Put the sources in json schema mentioned above."
        # )

        prompt = PromptTemplate.from_template(
            "You are an assistant for classifying user queries into one of the following "
            "four categories: \n"
            "Factual Questions (What is?, Who invented?)\n"
            "Analytical Questions (How does?, Why do?)\n"
            "Comparison Questions (What's the difference between?)\n"
            "Definition Requests (Define?, Explain?)\n"
            "Question: {question} \n"
        )

        # self.query_classifier_prompt = PromptTemplate.from_template("Tell me a joke about {topic}")

        rag_chain = (
            {"question": RunnablePassthrough()}
            | prompt 
            | self.llm
        )
        question = input("Enter a question: ")
        print(rag_chain.invoke({"question": question}))
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message using query understanding.
        
        Students should:
        - Classify the query type
        - Generate appropriate response
        - Format based on query type
        
        Args:
            message: The user's input message
            chat_history: Not used in Part 1
            
        Returns:
            str: The assistant's response
        """
        # TODO: Students implement query understanding
        return "hello"

if __name__ == "__main__":
    chat = QueryUnderstandingChat()
    chat.initialize()
    chat.process_message("hello, world!")
