import streamlit as st
import openai
import requests
import os
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# Set API keys
openai.api_key = st.secrets["OPENAI_API_KEY"]
serper_api_key = st.secrets["SERPER_API_KEY"]

'''
st.title("Searching Whale")
st.image("Whale.png", width=300)
'''

# Define SerperApiSearchResults class
class SerperApiSearchResults:
    def __init__(self, api_key, num_results=5):
        self.api_key = api_key
        self.num_results = num_results

    def search(self, query):
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "q": query,
            "num": self.num_results
        }
        response = requests.post(url, json=payload, headers=headers)

        if response.status_code == 200:
            results = response.json()
            if 'organic' in results:
                return results['organic']
            else:
                return results
        else:
            response.raise_for_status()

# Initialize Serper API
serper_api = SerperApiSearchResults(api_key=serper_api_key, num_results=5)

# Define the OpenAI query function
def openai_query(prompt, model="gpt-3.5-turbo"):
    chat_openai = ChatOpenAI(model=model, openai_api_key=openai.api_key)
    messages = [
        [SystemMessage(content="You are a helpful assistant."), HumanMessage(content=f"Given the information: '{prompt}', can you summerize the given informations and provide a clear and concise conclusion in korean?")]
    ]
    response = chat_openai.generate(messages=messages)
    
    # Check if 'generations' is in response and it contains at least one item
    if hasattr(response, 'generations') and len(response.generations) > 0:
        return response.generations[0][0].text.strip()
    else:
        raise ValueError("Unexpected response format: {}".format(response))

def main():
    st.title("Searching Whale")
    st.image("Whale.png", width=300)
    
    # Input from user
    user_input = st.text_input("Enter your query:")
    
    if st.button("Search"):
        # Display the thinking chain
        st.write("### Thinking Chain")
        
        # Perform Serper search
        st.write("**Step 1: Performing Google Search...**")
        search_results = serper_api.search(user_input)
        
        # Display search results
        st.write("### Google Search Results")
        for result in search_results:
            st.write(f"**{result['title']}**")
            st.write(result['link'])
            st.write(result['snippet'])
        
        # Use LangChain to process the input
        st.write("**Step 2: Processing Input...**")
        llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai.api_key)
        prompt = PromptTemplate(template="{input}", input_variables=["input"])
        chain = LLMChain(llm=llm, prompt=prompt)
        
        react_result = chain.run(input=user_input)
        st.write(f"Chain Output: {react_result}")
        
        # Query OpenAI API with chain result
        st.write("**Step 3: Querying OpenAI API...**")
        openai_result = openai_query(react_result)
        
        # Display OpenAI result
        st.write("### OpenAI Response")
        st.write(openai_result)

if __name__ == "__main__":
    main()
