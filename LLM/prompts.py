from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

llm=HuggingFacePipeline.from_model_id(
    model_id='Qwen/Qwen2.5-7B-Instruct',
    task='text-generation',
    pipeline_kwargs=dict(max_new_tokens=1000)
)

model=ChatHuggingFace(llm=llm)



user_input=input("Travel Planner: What is your travel destination, duration, interests (eg: food, history, hiking), and style of stay(eg: beach, mountain, city,relaxed,active,luxury)? (Type 'exit' or 'quit' to end the chat): ").split()
Destination_input, Duration_input, Interests_input, Style_input = user_input



# Creating a prompt template 
template = PromptTemplate(
    template= """Act as a professional travel planner. 
    Create a detailed {Duration_input}-day itinerary for {Destination_input} focusing on {Interests_input}. Style of stay :{Style_input}
    **Output Requirements:**

    Provide a structured plan including:
    1. **Daily Schedule:** Time-blocked activities (Morning/Afternoon/Evening) with specific locations to visit in {Destination_input} with specific breakdown on {Duration_input} days in which day to do and go where. 
    2. **Logistics:** Recommended local transportation, accommodations in {Destination_input}. 
    3. **Dining:** 2-3 specific restaurant or food recommendations in {Destination_input} per day breakdown aligned with interests. 
    4. **Practical Info:** Key cultural tips of {Destination_input}, Specific emergency contacts of {Destination_input}, and a packing list for the season.
    5. **Additional Notes:** Any special instructions or recommendations for the trip.""",
    input_variables=['Duration_input','Destination_input','Intrest_input','Style_input']

)

prompt = template.invoke({
    'Duration_input': Duration_input,
    'Destination_input': Destination_input,
    'Interests_input': Interests_input,
    'Style_input': Style_input
})

result=model.invoke(prompt)
print(result.content)

# , e.g., food, history, hiking]