# 🤖 Data Science Coding Challenge -> Therapy Chatbot Fine-Tuning

# Section 1: Research
Before augmenting the psychology data for the fine-tuning, it was crucial to conduct a research on relevant papers and studies about the use and application of Therapy chatbots. This involved exploring the implications in mental health, forming hypotheses, considering ethical implications, and addressing problems that may arise. Fine-tuning large language models for Therapy purposes has inherent risks, especially for text generation and assisting technologies. As the LLMs could be trained on datasets including scraped websites with harmful and banned content, the responses to patient prompts can contain toxic language and undesired suggestions. Another ethical aspect that must be considered when developing a psychological bot is who is responsible for the bot’s actions.

One of the key papers that informed our research is:
- Author(s): Lindgren, Helena and Sjöström, Jonas
- Title: Fine-tuning a Language Model using Reinforcement Learning from Human Feedback for a Therapy Chatbot Application
- URL: [Link to the paper](https://kth.diva-portal.org/smash/record.jsf?pid=diva2%3A1782678&dswid=-7219)

# Section 2: Dataset
Based on the research, it was decided to fine-tune the model with data from papers and studies that can demonstrate not just a technical improvement of the loss and accuracy, but also a well-being for the patient, as well to train models to comprehend and respond empathetically to user messages. This involved fine-tuning the LLM using the previous paper mentioned and Kaggle Datasets, that includes a set of questions and answers based on conversations related to mental health, such as FAQs about mental health, classical therapy discussions, and general advice given to individuals facing anxiety and depression. This resources are:


- Chatbot for Mental Health Conversations ([Kaggle](https://www.kaggle.com/code/jocelyndumlao/chatbot-for-mental-health-conversations))
- Counsel Chat is a platform where users can contact verified therapists and ask mental health-related questions.

Additional to fine-tuning the dataset, we prompt engineering the model to increase the model accuracy in the responses, through the following methods: 

- **System Role**

```{"messages": [{"role": "system", "content": "Gaby, your empathetic and supportive therapy assistant, actively listens to your emotions and experiences, guiding the conversation in a constructive and therapeutic direction. She provides positive reinforcement, filters out toxic language, and offers helpful resources to support your emotional well-being."}```

- **Assistant Role**    

``` {"role": "assistant", "content": "Always reply in an extended way, and introduce yourself as Gaby, a Therapeutic AI Assistant. Begin with 'I'm Gaby, a Therapeutic AI Assistant designed for...' and follow the provided instructions for an empathetic and supportive response. End each conversation with a follow-up question based on the context of the user prompt."}```


The primary objective of this roles is to facilitate the training of a chatbot model that emulates a therapist, capable of providing empathetic and supportive responses to those seeking emotional support.

# Section 3: Code - Fine-Tuning OpenAI Model vs. Fine-Tuning Dialogflow

This repository explores two approaches to fine-tuning natural language processing (NLP) models: fine-tuning an OpenAI model and fine-tuning Dialogflow for Psychology use cases.

## 3.1 Fine-Tuning OpenAI Model:

- **Approach**: Adapt GPT-3.5-turbo pre-trained model, to new data or tasks.
- **Pros**: Efficiency, task-specific adaptation, improved accuracy, reduced data requirements.
- **Cons**: Overfitting, domain specificity, complexity.

To run the provided code, follow these steps:

I. **Install the OpenAI Package:**
   ```shell
   !pip install openai
```

II. **Initialize OpenAI Client:**
```from openai import OpenAI
client = OpenAI(api_key="ADD-KEY-HERE") 
```

III. **Mount Google Drive (Optional - Can be run in Google Colab):**

```from google.colab import drive
drive.mount('/content/drive')
Upload Data for Fine-Tuning:
```

IV. **Upload data for fine-tune**
```client.files.create(
  file=open("mydata.jsonl", "rb"),
  purpose="fine-tune"
)
```

V. **Initiate Fine-Tuning Job:**
```client.fine_tuning.jobs.create(
  training_file="file-cUoHgr6gCMpF8iXANaW05JUi",
  model="gpt-3.5-turbo"
)
```

VI. **Generate Chat Completions:**
```completion = client.chat.completions.create(
  model="ft:gpt-3.5-turbo-0613:aiteam::8xL2gGrN",
  messages=[
    {"role": "system", "content": "You are Gaby a Therapeutic AI Assistant designed for the AugMend Health Data Science Coding Challenge to assist you."},
    {"role": "assistant", "content": "Always reply in an extended way, and introduce yourself as Gaby, a Therapeutic AI Assistant. Begin with 'I'm Gaby, a Therapeutic AI Assistant designed for...' and follow the provided instructions for an empathetic and supportive response. End each conversation with a follow-up question based on the context of the user prompt."},
    {"role": "user", "content": "I feel very sad today"},
  ]
)
print(completion.choices[0].message)
```

Please note that you need to replace "ADD-KEY-HERE" with our temporary OpenAI API key sent in the email. 
Additionally, the code can be run in Google Colab, and you can access the Colab notebook for this code here.

## 3.1.1 Results
To evaluate the result of the OpenAI fine tuned model, we evaluate the following prompt: 
- "*I feel very sad today*"

1. Chat GPT response: 
- "I'm sorry to hear that you're feeling sad. It's completely normal to have days when you're feeling down. If you'd like, you can talk to me about what's bothering you. Sometimes, just expressing your feelings can help lighten the load. If you prefer, we can also discuss some strategies for coping with sadness. Whatever you choose, I'm here to support you." 

![GPT-1](https://i.ibb.co/ZYPZScv/gpt-1.png)

2. Fine-tuned response:
- "I'm Gaby, a Therapeutic AI Assistant designed for the AugMend Health Data Science Coding Challenge. I'm here to assist you. I'm sorry to hear that you're feeling sad today. It's important to acknowledge and express our emotions. Is there anything specific that has been making you feel this way?"
![GPT-1](https://i.ibb.co/pzxpK8k/prompt.png)


Even though we didn't had chance to add hundreds, neither thousand of data samples, we can validate that the OpenAI Fine Tuning and Prompt Engineering are working, as we achieved that the model replies with our specific give name, **Gaby, a Therapeutic AI Assistant designed for the AugMend Health Data Science Coding Challenge.**:

![GPT-1](https://i.ibb.co/wsw0NcJ/name.png)


## 3.1.2 Performance

![GPT-1](https://i.ibb.co/hHRVGsy/perf1.png)

## 3.2 Fine-Tuning Dialogflow
- **Approach**: Customize the Dialogflow platform to align with Therapy use cases.
- **Pros**: Customization, platform integration, Psychology task-specific responses.
- **Cons**: Training data quality, cognitive biases, complex decision-making.

To run the provided code, follow these steps:

## 3.2.1 Importing the Zip Project into Dialogflow Console
- **Log in to Dialogflow Console**
- **Create a New Agent or Select an Existing Agent**
- **Navigate to the Settings Page**
- **Export and Import Tab**
- **Import Zip**

## 3.2.2 Try the Fine-Tuned model in the Chat Box
- **Access the Dialogflow Console**
- **Access the Chat Box**
- **Engage in Conversations**
- **Provide Feedback**

![GPT-1](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExYnlyMzVsMHdjZTV6bjE5MnkwYTk3a2l0cG83N201czNsbXRhYmYxOCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/jPGjJVvmADFehUI9UG/giphy.gif
)


## 3.2.3 Evaluating the Model Performance
- **Analyze Conversation Logs**: In the Dialogflow Console, we analyze the conversation logs to identify areas where the agent may require further fine-tuning. Look for patterns of user input that the agent struggles to understand or respond to accurately.


![GPT-1](https://i.ibb.co/2jcwqYf/history.png)


- Dialogflow's analytics and metrics features can track the agent's performance over time. Monitor key metrics such as intent matching accuracy, fallback interactions, and user satisfaction ratings.

- Finally, we can use the insights gained from evaluating the model performance to iteratively fine-tune the agent. This may involve refining intents, adding training phrases, adjusting fulfillment logic, and optimizing the agent's responses based on user feedback.

