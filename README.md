# 🤖 Data Science Coding Challenge -> Therapy Chatbot Fine-Tuning

# Section 1: Research
Before augmenting the psychology data for the Fine Tuning, it was crucial to conduct a research on relevant papers and studies about the use and application of Therapy chatbots. This involved exploring the implications in mental health, forming hypotheses, considering ethical implications, and addressing problems that may arise. Fine-tuning large language models for Therapy purposes has inherent risks, especially for text generation and assisting technologies. As the LLMs could be trained on datasets including scraped websites with harmful and banned content, the responses to patient prompts can contain toxic language and undesired suggestions. Another ethical aspect that must be considered when developing a psychological bot is who is responsible for the bot’s actions.

One of the key papers that informed our research is:
- Author(s): Lindgren, Helena and Sjöström, Jonas
- Title: Fine-tuning a Language Model using Reinforcement Learning from Human Feedback for a Therapy Chatbot Application
- URL: [Link to the paper](https://www.diva-portal.org/smash/get/diva2:1782678/FULLTEXT01.pdf)

# Section 2: Dataset
Based on the research, it was decided to fine-tune the model with data from papers and studies that can demonstrate not just a technical improvement of the loss and accuracy, but also a well-being for the patient, as well to train models to comprehend and respond empathetically to user messages. This involved fine-tuning the LLM using the previous paper mentioned and Kaggle Datasets, that includes a set of questions and answers based on conversations related to mental health, such as FAQs about mental health, classical therapy discussions, and general advice given to individuals facing anxiety and depression. This resources are:

- Counsel Chat is a platform where users can contact verified therapists and ask mental health-related questions.
- Patient Health Questionnaire-9
- Chatbot for Mental Health Conversations ([Kaggle](https://www.kaggle.com/code/jocelyndumlao/chatbot-for-mental-health-conversations))

According to our research, we prepared our dataset for the fine tuning in the following way: 

- **System Role**

```{"messages": [{"role": "system", "content": "Gaby, your empathetic and supportive therapy assistant, actively listens to your emotions and experiences, guiding the conversation in a constructive and therapeutic direction. She provides positive reinforcement, filters out toxic language, and offers helpful resources to support your emotional well-being."}```

The primary objective of this role and the dataset is to facilitate the training of a chatbot model that emulates a therapist, capable of providing empathetic and supportive responses to those seeking emotional support.

# Section 3: Code - Fine-Tuning OpenAI Model vs. Fine-Tuning Dialogflow

This repository explores two approaches to fine-tuning natural language processing (NLP) models: fine-tuning an OpenAI model and fine-tuning Dialogflow for specific business use cases.

## 3.1 Fine-Tuning OpenAI Model:

- **Approach**: Adapt GPT-3.5 pre-trained model, to new data or tasks.
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
  model="ft:gpt-3.5-turbo-0613:aiteam::8x3qmPzI",
  messages=[
    {"role": "user", "content": "I want to cry"}
  ]
)
print(completion.choices[0].message)
```

Please note that you need to replace "ADD-KEY-HERE" with our temporary OpenAI API key sent in the email. 
Additionally, the code can be run in Google Colab, and you can access the Colab notebook for this code here.

## 3.1.1 Results
To evaluate the result of the fine tuned model, we evaluate the following prompt: 
- "*I feel very sad today*"

1. Chat GPT response:
![GPT-1](https://i.ibb.co/ZYPZScv/gpt-1.png)

2. Fine-tuned response:
![GPT-1](https://i.ibb.co/qjbnzpX/finetun-2.png)

## 3.1.2 Performance

![GPT-1](https://i.ibb.co/hHRVGsy/perf1.png)

![GPT-1](https://i.ibb.co/yshpKsq/perf2.png)

## 3.2 Fine-Tuning Dialogflow
- **Approach**: Customize the Dialogflow platform to align with specific business use cases.
- **Pros**: Customization, platform integration, user understanding, task-specific responses.
- **Cons**: Training data quality, cognitive biases, complex decision-making.

To run the provided code, follow these steps:

## 3.2.1 Importing the Zip Project into Dialogflow Console
- **Log in to Dialogflow Console**: Go to the Dialogflow Console and log in with your credentials.
- **Create a New Agent or Select an Existing Agent**: If you don't have an existing agent, create a new agent. Otherwise, select the existing agent where you want to import the zip project.
- **Navigate to the Settings Page**: In the left navigation bar, click on the gear icon to navigate to the settings page of the selected agent.
- **Export and Import Tab**: Click on the "Export and Import" tab in the settings page.
- **Import Zip**: Click on the "Import Zip" button and select the zip project file to import into the agent. Follow the on-screen instructions to complete the import process.

## 3.2.2 Fine-Tuning in the Chat Box
- **Access the Dialogflow Console**: Navigate to the Dialogflow Console and open the agent where you imported the zip project.
- **Access the Chat Box**: Use the built-in chat box or integrate the agent with a supported platform to access the chat interface.
- **Engage in Conversations**: Engage in conversations with the agent by typing or speaking input into the chat box. This allows you to fine-tune the agent by providing real user inputs and evaluating its responses.
- **Provide Feedback**: During the conversations, provide feedback on the agent's responses to help fine-tune its performance. This feedback can include correcting misunderstood intents, providing additional training phrases, and evaluating the relevance of responses.

## 3.2.3 Evaluating the Model Performance
- **Review User Interactions**: Review the user interactions with the agent to assess its performance in understanding and responding to user inputs.
- **Analyze Conversation Logs**: In the Dialogflow Console, analyze the conversation logs to identify areas where the agent may require further fine-tuning. Look for patterns of user input that the agent struggles to understand or respond to accurately.
- **Utilize Analytics and Metrics**: Leverage Dialogflow's analytics and metrics features to track the agent's performance over time. Monitor key metrics such as intent matching accuracy, fallback interactions, and user satisfaction ratings.
- **Iterative Fine-Tuning**: Use the insights gained from evaluating the model performance to iteratively fine-tune the agent. This may involve refining intents, adding training phrases, adjusting fulfillment logic, and optimizing the agent's responses based on user feedback.

