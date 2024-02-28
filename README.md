# 🤖 Data Science Coding Challenge -> Therapy Chatbot Fine-Tuning

## Research
Before augmenting the psychology data for the Fine Tuning, it was crucial to conduct a research on relevant papers and studies about the use and application of Therapy chatbots. This involved exploring the implications in mental health, forming hypotheses, considering ethical implications, and addressing problems that may arise. Fine-tuning large language models has inherent risks, especially for text generation and assisting technologies. As the LLMs could be trained on datasets including scraped websites with harmful and banned content, so the responses to prompts can contain toxic language. Another ethical aspect that must be considered when developing a psychological bot is who is responsible for the bot’s actions.

One of the key papers that informed our research is:
- Author(s): Lindgren, Helena and Sjöström, Jonas
- Title: Fine-tuning a Language Model using Reinforcement Learning from Human Feedback for a Therapy Chatbot Application
- URL: [Link to the paper](https://www.diva-portal.org/smash/get/diva2:1782678/FULLTEXT01.pdf)

## Database
Based on the research, it was decided to fine-tune the model with data from papers and studies that can demonstrate not just a technical improvement of the loss and accuracy, but also a well-being for the patient. This involved fine-tuning the LLM using the previous paper mentioned, that includes a set of questions and answers from Counsel Chat. Counsel Chat is a platform where users can contact verified therapists and ask mental health-related questions.

## Code

To run the provided code, follow these steps:

1. **Install the OpenAI Package:**
   ```shell
   !pip install openai

Initialize OpenAI Client:

from openai import OpenAI
client = OpenAI(api_key="ADD-KEY-HERE")
Mount Google Drive (Optional - Can be run in Google Colab):

from google.colab import drive
drive.mount('/content/drive')
Upload Data for Fine-Tuning:

client.files.create(
  file=open("mydata.jsonl", "rb"),
  purpose="fine-tune"
)
Initiate Fine-Tuning Job:

client.fine_tuning.jobs.create(
  training_file="file-cUoHgr6gCMpF8iXANaW05JUi",
  model="gpt-3.5-turbo"
)
Generate Chat Completions:

completion = client.chat.completions.create(
  model="ft:gpt-3.5-turbo-0613:aiteam::8x3qmPzI",
  messages=[
    {"role": "user", "content": "I want to cry"}
  ]
)
print(completion.choices[0].message)

Please note that you need to replace "ADD-KEY-HERE" with your actual OpenAI API key. Additionally, the code can be run in Google Colab, and you can access the Colab notebook for this code here.

## Results

## Performance
