# 🤖SmartPilot:Neurosymbolic Customized and Compact CoPilot for Manufacturing

We introduce SmartPilot: 𝘼 𝘾𝙪𝙨𝙩𝙤𝙢, 𝘾𝙤𝙢𝙥𝙖𝙘𝙩 𝙖𝙣𝙙 𝙉𝙚𝙪𝙧𝙤𝙨𝙮𝙢𝙗𝙤𝙡𝙞𝙘 𝘼𝙄 𝙢𝙤𝙙𝙚𝙡 - The co-pilot that leverages a #custom #right_sized #neurosymbolic AI model to transform manufacturing processes.

Visit here for the demo: [Demo link](https://lnkd.in/eBJhiBDJ)

🔧 𝘾𝙪𝙨𝙩𝙤𝙢: Tailored to solve specific industry challenges (here focused on rare events in assembly processes), providing focused and practical solutions.

⚙️ 𝘾𝙤𝙢𝙥𝙖𝙘𝙩: Lightweight and cost-effective, optimized for real-time deployment on #edge devices.

🧠 𝙉𝙚𝙪𝙧𝙤𝙨𝙮𝙢𝙗𝙤𝙡𝙞𝙘: Integrates curated data, manufacturing knowledge, and human expertise (subject matters) for enhanced reliability and safety.

## Method and Applications Details

### Key Features for The SmartPilot

| **Feature**           | **Feature Description**                                                                                                                                                             | **Example and How we achieved?**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
|-------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Alignment**                       | Ensuring that AI systems are designed and operate consistently with the enterprise's goals, values, and ethics, while also providing metrics for evaluating alignment with objectives. | SmartPilot ensure the model's actions and decisions support the company's objectives. Metrics such as reduced downtime, fewer production losses, and improved machine efficiency ensure the anomaly prediction agent’s actions align with business objectives.                                                                                                                                                                                                                                                                                                                       |
| **Grounding**                       | Refers to connecting AI outputs to real-world concepts, data, and actions to ensure relevance and accuracy, which includes training AI on domain-specific data.                        | Analyses ingredients and cooking actions in the context of specific health guidelines like diabetes.                                                                                                 Nourich grounds its recommendations by incorporating health-specific contexts (like diabetes) into its analysis, mapping the data to real-world dietary needs. This ensures that the AI recommendations are relevant to the user’s health status, thereby improving decision-making around food choices based on grounded, domain-specific data.                                |
| **Instructability**                 | The ability of an AI system to accept instructions from users and adapt its behavior, often by allowing symbolic input to override statistical models when needed.                     | **SmartPilot**: Allows for domain experts in manufacturing to influence AI outputs by integrating human expertise for handling rare events in assembly processes.                                                 The SmartPilot system incorporates expert knowledge and provides mechanisms for human operators to adjust AI behavior when encountering rare events in the assembly process. This enables instructability by letting users guide the AI, ensuring decisions remain aligned with operational realities and domain-specific nuances in manufacturing.                |
| **Explainability and Interpretability** | The capability of an AI system to provide understandable reasons for its decisions or predictions, ensuring transparency for both technical and non-technical users.                    | Provides reasoning and explanations behind the suitability of a recipe, including ingredient alternatives and the reasoning behind recommendations.                                                  Nourich focuses on explainability by offering clear justifications for its decisions. For instance, if a recipe does not suit a user’s dietary guidelines, the system provides alternative ingredients and a detailed reasoning process. This aligns with the need for AI systems to be transparent and interpretable, especially in sensitive areas like health and nutrition. |
| **Safety**                          | Ensuring AI systems operate without causing unintended harm, including implementing fail-safe mechanisms and rigorous testing before deployment.                                      | Ensures that interventions are safe and reliable, following expert guidelines when supporting student behavior.                                                                                 MTSS-CoPilot emphasizes safety by adhering to carefully curated expert knowledge, ensuring that the recommendations it provides are safe, effective, and in line with educational and behavioral standards. It includes validation mechanisms to avoid unintended consequences in student support.                                                                                   |
| **Abstraction**                     | The process of simplifying complex real-world data and situations to allow AI systems to focus on relevant features for decision-making.                                               | **SmartPilot**: Handles complex scenarios such as rare manufacturing events by abstracting key data from historical and real-time sources for actionable insights.                                                SmartPilot utilizes abstraction by synthesizing real-time data and historical information, simplifying complex manufacturing scenarios (e.g., rare events) into actionable insights. This allows the system to focus on the relevant features of a process, offering a streamlined approach to problem-solving in complex environments like manufacturing.         |
| **Robustness**                      | The ability of AI systems to operate reliably in diverse, unpredictable, and adverse conditions while maintaining accuracy and stability.                                               | Built to adapt to various case complexities in behavioral management, ensuring the system can handle diverse, unpredictable student cases.                                                      The MTSS-CoPilot demonstrates robustness by being able to adapt to diverse and unpredictable student cases within the behavioral health context. Its capacity to handle different levels of intervention based on the unique needs of each student reflects the system's ability to manage uncertainty and maintain reliability in real-world scenarios.                             |
| **Custom, Neurosymbolic, and Compact**| Describes methods focused on creating AI systems that are tailored to specific industry needs, use neurosymbolic reasoning for explainability, and are optimized for lightweight deployment. | **Nourich, MTSS-CoPilot, and SmartPilot**: All systems are built as custom solutions tailored to specific industries, using neurosymbolic reasoning for explainability, and optimized for lightweight deployment.  Each copilot is custom-built to address specific industry needs, using neurosymbolic AI for enhanced reasoning and explanation. The systems are compact, meaning they are lightweight and optimized for real-time use on consumer-grade hardware, aligning with the principles of being cost-effective and accessible for real-world applications.                |


## 📰 Slides

## 💻 Preliminary Experiments - Neuro-Symbolic Question-Answer Knowledge Search System 

This experiment implements a **Question-Answer Knowledge Search System** using **neuro-symbolic AI** principles. The system processes queries, retrieves relevant questions from a user-provided QA dataset, and generates a knowledge panel with extracted entities and relationships. The architecture effectively combines **neural network-based** language understanding and **symbolic reasoning** for precise query answering.

### Key Features

- **Neuro-Symbolic Design**: Combines the strengths of neural methods (deep learning models for sentence embeddings) with symbolic reasoning (knowledge graph-based similarity).
- **User-Customizable**: The system dynamically adapts to the user's **QA dataset** in JSON format and extracts insights from the provided questions and answers.
- **Compact and Efficient**: Utilizes **lightweight models** such as `MiniLM-L6-v2` with relatively modest parameters, ensuring fast performance and lower computational costs.

### Architecture Overview

#### 1. **Neuro-Symbolic Design**
The system leverages a **neuro-symbolic approach**, which combines:
   - **Neural Networks**: We use **SentenceTransformers** to generate dense embeddings for the questions and queries. This neural model helps compute similarity scores between the query and dataset questions based on the semantic meaning of sentences.
   - **Symbolic Knowledge Representation**: The system uses **Spacy** to extract **subject-verb-object (SVO)** triples from both the query and the top matched answers. These triples represent relationships and entities in a structured, symbolic manner, forming a lightweight **knowledge graph**.

The combination of these two approaches ensures that the system can:
   - Understand the **semantic meaning** of queries and answers (neural).
   - Reason about the **explicit relationships** between entities (symbolic).

#### 2. **Customization to the User's QA Dataset**

The system is designed to be highly flexible and customizable:
   - It accepts any **JSON-formatted dataset** of questions and answers, which is dynamically processed during runtime. The dataset can include any domain-specific questions and answers, making it adaptable to various contexts (e.g., **tax assistance**, **medical support**, etc.).
   - The user can update the dataset with new questions and answers without modifying the codebase. The system automatically adjusts to the new content and generates results accordingly.

#### 3. **Compactness: Lightweight and Efficient Models**

Despite its powerful neuro-symbolic design, the system is highly efficient:
   - The **SentenceTransformer model** used in this system, `MiniLM-L6-v2`, is a compact and efficient transformer-based language model with only **33 million parameters**, significantly smaller than many other transformer models like BERT or GPT.
   - The symbolic reasoning component using **Spacy** operates efficiently, extracting relationships from text with minimal computational overhead.

### Design Components

#### Neural Component
- **SentenceTransformers** (`MiniLM-L6-v2`): This model is a **smaller variant** of transformer-based language models. It is optimized for efficiency and speed, making it a great choice for systems where performance and low resource consumption are important.
- The model generates **semantic embeddings** for both the user's query and the dataset questions. These embeddings allow the system to measure **semantic similarity** using **cosine similarity**.

#### Symbolic Component
- **Spacy**: We use Spacy to extract **SVO (subject-verb-object) triples** from the query and the matched answers. These triples are then compared to calculate a **knowledge-graph-based similarity** between the query and answers.
- This **symbolic reasoning** enhances the system's interpretability and allows it to handle **structural relationships** in the text.

### Query Filtering Mechanism

To ensure the system only provides relevant answers, a filtering mechanism is applied:
- **Neural Similarity**: Queries must pass a neural similarity threshold. If the neural similarity score between the query and the dataset questions is below a certain threshold (e.g., **0.6**), the system responds with a boilerplate message.
- **Knowledge Graph Similarity**: If the query passes the neural threshold, the system checks the similarity between **SVO triples** extracted from the query and the matched answer. If this knowledge-graph similarity score is below a threshold (e.g., **0.5**), the system returns a boilerplate response: *"Sorry, we don't have sufficient information to answer this query."*

This dual filtering ensures that only **relevant and accurate answers** are provided to the user.
